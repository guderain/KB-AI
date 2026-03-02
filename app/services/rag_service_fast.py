import hashlib
import json
import logging
import re
import time
from html import unescape as html_unescape
from typing import Iterator
from urllib.parse import parse_qs, unquote, urlparse

from langchain_core.documents import Document

from app.core.config import get_settings
from app.db.postgres import SessionLocal
from app.models.chat_log import ChatLog
from app.services.dependencies import get_llm, get_redis_client, get_vector_store
from app.services.safety_service import (
    audit_model_output,
    audit_user_question,
    prompt_injection_guard_instruction,
    sanitize_untrusted_context,
)

logger = logging.getLogger(__name__)

UNKNOWN_MARKERS = (
    "i don't know",
    'i do not know',
    'unknown',
    'not sure',
    '我不知道',
    '不知道',
    '不清楚',
    '无法确定',
)


def _trim_doc_text(text: str, limit: int) -> str:
    clean = ' '.join(text.split())
    if len(clean) <= limit:
        return clean
    return clean[:limit] + ' ...'


def _build_prompt(question: str, docs: list[Document], context_chars_per_doc: int) -> str:
    context_lines: list[str] = []
    for idx, doc in enumerate(docs, start=1):
        source = doc.metadata.get('source', 'unknown')
        excerpt = sanitize_untrusted_context(_trim_doc_text(doc.page_content, context_chars_per_doc))
        context_lines.append(f'[{idx}] source={source}\n{excerpt}')
    context = '\n\n'.join(context_lines)
    guard_rule = prompt_injection_guard_instruction()
    return (
        f'{guard_rule}'
        '你是知识库问答助手。必须使用简体中文回答。'
        '仅基于给定上下文作答；若上下文不足，请只回复：我不知道。'
        '回答保持简洁，并在句末标注引用编号，如 [1][2]。\n\n'
        f'问题：{question}\n\n'
        f'上下文：\n{context}'
    )


def _build_web_prompt(question: str, web_results: list[dict[str, str]]) -> str:
    references: list[str] = []
    for idx, item in enumerate(web_results, start=1):
        title = sanitize_untrusted_context(item.get('title', 'untitled'))
        url = item.get('url', '')
        snippet = sanitize_untrusted_context(item.get('snippet', ''))
        references.append(f'[{idx}] title={title}\nurl={url}\nsnippet={snippet}')

    context = '\n\n'.join(references)
    guard_rule = prompt_injection_guard_instruction()
    return (
        f'{guard_rule}'
        '你是联网检索问答助手。必须使用简体中文回答。'
        '请基于提供的网页摘要回答问题，优先给出事实性、直接的结论。'
        '若信息冲突，请简要说明不确定性。'
        '答案末尾附上引用编号及对应URL。\n\n'
        f'问题：{question}\n\n'
        f'网页摘要：\n{context}'
    )


def _cache_key(question: str) -> str:
    normalized_question = ' '.join(question.strip().split())
    digest = hashlib.md5(normalized_question.encode('utf-8')).hexdigest()
    return f'qa:{digest}'


def _looks_unknown(answer: str) -> bool:
    normalized = ' '.join(answer.lower().split())
    return any(marker in normalized for marker in UNKNOWN_MARKERS)


def _normalize_cn_answer(answer: str) -> str:
    if not answer:
        return answer
    if _looks_unknown(answer):
        return '我不知道'
    return answer


def _clean_html_text(raw: str) -> str:
    text = re.sub(r'<[^>]+>', '', raw or '')
    text = html_unescape(text)
    return ' '.join(text.split()).strip()


def _query_tokens(query: str) -> list[str]:
    q = (query or '').lower()
    words = [w for w in re.split(r'[^0-9a-z\u4e00-\u9fff]+', q) if len(w) >= 2]
    chinese = ''.join(ch for ch in q if '\u4e00' <= ch <= '\u9fff')
    bigrams = [chinese[i : i + 2] for i in range(len(chinese) - 1)]
    # Keep uniqueness but preserve order.
    seen: set[str] = set()
    tokens: list[str] = []
    for t in words + bigrams:
        if t not in seen:
            seen.add(t)
            tokens.append(t)
    return tokens


def _result_score(query: str, item: dict[str, str]) -> int:
    hay = f"{item.get('title', '')} {item.get('snippet', '')}".lower()
    score = 0
    for token in _query_tokens(query):
        if token and token in hay:
            score += 2 if len(token) >= 4 else 1
    return score


def _filter_relevant_results(query: str, results: list[dict[str, str]]) -> list[dict[str, str]]:
    if not results:
        return []
    scored = [(item, _result_score(query, item)) for item in results]
    best = max(score for _, score in scored)
    # If all scores are 0, treat as irrelevant provider output and fallback.
    if best <= 0:
        return []
    # Keep reasonably related results.
    return [item for item, score in scored if score > 0]


def _parse_bing_results(html: str, max_results: int) -> list[dict[str, str]]:
    blocks = re.findall(r'<li[^>]*class="[^\"]*b_algo[^\"]*"[^>]*>(.*?)</li>', html, flags=re.IGNORECASE | re.DOTALL)
    results: list[dict[str, str]] = []
    seen: set[str] = set()
    for block in blocks:
        link_match = re.search(r'<h2[^>]*>\s*<a[^>]*href="([^\"]+)"[^>]*>(.*?)</a>', block, flags=re.IGNORECASE | re.DOTALL)
        if not link_match:
            continue
        url = link_match.group(1).strip()
        if not (url.startswith('http://') or url.startswith('https://')):
            continue
        if url in seen:
            continue
        seen.add(url)
        title = _clean_html_text(link_match.group(2))
        snippet_match = re.search(r'<p[^>]*>(.*?)</p>', block, flags=re.IGNORECASE | re.DOTALL)
        snippet = _clean_html_text(snippet_match.group(1)) if snippet_match else ''
        results.append({'title': title, 'url': url, 'snippet': snippet})
        if len(results) >= max_results:
            break
    return results


def _parse_baidu_results(html: str, max_results: int) -> list[dict[str, str]]:
    blocks = re.findall(r'<h3[^>]*class="[^\"]*t[^\"]*"[^>]*>(.*?)</h3>', html, flags=re.IGNORECASE | re.DOTALL)
    results: list[dict[str, str]] = []
    seen: set[str] = set()
    for block in blocks:
        link_match = re.search(r'<a[^>]*href="([^\"]+)"[^>]*>(.*?)</a>', block, flags=re.IGNORECASE | re.DOTALL)
        if not link_match:
            continue
        url = link_match.group(1).strip()
        if not (url.startswith('http://') or url.startswith('https://')):
            continue
        if url in seen:
            continue
        seen.add(url)
        title = _clean_html_text(link_match.group(2))
        results.append({'title': title, 'url': url, 'snippet': ''})
        if len(results) >= max_results:
            break
    return results


def _parse_duckduckgo_results(html: str, max_results: int) -> list[dict[str, str]]:
    matches = re.findall(
        r'<a[^>]*class="[^\"]*result__a[^\"]*"[^>]*href="([^\"]+)"[^>]*>(.*?)</a>',
        html,
        flags=re.IGNORECASE | re.DOTALL,
    )
    results: list[dict[str, str]] = []
    seen: set[str] = set()
    for raw_url, raw_title in matches:
        url = raw_url.strip()
        if '/l/?' in url:
            parsed = urlparse(url)
            qs = parse_qs(parsed.query)
            uddg = qs.get('uddg', [])
            if uddg:
                url = unquote(uddg[0])
        if not (url.startswith('http://') or url.startswith('https://')):
            continue
        if url in seen:
            continue
        seen.add(url)
        title = _clean_html_text(raw_title)
        results.append({'title': title, 'url': url, 'snippet': ''})
        if len(results) >= max_results:
            break
    return results


def _search_tavily(question: str, max_results: int, timeout_seconds: float | None = None) -> list[dict[str, str]]:
    settings = get_settings()
    if not settings.TAVILY_API_KEY:
        return []

    try:
        import httpx

        response = httpx.post(
            'https://api.tavily.com/search',
            json={
                'api_key': settings.TAVILY_API_KEY,
                'query': question,
                'max_results': max_results,
                'include_answer': False,
                'include_images': False,
                'search_depth': 'basic',
            },
            timeout=timeout_seconds or settings.WEB_SEARCH_TIMEOUT_SECONDS,
            follow_redirects=True,
            trust_env=settings.WEB_SEARCH_TRUST_ENV,
        )
        if response.status_code >= 400:
            logger.warning('tavily search failed status=%s', response.status_code)
            return []
        payload = response.json()
        rows = payload.get('results', []) if isinstance(payload, dict) else []
        results: list[dict[str, str]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            url = str(row.get('url') or '').strip()
            if not url:
                continue
            results.append(
                {
                    'title': str(row.get('title') or ''),
                    'url': url,
                    'snippet': str(row.get('content') or ''),
                }
            )
            if len(results) >= max_results:
                break
        return results
    except Exception:
        logger.exception('tavily search failed')
        return []


def _search_web(question: str, max_results: int) -> list[dict[str, str]]:
    settings = get_settings()
    provider = settings.WEB_SEARCH_PROVIDER.lower().strip()
    started_at = time.monotonic()
    total_budget = max(1.0, settings.WEB_SEARCH_TOTAL_TIMEOUT_SECONDS)

    def _remaining_timeout() -> float:
        elapsed = time.monotonic() - started_at
        left = total_budget - elapsed
        # Keep a small but usable timeout slice for each provider call.
        return max(0.6, min(settings.WEB_SEARCH_PROVIDER_TIMEOUT_SECONDS, left))

    # Optional SDK path for DuckDuckGo when installed.
    try:
        if provider in ('auto', 'duckduckgo', 'ddg'):
            from duckduckgo_search import DDGS

            with DDGS() as ddgs:
                rows = ddgs.text(question, max_results=max_results)
                if rows:
                    results: list[dict[str, str]] = []
                    for row in rows:
                        if not isinstance(row, dict):
                            continue
                        url = str(row.get('href') or '').strip()
                        if not url:
                            continue
                        results.append(
                            {
                                'title': str(row.get('title') or ''),
                                'url': url,
                                'snippet': str(row.get('body') or ''),
                            }
                        )
                    if results:
                        return results[:max_results]
    except Exception:
        logger.info('duckduckgo sdk unavailable, continue with html providers')

    providers = [provider] if provider != 'auto' else ['bing', 'tavily', 'baidu', 'duckduckgo']
    try:
        import httpx

        with httpx.Client(
            timeout=settings.WEB_SEARCH_PROVIDER_TIMEOUT_SECONDS,
            follow_redirects=True,
            trust_env=settings.WEB_SEARCH_TRUST_ENV,
            headers={'User-Agent': 'Mozilla/5.0'},
        ) as client:
            for p in providers:
                if (time.monotonic() - started_at) >= total_budget:
                    logger.warning('web_search timeout budget exhausted, stop providers')
                    break
                try:
                    request_timeout = _remaining_timeout()
                    if p == 'baidu':
                        r = client.get('https://www.baidu.com/s', params={'wd': question}, timeout=request_timeout)
                        results = _filter_relevant_results(
                            question, _parse_baidu_results(r.text, max_results=max_results)
                        )
                        if results:
                            logger.info('web_search provider=baidu results=%d', len(results))
                            return results
                    elif p == 'bing':
                        market = 'zh-CN' if settings.WEB_SEARCH_LOCALE.startswith('zh') else 'en-US'
                        r = client.get(
                            'https://www.bing.com/search',
                            params={'q': question, 'setlang': market, 'mkt': market},
                            timeout=request_timeout,
                        )
                        results = _filter_relevant_results(
                            question, _parse_bing_results(r.text, max_results=max_results)
                        )
                        if results:
                            logger.info('web_search provider=bing results=%d', len(results))
                            return results
                    elif p == 'tavily':
                        results = _filter_relevant_results(
                            question,
                            _search_tavily(
                                question,
                                max_results=max_results,
                                timeout_seconds=request_timeout,
                            ),
                        )
                        if results:
                            logger.info('web_search provider=tavily results=%d', len(results))
                            return results
                    elif p in ('duckduckgo', 'ddg'):
                        r = client.get(
                            'https://duckduckgo.com/html/',
                            params={'q': question, 'kl': 'cn-zh'},
                            timeout=request_timeout,
                        )
                        results = _filter_relevant_results(
                            question, _parse_duckduckgo_results(r.text, max_results=max_results)
                        )
                        if results:
                            logger.info('web_search provider=duckduckgo_html results=%d', len(results))
                            return results
                    else:
                        logger.warning('web_search unknown provider=%s', p)
                except Exception:
                    logger.exception('web_search provider failed: %s', p)
                    continue
        return []
    except Exception:
        logger.exception('web_search failed')
        return []


def _answer_with_fallback(question: str, docs: list[Document]) -> tuple[str, list[str], str]:
    settings = get_settings()
    llm = get_llm()

    if docs:
        selected_docs = docs[: settings.ANSWER_TOP_K]
        prompt = _build_prompt(question, selected_docs, settings.CONTEXT_CHARS_PER_DOC)
        answer = llm.invoke(prompt).content or ''
        answer = _normalize_cn_answer(answer)
        sources = list(dict.fromkeys(doc.metadata.get('source', 'unknown') for doc in docs))
        if not settings.ENABLE_WEB_FALLBACK or not _looks_unknown(answer):
            return answer, sources, 'kb'
    else:
        answer = ''

    if settings.ENABLE_WEB_FALLBACK:
        web_results = _search_web(question, max_results=settings.WEB_SEARCH_MAX_RESULTS)
        if web_results:
            web_prompt = _build_web_prompt(question, web_results)
            web_answer = llm.invoke(web_prompt).content or ''
            web_answer = _normalize_cn_answer(web_answer)
            web_sources = [item['url'] for item in web_results]
            return web_answer, web_sources, 'web'

    if answer:
        return _normalize_cn_answer(answer), [], 'kb_unknown'
    return '未命中本地知识库，且联网检索暂不可用，请稍后重试。', [], 'fallback_unavailable'


def ask(question: str) -> tuple[str, list[str], bool]:
    total_start = time.perf_counter()
    settings = get_settings()
    input_audit = audit_user_question(question)
    normalized_question = input_audit.sanitized_text or question

    if input_audit.blocked:
        logger.warning(
            'rag.ask blocked_input reason=%s labels=%s',
            input_audit.reason,
            ','.join(input_audit.labels),
        )
        return input_audit.block_message, [], False

    redis_client = get_redis_client()
    cache_key = _cache_key(normalized_question)
    cached = redis_client.get(cache_key)
    if cached:
        data = json.loads(cached)
        logger.info('rag.ask cache_hit=True total_ms=%.2f', (time.perf_counter() - total_start) * 1000)
        return data['answer'], data['sources'], True

    retrieve_start = time.perf_counter()
    vector_store = get_vector_store()
    retriever = vector_store.as_retriever(search_kwargs={'k': settings.TOP_K})
    docs = retriever.invoke(normalized_question)
    retrieve_ms = (time.perf_counter() - retrieve_start) * 1000

    answer, sources, mode = _answer_with_fallback(normalized_question, docs)
    answer = _normalize_cn_answer(answer)
    output_audit = audit_model_output(answer)
    if output_audit.blocked:
        logger.warning(
            'rag.ask blocked_output reason=%s labels=%s',
            output_audit.reason,
            ','.join(output_audit.labels),
        )
        answer = output_audit.block_message
        sources = []

    # Do not cache unknown answers; otherwise temporary network issues can poison cache.
    if not _looks_unknown(answer) and not output_audit.blocked:
        redis_client.setex(cache_key, settings.CACHE_TTL_SECONDS, json.dumps({'answer': answer, 'sources': sources}))
    logger.info(
        'rag.ask cache_hit=False mode=%s retrieve_ms=%.2f total_ms=%.2f docs=%d',
        mode,
        retrieve_ms,
        (time.perf_counter() - total_start) * 1000,
        len(docs),
    )
    return answer, sources, False


def _chunk_text(text: str, chunk_size: int = 24) -> Iterator[str]:
    if not text:
        return
    for i in range(0, len(text), chunk_size):
        yield text[i:i + chunk_size]


def ask_stream(question: str) -> tuple[list[str], Iterator[str]]:
    total_start = time.perf_counter()
    settings = get_settings()
    input_audit = audit_user_question(question)
    normalized_question = input_audit.sanitized_text or question

    if input_audit.blocked:
        logger.warning(
            'rag.ask_stream blocked_input reason=%s labels=%s',
            input_audit.reason,
            ','.join(input_audit.labels),
        )

        def _blocked_stream() -> Iterator[str]:
            yield input_audit.block_message

        return [], _blocked_stream()

    vector_store = get_vector_store()
    retriever = vector_store.as_retriever(search_kwargs={'k': settings.TOP_K})

    retrieve_start = time.perf_counter()
    docs = retriever.invoke(normalized_question)
    retrieve_ms = (time.perf_counter() - retrieve_start) * 1000

    answer, sources, mode = _answer_with_fallback(normalized_question, docs)
    answer = _normalize_cn_answer(answer)
    output_audit = audit_model_output(answer)
    if output_audit.blocked:
        logger.warning(
            'rag.ask_stream blocked_output reason=%s labels=%s',
            output_audit.reason,
            ','.join(output_audit.labels),
        )
        answer = output_audit.block_message
        sources = []

    def _stream() -> Iterator[str]:
        for chunk in _chunk_text(answer):
            yield chunk

        logger.info(
            'rag.ask_stream mode=%s retrieve_ms=%.2f total_ms=%.2f docs=%d',
            mode,
            retrieve_ms,
            (time.perf_counter() - total_start) * 1000,
            len(docs),
        )

    return sources, _stream()


def _save_chat_log(session_id: str, question: str, answer: str, sources: list[str]) -> None:
    db = SessionLocal()
    try:
        row = ChatLog(
            session_id=session_id,
            question=question,
            answer=answer,
            sources='\n'.join(sources),
        )
        db.add(row)
        db.commit()
    finally:
        db.close()


def save_chat_log(session_id: str, question: str, answer: str, sources: list[str]) -> None:
    try:
        _save_chat_log(session_id=session_id, question=question, answer=answer, sources=sources)
    except Exception:
        logger.exception('save_chat_log failed')
