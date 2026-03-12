import asyncio
import hashlib
import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from html import unescape as html_unescape
from typing import Iterator
from urllib.parse import parse_qs, unquote, urlparse

from langchain_core.documents import Document
from sqlalchemy import or_, select

from app.core.config import get_settings
from app.db.postgres import SessionLocal
from app.models.chat_log import ChatLog
from app.models.chunk import ChunkMetadata
from app.services.dependencies import get_http_client, get_llm, get_redis_client, get_vector_store
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


# 把用户问题拆成一组可用于检索匹配的关键词 token
def _query_tokens(query: str) -> list[str]:
    # 防空：None 会变 ''，全部转小写：让匹配大小写不敏感（对英文有意义）
    # q："rag 检索增强 方案怎么做？"
    q = (query or '').lower()
    # 按“非字母数字非中文字符”切分，也就是把空格、标点、符号当分隔符，过滤长度小于2的片段
    # 返回：['rag', '检索增强', '方案', '怎么做']
    words = [w for w in re.split(r'[^0-9a-z\u4e00-\u9fff]+', q) if len(w) >= 2]
    # 提取中文字符 "检索增强方案怎么做"
    chinese = ''.join(ch for ch in q if '\u4e00' <= ch <= '\u9fff')
    # 对中文串做 2-gram（双字切片）中文没有天然空格分词，用双字片段能提高匹配召回率
    # 返回 ["检索","索增","增强","强方","方案","案怎","怎么","么做"]
    bigrams = [chinese[i : i + 2] for i in range(len(chinese) - 1)]
    # 返回 tokens：先 words，再补 bigrams 里未出现的项
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


def _doc_key(doc: Document) -> tuple[str, str]:
    source = str(doc.metadata.get('source', 'unknown'))
    content = ' '.join((doc.page_content or '').split())
    return source, content


def _keyword_retrieve(question: str, top_k: int) -> list[Document]:
    if top_k <= 0:
        return []
    tokens = [t for t in _query_tokens(question) if t]
    if not tokens:
        return []

    # 只取前 8 个关键词，限制查询条件数量，避免全表扫描
    tokens = tokens[:8] 
    # ChunkMetadata.content：拿到 ORM 列对象（不是普通字符串）
    # .ilike(...)：在这个列上构造“大小写不敏感 LIKE 条件”
    # 所以 conditions 最终是一个“条件表达式列表”
    conditions = [ChunkMetadata.content.ilike(f'%{token}%') for token in tokens]
    if not conditions:
        return []

    db = SessionLocal()
    try:
        # 限制候选集数量
        candidate_limit = max(top_k * 8, 30)
        stmt = select(ChunkMetadata).where(or_(*conditions)).limit(candidate_limit)
        # ChunkMetadata实例是ORM对象，存储的数据有：source, title, content
        rows = db.execute(stmt).scalars().all()
    except Exception:
        logger.exception('keyword retrieve failed')
        return []
    finally:
        db.close()
    # 记录得分和匹配到的 ChunkMetadata 实例
    scored: list[tuple[int, ChunkMetadata]] = []
    for row in rows:
        # 拼接 title 和 content，转为小写
        hay = f'{row.title or ""} {row.content or ""}'.lower()
        # 初始化得分
        score = 0
        for token in tokens:
            # 如果 关键词 token 在 hay 中，得分加 2 或 1
            if token in hay:
                # 如果 关键词 token 长度 >= 4，得分加 2，否则加 1
                score += 2 if len(token) >= 4 else 1
        # 如果得分大于 0，记录得分和匹配到的 ChunkMetadata 实例到 scored 列表
        if score > 0:
            scored.append((score, row))

    # 对 scored 列表按得分降序排序
    scored.sort(key=lambda item: item[0], reverse=True)
    # 初始化 docs 列表
    docs: list[Document] = []
    # 遍历 scored 列表前 top_k 个元素
    for _, row in scored[:top_k]:
        # 创建 Document 实例
        docs.append(
            Document(
                page_content=row.content or '',
                metadata={'source': row.source or 'unknown', 'title': row.title or '', 'retrieval': 'keyword'},
            )
        )
    return docs


# 线程池用于执行同步的数据库和向量检索操作
_retrieval_executor = ThreadPoolExecutor(max_workers=4)


async def _keyword_retrieve_async(question: str, top_k: int) -> list[Document]:
    """异步版本的关键词检索，在线程池中执行同步数据库查询。"""
    # 获取当前线程的事件循环
    loop = asyncio.get_running_loop()
    # 在事件循环中执行同步关键词检索
    return await loop.run_in_executor(_retrieval_executor, _keyword_retrieve, question, top_k)


async def _vector_retrieve_async(question: str, top_k: int) -> list[Document]:
    """异步版本的向量检索，在线程池中执行同步向量查询。"""
    settings = get_settings()
    vector_store = get_vector_store()
    vector_retriever = vector_store.as_retriever(search_kwargs={'k': top_k})

    def _sync_invoke() -> list[Document]:
        return vector_retriever.invoke(question)
    # 获取当前线程的事件循环
    loop = asyncio.get_running_loop()
    # 在事件循环中执行同步查询
    return await loop.run_in_executor(_retrieval_executor, _sync_invoke)


def _fuse_retrieved_docs(vector_docs: list[Document], keyword_docs: list[Document], top_k: int) -> list[Document]:
    if top_k <= 0:
        return []
    # 倒排得分融合：对不同检索器的结果进行融合，提高召回率
    k = 60.0
    # score_map：每个文档 key 的融合分
    score_map: dict[tuple[str, str], float] = {}
    # doc_map：key 到 Document 的映射，key 是文档的 source 和 content 的元组
    doc_map: dict[tuple[str, str], Document] = {}

    # 遍历 vector_docs 列表
    for rank, doc in enumerate(vector_docs):
        # 创建 Document 实例
        key = _doc_key(doc)
        # 计算得分
        # 得分规则是：1 / (k + rank + 1)，k 是常数，rank 是文档在列表中的位置，+1 是为了避免分母为 0
        # 这个规则的目的是：让距离越近的文档得分越高，距离越远的文档得分越低
        score_map[key] = score_map.get(key, 0.0) + (1.0 / (k + rank + 1))
        doc_map[key] = doc

    # 遍历 keyword_docs 列表
    for rank, doc in enumerate(keyword_docs):
        # 创建 Document 实例
        key = _doc_key(doc)
        # 计算得分
        score_map[key] = score_map.get(key, 0.0) + (1.0 / (k + rank + 1))
        if key not in doc_map:
            doc_map[key] = doc

    # 对 score_map 按得分降序排序
    ranked_keys = sorted(score_map.keys(), key=lambda item: score_map[item], reverse=True)
    # 返回前 top_k 个 Document 实例
    return [doc_map[key] for key in ranked_keys[:top_k]]

def _rerank_docs(question: str, docs: list[Document], top_k: int, candidate_k: int) -> list[Document]:
    settings = get_settings()
    if not docs or top_k <= 0:
        return []
    # 截取前 候选集数量 个文档
    candidates = docs[:candidate_k]
    # 截取候选集中每个文档的前 最大文档字符数 个字符
    # Reranker 模型有输入长度限制（通常是 512/1024 tokens）
        # 过长的文档会增加 API 成本和延迟
        # 超过 2000 字符的部分对相关性判断影响很小
    texts = [(doc.page_content or '')[: settings.RETRIEVAL_RERANK_MAX_DOC_CHARS] for doc in candidates]
    # 如果文本为空，返回候选集前 top_k 个文档
    if not texts:
        return candidates[:top_k]

    api_key = settings.OPENAI_API_KEY.strip()
    if not api_key:
        logger.warning('reranker api_key empty, fallback to fused order')
        return candidates[:top_k]

    # 只返回排序后的索引列表，不返回文档内容，减少网络传输
    request_body = {
        'model': settings.RETRIEVAL_RERANK_MODEL,
        'input': {'query': question, 'documents': texts},
        'parameters': {'top_n': len(texts), 'return_documents': False},
    }
    try:
        # 发送请求到 reranker 服务
        resp = get_http_client().post(
            settings.RETRIEVAL_RERANK_ENDPOINT,
            headers={'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'},
            json=request_body,
            timeout=settings.RETRIEVAL_RERANK_TIMEOUT_SECONDS,
        )
        # 如果响应状态码大于等于 400，记录错误日志
        if resp.status_code >= 400:
            logger.warning(
                'reranker http_failed status=%s body=%s',
                resp.status_code,
                (resp.text or '')[:200],
            )
            return candidates[:top_k]
 
        payload = resp.json()
    except Exception:
        logger.exception('reranker request failed, fallback to fused order')
        return candidates[:top_k]

    # 解析响应体
    # API 返回格式示例：
    # {
    #     "output": {
    #     "results": [
    #         {"index": 3, "relevance_score": 0.95},  // 原文档索引 3，最相关
    #         {"index": 0, "relevance_score": 0.87},  // 原文档索引 0
    #         {"index": 5, "relevance_score": 0.82}   // 原文档索引 5
    #     ]
    #     }
    # }
    results = ((payload.get('output') or {}).get('results') or []) if isinstance(payload, dict) else []
    # 初始化 ranked 列表
    ranked: list[Document] = []
    # 初始化 used 集合
    used: set[int] = set()
    # 遍历 results 列表
    for row in results:
        # 获取索引
        idx = row.get('index') if isinstance(row, dict) else None
        # 如果索引是整数，且在候选集中，且未被使用过，记录文档到 ranked 列表
        if isinstance(idx, int) and 0 <= idx < len(candidates) and idx not in used:
            # 按 Reranker 顺序取文档
            ranked.append(candidates[idx])
            # 记录索引到 used 集合
            used.add(idx)

    if not ranked:
        logger.warning('reranker empty_results, fallback to fused order')
        return candidates[:top_k]

    # 如果 Reranker 返回不足，按原顺序补足
    if len(ranked) < top_k:
        # 遍历候选集
        for idx, doc in enumerate(candidates):
            # 如果索引未被使用过，记录文档到 ranked 列表
            if idx not in used:
                ranked.append(doc)
            # 如果 ranked 列表长度大于等于 top_k，退出循环
            if len(ranked) >= top_k:
                break
    return ranked[:top_k]


async def _hybrid_retrieve_async(question: str, top_k: int) -> tuple[list[Document], int, int, int]:
    """异步版本的混合检索，并行执行向量召回和关键词召回。"""
    settings = get_settings()

    # 并行执行向量召回和关键词召回
    vector_task = _vector_retrieve_async(question, settings.RETRIEVAL_VECTOR_TOP_K)
    keyword_task = _keyword_retrieve_async(question, settings.RETRIEVAL_KEYWORD_TOP_K)

    vector_docs, keyword_docs = await asyncio.gather(vector_task, keyword_task)

    # 召回结果融合 给reranker候选
    fused_docs = _fuse_retrieved_docs(
        vector_docs=vector_docs,
        keyword_docs=keyword_docs,
        top_k=max(top_k, settings.RETRIEVAL_RERANK_CANDIDATE_K),
    )
    # 重排序
    if settings.ENABLE_RETRIEVAL_RERANKER:
        final_docs = _rerank_docs(
            question=question,
            docs=fused_docs,
            top_k=top_k,
            candidate_k=settings.RETRIEVAL_RERANK_CANDIDATE_K,
        )
    # 保底：reranker失败，返回融合打分结果作为最终结果
    else:
        final_docs = fused_docs[:top_k]
    # 返回最终结果、向量召回结果数量、关键词召回结果数量、融合结果数量
    return final_docs, len(vector_docs), len(keyword_docs), len(fused_docs)


def _hybrid_retrieve(question: str, top_k: int) -> tuple[list[Document], int, int, int]:
    """混合检索（同步入口），优先走异步并行；检测到运行中 loop 时回退同步。"""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # 当前线程没有运行中的事件循环，可安全使用 asyncio.run。
        return asyncio.run(_hybrid_retrieve_async(question, top_k))

    # 有运行中 event loop：不再用 run_coroutine_threadsafe(...).result()，改为同步回退路径，避免潜在死锁/阻塞
    logger.warning('hybrid_retrieve running_loop_detected fallback=sync')
    settings = get_settings()
    vector_store = get_vector_store()
    vector_retriever = vector_store.as_retriever(search_kwargs={'k': settings.RETRIEVAL_VECTOR_TOP_K})
    vector_docs = vector_retriever.invoke(question)
    keyword_docs = _keyword_retrieve(question=question, top_k=settings.RETRIEVAL_KEYWORD_TOP_K)
    fused_docs = _fuse_retrieved_docs(
        vector_docs=vector_docs,
        keyword_docs=keyword_docs,
        top_k=max(top_k, settings.RETRIEVAL_RERANK_CANDIDATE_K),
    )
    if settings.ENABLE_RETRIEVAL_RERANKER:
        final_docs = _rerank_docs(
            question=question,
            docs=fused_docs,
            top_k=top_k,
            candidate_k=settings.RETRIEVAL_RERANK_CANDIDATE_K,
        )
    else:
        final_docs = fused_docs[:top_k]
    return final_docs, len(vector_docs), len(keyword_docs), len(fused_docs)


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
        # 只返回实际用于生成答案的文档来源
        sources = list(dict.fromkeys(doc.metadata.get('source', 'unknown') for doc in selected_docs))
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
    docs, vector_count, keyword_count, fused_count = _hybrid_retrieve(normalized_question, settings.TOP_K)
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
        'rag.ask cache_hit=False mode=%s retrieve_ms=%.2f total_ms=%.2f docs=%d vector_docs=%d keyword_docs=%d fused_docs=%d',
        mode,
        retrieve_ms,
        (time.perf_counter() - total_start) * 1000,
        len(docs),
        vector_count,
        keyword_count,
        fused_count,
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

    retrieve_start = time.perf_counter()
    docs, vector_count, keyword_count, fused_count = _hybrid_retrieve(normalized_question, settings.TOP_K)
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
            'rag.ask_stream mode=%s retrieve_ms=%.2f total_ms=%.2f docs=%d vector_docs=%d keyword_docs=%d fused_docs=%d',
            mode,
            retrieve_ms,
            (time.perf_counter() - total_start) * 1000,
            len(docs),
            vector_count,
            keyword_count,
            fused_count,
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
