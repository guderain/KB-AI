import hashlib
import json
import logging
import time
from typing import Iterator

from langchain_core.documents import Document

from app.core.config import get_settings
from app.db.postgres import SessionLocal
from app.models.chat_log import ChatLog
from app.services.dependencies import get_llm, get_redis_client, get_streaming_llm, get_vector_store

logger = logging.getLogger(__name__)


def _trim_doc_text(text: str, limit: int) -> str:
    clean = ' '.join(text.split())
    if len(clean) <= limit:
        return clean
    return clean[:limit] + ' ...'


def _build_prompt(question: str, docs: list[Document], context_chars_per_doc: int) -> str:
    context_lines: list[str] = []
    for idx, doc in enumerate(docs, start=1):
        source = doc.metadata.get('source', 'unknown')
        excerpt = _trim_doc_text(doc.page_content, context_chars_per_doc)
        context_lines.append(f'[{idx}] source={source}\n{excerpt}')
    context = '\n\n'.join(context_lines)
    return (
        'You are a knowledge-base assistant. Answer strictly based on the given context. '
        "If the context is insufficient, reply exactly: I don't know. "
        'Keep the answer concise and cite source indices like [1][2].\n\n'
        f'Question: {question}\n\n'
        f'Context:\n{context}'
    )


def _cache_key(question: str) -> str:
    normalized_question = ' '.join(question.strip().split())
    digest = hashlib.md5(normalized_question.encode('utf-8')).hexdigest()
    return f'qa:{digest}'


def ask(question: str) -> tuple[str, list[str], bool]:
    total_start = time.perf_counter()
    settings = get_settings()
    redis_client = get_redis_client()
    cache_key = _cache_key(question)
    cached = redis_client.get(cache_key)
    if cached:
        data = json.loads(cached)
        logger.info('rag.ask cache_hit=True total_ms=%.2f', (time.perf_counter() - total_start) * 1000)
        return data['answer'], data['sources'], True

    retrieve_start = time.perf_counter()
    vector_store = get_vector_store()
    retriever = vector_store.as_retriever(search_kwargs={'k': settings.TOP_K})
    docs = retriever.invoke(question)
    retrieve_ms = (time.perf_counter() - retrieve_start) * 1000
    if not docs:
        answer = 'Knowledge base is not indexed yet. Please run /api/v1/ingest/reindex.'
        logger.warning('rag.ask empty_docs=True retrieve_ms=%.2f', retrieve_ms)
        return answer, [], False

    selected_docs = docs[: settings.ANSWER_TOP_K]
    prompt = _build_prompt(question, selected_docs, settings.CONTEXT_CHARS_PER_DOC)

    llm_start = time.perf_counter()
    llm = get_llm()
    answer = llm.invoke(prompt).content
    llm_ms = (time.perf_counter() - llm_start) * 1000
    sources = list(dict.fromkeys(doc.metadata.get('source', 'unknown') for doc in docs))

    redis_client.setex(cache_key, settings.CACHE_TTL_SECONDS, json.dumps({'answer': answer, 'sources': sources}))
    logger.info(
        'rag.ask cache_hit=False retrieve_ms=%.2f llm_ms=%.2f total_ms=%.2f docs=%d selected_docs=%d top_k=%d answer_top_k=%d',
        retrieve_ms,
        llm_ms,
        (time.perf_counter() - total_start) * 1000,
        len(docs),
        len(selected_docs),
        settings.TOP_K,
        settings.ANSWER_TOP_K,
    )
    return answer, sources, False


def ask_stream(question: str) -> tuple[list[str], Iterator[str]]:
    total_start = time.perf_counter()
    settings = get_settings()
    vector_store = get_vector_store()
    retriever = vector_store.as_retriever(search_kwargs={'k': settings.TOP_K})

    retrieve_start = time.perf_counter()
    docs = retriever.invoke(question)
    retrieve_ms = (time.perf_counter() - retrieve_start) * 1000
    if not docs:
        logger.warning('rag.ask_stream empty_docs=True retrieve_ms=%.2f', retrieve_ms)

        def _empty_stream() -> Iterator[str]:
            yield 'Knowledge base is not indexed yet. Please run /api/v1/ingest/reindex.'

        return [], _empty_stream()

    selected_docs = docs[: settings.ANSWER_TOP_K]
    prompt = _build_prompt(question, selected_docs, settings.CONTEXT_CHARS_PER_DOC)
    llm = get_streaming_llm()
    sources = list(dict.fromkeys(doc.metadata.get('source', 'unknown') for doc in docs))

    def _stream() -> Iterator[str]:
        llm_start = time.perf_counter()
        first_token_ms: float | None = None

        for chunk in llm.stream(prompt):
            text = chunk.content or ''
            if text:
                if first_token_ms is None:
                    first_token_ms = (time.perf_counter() - llm_start) * 1000
                yield text

        logger.info(
            'rag.ask_stream retrieve_ms=%.2f first_token_ms=%.2f total_ms=%.2f docs=%d selected_docs=%d top_k=%d answer_top_k=%d',
            retrieve_ms,
            first_token_ms or -1.0,
            (time.perf_counter() - total_start) * 1000,
            len(docs),
            len(selected_docs),
            settings.TOP_K,
            settings.ANSWER_TOP_K,
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
