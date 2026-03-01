import hashlib
import json
import logging
import time
from typing import Iterator

from langchain_core.documents import Document

from app.core.config import get_settings
from app.db.postgres import SessionLocal
from app.models.chat_log import ChatLog
from app.services.dependencies import get_llm, get_redis_client, get_vector_store

logger = logging.getLogger(__name__)


def _build_prompt(question: str, docs: list[Document]) -> str:
    context_lines: list[str] = []
    for idx, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown")
        context_lines.append(f"[{idx}] source={source}\n{doc.page_content}")
    context = "\n\n".join(context_lines)
    return (
        "你是知识库问答助手。请只基于给定上下文回答，无法确认时明确说不知道。"
        "回答要简洁，并在末尾附上引用编号。\n\n"
        f"问题：{question}\n\n"
        f"上下文：\n{context}"
    )


def _cache_key(question: str) -> str:
    normalized_question = " ".join(question.strip().split())
    digest = hashlib.md5(normalized_question.encode("utf-8")).hexdigest()
    return f"qa:{digest}"


def ask(question: str) -> tuple[str, list[str], bool]:
    total_start = time.perf_counter()
    settings = get_settings()
    redis_client = get_redis_client()
    cache_key = _cache_key(question)
    cached = redis_client.get(cache_key)
    if cached:
        data = json.loads(cached)
        logger.info("rag.ask cache_hit=True total_ms=%.2f", (time.perf_counter() - total_start) * 1000)
        return data["answer"], data["sources"], True

    retrieve_start = time.perf_counter()
    vector_store = get_vector_store()
    retriever = vector_store.as_retriever(search_kwargs={"k": settings.TOP_K})
    docs = retriever.invoke(question)
    retrieve_ms = (time.perf_counter() - retrieve_start) * 1000
    prompt = _build_prompt(question, docs)

    llm_start = time.perf_counter()
    llm = get_llm()
    answer = llm.invoke(prompt).content
    llm_ms = (time.perf_counter() - llm_start) * 1000
    sources = list(dict.fromkeys(doc.metadata.get("source", "unknown") for doc in docs))

    redis_client.setex(cache_key, settings.CACHE_TTL_SECONDS, json.dumps({"answer": answer, "sources": sources}))
    logger.info(
        "rag.ask cache_hit=False retrieve_ms=%.2f llm_ms=%.2f total_ms=%.2f docs=%d top_k=%d",
        retrieve_ms,
        llm_ms,
        (time.perf_counter() - total_start) * 1000,
        len(docs),
        settings.TOP_K,
    )
    return answer, sources, False


def ask_stream(question: str) -> tuple[list[str], Iterator[str]]:
    settings = get_settings()
    vector_store = get_vector_store()
    retriever = vector_store.as_retriever(search_kwargs={"k": settings.TOP_K})
    docs = retriever.invoke(question)
    prompt = _build_prompt(question, docs)
    llm = get_llm()
    sources = list(dict.fromkeys(doc.metadata.get("source", "unknown") for doc in docs))

    def _stream() -> Iterator[str]:
        for chunk in llm.stream(prompt):
            text = chunk.content or ""
            if text:
                yield text

    return sources, _stream()


def _save_chat_log(session_id: str, question: str, answer: str, sources: list[str]) -> None:
    db = SessionLocal()
    try:
        row = ChatLog(
            session_id=session_id,
            question=question,
            answer=answer,
            sources="\n".join(sources),
        )
        db.add(row)
        db.commit()
    finally:
        db.close()


def save_chat_log(session_id: str, question: str, answer: str, sources: list[str]) -> None:
    try:
        _save_chat_log(session_id=session_id, question=question, answer=answer, sources=sources)
    except Exception:
        logger.exception("save_chat_log failed")

