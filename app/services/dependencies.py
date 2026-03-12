from functools import lru_cache

import httpx
import redis
from langchain_milvus import Milvus
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from app.core.config import get_settings

# lru_cache 是 Python 内置的缓存装饰器，用于缓存函数的结果，避免重复计算
# 这里缓存 httpx.Client 和 httpx.AsyncClient 实例，避免重复创建
@lru_cache
def get_http_client() -> httpx.Client:
    # 忽略系统代理环境变量，避免意外的代理劫持
    return httpx.Client(trust_env=False, timeout=60.0)


@lru_cache
def get_async_http_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(trust_env=False, timeout=60.0)


@lru_cache
def get_embeddings() -> OpenAIEmbeddings:
    settings = get_settings()
    return OpenAIEmbeddings(
        model=settings.OPENAI_EMBEDDING_MODEL,
        api_key=settings.OPENAI_API_KEY,
        base_url=settings.OPENAI_BASE_URL,
        # DashScope OpenAI-compatible embedding endpoint expects string input.
        # Disable token-length pre-processing to avoid sending token arrays.
        check_embedding_ctx_length=False,
        chunk_size=10,
        http_client=get_http_client(),
        http_async_client=get_async_http_client(),
    )


@lru_cache
def get_llm() -> ChatOpenAI:
    settings = get_settings()
    return ChatOpenAI(
        model=settings.OPENAI_CHAT_MODEL,
        api_key=settings.OPENAI_API_KEY,
        base_url=settings.OPENAI_BASE_URL,
        temperature=0.2,
        timeout=settings.OPENAI_TIMEOUT_SECONDS,
        max_retries=settings.OPENAI_MAX_RETRIES,
        max_tokens=settings.OPENAI_MAX_TOKENS,
        http_client=get_http_client(),
        http_async_client=get_async_http_client(),
    )


@lru_cache
def get_streaming_llm() -> ChatOpenAI:
    settings = get_settings()
    return ChatOpenAI(
        model=settings.OPENAI_CHAT_MODEL,
        api_key=settings.OPENAI_API_KEY,
        base_url=settings.OPENAI_BASE_URL,
        temperature=0.2,
        timeout=settings.OPENAI_TIMEOUT_SECONDS,
        max_retries=settings.OPENAI_MAX_RETRIES,
        max_tokens=settings.OPENAI_MAX_TOKENS,
        streaming=True,
        http_client=get_http_client(),
        http_async_client=get_async_http_client(),
    )


@lru_cache
def get_vector_store() -> Milvus:
    settings = get_settings()
    return Milvus(
        embedding_function=get_embeddings(),
        collection_name=settings.MILVUS_COLLECTION,
        connection_args={"uri": settings.MILVUS_URI, "token": settings.MILVUS_TOKEN},
        auto_id=False,
    )


@lru_cache
def get_redis_client() -> redis.Redis:
    settings = get_settings()
    return redis.from_url(settings.REDIS_URL, decode_responses=True)
