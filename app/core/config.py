from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', extra='ignore')

    APP_ENV: str = 'dev'
    APP_NAME: str = 'KB-AI-RAG'
    APP_HOST: str = '0.0.0.0'
    APP_PORT: int = 8000
    API_V1_PREFIX: str = '/api/v1'

    OPENAI_API_KEY: str = ''
    OPENAI_BASE_URL: str = 'https://api.openai.com/v1'
    OPENAI_CHAT_MODEL: str = 'gpt-4o-mini'
    OPENAI_EMBEDDING_MODEL: str = 'text-embedding-3-small'
    OPENAI_TIMEOUT_SECONDS: float = 45.0
    OPENAI_MAX_RETRIES: int = 1
    OPENAI_MAX_TOKENS: int = 512

    MILVUS_URI: str = 'http://localhost:19530'
    MILVUS_TOKEN: str = ''
    MILVUS_COLLECTION: str = 'kb_chunks'

    REDIS_URL: str = 'redis://localhost:6379/0'
    CACHE_TTL_SECONDS: int = 3600

    POSTGRES_URL: str = 'postgresql+psycopg://postgres:postgres@localhost:5432/kb_ai'

    KNOWLEDGE_BASE_DIR: str = 'D:/\u77e5\u8bc6\u5e93'
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 160
    TOP_K: int = 6
    ANSWER_TOP_K: int = 4
    CONTEXT_CHARS_PER_DOC: int = 800
    ENABLE_WEB_FALLBACK: bool = True
    WEB_SEARCH_PROVIDER: str = 'auto'
    WEB_SEARCH_LOCALE: str = 'zh-CN'
    WEB_SEARCH_MAX_RESULTS: int = 5
    WEB_SEARCH_TIMEOUT_SECONDS: float = 12.0
    WEB_SEARCH_PROVIDER_TIMEOUT_SECONDS: float = 30.0
    WEB_SEARCH_TOTAL_TIMEOUT_SECONDS: float = 40.0
    WEB_SEARCH_TRUST_ENV: bool = True
    TAVILY_API_KEY: str = ''

    CORS_ORIGINS: str = 'http://localhost:3000,http://127.0.0.1:3000'
    API_KEYS: str = ''
    RATE_LIMIT_PER_MINUTE: int = 60
    INGEST_RATE_LIMIT_PER_MINUTE: int = 10
    AUTO_INGEST_ON_STARTUP: bool = True

    @property
    def cors_origins_list(self) -> list[str]:
        return [item.strip() for item in self.CORS_ORIGINS.split(',') if item.strip()]

    @property
    def api_keys_list(self) -> list[str]:
        return [item.strip() for item in self.API_KEYS.split(',') if item.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()


# Backward-compatible singleton for old imports:
# `from app.core.config import settings`
settings = get_settings()
