from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from app.core.config import get_settings


settings = get_settings()
engine = create_engine(settings.POSTGRES_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


class Base(DeclarativeBase):
    pass


def init_db() -> None:
    from app.models.chat_log import ChatLog  # noqa: F401
    from app.models.chunk import ChunkMetadata  # noqa: F401
    from app.models.doc_index import DocIndex  # noqa: F401

    Base.metadata.create_all(bind=engine)
