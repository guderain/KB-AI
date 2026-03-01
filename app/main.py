import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.router import api_router
from app.core.config import get_settings
from app.core.logging_setup import setup_logging
from app.db.postgres import init_db
from app.services.ingestion_service import ensure_index_ready_on_startup

setup_logging()
settings = get_settings()
logger = logging.getLogger(__name__)
app = FastAPI(title=settings.APP_NAME)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup() -> None:
    init_db()
    if settings.AUTO_INGEST_ON_STARTUP:
        result = ensure_index_ready_on_startup()
        logger.info(
            "startup.auto_ingest triggered=%s reason=%s files_total=%s chunks_inserted=%s",
            result.get("triggered"),
            result.get("reason"),
            result.get("files_total"),
            result.get("chunks_inserted"),
        )


app.include_router(api_router, prefix=settings.API_V1_PREFIX)
