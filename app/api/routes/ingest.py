from fastapi import APIRouter, Depends

from app.api.deps.security import guard_ingest
from app.core.config import get_settings
from app.schemas.ingest import IncrementalReindexResponse, ReindexResponse
from app.services.ingestion_service import incremental_reindex, reindex

router = APIRouter(prefix="/ingest", tags=["ingest"], dependencies=[Depends(guard_ingest)])


@router.post("/reindex", response_model=ReindexResponse)
def reindex_knowledge() -> ReindexResponse:
    files_loaded, chunks_indexed = reindex()
    settings = get_settings()
    return ReindexResponse(
        files_loaded=files_loaded,
        chunks_indexed=chunks_indexed,
        collection=settings.MILVUS_COLLECTION,
    )


@router.post("/incremental", response_model=IncrementalReindexResponse)
def incremental_reindex_knowledge() -> IncrementalReindexResponse:
    result = incremental_reindex(allow_destructive_migration=False)
    settings = get_settings()
    return IncrementalReindexResponse(
        files_total=result["files_total"],
        files_changed=result["files_changed"],
        files_removed=result["files_removed"],
        chunks_inserted=result["chunks_inserted"],
        chunks_deleted=result["chunks_deleted"],
        collection=settings.MILVUS_COLLECTION,
        migrated_full_reindex=bool(result.get("migrated_full_reindex", 0)),
        needs_full_reindex=bool(result.get("needs_full_reindex", 0)),
    )
