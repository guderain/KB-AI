from pydantic import BaseModel


class ReindexResponse(BaseModel):
    files_loaded: int
    chunks_indexed: int
    collection: str


class IncrementalReindexResponse(BaseModel):
    files_total: int
    files_changed: int
    files_removed: int
    chunks_inserted: int
    chunks_deleted: int
    collection: str
    migrated_full_reindex: bool = False
    needs_full_reindex: bool = False
