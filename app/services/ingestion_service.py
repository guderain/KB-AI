import hashlib
import logging
from datetime import datetime
from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pymilvus import MilvusClient
from sqlalchemy import delete, select

from app.core.config import get_settings
from app.db.postgres import SessionLocal
from app.models.chunk import ChunkMetadata
from app.models.doc_index import DocIndex
from app.services.dependencies import get_vector_store

logger = logging.getLogger(__name__)


def _safe_title(file_path: str) -> str:
    return Path(file_path).stem[:500]


def _hash_text(content: str) -> str:
    return hashlib.sha256(content.encode('utf-8')).hexdigest()


def _chunk_id(source: str, file_hash: str, index: int) -> str:
    raw = f'{source}|{file_hash}|{index}'
    return hashlib.sha1(raw.encode('utf-8')).hexdigest()


def _scan_md_files(root: str) -> list[Path]:
    return [p for p in Path(root).rglob('*.md') if p.is_file()]


def _load_file_doc(path: Path) -> Document:
    loader = TextLoader(str(path), encoding='utf-8')
    docs = loader.load()
    if not docs:
        return Document(page_content='', metadata={'source': str(path)})
    return docs[0]


def _build_splitter() -> RecursiveCharacterTextSplitter:
    settings = get_settings()
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
    )


def _drop_collection(collection_name: str) -> None:
    settings = get_settings()
    client = MilvusClient(uri=settings.MILVUS_URI, token=settings.MILVUS_TOKEN or None)
    if client.has_collection(collection_name=collection_name):
        client.drop_collection(collection_name=collection_name)


def _needs_schema_migration(collection_name: str) -> bool:
    settings = get_settings()
    client = MilvusClient(uri=settings.MILVUS_URI, token=settings.MILVUS_TOKEN or None)
    if not client.has_collection(collection_name=collection_name):
        return False
    description = client.describe_collection(collection_name=collection_name)
    return bool(description.get('auto_id', False))


def _get_collection_row_count(collection_name: str) -> tuple[bool, int]:
    settings = get_settings()
    client = MilvusClient(uri=settings.MILVUS_URI, token=settings.MILVUS_TOKEN or None)
    if not client.has_collection(collection_name=collection_name):
        return False, 0
    try:
        stats = client.get_collection_stats(collection_name=collection_name)
        row_count = int(stats.get('row_count', 0))
        return True, row_count
    except Exception:
        # Keep startup resilient even when stats API is temporarily unavailable.
        return True, 0


def _delete_vector_ids(vector_ids: list[str]) -> int:
    if not vector_ids:
        return 0
    try:
        store = get_vector_store()
        store.delete(ids=vector_ids)
        return len(vector_ids)
    except Exception:
        return 0


def _upsert_doc_index(db, source: str, file_hash: str, chunks_count: int) -> None:
    exists = db.execute(select(DocIndex).where(DocIndex.source == source)).scalar_one_or_none()
    if exists:
        exists.file_hash = file_hash
        exists.chunks_count = chunks_count
        exists.updated_at = datetime.utcnow()
    else:
        db.add(
            DocIndex(
                source=source,
                file_hash=file_hash,
                chunks_count=chunks_count,
                updated_at=datetime.utcnow(),
            )
        )


def reindex() -> tuple[int, int]:
    settings = get_settings()
    files = _scan_md_files(settings.KNOWLEDGE_BASE_DIR)
    docs: list[Document] = []
    file_hash_map: dict[str, str] = {}

    for path in files:
        doc = _load_file_doc(path)
        docs.append(doc)
        file_hash_map[str(path)] = _hash_text(doc.page_content)

    splitter = _build_splitter()
    chunks = splitter.split_documents(docs)

    _drop_collection(settings.MILVUS_COLLECTION)
    store = get_vector_store()

    ids: list[str] = []
    for index, chunk in enumerate(chunks):
        source = chunk.metadata.get('source', 'unknown')
        file_hash = file_hash_map.get(source, 'na')
        ids.append(_chunk_id(source, file_hash, index))

    if chunks:
        store.add_documents(chunks, ids=ids)

    db = SessionLocal()
    try:
        db.execute(delete(ChunkMetadata))
        db.execute(delete(DocIndex))

        source_chunk_count: dict[str, int] = {}
        for idx, chunk in enumerate(chunks):
            source = chunk.metadata.get('source', 'unknown')
            source_chunk_count[source] = source_chunk_count.get(source, 0) + 1
            row = ChunkMetadata(
                chunk_id=ids[idx],
                source=source,
                title=_safe_title(source),
                content=chunk.page_content,
            )
            db.add(row)

        for source, file_hash in file_hash_map.items():
            _upsert_doc_index(db, source, file_hash, source_chunk_count.get(source, 0))

        db.commit()
    finally:
        db.close()

    return len(files), len(chunks)


def incremental_reindex(allow_destructive_migration: bool = False) -> dict[str, int]:
    settings = get_settings()
    if _needs_schema_migration(settings.MILVUS_COLLECTION):
        _, row_count = _get_collection_row_count(settings.MILVUS_COLLECTION)
        # If collection is empty, destructive rebuild is safe.
        if allow_destructive_migration or row_count <= 0:
            files_loaded, chunks_indexed = reindex()
            return {
                'files_total': files_loaded,
                'files_changed': files_loaded,
                'files_removed': 0,
                'chunks_inserted': chunks_indexed,
                'chunks_deleted': 0,
                'migrated_full_reindex': 1,
                'needs_full_reindex': 0,
            }
        logger.warning(
            'incremental_reindex skipped due to schema migration requirement; collection=%s row_count=%d',
            settings.MILVUS_COLLECTION,
            row_count,
        )
        return {
            'files_total': len(_scan_md_files(settings.KNOWLEDGE_BASE_DIR)),
            'files_changed': 0,
            'files_removed': 0,
            'chunks_inserted': 0,
            'chunks_deleted': 0,
            'migrated_full_reindex': 0,
            'needs_full_reindex': 1,
        }

    files = _scan_md_files(settings.KNOWLEDGE_BASE_DIR)
    current_docs: dict[str, Document] = {}
    current_hashes: dict[str, str] = {}

    for path in files:
        doc = _load_file_doc(path)
        source = str(path)
        current_docs[source] = doc
        current_hashes[source] = _hash_text(doc.page_content)

    db = SessionLocal()
    try:
        db_rows = db.execute(select(DocIndex)).scalars().all()
        old_hashes = {row.source: row.file_hash for row in db_rows}

        current_sources = set(current_hashes.keys())
        old_sources = set(old_hashes.keys())

        removed_sources = sorted(old_sources - current_sources)
        changed_sources = sorted(
            source for source in current_sources if source not in old_hashes or old_hashes[source] != current_hashes[source]
        )

        deleted_chunks = 0
        for source in removed_sources + changed_sources:
            chunk_ids = db.execute(select(ChunkMetadata.chunk_id).where(ChunkMetadata.source == source)).scalars().all()
            deleted_chunks += _delete_vector_ids(chunk_ids)
            db.execute(delete(ChunkMetadata).where(ChunkMetadata.source == source))
            db.execute(delete(DocIndex).where(DocIndex.source == source))

        splitter = _build_splitter()
        changed_docs = [current_docs[source] for source in changed_sources]
        changed_chunks = splitter.split_documents(changed_docs) if changed_docs else []

        inserted_chunks = 0
        if changed_chunks:
            store = get_vector_store()
            ids: list[str] = []
            source_counter: dict[str, int] = {}
            for chunk in changed_chunks:
                source = chunk.metadata.get('source', 'unknown')
                source_counter[source] = source_counter.get(source, 0) + 1
                ids.append(_chunk_id(source, current_hashes.get(source, 'na'), source_counter[source]))
            store.add_documents(changed_chunks, ids=ids)

            for idx, chunk in enumerate(changed_chunks):
                source = chunk.metadata.get('source', 'unknown')
                db.add(
                    ChunkMetadata(
                        chunk_id=ids[idx],
                        source=source,
                        title=_safe_title(source),
                        content=chunk.page_content,
                    )
                )
            inserted_chunks = len(changed_chunks)

        for source in changed_sources:
            source_chunks = db.execute(select(ChunkMetadata).where(ChunkMetadata.source == source)).scalars().all()
            _upsert_doc_index(db, source, current_hashes[source], len(source_chunks))

        db.commit()

        return {
            'files_total': len(files),
            'files_changed': len(changed_sources),
            'files_removed': len(removed_sources),
            'chunks_inserted': inserted_chunks,
            'chunks_deleted': deleted_chunks,
            'migrated_full_reindex': 0,
            'needs_full_reindex': 0,
        }
    finally:
        db.close()


def ensure_index_ready_on_startup() -> dict[str, int | bool | str]:
    settings = get_settings()
    files = _scan_md_files(settings.KNOWLEDGE_BASE_DIR)
    if not files:
        return {
            'triggered': False,
            'reason': 'no_markdown_files',
            'files_total': 0,
            'chunks_inserted': 0,
            'chunks_deleted': 0,
            'files_changed': 0,
            'files_removed': 0,
        }

    collection_exists, row_count = _get_collection_row_count(settings.MILVUS_COLLECTION)

    if not collection_exists or row_count <= 0:
        result = incremental_reindex(allow_destructive_migration=False)
        logger.info(
            'startup.ingest triggered=True reason=collection_empty_or_missing files_total=%d chunks_inserted=%d chunks_deleted=%d',
            result['files_total'],
            result['chunks_inserted'],
            result['chunks_deleted'],
        )
        return {
            'triggered': True,
            'reason': 'collection_empty_or_missing',
            **result,
        }

    return {
        'triggered': False,
        'reason': 'collection_ready',
        'files_total': len(files),
        'chunks_inserted': 0,
        'chunks_deleted': 0,
        'files_changed': 0,
        'files_removed': 0,
    }
