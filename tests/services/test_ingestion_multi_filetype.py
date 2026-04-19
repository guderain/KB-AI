import os
import shutil
import tempfile

import pytest

from app.core.config import get_settings
from app.db.postgres import SessionLocal
from app.models.doc_index import DocIndex
from app.services.ingestion_service import reindex


@pytest.fixture
def temp_kb_dir():
    settings = get_settings()
    original_dir = settings.KNOWLEDGE_BASE_DIR
    with tempfile.TemporaryDirectory() as tmpdir:
        settings.KNOWLEDGE_BASE_DIR = tmpdir
        yield tmpdir
        settings.KNOWLEDGE_BASE_DIR = original_dir


def test_reindex_ingests_excel_and_image(temp_kb_dir):
    for name in ['sample.xlsx', 'sample.png']:
        shutil.copy(f'tests/fixtures/{name}', os.path.join(temp_kb_dir, name))

    files_count, chunks_count = reindex()
    assert files_count == 2
    assert chunks_count > 0

    db = SessionLocal()
    try:
        rows = db.query(DocIndex).all()
        sources = {r.source for r in rows}
        assert any('sample.xlsx' in s for s in sources)
        assert any('sample.png' in s for s in sources)
    finally:
        db.close()
