from pathlib import Path

from app.services.parsers import ParserPipeline


def test_pipeline_parses_excel():
    pipeline = ParserPipeline()
    docs = pipeline.parse_file(Path('tests/fixtures/sample.xlsx'))
    assert any(d.metadata['doc_type'] == 'excel' for d in docs)


def test_pipeline_parses_image():
    pipeline = ParserPipeline()
    docs = pipeline.parse_file(Path('tests/fixtures/sample.png'))
    assert len(docs) == 1
    assert docs[0].metadata['doc_type'] == 'image'


def test_pipeline_parses_pdf():
    pipeline = ParserPipeline()
    docs = pipeline.parse_file(Path('tests/fixtures/sample.pdf'))
    assert any(d.metadata['doc_type'] == 'pdf' for d in docs)


def test_pipeline_parses_word():
    pipeline = ParserPipeline()
    docs = pipeline.parse_file(Path('tests/fixtures/sample.docx'))
    assert any(d.metadata['doc_type'] == 'word' for d in docs)


def test_pipeline_fallback_for_plaintext():
    pipeline = ParserPipeline()
    docs = pipeline.parse_file(Path('tests/services/parsers/test_pipeline.py'))
    assert len(docs) >= 1
    assert docs[0].metadata.get('source', '').endswith('test_pipeline.py')
