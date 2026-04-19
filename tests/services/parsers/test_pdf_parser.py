from pathlib import Path

from app.services.parsers.pdf_parser import PDFParser


def test_pdf_parser_extracts_text():
    parser = PDFParser()
    docs = parser.parse(Path('tests/fixtures/sample.pdf'))
    assert len(docs) >= 1
    combined = ' '.join(d.page_content for d in docs)
    assert 'Chapter 1' in combined or 'Introduction' in combined
    assert all(d.metadata['doc_type'] == 'pdf' for d in docs)
