from pathlib import Path

from app.services.parsers.excel_parser import ExcelParser


def test_excel_parser_outputs_row_chunks_and_summary():
    parser = ExcelParser()
    docs = parser.parse(Path('tests/fixtures/sample.xlsx'))

    row_docs = [d for d in docs if d.metadata.get('element_type') == 'row']
    summary_docs = [d for d in docs if d.metadata.get('element_type') == 'summary']

    assert len(row_docs) == 3
    assert len(summary_docs) == 1

    first = row_docs[0]
    assert 'A产品' in first.page_content
    assert first.metadata['doc_type'] == 'excel'
    assert first.metadata['sheet'] == '销售数据'
    assert first.metadata['row_index'] == 0

    summary = summary_docs[0]
    assert '销售数据' in summary.page_content
    assert '3 行数据' in summary.page_content
