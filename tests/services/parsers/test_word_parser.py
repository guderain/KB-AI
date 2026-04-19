from pathlib import Path

from app.services.parsers.word_parser import WordParser


def test_word_parser_extracts_text_and_table():
    parser = WordParser()
    docs = parser.parse(Path('tests/fixtures/sample.docx'))
    combined = ' '.join(d.page_content for d in docs)
    assert '第一章 简介' in combined
    assert '背景介绍内容' in combined
    assert all(d.metadata['doc_type'] == 'word' for d in docs)
