from pathlib import Path

from app.core.config import get_settings
from app.services.parsers.image_parser import ImageParser


def test_image_parser_fallback_when_no_apis():
    parser = ImageParser()
    docs = parser.parse(Path('tests/fixtures/sample.png'))
    assert len(docs) == 1
    assert '[图片解析失败]' in docs[0].page_content
    assert docs[0].metadata['doc_type'] == 'image'


def test_image_parser_merge_ocr_and_caption(monkeypatch):
    parser = ImageParser()
    settings = get_settings()

    def mock_ocr(self, b64, mime):
        return 'OCR text here'

    def mock_caption(self, b64, mime):
        return 'Caption text here'

    monkeypatch.setattr(ImageParser, '_call_ocr', mock_ocr)
    monkeypatch.setattr(ImageParser, '_call_caption', mock_caption)
    monkeypatch.setattr(settings, 'OCR_API_ENDPOINT', 'http://mock-ocr')
    monkeypatch.setattr(settings, 'VISION_API_ENDPOINT', 'http://mock-vision')

    docs = parser.parse(Path('tests/fixtures/sample.png'))
    assert len(docs) == 1
    content = docs[0].page_content
    assert '[图片描述]' in content
    assert 'Caption text here' in content
    assert '[图片文字]' in content
    assert 'OCR text here' in content
