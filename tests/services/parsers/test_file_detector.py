from pathlib import Path

from app.services.parsers.file_detector import FileType, FileTypeDetector


def test_detect_markdown():
    assert FileTypeDetector.detect(Path('doc.md')) == FileType.MARKDOWN


def test_detect_pdf():
    assert FileTypeDetector.detect(Path('doc.pdf')) == FileType.PDF


def test_detect_word_docx():
    assert FileTypeDetector.detect(Path('doc.docx')) == FileType.WORD


def test_detect_excel_xlsx():
    assert FileTypeDetector.detect(Path('data.xlsx')) == FileType.EXCEL


def test_detect_image_png():
    assert FileTypeDetector.detect(Path('img.png')) == FileType.IMAGE


def test_detect_unknown():
    assert FileTypeDetector.detect(Path('archive.zip')) == FileType.UNKNOWN


def test_detect_word_doc():
    assert FileTypeDetector.detect(Path('doc.doc')) == FileType.WORD


def test_detect_excel_xls():
    assert FileTypeDetector.detect(Path('data.xls')) == FileType.EXCEL


def test_detect_image_jpg():
    assert FileTypeDetector.detect(Path('img.jpg')) == FileType.IMAGE


def test_detect_image_jpeg():
    assert FileTypeDetector.detect(Path('img.jpeg')) == FileType.IMAGE


def test_detect_image_bmp():
    assert FileTypeDetector.detect(Path('img.bmp')) == FileType.IMAGE


def test_detect_image_gif():
    assert FileTypeDetector.detect(Path('img.gif')) == FileType.IMAGE


def test_detect_image_webp():
    assert FileTypeDetector.detect(Path('img.webp')) == FileType.IMAGE


def test_detect_case_insensitive():
    assert FileTypeDetector.detect(Path('doc.PDF')) == FileType.PDF
    assert FileTypeDetector.detect(Path('img.JPG')) == FileType.IMAGE
