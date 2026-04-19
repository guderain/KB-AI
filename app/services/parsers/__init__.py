from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document

from app.services.parsers.excel_parser import ExcelParser
from app.services.parsers.file_detector import FileType, FileTypeDetector
from app.services.parsers.image_parser import ImageParser
from app.services.parsers.pdf_parser import PDFParser
from app.services.parsers.word_parser import WordParser


class ParserPipeline:
    def __init__(self) -> None:
        self._detector = FileTypeDetector()
        self._parsers = {
            FileType.EXCEL: ExcelParser(),
            FileType.IMAGE: ImageParser(),
            FileType.PDF: PDFParser(),
            FileType.WORD: WordParser(),
        }

    def parse_file(self, path: Path) -> list[Document]:
        file_type = self._detector.detect(path)

        if file_type == FileType.MARKDOWN or file_type == FileType.UNKNOWN:
            return self._load_plaintext(path)

        parser = self._parsers.get(file_type)
        if parser is None:
            return self._load_plaintext(path)

        return parser.parse(path)

    def _load_plaintext(self, path: Path) -> list[Document]:
        loader = TextLoader(str(path), encoding='utf-8')
        docs = loader.load()
        if not docs:
            return [Document(page_content='', metadata={'source': str(path)})]
        return docs
