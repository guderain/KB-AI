import logging
from pathlib import Path

from langchain_core.documents import Document
from pypdf import PdfReader

logger = logging.getLogger(__name__)


class PDFParser:
    def parse(self, path: Path) -> list[Document]:
        docs: list[Document] = []
        try:
            reader = PdfReader(str(path))
        except Exception as exc:
            logger.warning('Failed to open PDF %s: %s', path, exc)
            return docs

        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ''
            if not text.strip():
                continue
            metadata = {
                'source': str(path),
                'doc_type': 'pdf',
                'page': page_num,
                'element_type': 'text',
            }
            docs.append(Document(page_content=text.strip(), metadata=metadata))

        return docs
