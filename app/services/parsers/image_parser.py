import base64
import logging
from pathlib import Path

from langchain_core.documents import Document

from app.core.config import get_settings

logger = logging.getLogger(__name__)


class ImageParser:
    def parse(self, path: Path) -> list[Document]:
        settings = get_settings()
        if not settings.ENABLE_IMAGE_OCR and not settings.ENABLE_IMAGE_CAPTION:
            return []

        mime = self._guess_mime(path)
        try:
            b64 = self._encode_image(path)
        except Exception as exc:
            logger.warning('Failed to read image %s: %s', path, exc)
            return self._fallback_doc(path)

        text_ocr = ''
        text_caption = ''

        if settings.ENABLE_IMAGE_OCR and settings.OCR_API_ENDPOINT:
            try:
                text_ocr = self._call_ocr(b64, mime)
            except Exception as exc:
                logger.warning('OCR failed for %s: %s', path, exc)

        if settings.ENABLE_IMAGE_CAPTION and settings.VISION_API_ENDPOINT:
            try:
                text_caption = self._call_caption(b64, mime)
            except Exception as exc:
                logger.warning('Vision caption failed for %s: %s', path, exc)

        if not text_ocr and not text_caption:
            return self._fallback_doc(path)

        page_content = ''
        if text_caption:
            page_content += f'[图片描述]\n{text_caption}\n'
        if text_ocr:
            page_content += f'[图片文字]\n{text_ocr}'

        return [
            Document(
                page_content=page_content.strip(),
                metadata={
                    'source': str(path),
                    'doc_type': 'image',
                    'element_type': 'image',
                },
            )
        ]

    def _guess_mime(self, path: Path) -> str:
        suffix = path.suffix.lower()
        mapping = {
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.bmp': 'image/bmp',
            '.gif': 'image/gif',
            '.webp': 'image/webp',
        }
        return mapping.get(suffix, 'image/jpeg')

    def _encode_image(self, path: Path) -> str:
        return base64.b64encode(path.read_bytes()).decode('utf-8')

    def _fallback_doc(self, path: Path) -> list[Document]:
        return [
            Document(
                page_content=f'[图片解析失败] {path.name}',
                metadata={
                    'source': str(path),
                    'doc_type': 'image',
                    'element_type': 'image',
                },
            )
        ]

    def _call_ocr(self, b64: str, mime: str) -> str:
        """Placeholder: actual OCR API call to be implemented in Task 7."""
        return ''

    def _call_caption(self, b64: str, mime: str) -> str:
        """Placeholder: actual Vision API call to be implemented in Task 7."""
        return ''
