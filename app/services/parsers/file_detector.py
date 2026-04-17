from enum import Enum, auto
from pathlib import Path


class FileType(Enum):
    MARKDOWN = auto()
    PDF = auto()
    WORD = auto()
    EXCEL = auto()
    IMAGE = auto()
    UNKNOWN = auto()


class FileTypeDetector:
    _ext_map: dict[str, FileType] = {
        '.md': FileType.MARKDOWN,
        '.pdf': FileType.PDF,
        '.doc': FileType.WORD,
        '.docx': FileType.WORD,
        '.xls': FileType.EXCEL,
        '.xlsx': FileType.EXCEL,
        '.png': FileType.IMAGE,
        '.jpg': FileType.IMAGE,
        '.jpeg': FileType.IMAGE,
        '.bmp': FileType.IMAGE,
        '.gif': FileType.IMAGE,
        '.webp': FileType.IMAGE,
    }

    @classmethod
    def detect(cls, path: Path) -> FileType:
        ext = path.suffix.lower()
        return cls._ext_map.get(ext, FileType.UNKNOWN)
