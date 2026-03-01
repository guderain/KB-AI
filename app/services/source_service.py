from pathlib import Path

from app.core.config import get_settings


def _resolve_safe_path(path: str) -> Path:
    settings = get_settings()
    kb_root = Path(settings.KNOWLEDGE_BASE_DIR).resolve()
    source_path = Path(path).resolve()
    if kb_root not in source_path.parents and source_path != kb_root:
        raise ValueError("source path is outside knowledge base directory")
    return source_path


def get_source_content(path: str) -> tuple[str, str, str]:
    source_path = _resolve_safe_path(path)
    if not source_path.exists() or not source_path.is_file():
        raise FileNotFoundError("source file not found")

    content = source_path.read_text(encoding="utf-8")
    return str(source_path), source_path.name, content

