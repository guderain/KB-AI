from pathlib import Path

from app.core.config import get_settings


def _resolve_safe_path(path: str) -> Path:
    settings = get_settings()
    kb_root = Path(settings.KNOWLEDGE_BASE_DIR).resolve()
    source_path = Path(path).resolve()
    if kb_root in source_path.parents or source_path == kb_root:
        return source_path

    # Compatibility: old index records may contain absolute paths from another OS
    # (for example D:\knowledge\*.md). In that case, fall back to matching filename
    # inside the current knowledge base root.
    legacy_name = Path(path.replace("\\", "/")).name.strip()
    if legacy_name:
        candidates = [p for p in kb_root.rglob(legacy_name) if p.is_file()]
        if len(candidates) == 1:
            return candidates[0].resolve()

    raise ValueError("source path is outside knowledge base directory")


def get_source_content(path: str) -> tuple[str, str, str]:
    source_path = _resolve_safe_path(path)
    if not source_path.exists() or not source_path.is_file():
        raise FileNotFoundError("source file not found")

    content = source_path.read_text(encoding="utf-8")
    return str(source_path), source_path.name, content
