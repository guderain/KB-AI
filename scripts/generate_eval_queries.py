import json
import re
from pathlib import Path

from app.core.config import get_settings


def source_title(path: Path) -> str:
    name = path.stem
    # Remove trailing notion-style uuid/hash suffix.
    return re.sub(r"\s+[0-9a-f]{32,}$", "", name, flags=re.IGNORECASE).strip() or name


def main() -> None:
    settings = get_settings()
    root = Path(settings.KNOWLEDGE_BASE_DIR)
    files = sorted(root.rglob("*.md"))[:30]
    rows: list[dict[str, str]] = []
    for path in files:
        title = source_title(path)
        rows.append(
            {
                "question": f"请概括《{title}》的核心内容。",
                "expected_source_keyword": title[:40],
            }
        )

    output = Path("scripts/eval_queries.auto30.json")
    output.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"generated={len(rows)} -> {output}")


if __name__ == "__main__":
    main()

