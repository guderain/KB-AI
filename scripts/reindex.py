import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.db.postgres import init_db
from app.services.ingestion_service import reindex


def main() -> None:
    init_db()
    files_loaded, chunks_indexed = reindex()
    print(f"files_loaded={files_loaded}, chunks_indexed={chunks_indexed}")


if __name__ == "__main__":
    main()
