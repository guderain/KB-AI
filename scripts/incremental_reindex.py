import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.db.postgres import init_db
from app.services.ingestion_service import incremental_reindex


def main() -> None:
    init_db()
    result = incremental_reindex()
    print(result)


if __name__ == "__main__":
    main()

