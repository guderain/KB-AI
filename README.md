# KB-AI (FastAPI + LangChain RAG)

## Quick Start

1. Copy env:

```powershell
Copy-Item .env.example .env
```

2. Start dependencies:

```powershell
docker compose up -d
```

3. Install Python deps:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

4. Run API:

```powershell
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

5. Build index (manual full rebuild):

```powershell
python scripts/reindex.py
```

## API
- `GET /api/v1/health`
- `POST /api/v1/chat`
- `POST /api/v1/chat/stream`
- `POST /api/v1/ingest/reindex`
- `POST /api/v1/ingest/incremental`

## Notes
- Knowledge base defaults to `D:/知识库`.
- Frontend (NotionNext) can call this service through `KB_AI_BASE_URL` in Next.js server env.
- Set `AUTO_INGEST_ON_STARTUP=true` in production to auto-ingest when Milvus collection is missing or empty.
- If you need to force rebuild online data, call `POST /api/v1/ingest/reindex` with `X-API-Key`.
