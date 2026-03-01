from fastapi import APIRouter, Depends, HTTPException, Query

from app.api.deps.security import guard_read_write
from app.schemas.source import SourceContentResponse
from app.services.source_service import get_source_content

router = APIRouter(prefix="/sources", tags=["sources"], dependencies=[Depends(guard_read_write)])


@router.get("/content", response_model=SourceContentResponse)
def source_content(path: str = Query(..., min_length=1)) -> SourceContentResponse:
    try:
        safe_path, title, content = get_source_content(path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return SourceContentResponse(path=safe_path, title=title, content=content)
