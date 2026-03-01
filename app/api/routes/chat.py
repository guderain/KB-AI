import json

from fastapi import APIRouter, BackgroundTasks, Depends
from fastapi.responses import StreamingResponse

from app.api.deps.security import guard_read_write
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.rag_service_fast import ask, ask_stream, save_chat_log

router = APIRouter(prefix="/chat", tags=["chat"], dependencies=[Depends(guard_read_write)])


@router.post("", response_model=ChatResponse)
def chat(payload: ChatRequest, background_tasks: BackgroundTasks) -> ChatResponse:
    answer, sources, cache_hit = ask(payload.question)
    background_tasks.add_task(
        save_chat_log,
        payload.session_id,
        payload.question,
        answer,
        sources,
    )
    return ChatResponse(answer=answer, sources=sources, cache_hit=cache_hit)


@router.post("/stream")
def chat_stream(payload: ChatRequest) -> StreamingResponse:
    sources, chunks = ask_stream(payload.question)

    def event_stream():
        yield f"data: {json.dumps({'type': 'sources', 'data': sources}, ensure_ascii=False)}\n\n"
        for text in chunks:
            yield f"data: {json.dumps({'type': 'token', 'data': text}, ensure_ascii=False)}\n\n"
        yield "data: {\"type\":\"done\"}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
