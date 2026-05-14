from typing import Optional
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from fastapi.responses import StreamingResponse
from opentelemetry import context as otel_context
from sqlmodel import Session

from backend.api.dependencies import (
    get_chat_service,
    get_db,
    get_pipeline_for_model,
    settings,
)
from backend.api.schemas import ChatRequest, ChatResponse, ContextResponse
from backend.core import SessionNotFoundError
from backend.services.chat_service import ChatService


router = APIRouter()


@router.get("/")
async def root():
    return {"message": "RAG API is running", "docs": "/docs"}


@router.get("/health")
async def health():
    return {"status": "healthy"}


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(
    request: ChatRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    chat_service: ChatService = Depends(get_chat_service),
):
    try:
        current_context = otel_context.get_current()
        result = await chat_service.chat(
            query=request.query,
            session_id=request.session_id,
            k=request.k,
            model=request.model,
            db=db,
            task_spawner=background_tasks.add_task,
            parent_context=current_context,
        )
        return ChatResponse(
            id=result.id,
            session_id=result.session_id,
            query=result.query,
            response=result.response,
            source_documents=result.source_documents,
        )
    except SessionNotFoundError:
        raise HTTPException(status_code=404, detail="Session not found")
    except HTTPException:
        raise
    except Exception as error:
        print(f"Error in chat endpoint: {error}")
        raise HTTPException(status_code=500, detail=str(error))


@router.post("/context", response_model=ContextResponse)
async def context_endpoint(request: ChatRequest):
    pipeline = get_pipeline_for_model(request.model)
    docs = await pipeline.prepare_context(request.query, k=request.k)
    return ContextResponse(
        query=request.query,
        source_documents=[doc.page_content for doc in docs],
    )


@router.get("/chat/stream")
async def chat_stream_get_endpoint(
    query: str,
    k: int = 3,
    model: str = settings.openrouter_model,
    session_id: Optional[UUID] = None,
    db: Session = Depends(get_db),
    chat_service: ChatService = Depends(get_chat_service),
):
    try:
        stream_result = chat_service.chat_stream(
            query=query,
            session_id=session_id,
            k=k,
            model=model,
            db=db,
        )
    except SessionNotFoundError:
        raise HTTPException(status_code=404, detail="Session not found")

    return StreamingResponse(stream_result.stream, media_type="text/event-stream")


@router.post("/chat/stream")
async def chat_stream_endpoint(
    request: ChatRequest,
    db: Session = Depends(get_db),
    chat_service: ChatService = Depends(get_chat_service),
):
    try:
        stream_result = chat_service.chat_stream(
            query=request.query,
            session_id=request.session_id,
            k=request.k,
            model=request.model,
            db=db,
        )
    except SessionNotFoundError:
        raise HTTPException(status_code=404, detail="Session not found")

    return StreamingResponse(stream_result.stream, media_type="text/event-stream")
