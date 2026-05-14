import asyncio
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from datetime import datetime
from math import ceil
from time import monotonic
from typing import List, Optional
from uuid import UUID

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from opentelemetry import context as otel_context
from opentelemetry import trace as otel_trace
from pydantic import BaseModel, SecretStr
from sqlalchemy import asc, desc
from sqlmodel import Session, col, select

from backend.adapters import OpenRouterGenerator, PassThroughReranker, QdrantRetriever
from backend.core import (
    ChatMessage,
    ChatSession,
    Evaluation,
    InfrastructureFactory,
    RAGPipeline,
    SessionNotFoundError,
    Settings,
    TracingHook,
)
from backend.core.evaluation_store import (
    create_pending_evaluation,
)
from backend.services.chat_service import ChatService
from backend.services.evaluation_service import EvaluationService

from .evaluation import RagasEvaluator


settings = Settings()
factory = InfrastructureFactory(settings)
factory.setup_tracing(service_name="rag-backend")

if settings.openrouter_api_key is None:
    raise RuntimeError("OPENROUTER_API_KEY is required")

openrouter_api_key = settings.openrouter_api_key


def run_startup_migrations() -> None:
    engine = factory.get_engine()

    if engine.dialect.name != "sqlite":
        return

    with engine.connect() as connection:
        chatmessage_columns = {
            row[1]
            for row in connection.exec_driver_sql("PRAGMA table_info(chatmessage)")
        }
        if "latency_ms" not in chatmessage_columns:
            connection.exec_driver_sql(
                "ALTER TABLE chatmessage ADD COLUMN latency_ms INTEGER"
            )
        if "token_count" not in chatmessage_columns:
            connection.exec_driver_sql(
                "ALTER TABLE chatmessage ADD COLUMN token_count INTEGER"
            )

        evaluation_columns = {
            row[1]
            for row in connection.exec_driver_sql("PRAGMA table_info(evaluation)")
        }
        if "reasoning" not in evaluation_columns:
            connection.exec_driver_sql(
                "ALTER TABLE evaluation ADD COLUMN reasoning VARCHAR"
            )

        connection.commit()


@asynccontextmanager
async def lifespan(app: FastAPI):
    factory.init_db()
    run_startup_migrations()
    yield


app = FastAPI(title="Modular RAG API", lifespan=lifespan)


class InMemoryRateLimiter:
    def __init__(self, max_calls: int, period_seconds: float):
        self.max_calls = max_calls
        self.period_seconds = period_seconds
        self._calls: dict[str, deque[float]] = defaultdict(deque)
        self._lock = asyncio.Lock()

    async def check(self, key: str) -> tuple[bool, int]:
        now = monotonic()
        cutoff = now - self.period_seconds
        async with self._lock:
            bucket = self._calls[key]
            while bucket and bucket[0] < cutoff:
                bucket.popleft()

            if len(bucket) >= self.max_calls:
                retry_after = max(1, ceil(self.period_seconds - (now - bucket[0])))
                return False, retry_after

            bucket.append(now)
            return True, 0

    async def reset(self) -> None:
        async with self._lock:
            self._calls.clear()


reevaluate_rate_limiter = InMemoryRateLimiter(max_calls=3, period_seconds=60)


def get_db():
    yield from factory.get_session()


# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

tracer = otel_trace.get_tracer(__name__)
embeddings = factory.get_embeddings()
vectorstore = factory.get_vectorstore()

retriever_adapter = QdrantRetriever(vectorstore=vectorstore)
reranker_adapter = PassThroughReranker()

pipeline_cache: dict[str, RAGPipeline] = {}


def get_openrouter_api_key() -> SecretStr:
    return openrouter_api_key


def get_pipeline_for_model(model: str | None) -> RAGPipeline:
    selected_model = model or settings.openrouter_model
    cached = pipeline_cache.get(selected_model)
    if cached is not None:
        return cached

    generator_adapter = OpenRouterGenerator(
        api_key_provider=get_openrouter_api_key,
        default_model=selected_model,
    )

    pipeline = RAGPipeline(
        retriever=retriever_adapter,
        reranker=reranker_adapter,
        generator=generator_adapter,
        hooks=[TracingHook(tracer=tracer)],
    )

    pipeline_cache[selected_model] = pipeline
    return pipeline


evaluator = RagasEvaluator(
    api_key=openrouter_api_key.get_secret_value(),
    eval_model=settings.ragas_eval_model,
    embeddings=embeddings,
)

evaluation_service = EvaluationService(
    evaluator_provider=lambda: evaluator,
    pipeline_factory=lambda model: get_pipeline_for_model(model),
)


# Models
class ChatRequest(BaseModel):
    query: str
    session_id: Optional[UUID] = None
    k: int = 3
    model: str = settings.openrouter_model


class ChatResponse(BaseModel):
    id: UUID
    session_id: UUID
    query: str
    response: str
    source_documents: List[str]


class ContextResponse(BaseModel):
    query: str
    source_documents: List[str]


class SessionSummaryResponse(BaseModel):
    id: UUID
    title: str
    created_at: datetime
    updated_at: datetime


class SessionMessageResponse(BaseModel):
    id: UUID
    session_id: UUID
    role: str
    content: str
    latency_ms: int | None = None
    token_count: int | None = None
    faithfulness: float | None = None
    answer_relevancy: float | None = None
    reasoning: str | None = None
    evaluation_status: str | None = None
    evaluation_version: int | None = None
    created_at: datetime


class MessageEvaluationVersionResponse(BaseModel):
    id: UUID
    message_id: UUID
    version: int
    status: str
    faithfulness: float | None = None
    answer_relevancy: float | None = None
    reasoning: str | None = None
    error_message: str | None = None
    created_at: datetime
    updated_at: datetime


class ReevaluateRequest(BaseModel):
    k: int = 3
    model: str = settings.openrouter_model


def build_message_evaluation_response(
    evaluation: Evaluation,
) -> MessageEvaluationVersionResponse:
    return MessageEvaluationVersionResponse(
        id=evaluation.id,
        message_id=evaluation.message_id,
        version=evaluation.version,
        status=evaluation.status,
        faithfulness=evaluation.faithfulness,
        answer_relevancy=evaluation.answer_relevancy,
        reasoning=evaluation.reasoning,
        error_message=evaluation.error_message,
        created_at=evaluation.created_at,
        updated_at=evaluation.updated_at,
    )


def estimate_token_count(content: str) -> int:
    return len(content.split())


def get_chat_service() -> ChatService:
    return ChatService(
        tracer=tracer,
        pipeline_factory=get_pipeline_for_model,
        token_counter=estimate_token_count,
        create_pending_evaluation_fn=create_pending_evaluation,
        evaluation_service=evaluation_service,
    )


@app.get("/")
async def root():
    return {"message": "RAG API is running", "docs": "/docs"}


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(
    request: ChatRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    try:
        chat_service = get_chat_service()
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
    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/context", response_model=ContextResponse)
async def context_endpoint(request: ChatRequest):
    pipeline = get_pipeline_for_model(request.model)
    docs = await pipeline.prepare_context(request.query, k=request.k)

    return ContextResponse(
        query=request.query,
        source_documents=[doc.page_content for doc in docs],
    )


@app.get("/sessions", response_model=List[SessionSummaryResponse])
async def list_sessions(db: Session = Depends(get_db)):
    sessions = db.exec(
        select(ChatSession).order_by(desc(col(ChatSession.created_at)))
    ).all()

    return [
        SessionSummaryResponse(
            id=session.id,
            title=session.title,
            created_at=session.created_at,
            updated_at=session.updated_at,
        )
        for session in sessions
    ]


@app.get("/sessions/{session_id}/messages", response_model=List[SessionMessageResponse])
async def get_session_messages(session_id: UUID, db: Session = Depends(get_db)):
    session = db.get(ChatSession, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    messages = db.exec(
        select(ChatMessage)
        .where(ChatMessage.session_id == session_id)
        .order_by(asc(col(ChatMessage.created_at)))
    ).all()

    response_messages: list[SessionMessageResponse] = []

    for message in messages:
        latest_evaluation = db.exec(
            select(Evaluation)
            .where(Evaluation.message_id == message.id)
            .order_by(desc(col(Evaluation.version)))
        ).first()

        response_messages.append(
            SessionMessageResponse(
                id=message.id,
                session_id=message.session_id,
                role=message.role,
                content=message.content,
                latency_ms=message.latency_ms,
                token_count=message.token_count,
                faithfulness=None
                if latest_evaluation is None
                else latest_evaluation.faithfulness,
                answer_relevancy=None
                if latest_evaluation is None
                else latest_evaluation.answer_relevancy,
                reasoning=None
                if latest_evaluation is None
                else latest_evaluation.reasoning,
                evaluation_status=None
                if latest_evaluation is None
                else latest_evaluation.status,
                evaluation_version=None
                if latest_evaluation is None
                else latest_evaluation.version,
                created_at=message.created_at,
            )
        )

    return response_messages


@app.get(
    "/messages/{message_id}/evaluations",
    response_model=List[MessageEvaluationVersionResponse],
)
async def get_message_evaluations(message_id: UUID, db: Session = Depends(get_db)):
    message = db.get(ChatMessage, message_id)
    if not message:
        raise HTTPException(status_code=404, detail="Message not found")

    evaluations = db.exec(
        select(Evaluation)
        .where(Evaluation.message_id == message_id)
        .order_by(desc(col(Evaluation.version)))
    ).all()

    return [build_message_evaluation_response(evaluation) for evaluation in evaluations]


@app.post(
    "/messages/{message_id}/re-evaluate",
    response_model=MessageEvaluationVersionResponse,
    status_code=202,
)
async def reevaluate_assistant_message(
    message_id: UUID,
    reevaluate_request: ReevaluateRequest,
    http_request: Request,
    db: Session = Depends(get_db),
):
    client_host = (
        "unknown"
        if http_request.client is None or http_request.client.host is None
        else http_request.client.host
    )
    allowed, retry_after = await reevaluate_rate_limiter.check(
        f"{client_host}:{message_id}"
    )
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail="Too many re-evaluation requests. Try again shortly.",
            headers={"Retry-After": str(retry_after)},
        )

    message = db.get(ChatMessage, message_id)
    if not message:
        raise HTTPException(status_code=404, detail="Message not found")
    if message.role != "assistant":
        raise HTTPException(
            status_code=400,
            detail="Only assistant messages can be re-evaluated",
        )

    evaluation_record = create_pending_evaluation(
        db=db,
        message_id=message.id,
    )
    await evaluation_service.publish_score_event_for_evaluation(evaluation_record, message)

    asyncio.create_task(
        evaluation_service.trigger_reevaluation(
            message_id=message.id,
            k=reevaluate_request.k,
            model=reevaluate_request.model,
            evaluation_id=evaluation_record.id,
            db_bind=db.get_bind(),
        )
    )

    return build_message_evaluation_response(evaluation_record)


@app.get("/chat/stream")
async def chat_stream_get_endpoint(
    query: str,
    k: int = 3,
    model: str = settings.openrouter_model,
    session_id: Optional[UUID] = None,
    db: Session = Depends(get_db),
):
    try:
        chat_service = get_chat_service()
        stream_result = chat_service.chat_stream(
            query=query,
            session_id=session_id,
            k=k,
            model=model,
            db=db,
        )
    except SessionNotFoundError:
        raise HTTPException(status_code=404, detail="Session not found")

    return StreamingResponse(
        stream_result.stream,
        media_type="text/event-stream",
    )


@app.post("/chat/stream")
async def chat_stream_endpoint(request: ChatRequest, db: Session = Depends(get_db)):
    try:
        chat_service = get_chat_service()
        stream_result = chat_service.chat_stream(
            query=request.query,
            session_id=request.session_id,
            k=request.k,
            model=request.model,
            db=db,
        )
    except SessionNotFoundError:
        raise HTTPException(status_code=404, detail="Session not found")

    return StreamingResponse(
        stream_result.stream,
        media_type="text/event-stream",
    )


@app.get("/scores/stream")
async def scores_stream_endpoint():
    return StreamingResponse(
        evaluation_service.score_event_stream(),
        media_type="text/event-stream",
    )


@app.get("/health")
async def health():
    return {"status": "healthy"}


if __name__ == "__main__":
    import sys

    import uvicorn

    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    uvicorn.run(app, host="0.0.0.0", port=8000)
