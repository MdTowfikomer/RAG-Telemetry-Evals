import asyncio
import json
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from math import ceil
from time import monotonic, perf_counter
from typing import Any, AsyncGenerator, List, Optional
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
    mark_evaluation_completed,
    mark_evaluation_failed,
)
from backend.services.chat_service import ChatService

from .evaluation import EvalContext, RagasEvaluator


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


class ScoreBroadcaster:
    def __init__(self):
        self.subscribers: list[asyncio.Queue[dict[str, Any]]] = []

    def subscribe(self) -> asyncio.Queue[dict[str, Any]]:
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self.subscribers.append(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[dict[str, Any]]) -> None:
        if queue in self.subscribers:
            self.subscribers.remove(queue)

    async def publish(self, event: dict[str, Any]) -> None:
        for subscriber in list(self.subscribers):
            await subscriber.put(event)


score_broadcaster = ScoreBroadcaster()


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


async def publish_score_event_for_evaluation(
    evaluation: Evaluation,
    message: ChatMessage | None,
) -> None:
    await score_broadcaster.publish(
        {
            "type": "score",
            "message_id": str(evaluation.message_id),
            "faithfulness": evaluation.faithfulness,
            "answer_relevancy": evaluation.answer_relevancy,
            "reasoning": evaluation.reasoning,
            "status": evaluation.status,
            "version": evaluation.version,
            "latency_ms": None if message is None else message.latency_ms,
            "token_count": None if message is None else message.token_count,
        }
    )


async def evaluate_ragas(
    query: str,
    answer: str,
    contexts: List[str],
    evaluation_id: UUID,
    db_bind,
    parent_context=None,
):
    """
    Background task to compute Ragas metrics using the Evaluation Engine Service.
    """
    if parent_context:
        otel_context.attach(parent_context)

    try:
        print("Delegating to Evaluation Engine Service...")
        scores = await evaluator.evaluate(
            EvalContext(query=query, answer=answer, contexts=contexts)
        )
        raw_faithfulness = scores.get("faithfulness")
        faithfulness = (
            float(raw_faithfulness)
            if isinstance(raw_faithfulness, (int, float))
            else None
        )
        raw_answer_relevancy = scores.get("answer_relevancy")
        answer_relevancy = (
            float(raw_answer_relevancy)
            if isinstance(raw_answer_relevancy, (int, float))
            else None
        )
        raw_reasoning = scores.get("reasoning")
        reasoning = (
            raw_reasoning
            if isinstance(raw_reasoning, str) and raw_reasoning.strip()
            else "Ragas evaluation completed successfully."
        )

        with Session(db_bind) as evaluation_db:
            updated_eval = mark_evaluation_completed(
                db=evaluation_db,
                evaluation_id=evaluation_id,
                faithfulness=faithfulness,
                answer_relevancy=answer_relevancy,
                reasoning=reasoning,
            )

            if updated_eval is not None:
                message = evaluation_db.get(ChatMessage, updated_eval.message_id)
                await publish_score_event_for_evaluation(updated_eval, message)
    except Exception as e:
        with Session(db_bind) as evaluation_db:
            updated_eval = mark_evaluation_failed(
                db=evaluation_db,
                evaluation_id=evaluation_id,
                error_message=str(e),
            )

            if updated_eval is not None:
                await publish_score_event_for_evaluation(updated_eval, None)

        print(f"Error in Ragas evaluation delegation: {e}")


def estimate_token_count(content: str) -> int:
    return len(content.split())


def get_chat_service() -> ChatService:
    return ChatService(
        tracer=tracer,
        pipeline_factory=get_pipeline_for_model,
        token_counter=estimate_token_count,
        create_pending_evaluation_fn=create_pending_evaluation,
        evaluate_ragas_fn=evaluate_ragas,
        stream_response_factory=sse_stream_response,
    )


async def persist_assistant_message(
    assistant_message_id: UUID,
    content: str,
    db_bind,
    latency_ms: int | None = None,
) -> None:
    with Session(db_bind) as persistence_db:
        assistant_msg = persistence_db.get(ChatMessage, assistant_message_id)
        if assistant_msg is None:
            return

        assistant_msg.content = content
        assistant_msg.latency_ms = latency_ms
        assistant_msg.token_count = estimate_token_count(content)
        persistence_db.add(assistant_msg)

        session = persistence_db.get(ChatSession, assistant_msg.session_id)
        if session is not None:
            session.updated_at = datetime.now(UTC)
            persistence_db.add(session)

        persistence_db.commit()


async def evaluate_ragas_for_stream(
    query: str,
    answer: str,
    k: int,
    model: str,
    evaluation_id: UUID,
    db_bind,
) -> None:
    try:
        pipeline = get_pipeline_for_model(model)
        docs = await pipeline.prepare_context(query, k=k)
        contexts = [doc.page_content for doc in docs]
        await evaluate_ragas(
            query=query,
            answer=answer,
            contexts=contexts,
            evaluation_id=evaluation_id,
            db_bind=db_bind,
        )
    except Exception as e:
        print(f"Error in streaming evaluation delegation: {e}")


async def evaluate_ragas_for_existing_message(
    message_id: UUID,
    k: int,
    model: str,
    evaluation_id: UUID,
    db_bind,
) -> None:
    with Session(db_bind) as evaluation_db:
        assistant_message = evaluation_db.get(ChatMessage, message_id)
        if assistant_message is None:
            updated_eval = mark_evaluation_failed(
                db=evaluation_db,
                evaluation_id=evaluation_id,
                error_message="Assistant message not found for re-evaluation.",
            )
            if updated_eval is not None:
                await publish_score_event_for_evaluation(updated_eval, None)
            return

        if assistant_message.role != "assistant":
            updated_eval = mark_evaluation_failed(
                db=evaluation_db,
                evaluation_id=evaluation_id,
                error_message="Only assistant messages can be re-evaluated.",
            )
            if updated_eval is not None:
                await publish_score_event_for_evaluation(updated_eval, assistant_message)
            return

        latest_user_message = evaluation_db.exec(
            select(ChatMessage)
            .where(
                ChatMessage.session_id == assistant_message.session_id,
                ChatMessage.role == "user",
                ChatMessage.created_at <= assistant_message.created_at,
            )
            .order_by(desc(col(ChatMessage.created_at)))
        ).first()

        if latest_user_message is None:
            updated_eval = mark_evaluation_failed(
                db=evaluation_db,
                evaluation_id=evaluation_id,
                error_message="No user message found to evaluate this assistant answer.",
            )
            if updated_eval is not None:
                await publish_score_event_for_evaluation(updated_eval, assistant_message)
            return

        if not assistant_message.content.strip():
            updated_eval = mark_evaluation_failed(
                db=evaluation_db,
                evaluation_id=evaluation_id,
                error_message="Assistant message content is empty.",
            )
            if updated_eval is not None:
                await publish_score_event_for_evaluation(updated_eval, assistant_message)
            return

        query = latest_user_message.content
        answer = assistant_message.content

    await evaluate_ragas_for_stream(
        query=query,
        answer=answer,
        k=k,
        model=model,
        evaluation_id=evaluation_id,
        db_bind=db_bind,
    )


async def sse_stream_response(
    query: str,
    k: int,
    model: str,
    session_id: UUID,
    user_message_id: UUID,
    assistant_message_id: UUID,
    db_bind,
    evaluation_id: UUID,
) -> AsyncGenerator[str, None]:
    pipeline = get_pipeline_for_model(model)
    generated_tokens: list[str] = []
    stream_started_at = perf_counter()

    stream_meta_payload = json.dumps(
        {
            "type": "meta",
            "session_id": str(session_id),
            "user_message_id": str(user_message_id),
            "assistant_message_id": str(assistant_message_id),
        }
    )
    yield f"data: {stream_meta_payload}\n\n"

    try:
        async for token in pipeline.stream(query, k=k):
            generated_tokens.append(token)
            payload = json.dumps({"type": "token", "token": token})
            yield f"data: {payload}\n\n"
    finally:
        final_answer = "".join(generated_tokens)
        elapsed_ms = int((perf_counter() - stream_started_at) * 1000)
        await persist_assistant_message(
            assistant_message_id=assistant_message_id,
            content=final_answer,
            db_bind=db_bind,
            latency_ms=elapsed_ms,
        )

        if final_answer.strip():
            asyncio.create_task(
                evaluate_ragas_for_stream(
                    query=query,
                    answer=final_answer,
                    k=k,
                    model=model,
                    evaluation_id=evaluation_id,
                    db_bind=db_bind,
                )
            )

    yield "data: [DONE]\n\n"


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
    await publish_score_event_for_evaluation(evaluation_record, message)

    asyncio.create_task(
        evaluate_ragas_for_existing_message(
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
    async def event_generator() -> AsyncGenerator[str, None]:
        subscriber = score_broadcaster.subscribe()
        try:
            while True:
                try:
                    event = await asyncio.wait_for(subscriber.get(), timeout=15)
                    yield f"data: {json.dumps(event)}\n\n"
                except TimeoutError:
                    # SSE keep-alive to prevent idle intermediaries from dropping the connection.
                    yield ": keep-alive\n\n"
        finally:
            score_broadcaster.unsubscribe(subscriber)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/health")
async def health():
    return {"status": "healthy"}


if __name__ == "__main__":
    import sys

    import uvicorn

    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    uvicorn.run(app, host="0.0.0.0", port=8000)
