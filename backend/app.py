import asyncio
import json
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from time import perf_counter
from typing import Any, AsyncGenerator, List, Optional
from uuid import UUID

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException
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
    Settings,
)
from backend.core.evaluation_store import (
    create_pending_evaluation,
    mark_evaluation_completed,
    mark_evaluation_failed,
)

try:
    from evaluation import EvalContext, RagasEvaluator
except ImportError:
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

retriever_adapter = QdrantRetriever(vectorstore=vectorstore, tracer=tracer)
reranker_adapter = PassThroughReranker(tracer=tracer)

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
        tracer=tracer,
    )

    pipeline = RAGPipeline(
        retriever=retriever_adapter,
        reranker=reranker_adapter,
        generator=generator_adapter,
        tracer=tracer,
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

        with Session(db_bind) as evaluation_db:
            updated_eval = mark_evaluation_completed(
                db=evaluation_db,
                evaluation_id=evaluation_id,
                faithfulness=scores.get("faithfulness"),
                answer_relevancy=scores.get("answer_relevancy"),
                reasoning=scores.get("reasoning")
                or "Ragas evaluation completed successfully.",
            )

            if updated_eval is not None:
                message = evaluation_db.get(ChatMessage, updated_eval.message_id)
                if message is not None:
                    await score_broadcaster.publish(
                        {
                            "type": "score",
                            "message_id": str(message.id),
                            "faithfulness": updated_eval.faithfulness,
                            "answer_relevancy": updated_eval.answer_relevancy,
                            "reasoning": updated_eval.reasoning,
                            "status": updated_eval.status,
                            "version": updated_eval.version,
                            "latency_ms": message.latency_ms,
                            "token_count": message.token_count,
                        }
                    )
    except Exception as e:
        with Session(db_bind) as evaluation_db:
            updated_eval = mark_evaluation_failed(
                db=evaluation_db,
                evaluation_id=evaluation_id,
                error_message=str(e),
            )

            if updated_eval is not None:
                await score_broadcaster.publish(
                    {
                        "type": "score",
                        "message_id": str(updated_eval.message_id),
                        "faithfulness": updated_eval.faithfulness,
                        "answer_relevancy": updated_eval.answer_relevancy,
                        "reasoning": updated_eval.reasoning,
                        "status": updated_eval.status,
                        "version": updated_eval.version,
                        "latency_ms": None,
                        "token_count": None,
                    }
                )

        print(f"Error in Ragas evaluation delegation: {e}")


def estimate_token_count(content: str) -> int:
    return len(content.split())


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
        with tracer.start_as_current_span("chat_flow") as span:
            span.set_attribute("chat.query", request.query)

            # 1. Get or Create Session
            if request.session_id:
                session = db.get(ChatSession, request.session_id)
                if not session:
                    raise HTTPException(status_code=404, detail="Session not found")
            else:
                session = ChatSession(title=request.query[:50] + "...")
                db.add(session)
                db.commit()
                db.refresh(session)

            # 2. Persist User Message
            user_msg = ChatMessage(
                session_id=session.id, role="user", content=request.query
            )
            db.add(user_msg)
            db.commit()
            db.refresh(user_msg)

            # 3. Execute RAG Pipeline
            pipeline = get_pipeline_for_model(request.model)
            response_started_at = perf_counter()
            answer, docs = await pipeline.execute(request.query, k=request.k)
            elapsed_ms = int((perf_counter() - response_started_at) * 1000)
            contexts = [doc.page_content for doc in docs]

            # 4. Persist Assistant Message
            assistant_msg = ChatMessage(
                session_id=session.id,
                role="assistant",
                content=answer,
                latency_ms=elapsed_ms,
                token_count=estimate_token_count(answer),
            )
            db.add(assistant_msg)
            db.commit()
            db.refresh(assistant_msg)

            # 5. Create Evaluation record and trigger async evaluation
            evaluation_record = create_pending_evaluation(
                db=db,
                message_id=assistant_msg.id,
            )

            current_context = otel_context.get_current()
            background_tasks.add_task(
                evaluate_ragas,
                request.query,
                answer,
                contexts,
                evaluation_record.id,
                db.get_bind(),
                current_context,
            )

            return ChatResponse(
                id=assistant_msg.id,
                session_id=session.id,
                query=request.query,
                response=answer,
                source_documents=contexts,
            )
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

    return [
        MessageEvaluationVersionResponse(
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
        for evaluation in evaluations
    ]


def prepare_stream_messages(
    db: Session,
    query: str,
    session_id: Optional[UUID],
) -> tuple[ChatSession, ChatMessage, ChatMessage, Evaluation]:
    if session_id:
        session = db.get(ChatSession, session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
    else:
        session = ChatSession(title=query[:50] + "...")
        db.add(session)
        db.commit()
        db.refresh(session)

    user_msg = ChatMessage(session_id=session.id, role="user", content=query)
    assistant_msg = ChatMessage(session_id=session.id, role="assistant", content="")
    db.add(user_msg)
    db.add(assistant_msg)

    session.updated_at = datetime.now(UTC)
    db.add(session)

    db.commit()
    db.refresh(user_msg)
    db.refresh(assistant_msg)

    evaluation_record = create_pending_evaluation(
        db=db,
        message_id=assistant_msg.id,
    )

    return session, user_msg, assistant_msg, evaluation_record


@app.get("/chat/stream")
async def chat_stream_get_endpoint(
    query: str,
    k: int = 3,
    model: str = settings.openrouter_model,
    session_id: Optional[UUID] = None,
    db: Session = Depends(get_db),
):
    session, user_msg, assistant_msg, evaluation_record = prepare_stream_messages(
        db=db,
        query=query,
        session_id=session_id,
    )

    return StreamingResponse(
        sse_stream_response(
            query=query,
            k=k,
            model=model,
            session_id=session.id,
            user_message_id=user_msg.id,
            assistant_message_id=assistant_msg.id,
            db_bind=db.get_bind(),
            evaluation_id=evaluation_record.id,
        ),
        media_type="text/event-stream",
    )


@app.post("/chat/stream")
async def chat_stream_endpoint(request: ChatRequest, db: Session = Depends(get_db)):
    session, user_msg, assistant_msg, evaluation_record = prepare_stream_messages(
        db=db,
        query=request.query,
        session_id=request.session_id,
    )

    return StreamingResponse(
        sse_stream_response(
            query=request.query,
            k=request.k,
            model=request.model,
            session_id=session.id,
            user_message_id=user_msg.id,
            assistant_message_id=assistant_msg.id,
            db_bind=db.get_bind(),
            evaluation_id=evaluation_record.id,
        ),
        media_type="text/event-stream",
    )


@app.get("/scores/stream")
async def scores_stream_endpoint():
    async def event_generator() -> AsyncGenerator[str, None]:
        subscriber = score_broadcaster.subscribe()
        try:
            while True:
                event = await subscriber.get()
                yield f"data: {json.dumps(event)}\n\n"
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
