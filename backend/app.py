import asyncio
import json
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import AsyncGenerator, List, Optional
from uuid import UUID

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from opentelemetry import context as otel_context
from opentelemetry import trace as otel_trace
from pydantic import BaseModel, SecretStr
from sqlmodel import Session, select

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


@asynccontextmanager
async def lifespan(app: FastAPI):
    factory.init_db()
    yield


app = FastAPI(title="Modular RAG API", lifespan=lifespan)


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
    created_at: datetime


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
            mark_evaluation_completed(
                db=evaluation_db,
                evaluation_id=evaluation_id,
                faithfulness=scores.get("faithfulness"),
                answer_relevancy=scores.get("answer_relevancy"),
            )
    except Exception as e:
        with Session(db_bind) as evaluation_db:
            mark_evaluation_failed(
                db=evaluation_db,
                evaluation_id=evaluation_id,
                error_message=str(e),
            )

        print(f"Error in Ragas evaluation delegation: {e}")


async def persist_assistant_message(
    assistant_message_id: UUID,
    content: str,
    db_bind,
) -> None:
    with Session(db_bind) as persistence_db:
        assistant_msg = persistence_db.get(ChatMessage, assistant_message_id)
        if assistant_msg is None:
            return

        assistant_msg.content = content
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
        await persist_assistant_message(
            assistant_message_id=assistant_message_id,
            content=final_answer,
            db_bind=db_bind,
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
            answer, docs = await pipeline.execute(request.query, k=request.k)
            contexts = [doc.page_content for doc in docs]

            # 4. Persist Assistant Message
            assistant_msg = ChatMessage(
                session_id=session.id, role="assistant", content=answer
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
        select(ChatSession).order_by(ChatSession.created_at.desc())
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
        .order_by(ChatMessage.created_at.asc())
    ).all()

    return [
        SessionMessageResponse(
            id=message.id,
            session_id=message.session_id,
            role=message.role,
            content=message.content,
            created_at=message.created_at,
        )
        for message in messages
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


@app.get("/health")
async def health():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
