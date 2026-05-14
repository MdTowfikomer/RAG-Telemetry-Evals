from dataclasses import dataclass
from time import perf_counter
from typing import Awaitable, Callable
from uuid import UUID

from fastapi import BackgroundTasks, HTTPException
from opentelemetry import context as otel_context
from sqlalchemy.orm import Session as SQLAlchemySession

from backend.core import ChatMessage, ChatSession


@dataclass
class ChatResult:
    id: UUID
    session_id: UUID
    query: str
    response: str
    source_documents: list[str]


class ChatService:
    def __init__(
        self,
        *,
        tracer,
        pipeline_factory: Callable[[str | None], object],
        token_counter: Callable[[str], int],
        create_pending_evaluation_fn: Callable[..., object],
        evaluate_ragas_fn: Callable[..., Awaitable[None]],
    ) -> None:
        self._tracer = tracer
        self._pipeline_factory = pipeline_factory
        self._token_counter = token_counter
        self._create_pending_evaluation = create_pending_evaluation_fn
        self._evaluate_ragas = evaluate_ragas_fn

    async def chat(
        self,
        *,
        query: str,
        session_id: UUID | None,
        k: int,
        model: str,
        db: SQLAlchemySession,
        background_tasks: BackgroundTasks,
    ) -> ChatResult:
        with self._tracer.start_as_current_span("chat_flow") as span:
            span.set_attribute("chat.query", query)

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
            db.add(user_msg)
            db.commit()
            db.refresh(user_msg)

            pipeline = self._pipeline_factory(model)
            response_started_at = perf_counter()
            answer, docs = await pipeline.execute(query, k=k)
            elapsed_ms = int((perf_counter() - response_started_at) * 1000)
            contexts = [doc.page_content for doc in docs]

            assistant_msg = ChatMessage(
                session_id=session.id,
                role="assistant",
                content=answer,
                latency_ms=elapsed_ms,
                token_count=self._token_counter(answer),
            )
            db.add(assistant_msg)
            db.commit()
            db.refresh(assistant_msg)

            evaluation_record = self._create_pending_evaluation(
                db=db,
                message_id=assistant_msg.id,
            )

            current_context = otel_context.get_current()
            background_tasks.add_task(
                self._evaluate_ragas,
                query,
                answer,
                contexts,
                evaluation_record.id,
                db.get_bind(),
                current_context,
            )

            return ChatResult(
                id=assistant_msg.id,
                session_id=session.id,
                query=query,
                response=answer,
                source_documents=contexts,
            )
