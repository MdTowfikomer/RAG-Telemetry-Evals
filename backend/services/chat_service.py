import asyncio
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from time import perf_counter
from typing import AsyncGenerator, Callable
from uuid import UUID

from sqlalchemy.orm import Session as SQLAlchemySession
from sqlmodel import Session as SQLModelSession

from backend.core import ChatMessage, ChatSession
from backend.core.exceptions import SessionNotFoundError


@dataclass
class ChatResult:
    id: UUID
    session_id: UUID
    query: str
    response: str
    source_documents: list[str]


@dataclass
class ChatStreamResult:
    stream: AsyncGenerator[str, None]


class ChatService:
    def __init__(
        self,
        *,
        tracer,
        pipeline_factory: Callable[[str | None], object],
        token_counter: Callable[[str], int],
        create_pending_evaluation_fn: Callable[..., object],
        evaluation_service,
    ) -> None:
        self._tracer = tracer
        self._pipeline_factory = pipeline_factory
        self._token_counter = token_counter
        self._create_pending_evaluation = create_pending_evaluation_fn
        self._evaluation_service = evaluation_service

    async def chat(
        self,
        *,
        query: str,
        session_id: UUID | None,
        k: int,
        model: str,
        db: SQLAlchemySession,
        task_spawner: Callable[..., None],
        parent_context=None,
    ) -> ChatResult:
        with self._tracer.start_as_current_span("chat_flow") as span:
            span.set_attribute("chat.query", query)

            if session_id:
                session = db.get(ChatSession, session_id)
                if not session:
                    raise SessionNotFoundError("Session not found")
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

            task_spawner(
                self._evaluation_service.trigger_evaluation,
                query,
                answer,
                contexts,
                evaluation_record.id,
                db.get_bind(),
                parent_context,
            )

            return ChatResult(
                id=assistant_msg.id,
                session_id=session.id,
                query=query,
                response=answer,
                source_documents=contexts,
            )

    def chat_stream(
        self,
        *,
        query: str,
        session_id: UUID | None,
        k: int,
        model: str,
        db: SQLAlchemySession,
    ) -> ChatStreamResult:
        with self._tracer.start_as_current_span("chat_stream_flow") as span:
            span.set_attribute("chat.query", query)

            if session_id:
                session = db.get(ChatSession, session_id)
                if not session:
                    raise SessionNotFoundError("Session not found")
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

            evaluation_record = self._create_pending_evaluation(
                db=db,
                message_id=assistant_msg.id,
            )

            stream = self._stream_response(
                query=query,
                k=k,
                model=model,
                session_id=session.id,
                user_message_id=user_msg.id,
                assistant_message_id=assistant_msg.id,
                db_bind=db.get_bind(),
                evaluation_id=evaluation_record.id,
            )

            return ChatStreamResult(stream=stream)

    async def _persist_assistant_message(
        self,
        *,
        assistant_message_id: UUID,
        content: str,
        db_bind,
        latency_ms: int | None = None,
    ) -> None:
        with SQLModelSession(db_bind) as persistence_db:
            assistant_msg = persistence_db.get(ChatMessage, assistant_message_id)
            if assistant_msg is None:
                return

            assistant_msg.content = content
            assistant_msg.latency_ms = latency_ms
            assistant_msg.token_count = self._token_counter(content)
            persistence_db.add(assistant_msg)

            session = persistence_db.get(ChatSession, assistant_msg.session_id)
            if session is not None:
                session.updated_at = datetime.now(UTC)
                persistence_db.add(session)

            persistence_db.commit()

    async def _stream_response(
        self,
        *,
        query: str,
        k: int,
        model: str,
        session_id: UUID,
        user_message_id: UUID,
        assistant_message_id: UUID,
        db_bind,
        evaluation_id: UUID,
    ) -> AsyncGenerator[str, None]:
        pipeline = self._pipeline_factory(model)
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
            await self._persist_assistant_message(
                assistant_message_id=assistant_message_id,
                content=final_answer,
                db_bind=db_bind,
                latency_ms=elapsed_ms,
            )

            if final_answer.strip():
                asyncio.create_task(
                    self._evaluation_service.trigger_stream_evaluation(
                        query=query,
                        answer=final_answer,
                        k=k,
                        model=model,
                        evaluation_id=evaluation_id,
                        db_bind=db_bind,
                    )
                )

        yield "data: [DONE]\n\n"
