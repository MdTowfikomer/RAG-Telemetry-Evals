from datetime import datetime
from typing import List, Optional
from uuid import UUID

from pydantic import BaseModel

from backend.api.dependencies import settings
from backend.core import Evaluation


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
