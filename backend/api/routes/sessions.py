from typing import List
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import asc, desc
from sqlmodel import Session, col, select

from backend.api.dependencies import get_db
from backend.api.schemas import SessionMessageResponse, SessionSummaryResponse
from backend.core import ChatMessage, ChatSession, Evaluation


router = APIRouter()


@router.get("/sessions", response_model=List[SessionSummaryResponse])
async def list_sessions(db: Session = Depends(get_db)):
    sessions = db.exec(select(ChatSession).order_by(desc(col(ChatSession.created_at)))).all()

    return [
        SessionSummaryResponse(
            id=session.id,
            title=session.title,
            created_at=session.created_at,
            updated_at=session.updated_at,
        )
        for session in sessions
    ]


@router.get("/sessions/{session_id}/messages", response_model=List[SessionMessageResponse])
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
                reasoning=None if latest_evaluation is None else latest_evaluation.reasoning,
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
