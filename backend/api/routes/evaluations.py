import asyncio
from typing import List
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy import desc
from sqlmodel import Session, col, select

from backend.api.dependencies import (
    get_db,
    get_evaluation_service,
    reevaluate_rate_limiter,
)
from backend.api.schemas import (
    MessageEvaluationVersionResponse,
    ReevaluateRequest,
    build_message_evaluation_response,
)
from backend.core import ChatMessage, Evaluation
from backend.core.evaluation_store import create_pending_evaluation
from backend.services.evaluation_service import EvaluationService


router = APIRouter()


@router.get(
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


@router.post(
    "/messages/{message_id}/re-evaluate",
    response_model=MessageEvaluationVersionResponse,
    status_code=202,
)
async def reevaluate_assistant_message(
    message_id: UUID,
    reevaluate_request: ReevaluateRequest,
    http_request: Request,
    db: Session = Depends(get_db),
    evaluation_service: EvaluationService = Depends(get_evaluation_service),
):
    client_host = (
        "unknown"
        if http_request.client is None or http_request.client.host is None
        else http_request.client.host
    )
    allowed, retry_after = await reevaluate_rate_limiter.check(f"{client_host}:{message_id}")
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
