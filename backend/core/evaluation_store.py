from datetime import UTC, datetime
from uuid import UUID

from sqlmodel import Session, select

from .models import Evaluation


def get_next_evaluation_version(db: Session, message_id: UUID) -> int:
    latest = db.exec(
        select(Evaluation)
        .where(Evaluation.message_id == message_id)
        .order_by(Evaluation.version.desc())
    ).first()

    if latest is None:
        return 1

    return latest.version + 1


def create_pending_evaluation(db: Session, message_id: UUID) -> Evaluation:
    evaluation = Evaluation(
        message_id=message_id,
        version=get_next_evaluation_version(db, message_id),
        status="pending",
    )
    db.add(evaluation)
    db.commit()
    db.refresh(evaluation)
    return evaluation


def mark_evaluation_completed(
    db: Session,
    evaluation_id: UUID,
    faithfulness: float | None,
    answer_relevancy: float | None,
) -> Evaluation | None:
    evaluation = db.get(Evaluation, evaluation_id)
    if evaluation is None:
        return None

    evaluation.status = "completed"
    evaluation.faithfulness = faithfulness
    evaluation.answer_relevancy = answer_relevancy
    evaluation.error_message = None
    evaluation.updated_at = datetime.now(UTC)

    db.add(evaluation)
    db.commit()
    db.refresh(evaluation)
    return evaluation


def mark_evaluation_failed(
    db: Session,
    evaluation_id: UUID,
    error_message: str,
) -> Evaluation | None:
    evaluation = db.get(Evaluation, evaluation_id)
    if evaluation is None:
        return None

    evaluation.status = "failed"
    evaluation.error_message = error_message
    evaluation.updated_at = datetime.now(UTC)

    db.add(evaluation)
    db.commit()
    db.refresh(evaluation)
    return evaluation
