import unittest
from unittest.mock import AsyncMock, MagicMock

from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine, select

from backend.core import ChatMessage, ChatSession, Evaluation
from backend.core.evaluation_store import create_pending_evaluation
from backend.services.evaluation_service import EvaluationService


class FakeBroadcaster:
    def __init__(self):
        self.events: list[dict] = []

    async def publish(self, event: dict) -> None:
        self.events.append(event)


class TestEvaluationService(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.engine = create_engine(
            "sqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        SQLModel.metadata.create_all(self.engine)

    async def test_trigger_evaluation_marks_completed_and_broadcasts(self):
        with Session(self.engine) as db:
            chat_session = ChatSession(title="Eval session")
            db.add(chat_session)
            db.commit()
            db.refresh(chat_session)

            assistant = ChatMessage(
                session_id=chat_session.id,
                role="assistant",
                content="Assistant answer",
                latency_ms=123,
                token_count=17,
            )
            db.add(assistant)
            db.commit()
            db.refresh(assistant)

            pending = create_pending_evaluation(db, assistant.id)

        mock_evaluator = MagicMock()
        mock_evaluator.evaluate = AsyncMock(
            return_value={
                "faithfulness": float("nan"),
                "answer_relevancy": 0.91,
                "reasoning": "Evaluator reasoning",
            }
        )
        broadcaster = FakeBroadcaster()
        service = EvaluationService(
            evaluator_provider=lambda: mock_evaluator,
            pipeline_factory=lambda _model: None,
            score_broadcaster=broadcaster,
        )

        await service.trigger_evaluation(
            query="What is RAG?",
            answer="RAG answer",
            contexts=["ctx"],
            evaluation_id=pending.id,
            db_bind=self.engine,
        )

        with Session(self.engine) as db:
            updated = db.exec(
                select(Evaluation).where(Evaluation.id == pending.id)
            ).first()

            self.assertIsNotNone(updated)
            assert updated is not None
            self.assertEqual(updated.status, "completed")
            self.assertIsNone(updated.faithfulness)
            self.assertEqual(updated.answer_relevancy, 0.91)
            self.assertEqual(updated.reasoning, "Evaluator reasoning")

        self.assertEqual(len(broadcaster.events), 1)
        event = broadcaster.events[0]
        self.assertEqual(event["type"], "score")
        self.assertEqual(event["message_id"], str(pending.message_id))
        self.assertEqual(event["status"], "completed")
        self.assertEqual(event["version"], 1)
        self.assertEqual(event["answer_relevancy"], 0.91)
        self.assertEqual(event["reasoning"], "Evaluator reasoning")
        self.assertEqual(event["latency_ms"], 123)
        self.assertEqual(event["token_count"], 17)


if __name__ == "__main__":
    unittest.main()
