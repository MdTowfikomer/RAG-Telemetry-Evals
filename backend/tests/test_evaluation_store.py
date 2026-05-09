import unittest

from sqlmodel import Session, SQLModel, create_engine

from backend.core.evaluation_store import (
    create_pending_evaluation,
    mark_evaluation_completed,
    mark_evaluation_failed,
)
from backend.core.models import ChatMessage, ChatSession


class TestEvaluationStore(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine("sqlite:///:memory:")
        SQLModel.metadata.create_all(self.engine)
        self.session = Session(self.engine)

        chat_session = ChatSession(title="Eval Session")
        self.session.add(chat_session)
        self.session.commit()
        self.session.refresh(chat_session)

        self.message = ChatMessage(
            session_id=chat_session.id,
            role="assistant",
            content="Answer",
        )
        self.session.add(self.message)
        self.session.commit()
        self.session.refresh(self.message)

    def tearDown(self):
        self.session.close()

    def test_pending_evaluation_starts_from_version_one(self):
        evaluation = create_pending_evaluation(self.session, self.message.id)

        self.assertEqual(evaluation.version, 1)
        self.assertEqual(evaluation.status, "pending")

    def test_pending_evaluation_increments_version(self):
        first = create_pending_evaluation(self.session, self.message.id)
        second = create_pending_evaluation(self.session, self.message.id)

        self.assertEqual(first.version, 1)
        self.assertEqual(second.version, 2)

    def test_mark_completed_updates_scores_and_status(self):
        evaluation = create_pending_evaluation(self.session, self.message.id)

        updated = mark_evaluation_completed(
            self.session,
            evaluation.id,
            faithfulness=0.91,
            answer_relevancy=0.88,
        )

        self.assertIsNotNone(updated)
        self.assertEqual(updated.status, "completed")
        self.assertEqual(updated.faithfulness, 0.91)
        self.assertEqual(updated.answer_relevancy, 0.88)
        self.assertIsNone(updated.error_message)

    def test_mark_failed_updates_status_and_error(self):
        evaluation = create_pending_evaluation(self.session, self.message.id)

        updated = mark_evaluation_failed(
            self.session,
            evaluation.id,
            error_message="Evaluator timeout",
        )

        self.assertIsNotNone(updated)
        self.assertEqual(updated.status, "failed")
        self.assertEqual(updated.error_message, "Evaluator timeout")


if __name__ == "__main__":
    unittest.main()
