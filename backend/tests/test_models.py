import unittest
from uuid import UUID, uuid4

from sqlmodel import Session, SQLModel, create_engine, select

from backend.core.models import ChatMessage, ChatSession, Evaluation


class TestModels(unittest.TestCase):
    def setUp(self):
        # Use in-memory SQLite for testing
        self.engine = create_engine("sqlite:///:memory:")
        SQLModel.metadata.create_all(self.engine)
        self.session = Session(self.engine)

    def tearDown(self):
        self.session.close()

    def test_create_session(self):
        session = ChatSession(title="Test Session")
        self.session.add(session)
        self.session.commit()
        self.session.refresh(session)

        self.assertIsInstance(session.id, UUID)
        self.assertEqual(session.title, "Test Session")
        self.assertIsNotNone(session.created_at)

    def test_create_message(self):
        session = ChatSession(title="Test Session")
        self.session.add(session)
        self.session.commit()
        self.session.refresh(session)

        message = ChatMessage(session_id=session.id, role="user", content="Hello world")
        self.session.add(message)
        self.session.commit()
        self.session.refresh(message)

        self.assertIsInstance(message.id, UUID)
        self.assertEqual(message.session_id, session.id)
        self.assertEqual(message.role, "user")
        self.assertEqual(message.content, "Hello world")

    def test_relational_integrity(self):
        session = ChatSession(title="Test Session")
        self.session.add(session)
        self.session.commit()
        self.session.refresh(session)

        msg1 = ChatMessage(session_id=session.id, role="user", content="msg1")
        msg2 = ChatMessage(session_id=session.id, role="assistant", content="msg2")
        self.session.add(msg1)
        self.session.add(msg2)
        self.session.commit()
        self.session.refresh(session)

        self.assertEqual(len(session.messages), 2)
        self.assertEqual(session.messages[0].content, "msg1")
        self.assertEqual(session.messages[1].content, "msg2")
        self.assertEqual(session.messages[0].session.id, session.id)

    def test_message_to_evaluations_relationship(self):
        session = ChatSession(title="Evaluation Session")
        self.session.add(session)
        self.session.commit()
        self.session.refresh(session)

        message = ChatMessage(session_id=session.id, role="assistant", content="Answer")
        self.session.add(message)
        self.session.commit()
        self.session.refresh(message)

        eval_v1 = Evaluation(message_id=message.id, version=1, status="completed")
        eval_v2 = Evaluation(message_id=message.id, version=2, status="failed")
        self.session.add(eval_v1)
        self.session.add(eval_v2)
        self.session.commit()
        self.session.refresh(message)

        self.assertEqual(len(message.evaluations), 2)
        self.assertEqual(message.evaluations[0].message_id, message.id)
        self.assertEqual(message.evaluations[1].message_id, message.id)

    def test_evaluation_fields(self):
        session = ChatSession(title="Eval Fields")
        self.session.add(session)
        self.session.commit()
        self.session.refresh(session)

        message = ChatMessage(session_id=session.id, role="assistant", content="Answer")
        self.session.add(message)
        self.session.commit()
        self.session.refresh(message)

        evaluation = Evaluation(message_id=message.id, version=1, status="pending")
        self.session.add(evaluation)
        self.session.commit()
        self.session.refresh(evaluation)

        self.assertEqual(evaluation.message_id, message.id)
        self.assertEqual(evaluation.version, 1)
        self.assertEqual(evaluation.status, "pending")
        self.assertIsNone(evaluation.faithfulness)
        self.assertIsNone(evaluation.answer_relevancy)

    def test_uuid_uniqueness(self):
        session1 = ChatSession(title="S1")
        session2 = ChatSession(title="S2")
        self.session.add(session1)
        self.session.add(session2)
        self.session.commit()

        self.assertNotEqual(session1.id, session2.id)

        msg1 = ChatMessage(session_id=session1.id, role="user", content="M1")
        msg2 = ChatMessage(session_id=session1.id, role="user", content="M2")
        self.session.add(msg1)
        self.session.add(msg2)
        self.session.commit()

        self.assertNotEqual(msg1.id, msg2.id)


if __name__ == "__main__":
    unittest.main()
