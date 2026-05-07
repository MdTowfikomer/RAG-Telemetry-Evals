import unittest
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine, select

from backend.app import app, factory, get_db
from backend.app import evaluator as real_evaluator
from backend.core import Document, Settings
from backend.core.models import ChatMessage, ChatSession
from backend.evaluation import MockEvaluator


class TestAPIWithMockEvaluator(unittest.TestCase):
    def setUp(self):
        # Override settings for testing
        self.settings = Settings(
            database_url="sqlite:///:memory:", openrouter_api_key="test-key"
        )
        self.engine = create_engine(
            "sqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        SQLModel.metadata.create_all(self.engine)

        def override_get_db():
            with Session(self.engine) as session:
                yield session

        app.dependency_overrides[get_db] = override_get_db

        self.client = TestClient(app)
        self.mock_evaluator = MockEvaluator(
            scores={"faithfulness": 0.95, "answer_relevancy": 0.9}
        )
        # Override the app's evaluator with our mock
        import backend.app

        backend.app.evaluator = self.mock_evaluator

    def tearDown(self):
        # Restore real evaluator
        import backend.app

        backend.app.evaluator = real_evaluator

    @patch("backend.app.get_pipeline_for_model")
    def test_chat_flow_triggers_evaluation(self, mock_get_pipeline_for_model):
        mock_pipeline = AsyncMock()
        mock_docs = [Document(page_content="RAG is cool.", metadata={})]
        mock_pipeline.execute.return_value = ("RAG is a system that...", mock_docs)
        mock_get_pipeline_for_model.return_value = mock_pipeline

        payload = {"query": "What is RAG?"}
        response = self.client.post("/chat", json=payload)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["response"], "RAG is a system that...")

        mock_pipeline.execute.assert_awaited_once()

        self.assertEqual(len(self.mock_evaluator.called_with), 1)
        self.assertEqual(self.mock_evaluator.called_with[0].query, "What is RAG?")
        self.assertEqual(
            self.mock_evaluator.called_with[0].answer,
            "RAG is a system that...",
        )
        self.assertEqual(self.mock_evaluator.called_with[0].contexts, ["RAG is cool."])

    @patch("backend.app.evaluate_ragas_for_stream", new_callable=AsyncMock)
    @patch("backend.app.get_pipeline_for_model")
    def test_chat_stream_persists_messages_linked_to_session(
        self,
        mock_get_pipeline_for_model,
        mock_evaluate_ragas_for_stream,
    ):
        async def mock_stream(_query, k=3):
            yield "RAG "
            yield "stream"

        mock_pipeline = AsyncMock()
        mock_pipeline.stream = mock_stream
        mock_get_pipeline_for_model.return_value = mock_pipeline

        response = self.client.get(
            "/chat/stream",
            params={"query": "What is RAG?", "k": 3, "model": "test-model"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn("[DONE]", response.text)

        with Session(self.engine) as session:
            sessions = session.exec(select(ChatSession)).all()
            self.assertEqual(len(sessions), 1)

            messages = session.exec(select(ChatMessage)).all()
            self.assertEqual(len(messages), 2)

            user_messages = [m for m in messages if m.role == "user"]
            assistant_messages = [m for m in messages if m.role == "assistant"]

            self.assertEqual(len(user_messages), 1)
            self.assertEqual(len(assistant_messages), 1)

            self.assertEqual(user_messages[0].session_id, sessions[0].id)
            self.assertEqual(assistant_messages[0].session_id, sessions[0].id)
            self.assertEqual(user_messages[0].content, "What is RAG?")
            self.assertEqual(assistant_messages[0].content, "RAG stream")

        mock_evaluate_ragas_for_stream.assert_called_once()
