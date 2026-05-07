import unittest
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from backend.app import app, factory, get_db
from backend.app import evaluator as real_evaluator
from backend.core import Document, Settings
from backend.core.models import ChatMessage, ChatSession
from backend.evaluation import MockEvaluator
from sqlmodel import SQLModel, Session, create_engine
from sqlalchemy.pool import StaticPool

class TestAPIWithMockEvaluator(unittest.TestCase):
    def setUp(self):
        # Override settings for testing
        self.settings = Settings(
            database_url="sqlite:///:memory:",
            openrouter_api_key="test-key"
        )
        self.engine = create_engine(
            "sqlite:///:memory:", 
            connect_args={"check_same_thread": False},
            poolclass=StaticPool
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
