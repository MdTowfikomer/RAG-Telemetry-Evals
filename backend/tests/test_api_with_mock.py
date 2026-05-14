import unittest
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID

from fastapi.testclient import TestClient
from pydantic import SecretStr
from sqlalchemy import asc, desc
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, col, create_engine, select

from backend.app import app
from backend.api.dependencies import get_db
import backend.api.dependencies as api_dependencies
from backend.core import Document, Settings
from backend.core.models import ChatMessage, ChatSession, Evaluation
from backend.evaluation import MockEvaluator


class TestAPIWithMockEvaluator(unittest.TestCase):
    def setUp(self):
        # Override settings for testing
        self.settings = Settings(
            database_url="sqlite:///:memory:", openrouter_api_key=SecretStr("test-key")
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
        self.real_evaluator = api_dependencies.evaluation_service._evaluator
        api_dependencies.evaluation_service._evaluator = self.mock_evaluator
        import asyncio

        asyncio.run(api_dependencies.reevaluate_rate_limiter.reset())

    def tearDown(self):
        api_dependencies.evaluation_service._evaluator = self.real_evaluator

    @patch("backend.api.dependencies.get_pipeline_for_model")
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

    @patch(
        "backend.api.dependencies.evaluation_service.trigger_stream_evaluation",
        new_callable=AsyncMock,
    )
    @patch("backend.api.dependencies.get_pipeline_for_model")
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
            self.assertIsNotNone(assistant_messages[0].latency_ms)
            self.assertGreaterEqual(assistant_messages[0].latency_ms, 0)

        mock_evaluate_ragas_for_stream.assert_called_once()

    @patch("backend.api.dependencies.get_pipeline_for_model")
    def test_chat_flow_updates_evaluation_record_to_completed(
        self,
        mock_get_pipeline_for_model,
    ):
        mock_pipeline = AsyncMock()
        mock_docs = [Document(page_content="RAG context", metadata={})]
        mock_pipeline.execute.return_value = ("RAG answer", mock_docs)
        mock_get_pipeline_for_model.return_value = mock_pipeline

        response = self.client.post("/chat", json={"query": "Explain RAG"})

        self.assertEqual(response.status_code, 200)

        with Session(self.engine) as session:
            assistant = session.exec(
                select(ChatMessage)
                .where(ChatMessage.role == "assistant")
                .order_by(desc(col(ChatMessage.created_at)))
            ).first()

            self.assertIsNotNone(assistant)

            evaluations = session.exec(
                select(Evaluation)
                .where(Evaluation.message_id == assistant.id)
                .order_by(asc(col(Evaluation.version)))
            ).all()

            self.assertEqual(len(evaluations), 1)
            self.assertEqual(evaluations[0].version, 1)
            self.assertEqual(evaluations[0].status, "completed")
            self.assertEqual(evaluations[0].faithfulness, 0.95)
            self.assertEqual(evaluations[0].answer_relevancy, 0.9)
            self.assertIsNotNone(evaluations[0].reasoning)
            self.assertGreater(len(evaluations[0].reasoning), 10)
            self.assertIsNotNone(assistant.latency_ms)
            self.assertGreaterEqual(assistant.latency_ms, 0)
            self.assertIsNotNone(assistant.token_count)
            self.assertGreater(assistant.token_count, 0)

    @patch("backend.api.routes.evaluations.asyncio.create_task")
    @patch(
        "backend.api.dependencies.evaluation_service.trigger_reevaluation",
        new_callable=MagicMock,
    )
    def test_reevaluate_message_creates_incremented_pending_version(
        self,
        mock_reevaluate_existing_message,
        mock_create_task,
    ):
        with Session(self.engine) as session:
            chat_session = ChatSession(title="Existing chat")
            session.add(chat_session)
            session.commit()
            session.refresh(chat_session)

            user_message = ChatMessage(
                session_id=chat_session.id,
                role="user",
                content="What is RAG?",
            )
            assistant_message = ChatMessage(
                session_id=chat_session.id,
                role="assistant",
                content="RAG uses retrieval and generation.",
                latency_ms=100,
                token_count=8,
            )
            session.add(user_message)
            session.add(assistant_message)
            session.commit()
            session.refresh(assistant_message)

            first_eval = Evaluation(
                message_id=assistant_message.id,
                version=1,
                status="completed",
                faithfulness=0.9,
                answer_relevancy=0.89,
                reasoning="Initial evaluation",
            )
            session.add(first_eval)
            session.commit()

            assistant_message_id = assistant_message.id

        response = self.client.post(
            f"/messages/{assistant_message_id}/re-evaluate",
            json={"k": 3, "model": "test-model"},
        )

        self.assertEqual(response.status_code, 202)
        payload = response.json()
        self.assertEqual(payload["status"], "pending")
        self.assertEqual(payload["version"], 2)
        self.assertEqual(payload["message_id"], str(assistant_message_id))

        with Session(self.engine) as session:
            evaluations = session.exec(
                select(Evaluation)
                .where(Evaluation.message_id == assistant_message_id)
                .order_by(asc(col(Evaluation.version)))
            ).all()

            self.assertEqual(len(evaluations), 2)
            self.assertEqual(evaluations[0].version, 1)
            self.assertEqual(evaluations[0].status, "completed")
            self.assertEqual(evaluations[1].version, 2)
            self.assertEqual(evaluations[1].status, "pending")

        mock_reevaluate_existing_message.assert_called_once()
        mock_create_task.assert_called_once()

    @patch("backend.api.routes.evaluations.asyncio.create_task")
    @patch(
        "backend.api.dependencies.evaluation_service.trigger_reevaluation",
        new_callable=MagicMock,
    )
    def test_reevaluate_message_rate_limited_after_three_requests(
        self,
        _mock_reevaluate_existing_message,
        _mock_create_task,
    ):
        with Session(self.engine) as session:
            chat_session = ChatSession(title="Rate limit chat")
            session.add(chat_session)
            session.commit()
            session.refresh(chat_session)

            user_message = ChatMessage(
                session_id=chat_session.id,
                role="user",
                content="What is RAG?",
            )
            assistant_message = ChatMessage(
                session_id=chat_session.id,
                role="assistant",
                content="RAG uses retrieval and generation.",
            )
            session.add(user_message)
            session.add(assistant_message)
            session.commit()
            session.refresh(assistant_message)
            assistant_message_id = assistant_message.id

        for _ in range(3):
            response = self.client.post(
                f"/messages/{assistant_message_id}/re-evaluate",
                json={"k": 3, "model": "test-model"},
            )
            self.assertEqual(response.status_code, 202)

        limited = self.client.post(
            f"/messages/{assistant_message_id}/re-evaluate",
            json={"k": 3, "model": "test-model"},
        )
        self.assertEqual(limited.status_code, 429)
        self.assertIn("Retry-After", limited.headers)
