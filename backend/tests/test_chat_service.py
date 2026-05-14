import unittest
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID

from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

from backend.core import ChatMessage, ChatSession, Document
from backend.core.evaluation_store import create_pending_evaluation
from backend.core.exceptions import SessionNotFoundError
from backend.services.chat_service import ChatService


class TestChatService(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.engine = create_engine(
            "sqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        SQLModel.metadata.create_all(self.engine)

        self.mock_tracer = MagicMock()
        self.mock_pipeline_factory = MagicMock()
        self.mock_token_counter = MagicMock(return_value=10)
        self.mock_create_pending_evaluation_fn = MagicMock(
            side_effect=lambda db, message_id: MagicMock(
                id=UUID("00000000-0000-0000-0000-000000000001"), message_id=message_id
            )
        )
        self.mock_evaluate_ragas_fn = AsyncMock()
        self.mock_stream_response_factory = MagicMock()

        self.chat_service = ChatService(
            tracer=self.mock_tracer,
            pipeline_factory=self.mock_pipeline_factory,
            token_counter=self.mock_token_counter,
            create_pending_evaluation_fn=self.mock_create_pending_evaluation_fn,
            evaluate_ragas_fn=self.mock_evaluate_ragas_fn,
            stream_response_factory=self.mock_stream_response_factory,
        )

    def tearDown(self):
        SQLModel.metadata.drop_all(self.engine)

    def get_db(self):
        with Session(self.engine) as session:
            yield session

    async def test_chat_creates_new_session_and_messages(self):
        mock_pipeline = AsyncMock()
        mock_pipeline.execute.return_value = (
            "Test response",
            [Document(page_content="context")],
        )
        self.mock_pipeline_factory.return_value = mock_pipeline

        task_spawner_calls = []

        def mock_task_spawner(func, *args, **kwargs):
            task_spawner_calls.append({"func": func, "args": args, "kwargs": kwargs})

        db_gen = self.get_db()
        db = next(db_gen)

        result = await self.chat_service.chat(
            query="test query",
            session_id=None,
            k=3,
            model="test-model",
            db=db,
            task_spawner=mock_task_spawner,
            parent_context=None,
        )

        self.assertIsNotNone(result.session_id)
        self.assertEqual(result.query, "test query")
        self.assertEqual(result.response, "Test response")
        self.assertEqual(len(result.source_documents), 1)
        self.assertEqual(result.source_documents[0], "context")

        db.close()

        with Session(self.engine) as session:
            sessions = session.query(ChatSession).all()
            self.assertEqual(len(sessions), 1)
            self.assertEqual(sessions[0].title, "test query...")

            messages = session.query(ChatMessage).all()
            self.assertEqual(len(messages), 2)  # User and assistant messages
            self.assertEqual(messages[0].role, "user")
            self.assertEqual(messages[0].content, "test query")
            self.assertEqual(messages[1].role, "assistant")
            self.assertEqual(messages[1].content, "Test response")

        self.assertEqual(len(task_spawner_calls), 1)
        self.assertEqual(task_spawner_calls[0]["func"], self.chat_service._evaluate_ragas)

    async def test_chat_uses_existing_session(self):
        db_gen = self.get_db()
        db = next(db_gen)

        existing_session = ChatSession(title="existing session")
        db.add(existing_session)
        db.commit()
        db.refresh(existing_session)

        mock_pipeline = AsyncMock()
        mock_pipeline.execute.return_value = (
            "Response to existing session",
            [Document(page_content="context")],
        )
        self.mock_pipeline_factory.return_value = mock_pipeline

        task_spawner_calls = []

        def mock_task_spawner(func, *args, **kwargs):
            task_spawner_calls.append({"func": func, "args": args, "kwargs": kwargs})

        result = await self.chat_service.chat(
            query="new query",
            session_id=existing_session.id,
            k=3,
            model="test-model",
            db=db,
            task_spawner=mock_task_spawner,
            parent_context=None,
        )

        self.assertEqual(result.session_id, existing_session.id)
        self.assertEqual(result.response, "Response to existing session")

        db.close()

        with Session(self.engine) as session:
            sessions = session.query(ChatSession).all()
            self.assertEqual(len(sessions), 1)  # No new session created

            messages = session.query(ChatMessage).all()
            self.assertEqual(len(messages), 2)  # User and assistant messages
            self.assertEqual(messages[0].session_id, existing_session.id)
            self.assertEqual(messages[1].session_id, existing_session.id)

    async def test_chat_raises_session_not_found_error(self):
        db_gen = self.get_db()
        db = next(db_gen)

        with self.assertRaises(SessionNotFoundError):
            await self.chat_service.chat(
                query="non-existent session",
                session_id=UUID("00000000-0000-0000-0000-000000000002"),
                k=3,
                model="test-model",
                db=db,
                task_spawner=MagicMock(),
                parent_context=None,
            )

    async def test_chat_stream_creates_new_session_and_messages(self):
        mock_pipeline = AsyncMock()
        mock_pipeline.stream.return_value = AsyncMock()
        mock_pipeline.stream.return_value.__aiter__.return_value = iter(
            ["chunk1", "chunk2"]
        )
        self.mock_pipeline_factory.return_value = mock_pipeline

        mock_stream_generator = AsyncMock()
        mock_stream_generator.__aiter__.return_value = iter(["token1", "token2"])
        self.mock_stream_response_factory.return_value = mock_stream_generator

        db_gen = self.get_db()
        db = next(db_gen)

        result = self.chat_service.chat_stream(
            query="stream query",
            session_id=None,
            k=3,
            model="test-model",
            db=db,
        )

        # Iterate through the stream to trigger the internal logic
        async for _ in result.stream:
            pass

        db.close()

        with Session(self.engine) as session:
            sessions = session.query(ChatSession).all()
            self.assertEqual(len(sessions), 1)
            self.assertEqual(sessions[0].title, "stream query...")

            messages = session.query(ChatMessage).all()
            self.assertEqual(len(messages), 2)  # User and assistant messages
            self.assertEqual(messages[0].role, "user")
            self.assertEqual(messages[0].content, "stream query")
            self.assertEqual(messages[1].role, "assistant")
            self.assertEqual(messages[1].content, "") # Content is empty as it's streamed

    async def test_chat_stream_uses_existing_session(self):
        db_gen = self.get_db()
        db = next(db_gen)

        existing_session = ChatSession(title="existing stream session")
        db.add(existing_session)
        db.commit()
        db.refresh(existing_session)

        mock_pipeline = AsyncMock()
        mock_pipeline.stream.return_value = AsyncMock()
        mock_pipeline.stream.return_value.__aiter__.return_value = iter(
            ["chunk1", "chunk2"]
        )
        self.mock_pipeline_factory.return_value = mock_pipeline

        mock_stream_generator = AsyncMock()
        mock_stream_generator.__aiter__.return_value = iter(["token1", "token2"])
        self.mock_stream_response_factory.return_value = mock_stream_generator

        result = self.chat_service.chat_stream(
            query="new stream query",
            session_id=existing_session.id,
            k=3,
            model="test-model",
            db=db,
        )

        async for _ in result.stream:
            pass

        db.close()

        with Session(self.engine) as session:
            sessions = session.query(ChatSession).all()
            self.assertEqual(len(sessions), 1)  # No new session created

            messages = session.query(ChatMessage).all()
            self.assertEqual(len(messages), 2)  # User and assistant messages
            self.assertEqual(messages[0].session_id, existing_session.id)
            self.assertEqual(messages[1].session_id, existing_session.id)

    async def test_chat_stream_raises_session_not_found_error(self):
        db_gen = self.get_db()
        db = next(db_gen)

        with self.assertRaises(SessionNotFoundError):
            self.chat_service.chat_stream(
                query="non-existent stream session",
                session_id=UUID("00000000-0000-0000-0000-000000000003"),
                k=3,
                model="test-model",
                db=db,
            )
