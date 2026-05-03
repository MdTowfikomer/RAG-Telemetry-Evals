import unittest
from unittest.mock import MagicMock, patch

from pydantic import SecretStr

from backend.core import InfrastructureFactory, Settings


class TestInfrastructureFactory(unittest.TestCase):
    def setUp(self):
        self.settings = Settings(
            qdrant_url="http://localhost:6333",
            collection_name="rag_collection",
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            phoenix_url="http://localhost:6006/v1/traces",
            openrouter_api_key=SecretStr("test-openrouter-key"),
            openrouter_model="google/gemini-2.0-flash-001",
            ragas_eval_model="openai/gpt-4o-mini",
        )

    @patch("backend.core.infrastructure.LangChainInstrumentor")
    @patch("backend.core.infrastructure.otel_trace.set_tracer_provider")
    @patch("backend.core.infrastructure.BatchSpanProcessor")
    @patch("backend.core.infrastructure.TracerProvider")
    @patch("backend.core.infrastructure.OTLPSpanExporter")
    @patch("backend.core.infrastructure.Resource")
    def test_setup_tracing_initializes_once(
        self,
        mock_resource,
        mock_exporter,
        mock_tracer_provider_cls,
        mock_batch_span_processor,
        mock_set_tracer_provider,
        mock_langchain_instrumentor,
    ):
        factory = InfrastructureFactory(self.settings)

        tracer_provider_instance = MagicMock()
        mock_tracer_provider_cls.return_value = tracer_provider_instance
        instrumentor_instance = MagicMock()
        mock_langchain_instrumentor.return_value = instrumentor_instance

        factory.setup_tracing("rag-backend")
        factory.setup_tracing("rag-backend")

        mock_resource.assert_called_once_with(
            attributes={"service.name": "rag-backend"}
        )
        mock_exporter.assert_called_once_with(endpoint=self.settings.phoenix_url)
        mock_tracer_provider_cls.assert_called_once()
        tracer_provider_instance.add_span_processor.assert_called_once()
        mock_set_tracer_provider.assert_called_once_with(tracer_provider_instance)
        instrumentor_instance.instrument.assert_called_once()

    @patch("backend.core.infrastructure.LCHuggingFaceEmbeddings")
    def test_get_embeddings_is_cached(self, mock_embeddings_cls):
        factory = InfrastructureFactory(self.settings)

        first = factory.get_embeddings()
        second = factory.get_embeddings()

        self.assertIs(first, second)
        mock_embeddings_cls.assert_called_once_with(
            model_name=self.settings.embedding_model
        )

    @patch("backend.core.infrastructure.QdrantClient")
    def test_get_qdrant_client_is_cached(self, mock_qdrant_client_cls):
        factory = InfrastructureFactory(self.settings)

        first = factory.get_qdrant_client()
        second = factory.get_qdrant_client()

        self.assertIs(first, second)
        mock_qdrant_client_cls.assert_called_once_with(url=self.settings.qdrant_url)

    @patch("backend.core.infrastructure.QdrantVectorStore")
    @patch("backend.core.infrastructure.QdrantClient")
    @patch("backend.core.infrastructure.LCHuggingFaceEmbeddings")
    def test_get_vectorstore_is_cached_and_uses_factory_dependencies(
        self,
        mock_embeddings_cls,
        mock_qdrant_client_cls,
        mock_vectorstore_cls,
    ):
        factory = InfrastructureFactory(self.settings)

        first = factory.get_vectorstore()
        second = factory.get_vectorstore()

        self.assertIs(first, second)
        mock_embeddings_cls.assert_called_once_with(
            model_name=self.settings.embedding_model
        )
        mock_qdrant_client_cls.assert_called_once_with(url=self.settings.qdrant_url)
        mock_vectorstore_cls.assert_called_once_with(
            client=mock_qdrant_client_cls.return_value,
            collection_name=self.settings.collection_name,
            embedding=mock_embeddings_cls.return_value,
        )

    @patch("backend.core.infrastructure.ChatOpenAI")
    def test_get_llm_uses_default_or_override_model(self, mock_chat_openai):
        factory = InfrastructureFactory(self.settings)

        factory.get_llm()
        factory.get_llm(model="openai/gpt-4o-mini")

        self.assertEqual(mock_chat_openai.call_count, 2)
        first_call = mock_chat_openai.call_args_list[0]
        second_call = mock_chat_openai.call_args_list[1]

        self.assertEqual(first_call.kwargs["model"], self.settings.openrouter_model)
        self.assertEqual(second_call.kwargs["model"], "openai/gpt-4o-mini")
        self.assertEqual(
            first_call.kwargs["api_key"],
            self.settings.openrouter_api_key,
        )
        self.assertEqual(
            first_call.kwargs["base_url"],
            "https://openrouter.ai/api/v1",
        )


if __name__ == "__main__":
    unittest.main()
