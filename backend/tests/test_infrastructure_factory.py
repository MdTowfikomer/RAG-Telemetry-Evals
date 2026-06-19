import unittest
from unittest.mock import ANY, patch

from pydantic import SecretStr

from backend.core import InfrastructureFactory, Settings


class TestInfrastructureFactory(unittest.TestCase):
    def setUp(self):
        self.settings = Settings(
            qdrant_url="http://localhost:6333",
            qdrant_api_key=None,
            collection_name="rag_collection",
            embedding_model="embed-english-v3.0",
            cohere_api_key=SecretStr("test-cohere-key"),
            openrouter_api_key=SecretStr("test-openrouter-key"),
            openrouter_model="openrouter/free",
            ragas_eval_model="openai/gpt-4o-mini",
            database_url="sqlite:///:memory:",
        )

    def test_setup_tracing_is_noop(self):
        factory = InfrastructureFactory(self.settings)

        factory.setup_tracing("rag-backend")
        factory.setup_tracing("rag-backend")

        self.assertTrue(factory._tracing_initialized)

    @patch("backend.core.infrastructure.CohereEmbeddings")
    def test_get_embeddings_is_cached(self, mock_embeddings_cls):
        factory = InfrastructureFactory(self.settings)

        first = factory.get_embeddings()
        second = factory.get_embeddings()

        self.assertIs(first, second)
        mock_embeddings_cls.assert_called_once_with(
            model=self.settings.embedding_model,
            cohere_api_key=self.settings.cohere_api_key.get_secret_value(),
            client=None,
            async_client=None,
        )

    @patch("backend.core.infrastructure.QdrantClient")
    def test_get_qdrant_client_is_cached(self, mock_qdrant_client_cls):
        factory = InfrastructureFactory(self.settings)

        first = factory.get_qdrant_client()
        second = factory.get_qdrant_client()

        self.assertIs(first, second)
        mock_qdrant_client_cls.assert_called_once_with(
            url=self.settings.qdrant_url,
            api_key=None,
        )

    @patch("backend.core.infrastructure.FastEmbedSparse")
    @patch("backend.core.infrastructure.QdrantVectorStore")
    @patch("backend.core.infrastructure.QdrantClient")
    @patch("backend.core.infrastructure.CohereEmbeddings")
    def test_get_vectorstore_is_cached_and_uses_factory_dependencies(
        self,
        mock_embeddings_cls,
        mock_qdrant_client_cls,
        mock_vectorstore_cls,
        mock_sparse_embeddings_cls,
    ):
        factory = InfrastructureFactory(self.settings)

        first = factory.get_vectorstore()
        second = factory.get_vectorstore()

        self.assertIs(first, second)
        mock_embeddings_cls.assert_called_once_with(
            model=self.settings.embedding_model,
            cohere_api_key=self.settings.cohere_api_key.get_secret_value(),
            client=None,
            async_client=None,
        )
        mock_qdrant_client_cls.assert_called_once_with(
            url=self.settings.qdrant_url,
            api_key=None,
        )
        mock_sparse_embeddings_cls.assert_called_once_with(
            model_name="Qdrant/bm42-all-minilm-l6-v2-attentions"
        )
        call_kwargs = mock_vectorstore_cls.call_args.kwargs
        self.assertIs(call_kwargs["client"], mock_qdrant_client_cls.return_value)
        self.assertEqual(call_kwargs["collection_name"], self.settings.collection_name)
        self.assertIs(call_kwargs["embedding"], mock_embeddings_cls.return_value)
        self.assertIs(call_kwargs["sparse_embedding"], mock_sparse_embeddings_cls.return_value)
        self.assertEqual(call_kwargs["sparse_vector_name"], "fastembed-sparse")
        self.assertEqual(call_kwargs["retrieval_mode"], ANY)

    @patch("backend.core.infrastructure.FastEmbedSparse")
    def test_get_sparse_embeddings_is_cached(self, mock_sparse_embeddings_cls):
        factory = InfrastructureFactory(self.settings)

        first = factory.get_sparse_embeddings()
        second = factory.get_sparse_embeddings()

        self.assertIs(first, second)
        mock_sparse_embeddings_cls.assert_called_once_with(
            model_name="Qdrant/bm42-all-minilm-l6-v2-attentions"
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
