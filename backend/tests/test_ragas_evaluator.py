import unittest
from unittest.mock import MagicMock, patch
from backend.evaluation import RagasEvaluator, EvalContext

class TestRagasEvaluator(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.api_key = "fake-key"
        self.embeddings = MagicMock()
        self.evaluator = RagasEvaluator(
            api_key=self.api_key,
            embeddings=self.embeddings
        )

    @patch("backend.evaluation.ragas_evaluator.evaluate")
    @patch("backend.evaluation.ragas_evaluator.OpenAI")
    @patch("backend.evaluation.ragas_evaluator.llm_factory")
    async def test_evaluate_success(self, mock_llm_factory, mock_openai, mock_evaluate):
        # Setup mocks
        import pandas as pd
        mock_df = pd.DataFrame({"faithfulness": [0.9], "answer_relevancy": [0.8]})
        mock_result = MagicMock()
        mock_result.to_pandas.return_value = mock_df
        mock_evaluate.return_value = mock_result

        ctx = EvalContext(
            query="What is RAG?",
            answer="Retrieval Augmented Generation",
            contexts=["RAG stands for..."]
        )

        scores = await self.evaluator.evaluate(ctx)

        self.assertEqual(scores["faithfulness"], 0.9)
        self.assertEqual(scores["answer_relevancy"], 0.8)
        mock_evaluate.assert_called_once()

    @patch("backend.evaluation.ragas_evaluator.evaluate")
    @patch("backend.evaluation.ragas_evaluator.OpenAI")
    async def test_evaluate_failure(self, mock_openai, mock_evaluate):
        mock_evaluate.side_effect = Exception("Ragas error")

        ctx = EvalContext(
            query="What is RAG?",
            answer="Retrieval Augmented Generation",
            contexts=["RAG stands for..."]
        )

        with self.assertRaises(Exception):
            await self.evaluator.evaluate(ctx)
