import unittest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
from backend.app import app, evaluator as real_evaluator
from backend.evaluation import MockEvaluator

class TestAPIWithMockEvaluator(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        self.mock_evaluator = MockEvaluator(scores={"faithfulness": 0.95, "answer_relevancy": 0.9})
        # Override the app's evaluator with our mock
        import backend.app
        backend.app.evaluator = self.mock_evaluator

    def tearDown(self):
        # Restore real evaluator if needed, or just let next test handle it
        import backend.app
        backend.app.evaluator = real_evaluator

    @patch("backend.app.RAGService.retrieve")
    @patch("backend.app.RAGService.generate")
    def test_chat_flow_triggers_evaluation(self, mock_generate, mock_retrieve):
        # Setup mocks for RAG components
        mock_doc = MagicMock()
        mock_doc.page_content = "RAG is cool."
        mock_retrieve.return_value = [mock_doc]
        mock_generate.return_value = "RAG is a system that..."

        payload = {"query": "What is RAG?"}
        response = self.client.post("/chat", json=payload)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["response"], "RAG is a system that...")
        
        # Check if evaluator was called (it's a background task, so we might need to wait or check directly)
        # In TestClient, background tasks run synchronously unless configured otherwise?
        # Actually TestClient runs background tasks before returning.
        
        self.assertEqual(len(self.mock_evaluator.called_with), 1)
        self.assertEqual(self.mock_evaluator.called_with[0].query, "What is RAG?")
        self.assertEqual(self.mock_evaluator.called_with[0].answer, "RAG is a system that...")
        self.assertEqual(self.mock_evaluator.called_with[0].contexts, ["RAG is cool."])
