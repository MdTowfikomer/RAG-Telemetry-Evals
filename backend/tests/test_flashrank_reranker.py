import unittest
from backend.core import Document
from backend.adapters.rerankers.flashrank_reranker import FlashRankReranker


class TestFlashRankReranker(unittest.IsolatedAsyncioTestCase):
    async def test_rerank_sorts_by_relevance(self):
        reranker = FlashRankReranker(score_threshold=0.0)
        docs = [
            Document(page_content="Berlin is the capital of Germany.", metadata={"id": 1}),
            Document(page_content="Paris is the capital of France.", metadata={"id": 2}),
            Document(page_content="Madrid is the capital of Spain.", metadata={"id": 3}),
        ]
        query = "What is the capital of France?"

        reranked = await reranker.rerank(query, docs)

        self.assertTrue(len(reranked) > 0)
        # Paris should be the top-scored document
        self.assertIn("Paris", reranked[0].page_content)
        self.assertEqual(reranked[0].metadata["id"], 2)

    async def test_rerank_filters_low_scores(self):
        # Using a threshold > 1.0 to filter out all documents
        reranker = FlashRankReranker(score_threshold=1.1)
        docs = [
            Document(page_content="Berlin is the capital of Germany.", metadata={"id": 1}),
            Document(page_content="Paris is the capital of France.", metadata={"id": 2}),
        ]
        query = "What is the capital of France?"

        reranked = await reranker.rerank(query, docs)

        # High threshold should filter out documents with lower scores
        self.assertEqual(len(reranked), 0)


if __name__ == "__main__":
    unittest.main()
