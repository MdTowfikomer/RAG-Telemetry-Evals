import asyncio
from flashrank import Ranker, RerankRequest
from backend.core import Document, Reranker


class FlashRankReranker(Reranker):
    def __init__(self, model_name: str = "ms-marco-MiniLM-L-12-v2", score_threshold: float = 0.05):
        self.ranker = Ranker(model_name=model_name)
        self.score_threshold = score_threshold

    async def rerank(self, query: str, docs: list[Document]) -> list[Document]:
        if not docs:
            return []

        # Convert documents to FlashRank input format
        passages = []
        for i, doc in enumerate(docs):
            passages.append({
                "id": str(i),
                "text": doc.page_content,
                "meta": doc.metadata
            })

        # Run reranking in a separate thread to prevent blocking the event loop
        rerank_request = RerankRequest(query=query, passages=passages)
        results = await asyncio.to_thread(self.ranker.rerank, rerank_request)

        # Reconstruct sorted and filtered Document list
        reranked_docs = []
        for result in results:
            score = result.get("score", 0.0)
            if score >= self.score_threshold:
                reranked_docs.append(
                    Document(
                        page_content=result["text"],
                        metadata=result.get("meta", {}) or {}
                    )
                )

        return reranked_docs
