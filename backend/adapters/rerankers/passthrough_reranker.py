from backend.core import Document, Reranker


class PassThroughReranker(Reranker):
    async def rerank(self, query: str, docs: list[Document]) -> list[Document]:
        return docs
