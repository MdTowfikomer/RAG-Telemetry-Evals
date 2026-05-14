import asyncio

from backend.core import Document, Retriever


class QdrantRetriever(Retriever):
    def __init__(self, vectorstore):
        self.vectorstore = vectorstore

    async def retrieve(self, query: str, k: int) -> list[Document]:
        raw_docs = await asyncio.to_thread(
            self.vectorstore.similarity_search,
            query,
            k,
        )

        docs = [
            Document(
                page_content=getattr(doc, "page_content", ""),
                metadata=getattr(doc, "metadata", {}) or {},
            )
            for doc in raw_docs
        ]

        return docs
