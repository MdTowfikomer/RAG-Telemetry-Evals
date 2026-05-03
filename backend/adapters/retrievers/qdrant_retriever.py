import asyncio
from contextlib import nullcontext

from backend.core import Document, Retriever


class QdrantRetriever(Retriever):
    def __init__(self, vectorstore, tracer=None):
        self.vectorstore = vectorstore
        self.tracer = tracer

    def _start_span(self, name: str):
        if self.tracer is None:
            return nullcontext()
        return self.tracer.start_as_current_span(name)

    async def retrieve(self, query: str, k: int) -> list[Document]:
        with self._start_span("retrieve") as span:
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

            if span is not None:
                span.set_attribute("retrieval.k", k)
                span.set_attribute("retrieval.num_docs", len(docs))

            return docs
