from contextlib import nullcontext
from typing import AsyncGenerator

from .interfaces import Generator, Reranker, Retriever
from .models import Document


class RAGPipeline:
    def __init__(
        self,
        retriever: Retriever,
        reranker: Reranker,
        generator: Generator,
        tracer=None,
    ):
        self.retriever = retriever
        self.reranker = reranker
        self.generator = generator
        self.tracer = tracer

    def _start_span(self, name: str):
        if self.tracer is None:
            return nullcontext()
        return self.tracer.start_as_current_span(name)

    async def prepare_context(self, query: str, k: int = 3) -> list[Document]:
        docs = await self.retriever.retrieve(query, k)
        reranked = await self.reranker.rerank(query, docs)
        return reranked

    async def execute(self, query: str, k: int = 3) -> tuple[str, list[Document]]:
        with self._start_span("rag_pipeline_execute") as span:
            docs = await self.retriever.retrieve(query, k)
            reranked = await self.reranker.rerank(query, docs)
            response = await self.generator.generate(query, reranked)

            if span is not None:
                span.set_attribute("rag.query", query)
                span.set_attribute("rag.k", k)
                span.set_attribute("rag.docs_retrieved", len(docs))
                span.set_attribute("rag.docs_reranked", len(reranked))

            return response, reranked

    async def stream(self, query: str, k: int = 3) -> AsyncGenerator[str, None]:
        with self._start_span("rag_pipeline_stream") as span:
            docs = await self.retriever.retrieve(query, k)
            reranked = await self.reranker.rerank(query, docs)

            if span is not None:
                span.set_attribute("rag.query", query)
                span.set_attribute("rag.k", k)
                span.set_attribute("rag.docs_retrieved", len(docs))
                span.set_attribute("rag.docs_reranked", len(reranked))

            async for chunk in self.generator.stream(query, reranked):
                yield chunk
