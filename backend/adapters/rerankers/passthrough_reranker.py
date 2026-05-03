from contextlib import nullcontext

from backend.core import Document, Reranker


class PassThroughReranker(Reranker):
    def __init__(self, tracer=None):
        self.tracer = tracer

    def _start_span(self, name: str):
        if self.tracer is None:
            return nullcontext()
        return self.tracer.start_as_current_span(name)

    async def rerank(self, query: str, docs: list[Document]) -> list[Document]:
        with self._start_span("rerank") as span:
            if span is not None:
                span.set_attribute("rerank.num_docs", len(docs))
                span.set_attribute("rerank.strategy", "pass_through")
            return docs
