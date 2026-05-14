from abc import ABC, abstractmethod
from contextvars import ContextVar
from dataclasses import dataclass
from typing import AsyncGenerator

from .interfaces import Generator, Reranker, Retriever
from .models import Document


class RAGHook(ABC):
    @abstractmethod
    async def on_start(self, *, query: str, k: int, is_stream: bool) -> None:
        raise NotImplementedError

    @abstractmethod
    async def after_retrieve(
        self,
        *,
        query: str,
        k: int,
        docs: list[Document],
        is_stream: bool,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    async def after_rerank(
        self,
        *,
        query: str,
        k: int,
        docs: list[Document],
        is_stream: bool,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    async def on_end(
        self,
        *,
        query: str,
        k: int,
        response: str | None,
        docs: list[Document],
        is_stream: bool,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    async def on_error(
        self,
        *,
        query: str,
        k: int,
        error: Exception,
        is_stream: bool,
    ) -> None:
        raise NotImplementedError


@dataclass
class _TraceState:
    manager: object
    span: object
    docs_retrieved: int = 0
    docs_reranked: int = 0


class TracingHook(RAGHook):
    def __init__(self, tracer=None):
        self.tracer = tracer
        self._state: ContextVar[_TraceState | None] = ContextVar(
            "rag_tracing_hook_state", default=None
        )

    async def on_start(self, *, query: str, k: int, is_stream: bool) -> None:
        if self.tracer is None:
            return

        span_name = "rag_pipeline_stream" if is_stream else "rag_pipeline_execute"
        manager = self.tracer.start_as_current_span(span_name)
        span = manager.__enter__()
        self._state.set(_TraceState(manager=manager, span=span))

    async def after_retrieve(
        self,
        *,
        query: str,
        k: int,
        docs: list[Document],
        is_stream: bool,
    ) -> None:
        state = self._state.get()
        if state is None:
            return
        state.docs_retrieved = len(docs)

    async def after_rerank(
        self,
        *,
        query: str,
        k: int,
        docs: list[Document],
        is_stream: bool,
    ) -> None:
        state = self._state.get()
        if state is None:
            return
        state.docs_reranked = len(docs)

    async def on_end(
        self,
        *,
        query: str,
        k: int,
        response: str | None,
        docs: list[Document],
        is_stream: bool,
    ) -> None:
        state = self._state.get()
        if state is None:
            return

        state.span.set_attribute("rag.query", query)
        state.span.set_attribute("rag.k", k)
        state.span.set_attribute("rag.docs_retrieved", state.docs_retrieved)
        state.span.set_attribute("rag.docs_reranked", state.docs_reranked)
        state.manager.__exit__(None, None, None)
        self._state.set(None)

    async def on_error(
        self,
        *,
        query: str,
        k: int,
        error: Exception,
        is_stream: bool,
    ) -> None:
        state = self._state.get()
        if state is None:
            return

        state.span.set_attribute("rag.query", query)
        state.span.set_attribute("rag.k", k)
        state.span.set_attribute("rag.docs_retrieved", state.docs_retrieved)
        state.span.set_attribute("rag.docs_reranked", state.docs_reranked)
        state.span.record_exception(error)
        state.manager.__exit__(type(error), error, error.__traceback__)
        self._state.set(None)


class RAGPipeline:
    def __init__(
        self,
        retriever: Retriever,
        reranker: Reranker,
        generator: Generator,
        hooks: list[RAGHook] | None = None,
    ):
        self.retriever = retriever
        self.reranker = reranker
        self.generator = generator
        self.hooks = hooks or []

    async def _trigger_hooks(self, hook_name: str, **kwargs) -> None:
        for hook in self.hooks:
            hook_fn = getattr(hook, hook_name)
            await hook_fn(**kwargs)

    async def prepare_context(self, query: str, k: int = 3) -> list[Document]:
        docs = await self.retriever.retrieve(query, k)
        reranked = await self.reranker.rerank(query, docs)
        return reranked

    async def execute(self, query: str, k: int = 3) -> tuple[str, list[Document]]:
        await self._trigger_hooks("on_start", query=query, k=k, is_stream=False)
        try:
            docs = await self.retriever.retrieve(query, k)
            await self._trigger_hooks(
                "after_retrieve",
                query=query,
                k=k,
                docs=docs,
                is_stream=False,
            )
            reranked = await self.reranker.rerank(query, docs)
            await self._trigger_hooks(
                "after_rerank",
                query=query,
                k=k,
                docs=reranked,
                is_stream=False,
            )
            response = await self.generator.generate(query, reranked)
            await self._trigger_hooks(
                "on_end",
                query=query,
                k=k,
                response=response,
                docs=reranked,
                is_stream=False,
            )
        except Exception as error:
            await self._trigger_hooks(
                "on_error",
                query=query,
                k=k,
                error=error,
                is_stream=False,
            )
            raise

        return response, reranked

    async def stream(self, query: str, k: int = 3) -> AsyncGenerator[str, None]:
        await self._trigger_hooks("on_start", query=query, k=k, is_stream=True)
        try:
            docs = await self.retriever.retrieve(query, k)
            await self._trigger_hooks(
                "after_retrieve",
                query=query,
                k=k,
                docs=docs,
                is_stream=True,
            )
            reranked = await self.reranker.rerank(query, docs)
            await self._trigger_hooks(
                "after_rerank",
                query=query,
                k=k,
                docs=reranked,
                is_stream=True,
            )

            response_parts: list[str] = []
            async for chunk in self.generator.stream(query, reranked):
                response_parts.append(chunk)
                yield chunk

            await self._trigger_hooks(
                "on_end",
                query=query,
                k=k,
                response="".join(response_parts),
                docs=reranked,
                is_stream=True,
            )
        except Exception as error:
            await self._trigger_hooks(
                "on_error",
                query=query,
                k=k,
                error=error,
                is_stream=True,
            )
            raise
