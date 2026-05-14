import unittest

from backend.core import (
    Document,
    Generator,
    RAGHook,
    RAGPipeline,
    Reranker,
    Retriever,
    TracingHook,
)


class MockRetriever(Retriever):
    def __init__(self):
        self.calls = []

    async def retrieve(self, query: str, k: int) -> list[Document]:
        self.calls.append((query, k))
        return [
            Document(page_content="doc-1", metadata={"rank": 1}),
            Document(page_content="doc-2", metadata={"rank": 2}),
        ]


class MockReranker(Reranker):
    def __init__(self):
        self.calls = []

    async def rerank(self, query: str, docs: list[Document]) -> list[Document]:
        self.calls.append((query, [doc.page_content for doc in docs]))
        return list(reversed(docs))


class MockGenerator(Generator):
    def __init__(self):
        self.generate_calls = []
        self.stream_calls = []

    async def generate(self, query: str, docs: list[Document]) -> str:
        self.generate_calls.append((query, [doc.page_content for doc in docs]))
        return "final-answer"

    def stream(self, query: str, docs: list[Document]):
        self.stream_calls.append((query, [doc.page_content for doc in docs]))

        async def _iterate_tokens():
            for token in ["tok-1", "tok-2", "tok-3"]:
                yield token

        return _iterate_tokens()


class FailingGenerator(Generator):
    async def generate(self, query: str, docs: list[Document]) -> str:
        raise RuntimeError("generation failed")

    def stream(self, query: str, docs: list[Document]):
        async def _unused_stream():
            yield "never-used"

        return _unused_stream()


class StreamFailingGenerator(Generator):
    async def generate(self, query: str, docs: list[Document]) -> str:
        return "unused"

    def stream(self, query: str, docs: list[Document]):
        async def _broken_stream():
            yield "tok-1"
            raise RuntimeError("stream failed")

        return _broken_stream()


class RecordingHook(RAGHook):
    def __init__(self):
        self.events: list[str] = []
        self.last_error: Exception | None = None

    async def on_start(self, *, query: str, k: int, is_stream: bool) -> None:
        self.events.append("on_start")

    async def after_retrieve(
        self,
        *,
        query: str,
        k: int,
        docs: list[Document],
        is_stream: bool,
    ) -> None:
        self.events.append("after_retrieve")

    async def after_rerank(
        self,
        *,
        query: str,
        k: int,
        docs: list[Document],
        is_stream: bool,
    ) -> None:
        self.events.append("after_rerank")

    async def on_end(
        self,
        *,
        query: str,
        k: int,
        response: str | None,
        docs: list[Document],
        is_stream: bool,
    ) -> None:
        self.events.append("on_end")

    async def on_error(
        self,
        *,
        query: str,
        k: int,
        error: Exception,
        is_stream: bool,
    ) -> None:
        self.events.append("on_error")
        self.last_error = error


class FakeSpan:
    def __init__(self):
        self.attributes: dict[str, object] = {}
        self.recorded_exceptions: list[Exception] = []
        self.exit_args: tuple[object, object, object] | None = None

    def set_attribute(self, key: str, value: object) -> None:
        self.attributes[key] = value

    def record_exception(self, error: Exception) -> None:
        self.recorded_exceptions.append(error)


class FakeSpanContextManager:
    def __init__(self, span: FakeSpan):
        self.span = span

    def __enter__(self) -> FakeSpan:
        return self.span

    def __exit__(self, exc_type, exc, tb) -> None:
        self.span.exit_args = (exc_type, exc, tb)


class FakeTracer:
    def __init__(self):
        self.started_spans: list[tuple[str, FakeSpan]] = []

    def start_as_current_span(self, name: str) -> FakeSpanContextManager:
        span = FakeSpan()
        self.started_spans.append((name, span))
        return FakeSpanContextManager(span)


class TestRAGPipeline(unittest.IsolatedAsyncioTestCase):
    async def test_execute_orchestrates_all_stages_in_order(self):
        retriever = MockRetriever()
        reranker = MockReranker()
        generator = MockGenerator()
        pipeline = RAGPipeline(
            retriever=retriever, reranker=reranker, generator=generator
        )

        response, docs = await pipeline.execute("what is rag", k=4)

        self.assertEqual(response, "final-answer")
        self.assertEqual([d.page_content for d in docs], ["doc-2", "doc-1"])
        self.assertEqual(retriever.calls, [("what is rag", 4)])
        self.assertEqual(reranker.calls, [("what is rag", ["doc-1", "doc-2"])])
        self.assertEqual(
            generator.generate_calls, [("what is rag", ["doc-2", "doc-1"])]
        )

    async def test_execute_bubbles_errors(self):
        retriever = MockRetriever()
        reranker = MockReranker()
        generator = FailingGenerator()
        pipeline = RAGPipeline(
            retriever=retriever, reranker=reranker, generator=generator
        )

        with self.assertRaisesRegex(RuntimeError, "generation failed"):
            await pipeline.execute("test")

    async def test_prepare_context_orchestrates_retrieval_and_rerank(self):
        retriever = MockRetriever()
        reranker = MockReranker()
        generator = MockGenerator()
        pipeline = RAGPipeline(
            retriever=retriever, reranker=reranker, generator=generator
        )

        docs = await pipeline.prepare_context("context only", k=5)

        self.assertEqual([d.page_content for d in docs], ["doc-2", "doc-1"])
        self.assertEqual(retriever.calls, [("context only", 5)])
        self.assertEqual(reranker.calls, [("context only", ["doc-1", "doc-2"])])
        self.assertEqual(generator.generate_calls, [])

    async def test_stream_yields_generator_tokens(self):
        retriever = MockRetriever()
        reranker = MockReranker()
        generator = MockGenerator()
        pipeline = RAGPipeline(
            retriever=retriever, reranker=reranker, generator=generator
        )

        received = []
        async for token in pipeline.stream("stream question", k=2):
            received.append(token)

        self.assertEqual(received, ["tok-1", "tok-2", "tok-3"])
        self.assertEqual(retriever.calls, [("stream question", 2)])
        self.assertEqual(reranker.calls, [("stream question", ["doc-1", "doc-2"])])
        self.assertEqual(
            generator.stream_calls, [("stream question", ["doc-2", "doc-1"])]
        )

    async def test_stream_bubbles_errors(self):
        retriever = MockRetriever()
        reranker = MockReranker()
        generator = StreamFailingGenerator()
        pipeline = RAGPipeline(
            retriever=retriever, reranker=reranker, generator=generator
        )

        with self.assertRaisesRegex(RuntimeError, "stream failed"):
            async for _ in pipeline.stream("broken stream", k=1):
                pass

    async def test_execute_triggers_hooks(self):
        retriever = MockRetriever()
        reranker = MockReranker()
        generator = MockGenerator()
        hook = RecordingHook()
        pipeline = RAGPipeline(
            retriever=retriever,
            reranker=reranker,
            generator=generator,
            hooks=[hook],
        )

        await pipeline.execute("hook test", k=3)

        self.assertEqual(
            hook.events,
            ["on_start", "after_retrieve", "after_rerank", "on_end"],
        )

    async def test_stream_triggers_hooks(self):
        retriever = MockRetriever()
        reranker = MockReranker()
        generator = MockGenerator()
        hook = RecordingHook()
        pipeline = RAGPipeline(
            retriever=retriever,
            reranker=reranker,
            generator=generator,
            hooks=[hook],
        )

        async for _ in pipeline.stream("hook stream", k=2):
            pass

        self.assertEqual(
            hook.events,
            ["on_start", "after_retrieve", "after_rerank", "on_end"],
        )

    async def test_execute_triggers_on_error_hook(self):
        retriever = MockRetriever()
        reranker = MockReranker()
        generator = FailingGenerator()
        hook = RecordingHook()
        pipeline = RAGPipeline(
            retriever=retriever,
            reranker=reranker,
            generator=generator,
            hooks=[hook],
        )

        with self.assertRaisesRegex(RuntimeError, "generation failed"):
            await pipeline.execute("hook error")

        self.assertEqual(
            hook.events,
            ["on_start", "after_retrieve", "after_rerank", "on_error"],
        )
        self.assertIsNotNone(hook.last_error)

    async def test_tracing_hook_traces_execute(self):
        retriever = MockRetriever()
        reranker = MockReranker()
        generator = MockGenerator()
        tracer = FakeTracer()
        pipeline = RAGPipeline(
            retriever=retriever,
            reranker=reranker,
            generator=generator,
            hooks=[TracingHook(tracer=tracer)],
        )

        await pipeline.execute("trace execute", k=4)

        self.assertEqual(len(tracer.started_spans), 1)
        span_name, span = tracer.started_spans[0]
        self.assertEqual(span_name, "rag_pipeline_execute")
        self.assertEqual(span.attributes["rag.query"], "trace execute")
        self.assertEqual(span.attributes["rag.k"], 4)
        self.assertEqual(span.attributes["rag.docs_retrieved"], 2)
        self.assertEqual(span.attributes["rag.docs_reranked"], 2)
        self.assertEqual(span.recorded_exceptions, [])
        self.assertEqual(span.exit_args, (None, None, None))

    async def test_tracing_hook_traces_stream(self):
        retriever = MockRetriever()
        reranker = MockReranker()
        generator = MockGenerator()
        tracer = FakeTracer()
        pipeline = RAGPipeline(
            retriever=retriever,
            reranker=reranker,
            generator=generator,
            hooks=[TracingHook(tracer=tracer)],
        )

        async for _ in pipeline.stream("trace stream", k=2):
            pass

        self.assertEqual(len(tracer.started_spans), 1)
        span_name, span = tracer.started_spans[0]
        self.assertEqual(span_name, "rag_pipeline_stream")
        self.assertEqual(span.attributes["rag.query"], "trace stream")
        self.assertEqual(span.attributes["rag.k"], 2)
        self.assertEqual(span.attributes["rag.docs_retrieved"], 2)
        self.assertEqual(span.attributes["rag.docs_reranked"], 2)
        self.assertEqual(span.recorded_exceptions, [])
        self.assertEqual(span.exit_args, (None, None, None))

    async def test_tracing_hook_records_errors(self):
        retriever = MockRetriever()
        reranker = MockReranker()
        generator = FailingGenerator()
        tracer = FakeTracer()
        pipeline = RAGPipeline(
            retriever=retriever,
            reranker=reranker,
            generator=generator,
            hooks=[TracingHook(tracer=tracer)],
        )

        with self.assertRaisesRegex(RuntimeError, "generation failed"):
            await pipeline.execute("trace error", k=3)

        self.assertEqual(len(tracer.started_spans), 1)
        span_name, span = tracer.started_spans[0]
        self.assertEqual(span_name, "rag_pipeline_execute")
        self.assertEqual(span.attributes["rag.query"], "trace error")
        self.assertEqual(span.attributes["rag.k"], 3)
        self.assertEqual(span.attributes["rag.docs_retrieved"], 2)
        self.assertEqual(span.attributes["rag.docs_reranked"], 2)
        self.assertEqual(len(span.recorded_exceptions), 1)
        self.assertIsInstance(span.recorded_exceptions[0], RuntimeError)
        self.assertEqual(span.exit_args[0], RuntimeError)


if __name__ == "__main__":
    unittest.main()
