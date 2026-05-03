import unittest

from backend.core import Document, Generator, RAGPipeline, Reranker, Retriever


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


if __name__ == "__main__":
    unittest.main()
