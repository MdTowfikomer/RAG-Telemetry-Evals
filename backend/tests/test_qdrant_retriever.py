import unittest
from types import SimpleNamespace

from backend.adapters.retrievers import QdrantRetriever


class FakeVectorStore:
    def __init__(self, docs):
        self.docs = docs
        self.calls = []

    def similarity_search(self, query, k):
        self.calls.append((query, k))
        return self.docs


class TestQdrantRetriever(unittest.IsolatedAsyncioTestCase):
    async def test_retrieve_maps_docs_into_core_document_model(self):
        raw_docs = [
            SimpleNamespace(page_content="first doc", metadata={"source": "a"}),
            SimpleNamespace(page_content="second doc", metadata={"source": "b"}),
        ]
        vectorstore = FakeVectorStore(docs=raw_docs)
        retriever = QdrantRetriever(vectorstore=vectorstore)

        docs = await retriever.retrieve("what is rag", 3)

        self.assertEqual(vectorstore.calls, [("what is rag", 9)])
        self.assertEqual(len(docs), 2)
        self.assertEqual(docs[0].page_content, "first doc")
        self.assertEqual(docs[0].metadata, {"source": "a"})
        self.assertEqual(docs[1].page_content, "second doc")
        self.assertEqual(docs[1].metadata, {"source": "b"})

    async def test_retrieve_handles_missing_doc_fields(self):
        raw_docs = [SimpleNamespace()]
        vectorstore = FakeVectorStore(docs=raw_docs)
        retriever = QdrantRetriever(vectorstore=vectorstore)

        docs = await retriever.retrieve("fallback", 1)

        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0].page_content, "")
        self.assertEqual(docs[0].metadata, {})

    async def test_retrieve_maps_parent_content_if_present(self):
        raw_docs = [
            SimpleNamespace(
                page_content="child content",
                metadata={"parent_content": "parent content", "parent_id": "p1"},
            )
        ]
        vectorstore = FakeVectorStore(docs=raw_docs)
        retriever = QdrantRetriever(vectorstore=vectorstore)

        docs = await retriever.retrieve("query", 1)

        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0].page_content, "parent content")
        self.assertEqual(docs[0].metadata["parent_id"], "p1")

    async def test_retrieve_deduplicates_parents(self):
        raw_docs = [
            SimpleNamespace(
                page_content="child 1",
                metadata={"parent_content": "parent A", "parent_id": "pA"},
            ),
            SimpleNamespace(
                page_content="child 2",
                metadata={"parent_content": "parent A", "parent_id": "pA"},
            ),
            SimpleNamespace(
                page_content="child 3",
                metadata={"parent_content": "parent B", "parent_id": "pB"},
            ),
        ]
        vectorstore = FakeVectorStore(docs=raw_docs)
        retriever = QdrantRetriever(vectorstore=vectorstore)

        docs = await retriever.retrieve("query", 5)

        self.assertEqual(len(docs), 2)
        self.assertEqual(docs[0].page_content, "parent A")
        self.assertEqual(docs[1].page_content, "parent B")


if __name__ == "__main__":
    unittest.main()
