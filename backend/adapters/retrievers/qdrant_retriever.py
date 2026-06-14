import asyncio

from backend.core import Document, Retriever


class QdrantRetriever(Retriever):
    def __init__(self, vectorstore):
        self.vectorstore = vectorstore

    async def retrieve(self, query: str, k: int) -> list[Document]:
        # Retrieve more documents to account for parent deduplication
        raw_docs = await asyncio.to_thread(
            self.vectorstore.similarity_search,
            query,
            k * 3,
        )

        seen_parents = set()
        docs = []
        for doc in raw_docs:
            metadata = getattr(doc, "metadata", {}) or {}
            parent_content = metadata.get("parent_content")
            parent_id = metadata.get("parent_id")

            if parent_content:
                dup_key = parent_id or parent_content
                if dup_key not in seen_parents:
                    seen_parents.add(dup_key)
                    docs.append(
                        Document(
                            page_content=parent_content,
                            metadata=metadata,
                        )
                    )
            else:
                docs.append(
                    Document(
                        page_content=getattr(doc, "page_content", ""),
                        metadata=metadata,
                    )
                )

            # Limit the output to k documents
            if len(docs) >= k:
                break

        return docs
