import os
from pathlib import Path

from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_core.documents import Document as LCDocument
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import SecretStr
from qdrant_client.http import models

from backend.core import InfrastructureFactory, Settings

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data"


def ingest(factory: InfrastructureFactory):
    print(f"Loading documents from {DATA_PATH}...")
    DATA_PATH.mkdir(parents=True, exist_ok=True)

    # Loaders for different file types
    text_loader = DirectoryLoader(
        str(DATA_PATH), glob="**/*.txt", loader_cls=TextLoader
    )
    md_loader = DirectoryLoader(str(DATA_PATH), glob="**/*.md", loader_cls=TextLoader)
    pdf_files = list(DATA_PATH.rglob("*.pdf"))

    docs = []
    docs.extend(text_loader.load())
    docs.extend(md_loader.load())
    for file_path in pdf_files:
        docs.extend(PyPDFLoader(str(file_path)).load())

    print(f"Loaded {len(docs)} documents.")
    if not docs:
        print("No documents found. Exiting.")
        return

    # Split using Parent-Child chunking strategy
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)

    splits = []
    parent_id_counter = 0
    for doc in docs:
        parent_docs = parent_splitter.split_documents([doc])
        for parent_doc in parent_docs:
            parent_id = f"{parent_doc.metadata.get('source', 'unknown')}_p{parent_id_counter}"
            parent_id_counter += 1

            # Split parent chunk into child chunks
            child_docs = child_splitter.split_documents([parent_doc])
            for child_doc in child_docs:
                child_metadata = dict(child_doc.metadata)
                child_metadata["parent_id"] = parent_id
                child_metadata["parent_content"] = parent_doc.page_content

                splits.append(
                    LCDocument(
                        page_content=child_doc.page_content,
                        metadata=child_metadata,
                    )
                )

    print(f"Created {len(splits)} child chunks associated with parents.")

    # Infra dependencies
    embeddings = factory.get_embeddings()
    sample_emb = embeddings.embed_query("test")
    vector_size = len(sample_emb)
    print(f"Detected embedding size: {vector_size}")
    client = factory.get_qdrant_client()

    # Recreate collection to ensure sparse vector configuration is active
    if client.collection_exists(factory.settings.collection_name):
        print(f"Deleting existing collection: {factory.settings.collection_name}")
        client.delete_collection(factory.settings.collection_name)

    print(f"Creating collection: {factory.settings.collection_name}")
    client.create_collection(
        collection_name=factory.settings.collection_name,
        vectors_config=models.VectorParams(
            size=vector_size, distance=models.Distance.COSINE
        ),
        sparse_vectors_config={
            "fastembed-sparse": models.SparseVectorParams(
                index=models.SparseIndexParams(
                    on_disk=True
                )
            )
        }
    )

    qdrant = factory.get_vectorstore()

    # Add documents
    print("Upserting to Qdrant...")
    qdrant.add_documents(splits)

    # Verification Search
    print("\n--- Verification ---")
    results = qdrant.similarity_search("RAG", k=1)
    if results:
        print(f"Search successful. Found: {results[0].page_content[:100]}...")
    else:
        print("Search failed. No documents found.")

    print("Ingestion complete.")


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    settings = Settings(
        openrouter_api_key=SecretStr(
            os.getenv("OPENROUTER_API_KEY", "ingest-not-required")
        ),
        cohere_api_key=SecretStr(os.getenv("COHERE_API_KEY", "ingest-not-required")),
    )
    factory = InfrastructureFactory(settings)
    factory.setup_tracing(service_name="rag-ingest")
    ingest(factory)
