import os
from pathlib import Path

from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import SecretStr
from qdrant_client.http import models

from backend.core import InfrastructureFactory, Settings

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data"


def ingest(factory: InfrastructureFactory):
    print(f"Loading documents from {DATA_PATH}...")

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

    # Split
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents(docs)
    print(f"Created {len(splits)} chunks.")

    # Infra dependencies
    _ = factory.get_embeddings()
    client = factory.get_qdrant_client()

    # Ensure collection exists
    if not client.collection_exists(factory.settings.collection_name):
        print(f"Creating collection: {factory.settings.collection_name}")
        client.create_collection(
            collection_name=factory.settings.collection_name,
            vectors_config=models.VectorParams(
                size=384, distance=models.Distance.COSINE
            ),
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
    settings = Settings(
        openrouter_api_key=SecretStr(
            os.getenv("OPENROUTER_API_KEY", "ingest-not-required")
        )
    )
    factory = InfrastructureFactory(settings)
    factory.setup_tracing(service_name="rag-ingest")
    ingest(factory)
