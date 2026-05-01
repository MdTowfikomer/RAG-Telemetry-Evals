import os
from dotenv import load_dotenv

from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http import models
import glob

# Tracing imports
from openinference.instrumentation.langchain import LangChainInstrumentor
from opentelemetry import trace as otel_trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# Load env 
load_dotenv()

# Config
DATA_PATH = "../data"
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = "rag_collection"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
PHOENIX_URL = os.getenv("PHOENIX_URL", "http://localhost:6006/v1/traces")

def setup_tracing():
    print(f"Setting up tracing to Phoenix at {PHOENIX_URL}...")
    endpoint = PHOENIX_URL
    exporter = OTLPSpanExporter(endpoint=endpoint)
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(BatchSpanProcessor(exporter))
    otel_trace.set_tracer_provider(tracer_provider)
    LangChainInstrumentor().instrument()

def ingest():
    print(f"Loading documents from {DATA_PATH}...")
    
    # Loaders for different file types
    text_loader = DirectoryLoader(DATA_PATH, glob="**/*.txt", loader_cls=TextLoader)
    md_loader = DirectoryLoader(DATA_PATH, glob="**/*.md", loader_cls=TextLoader)
    pdf_loader = glob.glob("data/**/*.pdf", recursive=True)
    
    docs = []
    docs.extend(text_loader.load())
    docs.extend(md_loader.load())
    for file in pdf_loader:
        loader = PyPDFLoader(file)
        docs.extend(loader.load())
    
    print(f"Loaded {len(docs)} documents.")
    if not docs:
        print("No documents found. Exiting.")
        return

    # Split
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents(docs)
    print(f"Created {len(splits)} chunks.")

    # Embeddings
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    # Qdrant Setup
    client = QdrantClient(url=QDRANT_URL)
    
    # Ensure collection exists
    if not client.collection_exists(COLLECTION_NAME):
        print(f"Creating collection: {COLLECTION_NAME}")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE),
        )

    # Vector Store
    qdrant = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )

    # Add documents
    print("Upserting to Qdrant...")
    qdrant.add_documents(splits)
    
    # Verification Search
    print("\n--- Verification ---")
    results = qdrant.similarity_search("RAG", k=1)
    if len(results) > 0:
        print(f"Search successful. Found: {results[0].page_content[:100]}...")
    else:
        print("Search failed. No documents found.")
    
    print("Ingestion complete.")

if __name__ == "__main__":
    setup_tracing()
    ingest()
