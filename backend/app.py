import os
from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Tracing imports
from openinference.instrumentation.langchain import LangChainInstrumentor
from opentelemetry import trace as otel_trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# Load env 
load_dotenv()

# Config
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = "rag_collection"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
PHOENIX_URL = os.getenv("PHOENIX_URL", "http://localhost:6006/v1/traces")

# OpenRouter / OpenAI Config
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "google/gemini-2.0-flash-001") # Default to Gemini 2.0 Flash

def setup_tracing():
    print(f"Setting up tracing to Phoenix at {PHOENIX_URL}...")
    exporter = OTLPSpanExporter(endpoint=PHOENIX_URL)
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(BatchSpanProcessor(exporter))
    otel_trace.set_tracer_provider(tracer_provider)
    LangChainInstrumentor().instrument()

app = FastAPI(title="Modular RAG API")

# Initialize components
setup_tracing()
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
client = QdrantClient(url=QDRANT_URL)
vectorstore = QdrantVectorStore(
    client=client,
    collection_name=COLLECTION_NAME,
    embedding=embeddings,
)

llm = ChatOpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
    model=OPENROUTER_MODEL,
)

# Models
class ChatRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    query: str
    response: str
    source_documents: List[str]

class RAGService:
    @staticmethod
    def retrieve(query: str, k: int = 3):
        print(f"Retrieving for: {query}")
        docs = vectorstore.similarity_search(query, k=k)
        return docs

    @staticmethod
    def generate(query: str, context_docs: List):
        print("Generating response...")
        context_text = "\n\n".join([doc.page_content for doc in context_docs])
        
        prompt = ChatPromptTemplate.from_template("""
        Answer the following question based ONLY on the provided context. 
        If the answer is not in the context, say that you don't know.
        
        Context:
        {context}
        
        Question: {question}
        """)
        
        chain = (
            {"context": lambda x: context_text, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        return chain.invoke(query)

@app.get("/")
async def root():
    return {"message": "RAG API is running", "docs": "/docs"}


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    if not OPENROUTER_API_KEY:
        raise HTTPException(status_code=500, detail="OPENROUTER_API_KEY not set")

    try:
        # Step 1: Retrieve
        docs = RAGService.retrieve(request.query)
        
        # Step 2: Generate
        answer = RAGService.generate(request.query, docs)
        
        return ChatResponse(
            query=request.query,
            response=answer,
            source_documents=[doc.page_content for doc in docs]
        )
    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
