import json
import os
from typing import AsyncGenerator, List

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_huggingface import HuggingFaceEmbeddings as LCHuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore

# Tracing imports
from openinference.instrumentation.langchain import LangChainInstrumentor
from opentelemetry import context as otel_context
from opentelemetry import trace as otel_trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from pydantic import BaseModel, SecretStr
from qdrant_client import QdrantClient

from backend.adapters import OpenRouterGenerator, PassThroughReranker, QdrantRetriever
from backend.core import RAGPipeline


from .evaluation import EvalContext, RagasEvaluator


# Load env
load_dotenv()

# Config
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = "rag_collection"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
PHOENIX_URL = os.getenv("PHOENIX_URL", "http://localhost:6006/v1/traces")

# OpenRouter / OpenAI Config
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "google/gemini-2.0-flash-001")
RAGAS_EVAL_MODEL = os.getenv("RAGAS_EVAL_MODEL", "openai/gpt-4o-mini")


def get_openrouter_api_key() -> SecretStr:
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY not set")
    return SecretStr(OPENROUTER_API_KEY)


def setup_tracing():
    print(f"Setting up tracing to Phoenix at {PHOENIX_URL}...")
    resource = Resource(attributes={"service.name": "rag-backend"})
    exporter = OTLPSpanExporter(endpoint=PHOENIX_URL)
    tracer_provider = TracerProvider(resource=resource)
    tracer_provider.add_span_processor(BatchSpanProcessor(exporter))
    otel_trace.set_tracer_provider(tracer_provider)
    LangChainInstrumentor().instrument()


app = FastAPI(title="Modular RAG API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
setup_tracing()
tracer = otel_trace.get_tracer(__name__)
embeddings = LCHuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
client = QdrantClient(url=QDRANT_URL)
vectorstore = QdrantVectorStore(
    client=client,
    collection_name=COLLECTION_NAME,
    embedding=embeddings,
)

retriever_adapter = QdrantRetriever(vectorstore=vectorstore, tracer=tracer)
reranker_adapter = PassThroughReranker(tracer=tracer)

pipeline_cache: dict[str, RAGPipeline] = {}


def get_pipeline_for_model(model: str | None) -> RAGPipeline:
    selected_model = model or OPENROUTER_MODEL
    cached = pipeline_cache.get(selected_model)
    if cached is not None:
        return cached

    generator_adapter = OpenRouterGenerator(
        api_key_provider=get_openrouter_api_key,
        default_model=selected_model,
        tracer=tracer,
    )

    pipeline = RAGPipeline(
        retriever=retriever_adapter,
        reranker=reranker_adapter,
        generator=generator_adapter,
        tracer=tracer,
    )

    pipeline_cache[selected_model] = pipeline
    return pipeline


evaluator = RagasEvaluator(
    api_key=get_openrouter_api_key().get_secret_value(),
    eval_model=RAGAS_EVAL_MODEL,
    embeddings=embeddings,
)


# Models
class ChatRequest(BaseModel):
    query: str
    k: int = 3
    model: str = OPENROUTER_MODEL


class ChatResponse(BaseModel):
    query: str
    response: str
    source_documents: List[str]


class ContextResponse(BaseModel):
    query: str
    source_documents: List[str]


async def evaluate_ragas(
    query: str, answer: str, contexts: List[str], parent_context=None
):
    """
    Background task to compute Ragas metrics using the Evaluation Engine Service.
    """
    if parent_context:
        otel_context.attach(parent_context)

    try:
        print("Delegating to Evaluation Engine Service...")
        await evaluator.evaluate(
            EvalContext(query=query, answer=answer, contexts=contexts)
        )
    except Exception as e:
        print(f"Error in Ragas evaluation delegation: {e}")


async def sse_stream_response(
    query: str,
    k: int,
    model: str,
) -> AsyncGenerator[str, None]:
    pipeline = get_pipeline_for_model(model)

    async for token in pipeline.stream(query, k=k):
        payload = json.dumps({"token": token})
        yield f"data: {payload}\n\n"

    yield "data: [DONE]\n\n"


@app.get("/")
async def root():
    return {"message": "RAG API is running", "docs": "/docs"}


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest, background_tasks: BackgroundTasks):
    if not OPENROUTER_API_KEY:
        raise HTTPException(status_code=500, detail="OPENROUTER_API_KEY not set")

    try:
        with tracer.start_as_current_span("chat_flow") as span:
            span.set_attribute("chat.query", request.query)
            pipeline = get_pipeline_for_model(request.model)
            answer, docs = await pipeline.execute(request.query, k=request.k)
            contexts = [doc.page_content for doc in docs]

            current_context = otel_context.get_current()
            background_tasks.add_task(
                evaluate_ragas,
                request.query,
                answer,
                contexts,
                current_context,
            )

            return ChatResponse(
                query=request.query,
                response=answer,
                source_documents=contexts,
            )
    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/context", response_model=ContextResponse)
async def context_endpoint(request: ChatRequest):
    pipeline = get_pipeline_for_model(request.model)
    docs = await pipeline.prepare_context(request.query, k=request.k)

    return ContextResponse(
        query=request.query,
        source_documents=[doc.page_content for doc in docs],
    )


@app.get("/chat/stream")
async def chat_stream_get_endpoint(
    query: str, k: int = 3, model: str = OPENROUTER_MODEL
):
    return StreamingResponse(
        sse_stream_response(query=query, k=k, model=model),
        media_type="text/event-stream",
    )


@app.post("/chat/stream")
async def chat_stream_endpoint(request: ChatRequest):
    return StreamingResponse(
        sse_stream_response(query=request.query, k=request.k, model=request.model),
        media_type="text/event-stream",
    )


@app.get("/health")
async def health():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
