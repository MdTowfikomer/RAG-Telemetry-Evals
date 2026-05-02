import asyncio
import json
import math
import os
from typing import Any, List, Sequence, cast

from datasets import Dataset
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings as LCHuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_qdrant import QdrantVectorStore
from openai import OpenAI

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
from ragas import evaluate
from ragas.embeddings.base import LangchainEmbeddingsWrapper
from ragas.llms import llm_factory
from ragas.metrics import AnswerRelevancy, Faithfulness
from ragas.metrics.base import Metric
from ragas.run_config import RunConfig

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

llm = ChatOpenAI(
    api_key=get_openrouter_api_key(),
    base_url="https://openrouter.ai/api/v1",
    model=OPENROUTER_MODEL,
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


class RAGService:
    @staticmethod
    def retrieve(query: str, k: int = 3) -> List:
        with tracer.start_as_current_span("retrieve") as span:
            print(f"Retrieving for: {query} with k={k}")
            docs = vectorstore.similarity_search(query, k=k)
            span.set_attribute("retrieval.k", k)
            span.set_attribute("retrieval.num_docs", len(docs))
            return docs

    @staticmethod
    def rerank(query: str, docs: List) -> List:
        with tracer.start_as_current_span("rerank") as span:
            print(f"Reranking {len(docs)} docs...")
            # Placeholder: just return docs
            span.set_attribute("rerank.num_docs", len(docs))
            return docs

    @staticmethod
    def evaluate_ragas(
        query: str, answer: str, contexts: List[str], parent_context=None
    ):
        """
        Background task to compute Ragas metrics and log them to Phoenix.
        """
        if parent_context:
            otel_context.attach(parent_context)

        with tracer.start_as_current_span("ragas_evaluation") as span:
            try:
                print("Starting Ragas evaluation...")

                # Prepare data with compatible schema
                dataset = Dataset.from_dict(
                    {
                        "question": [query],
                        "answer": [answer],
                        "contexts": [contexts],
                    }
                )

                # Setup modern Ragas judge LLM + embeddings
                judge_client = OpenAI(
                    api_key=get_openrouter_api_key().get_secret_value(),
                    base_url="https://openrouter.ai/api/v1",
                )
                ragas_llm = llm_factory(
                    RAGAS_EVAL_MODEL,
                    provider="openai",
                    client=judge_client,
                )
                # Note: LangchainEmbeddingsWrapper is required for compatibility with 
                # stable metrics in Ragas 0.4.3 despite the deprecation warning.
                ragas_embeddings = LangchainEmbeddingsWrapper(embeddings=embeddings)

                metrics: Sequence[Metric] = cast(
                    Sequence[Metric],
                    [
                        Faithfulness(llm=ragas_llm),
                        AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings),
                    ],
                )

                # Run evaluation
                result = evaluate(
                    dataset=dataset,
                    metrics=metrics,
                    run_config=RunConfig(timeout=45, max_retries=1, max_workers=2),
                    raise_exceptions=False,
                    show_progress=True,
                )

                # Extract scores
                result_df = cast(Any, result).to_pandas()
                f_score = float(result_df.loc[0, "faithfulness"])
                r_score = float(result_df.loc[0, "answer_relevancy"])

                if math.isnan(f_score) or math.isnan(r_score):
                    print(
                        "Ragas returned NaN scores. This usually means the eval model does not support required JSON/instructor mode. "
                        f"Current RAGAS_EVAL_MODEL={RAGAS_EVAL_MODEL}"
                    )
                    span.set_attribute("ragas.nan_scores", True)
                    span.set_attribute("ragas.eval_model", RAGAS_EVAL_MODEL)
                    return

                # Log to Phoenix as span attributes
                span.set_attribute("ragas.faithfulness", f_score)
                span.set_attribute("ragas.answer_relevancy", r_score)
                span.set_attribute("ragas.eval_model", RAGAS_EVAL_MODEL)

                print(
                    f"Ragas Evals Complete: Faithfulness={f_score:.2f}, Relevancy={r_score:.2f}, EvalModel={RAGAS_EVAL_MODEL}"
                )

            except Exception as e:
                print(f"Error in Ragas evaluation: {e}")
                span.record_exception(e)

    @staticmethod
    def generate(query: str, context_docs: List, model: str = OPENROUTER_MODEL):
        with tracer.start_as_current_span("generate") as span:
            print(f"Generating response with model={model}...")
            context_text = "\n\n".join([doc.page_content for doc in context_docs])

            prompt = ChatPromptTemplate.from_template("""
            Answer the following question based ONLY on the provided context.
            If the answer is not in the context, say that you don't know.

            Context:
            {context}

            Question: {question}
            """)

            # Dynamically set the model
            current_llm = ChatOpenAI(
                api_key=get_openrouter_api_key(),
                base_url="https://openrouter.ai/api/v1",
                model=model,
            )

            chain = (
                {"context": lambda x: context_text, "question": RunnablePassthrough()}
                | prompt
                | current_llm
                | StrOutputParser()
            )

            res = chain.invoke(query)
            span.set_attribute("generation.model", model)
            return res

    @staticmethod
    async def astream_generate(
        query: str, context_docs: List, model: str = OPENROUTER_MODEL
    ):
        # Note: Tracing a stream is more complex, keeping it simple for now
        context_text = "\n\n".join([doc.page_content for doc in context_docs])

        prompt = ChatPromptTemplate.from_template("""
            Answer the following question based ONLY on the provided context.
            If the answer is not in the context, say that you don't know.

            Context:{context}
            Question: {question}
            """)

        current_llm = ChatOpenAI(
            api_key=get_openrouter_api_key(),
            base_url="https://openrouter.ai/api/v1",
            model=model,
        )

        chain = (
            {"context": lambda x: context_text, "question": RunnablePassthrough()}
            | prompt
            | current_llm
            | StrOutputParser()
        )

        async for chunk in chain.astream(query):
            payload = json.dumps({"token": chunk})
            yield f"data: {payload}\n\n"
            await asyncio.sleep(0.01)  # Tiny delay for smoother UI streaming

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
            # Step 1: Retrieve
            docs = RAGService.retrieve(request.query, k=request.k)

            # Step 2: Rerank
            reranked_docs = RAGService.rerank(request.query, docs)

            # Step 3: Generate
            answer = RAGService.generate(
                request.query, reranked_docs, model=request.model
            )

            # Step 4: Trigger Evals in background
            contexts = [doc.page_content for doc in reranked_docs]
            current_context = otel_context.get_current()
            background_tasks.add_task(
                RAGService.evaluate_ragas,
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
    docs = RAGService.retrieve(request.query, k=request.k)
    reranked_docs = RAGService.rerank(request.query, docs)

    return ContextResponse(
        query=request.query,
        source_documents=[doc.page_content for doc in reranked_docs],
    )


@app.get("/chat/stream")
async def chat_stream_get_endpoint(
    query: str, k: int = 3, model: str = OPENROUTER_MODEL
):
    docs = RAGService.retrieve(query, k=k)
    reranked_docs = RAGService.rerank(query, docs)

    return StreamingResponse(
        RAGService.astream_generate(query, reranked_docs, model=model),
        media_type="text/event-stream",
    )


@app.post("/chat/stream")
async def chat_stream_endpoint(request: ChatRequest):
    docs = RAGService.retrieve(request.query, k=request.k)
    reranked_docs = RAGService.rerank(request.query, docs)

    return StreamingResponse(
        RAGService.astream_generate(request.query, reranked_docs, model=request.model),
        media_type="text/event-stream",
    )


@app.get("/health")
async def health():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
