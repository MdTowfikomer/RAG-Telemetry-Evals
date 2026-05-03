from langchain_huggingface import HuggingFaceEmbeddings as LCHuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_qdrant import QdrantVectorStore
from openinference.instrumentation.langchain import LangChainInstrumentor
from opentelemetry import trace as otel_trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from qdrant_client import QdrantClient

from .config import Settings


class InfrastructureFactory:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._embeddings = None
        self._qdrant_client = None
        self._vectorstore = None
        self._tracing_initialized = False

    def setup_tracing(self, service_name: str) -> None:
        if self._tracing_initialized:
            return

        resource = Resource(attributes={"service.name": service_name})
        exporter = OTLPSpanExporter(endpoint=self.settings.phoenix_url)
        tracer_provider = TracerProvider(resource=resource)
        tracer_provider.add_span_processor(BatchSpanProcessor(exporter))
        otel_trace.set_tracer_provider(tracer_provider)
        LangChainInstrumentor().instrument()
        self._tracing_initialized = True

    def get_embeddings(self):
        if self._embeddings is None:
            self._embeddings = LCHuggingFaceEmbeddings(
                model_name=self.settings.embedding_model
            )
        return self._embeddings

    def get_qdrant_client(self):
        if self._qdrant_client is None:
            self._qdrant_client = QdrantClient(url=self.settings.qdrant_url)
        return self._qdrant_client

    def get_vectorstore(self):
        if self._vectorstore is None:
            self._vectorstore = QdrantVectorStore(
                client=self.get_qdrant_client(),
                collection_name=self.settings.collection_name,
                embedding=self.get_embeddings(),
            )
        return self._vectorstore

    def get_llm(self, model: str | None = None):
        selected_model = model or self.settings.openrouter_model
        return ChatOpenAI(
            api_key=self.settings.openrouter_api_key,
            base_url="https://openrouter.ai/api/v1",
            model=selected_model,
        )
