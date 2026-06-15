from langchain_huggingface import HuggingFaceEmbeddings as LCHuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from openinference.instrumentation.langchain import LangChainInstrumentor
from opentelemetry import trace as otel_trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from qdrant_client import QdrantClient
from sqlmodel import Session, SQLModel, create_engine

from .config import Settings
from .models import ChatMessage, ChatSession, Evaluation, UploadedFile  # Ensure models are registered


class InfrastructureFactory:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._embeddings = None
        self._sparse_embeddings = None
        self._qdrant_client = None
        self._vectorstore = None
        self._engine = None
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

    def get_sparse_embeddings(self):
        if self._sparse_embeddings is None:
            self._sparse_embeddings = FastEmbedSparse(
                model_name="Qdrant/bm42-all-minilm-l6-v2-attentions"
            )
        return self._sparse_embeddings

    def get_qdrant_client(self):
        if self._qdrant_client is None:
            api_key = (
                self.settings.qdrant_api_key.get_secret_value()
                if self.settings.qdrant_api_key
                else None
            )
            self._qdrant_client = QdrantClient(
                url=self.settings.qdrant_url,
                api_key=api_key,
            )
        return self._qdrant_client

    def get_vectorstore(self):
        if self._vectorstore is None:
            self._vectorstore = QdrantVectorStore(
                client=self.get_qdrant_client(),
                collection_name=self.settings.collection_name,
                embedding=self.get_embeddings(),
                sparse_embedding=self.get_sparse_embeddings(),
                sparse_vector_name="fastembed-sparse",
                retrieval_mode=RetrievalMode.HYBRID,
            )
        return self._vectorstore

    def get_llm(self, model: str | None = None):
        selected_model = model or self.settings.openrouter_model
        return ChatOpenAI(
            api_key=self.settings.openrouter_api_key,
            base_url="https://openrouter.ai/api/v1",
            model=selected_model,
        )

    def get_engine(self):
        if self._engine is None:
            is_sqlite = self.settings.database_url.startswith("sqlite")
            connect_args = {"check_same_thread": False} if is_sqlite else {}
            pool_kwargs = {}
            if not is_sqlite:
                pool_kwargs = {
                    "pool_recycle": 300,
                    "pool_pre_ping": True,
                }
            self._engine = create_engine(
                self.settings.database_url,
                echo=False,
                connect_args=connect_args,
                **pool_kwargs
            )
        return self._engine

    def init_db(self):
        SQLModel.metadata.create_all(self.get_engine())

    def get_session(self):
        with Session(self.get_engine()) as session:
            yield session
