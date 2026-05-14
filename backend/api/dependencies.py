import asyncio
from collections import defaultdict, deque
from math import ceil
from time import monotonic

from opentelemetry import trace as otel_trace
from pydantic import SecretStr

from backend.adapters import OpenRouterGenerator, PassThroughReranker, QdrantRetriever
from backend.core import InfrastructureFactory, RAGPipeline, Settings, TracingHook
from backend.core.evaluation_store import (
    create_pending_evaluation,
    mark_evaluation_completed,
    mark_evaluation_failed,
)
from backend.evaluation import RagasEvaluator
from backend.services.chat_service import ChatService
from backend.services.evaluation_service import EvaluationService, ScoreBroadcaster


settings = Settings()
factory = InfrastructureFactory(settings)
factory.setup_tracing(service_name="rag-backend")

if settings.openrouter_api_key is None:
    raise RuntimeError("OPENROUTER_API_KEY is required")

openrouter_api_key = settings.openrouter_api_key
tracer = otel_trace.get_tracer(__name__)
embeddings = factory.get_embeddings()
vectorstore = factory.get_vectorstore()

retriever_adapter = QdrantRetriever(vectorstore=vectorstore)
reranker_adapter = PassThroughReranker()
pipeline_cache: dict[str, RAGPipeline] = {}


def get_openrouter_api_key() -> SecretStr:
    return openrouter_api_key


def get_pipeline_for_model(model: str | None) -> RAGPipeline:
    selected_model = model or settings.openrouter_model
    cached = pipeline_cache.get(selected_model)
    if cached is not None:
        return cached

    generator_adapter = OpenRouterGenerator(
        api_key_provider=get_openrouter_api_key,
        default_model=selected_model,
    )
    pipeline = RAGPipeline(
        retriever=retriever_adapter,
        reranker=reranker_adapter,
        generator=generator_adapter,
        hooks=[TracingHook(tracer=tracer)],
    )
    pipeline_cache[selected_model] = pipeline
    return pipeline


evaluator = RagasEvaluator(
    api_key=openrouter_api_key.get_secret_value(),
    eval_model=settings.ragas_eval_model,
    embeddings=embeddings,
)

score_broadcaster = ScoreBroadcaster()
evaluation_service = EvaluationService(
    evaluator=evaluator,
    score_broadcaster=score_broadcaster,
    pipeline_factory=get_pipeline_for_model,
    create_pending_evaluation_fn=create_pending_evaluation,
    mark_evaluation_completed_fn=mark_evaluation_completed,
    mark_evaluation_failed_fn=mark_evaluation_failed,
    token_counter=lambda content: len(content.split()),
)


def get_db():
    yield from factory.get_session()


def get_chat_service() -> ChatService:
    return ChatService(
        tracer=tracer,
        pipeline_factory=get_pipeline_for_model,
        token_counter=lambda content: len(content.split()),
        create_pending_evaluation_fn=create_pending_evaluation,
        evaluation_service=evaluation_service,
    )


def get_evaluation_service() -> EvaluationService:
    return evaluation_service


class InMemoryRateLimiter:
    def __init__(self, max_calls: int, period_seconds: float):
        self.max_calls = max_calls
        self.period_seconds = period_seconds
        self._calls: dict[str, deque[float]] = defaultdict(deque)
        self._lock = asyncio.Lock()

    async def check(self, key: str) -> tuple[bool, int]:
        now = monotonic()
        cutoff = now - self.period_seconds
        async with self._lock:
            bucket = self._calls[key]
            while bucket and bucket[0] < cutoff:
                bucket.popleft()

            if len(bucket) >= self.max_calls:
                retry_after = max(1, ceil(self.period_seconds - (now - bucket[0])))
                return False, retry_after

            bucket.append(now)
            return True, 0

    async def reset(self) -> None:
        async with self._lock:
            self._calls.clear()


reevaluate_rate_limiter = InMemoryRateLimiter(max_calls=3, period_seconds=60)
