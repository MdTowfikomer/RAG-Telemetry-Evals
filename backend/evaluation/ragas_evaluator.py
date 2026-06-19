import math
from typing import Any, Dict, Union, Callable

from openai import OpenAI
from opentelemetry import trace as otel_trace
from ragas import evaluate
from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
from ragas.embeddings.base import BaseRagasEmbedding
from ragas.llms import llm_factory
from ragas.llms.base import InstructorBaseRagasLLM
from ragas.metrics._faithfulness import Faithfulness
from ragas.metrics._answer_relevance import ResponseRelevancy
from ragas.run_config import RunConfig

from .interfaces import EvalContext, Evaluator


class RagasLangchainEmbeddings(BaseRagasEmbedding):
    """
    Thin wrapper adapting a LangChain embedding object to the ragas 0.4.x
    BaseRagasEmbedding interface (embed_text / aembed_text).
    """

    def __init__(self, embeddings: Any):
        super().__init__()
        self.embeddings = embeddings

    def embed_query(self, text: str) -> list[float]:
        return self.embeddings.embed_query(text)

    async def aembed_query(self, text: str) -> list[float]:
        return await self.embeddings.aembed_query(text)

    def embed_text(self, text: str, *args, **kwargs) -> list[float]:
        return self.embed_query(text)

    async def aembed_text(self, text: str, *args, **kwargs) -> list[float]:
        return await self.aembed_query(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embeddings.embed_documents(texts)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return await self.embeddings.aembed_documents(texts)


class RagasEvaluator(Evaluator):
    def __init__(
        self,
        api_key: Union[str, Callable[[], Any]],
        base_url: str = "https://openrouter.ai/api/v1",
        eval_model: str = "openrouter/free",
        embeddings: Any = None,
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.eval_model = eval_model
        self.embeddings = embeddings
        self.tracer = otel_trace.get_tracer(__name__)

    async def evaluate(self, ctx: EvalContext) -> Dict[str, float]:
        """
        Evaluate faithfulness and answer relevancy using ragas 0.4.x.

        ragas 0.4.x changes vs. older versions:
          - Metrics must be instances of ragas.metrics.base.Metric (not BaseMetric).
            Faithfulness  → ragas.metrics._faithfulness.Faithfulness
            AnswerRelevancy → ragas.metrics._answer_relevance.ResponseRelevancy
          - Dataset must be an EvaluationDataset of SingleTurnSample objects.
            Field names: user_input, response, retrieved_contexts.
          - llm/embeddings are passed to evaluate() so ragas handles .init() itself.
        """
        with self.tracer.start_as_current_span("ragas_evaluation") as span:
            try:
                # ── Build dataset ────────────────────────────────────────────
                sample = SingleTurnSample(
                    user_input=ctx.query,
                    response=ctx.answer,
                    retrieved_contexts=list(ctx.contexts),
                )
                dataset = EvaluationDataset(samples=[sample])

                # ── Build ragas LLM ──────────────────────────────────────────
                api_key_value = (
                    self.api_key() if callable(self.api_key) else self.api_key
                )
                if hasattr(api_key_value, "get_secret_value"):
                    api_key_value = api_key_value.get_secret_value()

                judge_client = OpenAI(
                    api_key=api_key_value,
                    base_url=self.base_url,
                )
                ragas_llm: InstructorBaseRagasLLM = llm_factory(
                    self.eval_model,
                    provider="openai",
                    client=judge_client,
                )

                # ── Build ragas embeddings ───────────────────────────────────
                ragas_embeddings = RagasLangchainEmbeddings(
                    embeddings=self.embeddings
                )

                # ── Metrics (must be Metric subclasses in 0.4.x) ────────────
                # Pass llm/embeddings via evaluate() so ragas calls .init()
                # correctly; do NOT pass them to the metric constructors.
                faithfulness_metric = Faithfulness()
                answer_relevancy_metric = ResponseRelevancy()

                # ── Run evaluation ───────────────────────────────────────────
                result = evaluate(
                    dataset=dataset,
                    metrics=[faithfulness_metric, answer_relevancy_metric],
                    llm=ragas_llm,
                    embeddings=ragas_embeddings,
                    run_config=RunConfig(timeout=45, max_retries=1, max_workers=2),
                    raise_exceptions=False,
                    show_progress=False,
                )

                # ── Extract scores ───────────────────────────────────────────
                result_df = result.to_pandas()
                f_score = float(result_df.loc[0, "faithfulness"])
                r_score = float(result_df.loc[0, "answer_relevancy"])

                scores = {
                    "faithfulness": f_score,
                    "answer_relevancy": r_score,
                    "reasoning": (
                        f"Faithfulness={f_score:.3f}, AnswerRelevancy={r_score:.3f}. "
                        "Scores are computed by Ragas judges over retrieved contexts."
                    ),
                }

                if math.isnan(f_score) or math.isnan(r_score):
                    span.set_attribute("ragas.nan_scores", True)

                span.set_attribute("ragas.faithfulness", f_score)
                span.set_attribute("ragas.answer_relevancy", r_score)
                span.set_attribute("ragas.eval_model", self.eval_model)

                return scores

            except Exception as e:
                span.record_exception(e)
                raise e
