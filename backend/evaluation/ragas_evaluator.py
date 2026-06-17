import math
from typing import Any, Dict, Sequence, Union, Callable, cast

from datasets import Dataset
from openai import OpenAI
from opentelemetry import trace as otel_trace
from ragas import evaluate
from ragas.embeddings.base import BaseRagasEmbedding
from ragas.llms import llm_factory
from ragas.metrics.collections import AnswerRelevancy, Faithfulness
from ragas.metrics.base import Metric
from ragas.run_config import RunConfig

from .interfaces import EvalContext, Evaluator


class RagasLangchainEmbeddings(BaseRagasEmbedding):
    """
    Custom wrapper to adapt LangChain embeddings for newer Ragas versions,
    replacing the deprecated LangchainEmbeddingsWrapper.
    """
    def __init__(self, embeddings: Any):
        super().__init__()
        self.embeddings = embeddings

    def embed_text(self, text: str) -> list[float]:
        return self.embeddings.embed_query(text)

    async def aembed_text(self, text: str) -> list[float]:
        return await self.embeddings.aembed_query(text)


class RagasEvaluator(Evaluator):
    def __init__(
        self,
        api_key: Union[str, Callable[[], Any]],
        base_url: str = "https://openrouter.ai/api/v1",
        eval_model: str = "google/gemini-2.0-flash-001",
        embeddings: Any = None,
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.eval_model = eval_model
        self.embeddings = embeddings
        self.tracer = otel_trace.get_tracer(__name__)

    async def evaluate(self, ctx: EvalContext) -> Dict[str, float]:
        """
        Implementation of evaluation using Ragas.
        """
        with self.tracer.start_as_current_span("ragas_evaluation") as span:
            try:
                # Prepare data
                dataset = Dataset.from_dict(
                    {
                        "question": [ctx.query],
                        "answer": [ctx.answer],
                        "contexts": [ctx.contexts],
                    }
                )

                # Setup Ragas LLM
                api_key_value = self.api_key() if callable(self.api_key) else self.api_key
                if hasattr(api_key_value, "get_secret_value"):
                    api_key_value = api_key_value.get_secret_value()

                judge_client = OpenAI(
                    api_key=api_key_value,
                    base_url=self.base_url,
                )
                ragas_llm = llm_factory(
                    self.eval_model,
                    provider="openai",
                    client=judge_client,
                )

                # Setup Ragas Embeddings using our custom wrapper to avoid deprecation warnings
                ragas_embeddings = RagasLangchainEmbeddings(embeddings=self.embeddings)

                metrics: Sequence[Metric] = cast(  # Understand what these metrics do
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

                # Log to Phoenix as span attributes
                span.set_attribute("ragas.faithfulness", f_score)
                span.set_attribute("ragas.answer_relevancy", r_score)
                span.set_attribute("ragas.eval_model", self.eval_model)

                return scores

            except Exception as e:
                span.record_exception(e)
                raise e
