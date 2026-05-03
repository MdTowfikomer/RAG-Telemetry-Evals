import math
from typing import Dict, Sequence, Any, cast
from datasets import Dataset
from opentelemetry import trace as otel_trace
from openai import OpenAI
from ragas import evaluate
from ragas.embeddings.base import LangchainEmbeddingsWrapper
from ragas.llms import llm_factory
from ragas.metrics import AnswerRelevancy, Faithfulness
from ragas.metrics.base import Metric
from ragas.run_config import RunConfig

from .interfaces import Evaluator, EvalContext

class RagasEvaluator(Evaluator):
    def __init__(
        self,
        api_key: str,
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
                judge_client = OpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url,
                )
                ragas_llm = llm_factory(
                    self.eval_model,
                    provider="openai",
                    client=judge_client,
                )
                
                # Setup Ragas Embeddings
                ragas_embeddings = LangchainEmbeddingsWrapper(embeddings=self.embeddings) # what does ragas_embeddings do?

                metrics: Sequence[Metric] = cast( # Understand what these metrics do
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
                    "answer_relevancy": r_score
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
