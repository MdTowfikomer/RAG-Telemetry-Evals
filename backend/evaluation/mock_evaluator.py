from typing import Dict
from .interfaces import Evaluator, EvalContext

class MockEvaluator(Evaluator):
    def __init__(self, scores: Dict[str, float]):
        self.scores = scores or {"faithfulness": 1.0, "answer_relevancy": 1.0}
        self.called_with = []

    async def evaluate(self, ctx: EvalContext) -> Dict[str, float]:
        self.called_with.append(ctx)
        return self.scores
