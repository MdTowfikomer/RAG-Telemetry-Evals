from abc import ABC, abstractmethod
from pydantic import BaseModel
from typing import List, Dict

class EvalContext(BaseModel):
    query: str
    answer: str
    contexts: List[str]

class Evaluator(ABC):
    @abstractmethod
    async def evaluate(self, ctx: EvalContext) -> Dict[str, float]:
        """
        Evaluate a RAG interaction and return a dictionary of metric names to scores.
        """
        pass
