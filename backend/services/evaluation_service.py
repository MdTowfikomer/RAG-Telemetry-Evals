import asyncio
import json
from collections import defaultdict, deque
from datetime import UTC, datetime
from math import ceil
from time import monotonic
from typing import Any, Callable, List, Optional
from uuid import UUID

from sqlmodel import Session as SQLModelSession, col, select

from backend.core import ChatMessage, ChatSession, Evaluation, RAGPipeline, Settings
from backend.core.evaluation_store import (
    create_pending_evaluation,
    mark_evaluation_completed,
    mark_evaluation_failed,
)
from backend.evaluation import EvalContext, RagasEvaluator


class ScoreBroadcaster:
    def __init__(self):
        self.subscribers: list[asyncio.Queue[dict[str, Any]]] = []

    def subscribe(self) -> asyncio.Queue[dict[str, Any]]:
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self.subscribers.append(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[dict[str, Any]]) -> None:
        if queue in self.subscribers:
            self.subscribers.remove(queue)

    async def publish(self, event: dict[str, Any]) -> None:
        for subscriber in list(self.subscribers): # Iterate over a copy to avoid "Set changed size during iteration"
            await subscriber.put(event)


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


class EvaluationService:
    def __init__(
        self,
        *,
        evaluator: RagasEvaluator,
        score_broadcaster: ScoreBroadcaster,
        pipeline_factory: Callable[[Optional[str]], RAGPipeline], # Corrected type hint
        create_pending_evaluation_fn: Callable[..., Any],
        mark_evaluation_completed_fn: Callable[..., Any],
        mark_evaluation_failed_fn: Callable[..., Any],
        token_counter: Callable[[str], int], # Moved here as ChatService doesn't depend on it
    ) -> None:
        self._evaluator = evaluator
        self._score_broadcaster = score_broadcaster
        self._pipeline_factory = pipeline_factory
        self._create_pending_evaluation = create_pending_evaluation_fn
        self._mark_evaluation_completed = mark_evaluation_completed_fn
        self._mark_evaluation_failed = mark_evaluation_failed_fn
        self._token_counter = token_counter


    async def _publish_score_event_for_evaluation(
        self,
        evaluation: Evaluation,
        message: ChatMessage | None,
    ) -> None:
        await self._score_broadcaster.publish(
            {
                "type": "score",
                "message_id": str(evaluation.message_id),
                "faithfulness": evaluation.faithfulness,
                "answer_relevancy": evaluation.answer_relevancy,
                "reasoning": evaluation.reasoning,
                "status": evaluation.status,
                "version": evaluation.version,
                "latency_ms": None if message is None else message.latency_ms,
                "token_count": None if message is None else message.token_count,
            }
        )

    async def trigger_evaluation(
        self,
        query: str,
        answer: str,
        contexts: List[str],
        evaluation_id: UUID,
        db_bind,
        parent_context=None, # otel_context.attach needs to be handled by caller
    ):
        """
        Background task to compute Ragas metrics using the Evaluation Engine Service.
        """
        # Removed otel_context.attach(parent_context) as per issue description
        # The caller (route handler) captures and passes the OTel context as a plain value, not imported inside the service.

        try:
            print("Delegating to Evaluation Engine Service...")
            scores = await self._evaluator.evaluate(
                EvalContext(query=query, answer=answer, contexts=contexts)
            )
            raw_faithfulness = scores.get("faithfulness")
            faithfulness = (
                float(raw_faithfulness)
                if isinstance(raw_faithfulness, (int, float))
                else None
            )
            raw_answer_relevancy = scores.get("answer_relevancy")
            answer_relevancy = (
                float(raw_answer_relevancy)
                if isinstance(raw_answer_relevancy, (int, float))
                else None
            )
            raw_reasoning = scores.get("reasoning")
            reasoning = (
                raw_reasoning
                if isinstance(raw_reasoning, str) and raw_reasoning.strip()
                else "Ragas evaluation completed successfully."
            )

            with SQLModelSession(db_bind) as evaluation_db:
                updated_eval = self._mark_evaluation_completed(
                    db=evaluation_db,
                    evaluation_id=evaluation_id,
                    faithfulness=faithfulness,
                    answer_relevancy=answer_relevancy,
                    reasoning=reasoning,
                )

                if updated_eval is not None:
                    message = evaluation_db.get(ChatMessage, updated_eval.message_id)
                    await self._publish_score_event_for_evaluation(updated_eval, message)
        except Exception as e:
            with SQLModelSession(db_bind) as evaluation_db:
                updated_eval = self._mark_evaluation_failed(
                    db=evaluation_db,
                    evaluation_id=evaluation_id,
                    error_message=str(e),
                )

                if updated_eval is not None:
                    await self._publish_score_event_for_evaluation(updated_eval, None)

            print(f"Error in Ragas evaluation delegation: {e}")

    async def trigger_stream_evaluation(
        self,
        query: str,
        answer: str,
        k: int,
        model: str,
        evaluation_id: UUID,
        db_bind,
    ) -> None:
        try:
            pipeline = self._pipeline_factory(model)
            docs = await pipeline.prepare_context(query, k=k)
            contexts = [doc.page_content for doc in docs]
            await self.trigger_evaluation(
                query=query,
                answer=answer,
                contexts=contexts,
                evaluation_id=evaluation_id,
                db_bind=db_bind,
            )
        except Exception as e:
            print(f"Error in streaming evaluation delegation: {e}")


    async def trigger_reevaluation(
        self,
        message_id: UUID,
        k: int,
        model: str,
        evaluation_id: UUID,
        db_bind,
    ) -> None:
        with SQLModelSession(db_bind) as evaluation_db:
            assistant_message = evaluation_db.get(ChatMessage, message_id)
            if assistant_message is None:
                updated_eval = self._mark_evaluation_failed(
                    db=evaluation_db,
                    evaluation_id=evaluation_id,
                    error_message="Assistant message not found for re-evaluation.",
                )
                if updated_eval is not None:
                    await self._publish_score_event_for_evaluation(updated_eval, None)
                return

            if assistant_message.role != "assistant":
                updated_eval = self._mark_evaluation_failed(
                    db=evaluation_db,
                    evaluation_id=evaluation_id,
                    error_message="Only assistant messages can be re-evaluated.",
                )
                if updated_eval is not None:
                    await self._publish_score_event_for_evaluation(updated_eval, assistant_message)
                return

            latest_user_message = evaluation_db.exec(
                select(ChatMessage)
                .where(
                    ChatMessage.session_id == assistant_message.session_id,
                    ChatMessage.role == "user",
                    ChatMessage.created_at <= assistant_message.created_at,
                )
                .order_by(col(ChatMessage.created_at).desc())
            ).first()

            if latest_user_message is None:
                updated_eval = self._mark_evaluation_failed(
                    db=evaluation_db,
                    evaluation_id=evaluation_id,
                    error_message="No user message found to evaluate this assistant answer.",
                )
                if updated_eval is not None:
                    await self._publish_score_event_for_evaluation(updated_eval, assistant_message)
                return

            if not assistant_message.content.strip():
                updated_eval = self._mark_evaluation_failed(
                    db=evaluation_db,
                    evaluation_id=evaluation_id,
                    error_message="Assistant message content is empty.",
                )
                if updated_eval is not None:
                    await self._publish_score_event_for_evaluation(updated_eval, assistant_message)
                return

            query = latest_user_message.content
            answer = assistant_message.content

        await self.trigger_stream_evaluation(
            query=query,
            answer=answer,
            k=k,
            model=model,
            evaluation_id=evaluation_id,
            db_bind=db_bind,
        )
