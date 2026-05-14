import {
  useEffect,
  type Dispatch,
  type MutableRefObject,
  type SetStateAction,
} from "react";
import { api } from "../lib/api";
import type { ChatMessage } from "../types";

type UseScoreSubscriptionArgs = {
  setMessages: Dispatch<SetStateAction<ChatMessage[]>>;
  assistantIdAliasRef: MutableRefObject<Record<string, string>>;
  selectedMessageIdRef: MutableRefObject<string | null>;
  setSelectedMessageId: Dispatch<SetStateAction<string | null>>;
  loadEvaluationHistory: (messageId: string) => Promise<void>;
};

export function useScoreSubscription({
  setMessages,
  assistantIdAliasRef,
  selectedMessageIdRef,
  setSelectedMessageId,
  loadEvaluationHistory,
}: UseScoreSubscriptionArgs) {
  useEffect(() => {
    const cleanup = api.streamScores(
      (event) => {
        const aliasId = assistantIdAliasRef.current[event.message_id];

        setMessages((previous) =>
          previous.map((message) => {
            if (message.id !== event.message_id && message.id !== aliasId) {
              return message;
            }

            return {
              ...message,
              faithfulness:
                typeof event.faithfulness === "number"
                  ? event.faithfulness
                  : message.faithfulness,
              answerRelevancy:
                typeof event.answer_relevancy === "number"
                  ? event.answer_relevancy
                  : message.answerRelevancy,
              evaluationStatus: event.status ?? message.evaluationStatus,
              evaluationVersion:
                typeof event.version === "number"
                  ? event.version
                  : message.evaluationVersion,
              reasoning:
                typeof event.reasoning === "string"
                  ? event.reasoning
                  : message.reasoning,
              latencyMs:
                typeof event.latency_ms === "number"
                  ? event.latency_ms
                  : message.latencyMs,
              tokenCount:
                typeof event.token_count === "number"
                  ? event.token_count
                  : message.tokenCount,
            };
          }),
        );

        if (
          selectedMessageIdRef.current !== null &&
          (selectedMessageIdRef.current === event.message_id ||
            selectedMessageIdRef.current === aliasId)
        ) {
          const messageIdForHistory =
            selectedMessageIdRef.current === aliasId
              ? event.message_id
              : selectedMessageIdRef.current;

          if (selectedMessageIdRef.current === aliasId) {
            setSelectedMessageId(event.message_id);
          }
          void loadEvaluationHistory(messageIdForHistory);
        }
      },
      () => {
        // Intentionally ignore score stream errors in UI state.
      },
    );

    return () => {
      cleanup();
    };
  }, [
    assistantIdAliasRef,
    loadEvaluationHistory,
    selectedMessageIdRef,
    setMessages,
    setSelectedMessageId,
  ]);
}
