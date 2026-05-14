import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type Dispatch,
  type SetStateAction,
} from "react";
import { api } from "../lib/api";
import type { MessageEvaluationVersion, ChatMessage, ChatSettings } from "../types";

function isUuid(value: string): boolean {
  return /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i.test(
    value,
  );
}

function mapEvaluationVersion(version: {
  id: string;
  message_id: string;
  version: number;
  status: "pending" | "completed" | "failed";
  faithfulness?: number | null;
  answer_relevancy?: number | null;
  reasoning?: string | null;
  error_message?: string | null;
  created_at: string;
  updated_at: string;
}): MessageEvaluationVersion {
  return {
    id: version.id,
    messageId: version.message_id,
    version: version.version,
    status: version.status,
    faithfulness:
      typeof version.faithfulness === "number" ? version.faithfulness : undefined,
    answerRelevancy:
      typeof version.answer_relevancy === "number"
        ? version.answer_relevancy
        : undefined,
    reasoning: typeof version.reasoning === "string" ? version.reasoning : undefined,
    errorMessage:
      typeof version.error_message === "string" ? version.error_message : undefined,
    createdAt: version.created_at,
    updatedAt: version.updated_at,
  };
}

type UseEvaluationsArgs = {
  settings: ChatSettings;
  setMessages: Dispatch<SetStateAction<ChatMessage[]>>;
  setErrorMessage: Dispatch<SetStateAction<string | null>>;
};

export function useEvaluations({
  settings,
  setMessages,
  setErrorMessage,
}: UseEvaluationsArgs) {
  const [selectedMessageId, setSelectedMessageId] = useState<string | null>(null);
  const [evaluationHistory, setEvaluationHistory] = useState<
    MessageEvaluationVersion[]
  >([]);
  const selectedMessageIdRef = useRef<string | null>(null);

  useEffect(() => {
    selectedMessageIdRef.current = selectedMessageId;
  }, [selectedMessageId]);

  const loadEvaluationHistory = useCallback(async (messageId: string) => {
    if (!isUuid(messageId)) {
      setEvaluationHistory([]);
      return;
    }
    try {
      const versions = await api.fetchMessageEvaluations(messageId);
      setEvaluationHistory(versions.map(mapEvaluationVersion));
    } catch {
      setEvaluationHistory([]);
    }
  }, []);

  const selectAssistantMessage = useCallback(
    async (messageId: string) => {
      setSelectedMessageId(messageId);
      await loadEvaluationHistory(messageId);
    },
    [loadEvaluationHistory],
  );

  const reevaluateAssistantMessage = useCallback(
    async (messageId: string) => {
      if (!isUuid(messageId)) return;

      try {
        const pending = await api.triggerMessageReevaluation(messageId, settings);
        setErrorMessage(null);
        setMessages((previous) =>
          previous.map((message) =>
            message.id === messageId
              ? {
                  ...message,
                  evaluationStatus: pending.status,
                  evaluationVersion: pending.version,
                }
              : message,
          ),
        );

        if (selectedMessageIdRef.current === messageId) {
          await loadEvaluationHistory(messageId);
        }
      } catch (error) {
        const message =
          error instanceof Error
            ? error.message
            : "Could not trigger re-evaluation for this message.";
        setErrorMessage(message);
      }
    },
    [loadEvaluationHistory, setErrorMessage, setMessages, settings],
  );

  const resetEvaluations = useCallback(() => {
    setSelectedMessageId(null);
    setEvaluationHistory([]);
  }, []);

  return {
    selectedMessageId,
    setSelectedMessageId,
    selectedMessageIdRef,
    evaluationHistory,
    setEvaluationHistory,
    loadEvaluationHistory,
    selectAssistantMessage,
    reevaluateAssistantMessage,
    resetEvaluations,
  };
}
