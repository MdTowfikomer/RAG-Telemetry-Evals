import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type Dispatch,
  type SetStateAction,
} from "react";
import { api } from "../lib/api";
import type {
  ChatMessage,
  ContextDoc,
  MessageEvaluationVersion,
  SessionSummary,
} from "../types";

type UseSessionsArgs = {
  initialMessages: ChatMessage[];
  setMessages: Dispatch<SetStateAction<ChatMessage[]>>;
  setContextDocs: Dispatch<SetStateAction<ContextDoc[]>>;
  setIsLoading: Dispatch<SetStateAction<boolean>>;
  setErrorMessage: Dispatch<SetStateAction<string | null>>;
  setSelectedMessageId: Dispatch<SetStateAction<string | null>>;
  setEvaluationHistory: Dispatch<SetStateAction<MessageEvaluationVersion[]>>;
};

export function useSessions({
  initialMessages,
  setMessages,
  setContextDocs,
  setIsLoading,
  setErrorMessage,
  setSelectedMessageId,
  setEvaluationHistory,
}: UseSessionsArgs) {
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);
  const sessionIdRef = useRef<string | null>(null);

  const refreshSessions = useCallback(async () => {
    setSessions(await api.fetchSessions());
  }, []);

  useEffect(() => {
    void refreshSessions();
  }, [refreshSessions]);

  const loadSession = useCallback(
    async (sessionId: string) => {
      setIsLoading(false);
      setErrorMessage(null);

      const sessionMessages = await api.fetchSessionMessages(sessionId);
      const mappedMessages: ChatMessage[] = sessionMessages.map((message) => ({
        id: message.id,
        role: message.role,
        content: message.content,
        latencyMs:
          typeof message.latency_ms === "number" ? message.latency_ms : undefined,
        tokenCount:
          typeof message.token_count === "number"
            ? message.token_count
            : undefined,
        faithfulness:
          typeof message.faithfulness === "number"
            ? message.faithfulness
            : undefined,
        answerRelevancy:
          typeof message.answer_relevancy === "number"
            ? message.answer_relevancy
            : undefined,
        reasoning:
          typeof message.reasoning === "string" ? message.reasoning : undefined,
        evaluationStatus: message.evaluation_status ?? undefined,
        evaluationVersion:
          typeof message.evaluation_version === "number"
            ? message.evaluation_version
            : undefined,
      }));

      setMessages(mappedMessages.length > 0 ? mappedMessages : initialMessages);
      sessionIdRef.current = sessionId;
      setActiveSessionId(sessionId);
      setContextDocs([]);
      setSelectedMessageId(null);
      setEvaluationHistory([]);
    },
    [
      initialMessages,
      setContextDocs,
      setErrorMessage,
      setEvaluationHistory,
      setIsLoading,
      setMessages,
      setSelectedMessageId,
    ],
  );

  const resetSessionState = useCallback(() => {
    sessionIdRef.current = null;
    setActiveSessionId(null);
  }, []);

  const resetToInitialMessages = useCallback(() => {
    setMessages(initialMessages);
  }, [initialMessages, setMessages]);

  return {
    sessions,
    activeSessionId,
    setActiveSessionId,
    sessionIdRef,
    refreshSessions,
    loadSession,
    resetSessionState,
    resetToInitialMessages,
  };
}
