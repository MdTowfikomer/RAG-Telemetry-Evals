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
  const sessionIdRef = useRef<string | null>(null);
  const loadSessionRequestIdRef = useRef(0);

  const refreshSessions = useCallback(async () => {
    setSessions(await api.fetchSessions());
  }, []);

  useEffect(() => {
    void refreshSessions();
  }, [refreshSessions]);

  const loadSession = useCallback(
    async (sessionId: string) => {
      const requestId = ++loadSessionRequestIdRef.current;
      sessionIdRef.current = sessionId;
      setIsLoading(false);
      setErrorMessage(null);

      try {
        const sessionMessages = await api.fetchSessionMessages(sessionId);
        if (loadSessionRequestIdRef.current !== requestId) {
          return;
        }
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

        setMessages(mappedMessages);
        setContextDocs([]);
        setSelectedMessageId(null);
        setEvaluationHistory([]);
      } catch (err: any) {
        if (loadSessionRequestIdRef.current !== requestId) {
          return;
        }
        console.error("Failed to load session:", err);
        setErrorMessage("Failed to load session history.");
        setMessages([]);
        setContextDocs([]);
        setSelectedMessageId(null);
        setEvaluationHistory([]);
      }
    },
    [setContextDocs, setErrorMessage, setEvaluationHistory, setIsLoading, setMessages, setSelectedMessageId],
  );

  const resetSessionState = useCallback(() => {
    loadSessionRequestIdRef.current += 1;
    sessionIdRef.current = null;
  }, []);

  const resetToInitialMessages = useCallback(() => {
    setMessages(initialMessages);
  }, [initialMessages, setMessages]);

  const deleteSession = useCallback(
    async (sessionId: string) => {
      try {
        await api.deleteSession(sessionId);
        setSessions((prev) => prev.filter((s) => s.id !== sessionId));
        if (sessionIdRef.current === sessionId) {
          resetToInitialMessages();
          resetSessionState();
        }
      } catch (error) {
        console.error("Failed to delete session:", error);
        setErrorMessage("Failed to delete session.");
      }
    },
    [resetSessionState, resetToInitialMessages, setErrorMessage]
  );

  return {
    sessions,
    sessionIdRef,
    refreshSessions,
    loadSession,
    resetSessionState,
    resetToInitialMessages,
    deleteSession,
  };
}
