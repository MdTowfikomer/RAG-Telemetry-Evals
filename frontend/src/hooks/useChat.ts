import { useEffect, useRef, useState } from "react";
import { api } from "../lib/api";
import type {
  ChatMessage,
  ChatSettings,
  ContextDoc,
  MessageEvaluationVersion,
  SessionSummary,
} from "../types";

const initialMessages: ChatMessage[] = [
  {
    id: "msg-welcome",
    role: "assistant",
    content:
      "Hi! I am your RAG assistant. Ask me about your indexed documents and I will answer with retrieved context.",
  },
];

function createId(prefix: string): string {
  return `${prefix}-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

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

export function useChat(settings: ChatSettings) {
  const [messages, setMessages] = useState<ChatMessage[]>(initialMessages);
  const [contextDocs, setContextDocs] = useState<ContextDoc[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);
  const [selectedMessageId, setSelectedMessageId] = useState<string | null>(
    null,
  );
  const [evaluationHistory, setEvaluationHistory] = useState<
    MessageEvaluationVersion[]
  >([]);

  const streamCleanupRef = useRef<(() => void) | null>(null);
  const scoresCleanupRef = useRef<(() => void) | null>(null);
  const sessionIdRef = useRef<string | null>(null);
  const assistantIdAliasRef = useRef<Record<string, string>>({});
  const selectedMessageIdRef = useRef<string | null>(null);

  const loadEvaluationHistory = async (messageId: string) => {
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
  };

  useEffect(() => {
    selectedMessageIdRef.current = selectedMessageId;
  }, [selectedMessageId]);

  const refreshSessions = async () => {
    const nextSessions = await api.fetchSessions();
    setSessions(nextSessions);
  };

  useEffect(() => {
    void refreshSessions();

    scoresCleanupRef.current = api.streamScores(
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
      streamCleanupRef.current?.();
      streamCleanupRef.current = null;

      scoresCleanupRef.current?.();
      scoresCleanupRef.current = null;
    };
  }, []);

  const clearChat = () => {
    streamCleanupRef.current?.();
    streamCleanupRef.current = null;
    setMessages(initialMessages);
    sessionIdRef.current = null;
    assistantIdAliasRef.current = {};
    setActiveSessionId(null);
    setContextDocs([]);
    setSelectedMessageId(null);
    setEvaluationHistory([]);
    setErrorMessage(null);
    setIsLoading(false);
  };

  const loadSession = async (sessionId: string) => {
    streamCleanupRef.current?.();
    streamCleanupRef.current = null;

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
  };

  const selectAssistantMessage = async (messageId: string) => {
    setSelectedMessageId(messageId);
    await loadEvaluationHistory(messageId);
  };

  const reevaluateAssistantMessage = async (messageId: string) => {
    if (!isUuid(messageId)) {
      return;
    }

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
  };

  const sendMessage = async (queryText: string) => {
    const trimmedQuery = queryText.trim();
    if (!trimmedQuery || isLoading) {
      return;
    }

    streamCleanupRef.current?.();
    streamCleanupRef.current = null;

    const userMessage: ChatMessage = {
      id: createId("msg-user"),
      role: "user",
      content: trimmedQuery,
    };

    const assistantMessageId = createId("msg-assistant");
    const assistantPlaceholder: ChatMessage = {
      id: assistantMessageId,
      role: "assistant",
      content: "",
    };

    setMessages((previous) => [...previous, userMessage, assistantPlaceholder]);
    setErrorMessage(null);
    setIsLoading(true);

    try {
      const nextContextDocs = await api.fetchContext(trimmedQuery, settings);
      setContextDocs(nextContextDocs);

      let activeUserMessageId = userMessage.id;
      let activeAssistantMessageId = assistantMessageId;

      const cleanup = api.streamChat(
        trimmedQuery,
        settings,
        (token) => {
          setMessages((previous) =>
            previous.map((message) =>
              message.id === activeAssistantMessageId ||
              message.id === assistantMessageId
                ? { ...message, content: `${message.content}${token}` }
                : message,
            ),
          );
        },
        () => {
          setIsLoading(false);
          if (streamCleanupRef.current === cleanup) {
            streamCleanupRef.current = null;
          }
        },
        (error) => {
          setErrorMessage(error.message);
          setIsLoading(false);

          setMessages((previous) =>
            previous.map((message) => {
              if (
                message.id !== activeAssistantMessageId &&
                message.id !== assistantMessageId
              ) {
                return message;
              }

              if (message.content.trim().length > 0) {
                return message;
              }

              return {
                ...message,
                content:
                  "I could not stream a response this time. Please retry.",
              };
            }),
          );

          if (streamCleanupRef.current === cleanup) {
            streamCleanupRef.current = null;
          }
        },
        (meta) => {
          sessionIdRef.current = meta.session_id;
          setActiveSessionId(meta.session_id);
          assistantIdAliasRef.current[meta.assistant_message_id] =
            activeAssistantMessageId;

          setMessages((previous) =>
            previous.map((message) => {
              if (message.id === activeUserMessageId) {
                return { ...message, id: meta.user_message_id };
              }

              if (message.id === activeAssistantMessageId) {
                return { ...message, id: meta.assistant_message_id };
              }

              return message;
            }),
          );

          if (
            selectedMessageIdRef.current === activeAssistantMessageId ||
            selectedMessageIdRef.current === assistantMessageId
          ) {
            setSelectedMessageId(meta.assistant_message_id);
          }

          activeUserMessageId = meta.user_message_id;
          activeAssistantMessageId = meta.assistant_message_id;
        },
        sessionIdRef.current ?? undefined,
      );

      streamCleanupRef.current = cleanup;
      await refreshSessions();
    } catch (error) {
      const message =
        error instanceof Error
          ? error.message
          : "Something went wrong while sending your query.";

      setErrorMessage(message);
      setIsLoading(false);

      setMessages((previous) =>
        previous.map((message) =>
          message.id === assistantMessageId
            ? {
                ...message,
                content:
                  "I could not start the response stream. Please try again.",
              }
            : message,
        ),
      );
    }
  };

  return {
    messages,
    contextDocs,
    isLoading,
    errorMessage,
    sessions,
    activeSessionId,
    selectedMessageId,
    evaluationHistory,
    sendMessage,
    clearChat,
    loadSession,
    refreshSessions,
    selectAssistantMessage,
    reevaluateAssistantMessage,
  };
}
