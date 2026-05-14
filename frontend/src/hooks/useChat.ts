import { useCallback, useState } from "react";
import { useChatStream } from "./useChatStream";
import { useEvaluations } from "./useEvaluations";
import { useScoreSubscription } from "./useScoreSubscription";
import { useSessions } from "./useSessions";
import type { ChatMessage, ChatSettings, ContextDoc } from "../types";

const initialMessages: ChatMessage[] = [
  {
    id: "msg-welcome",
    role: "assistant",
    content:
      "Hi! I am your RAG assistant. Ask me about your indexed documents and I will answer with retrieved context.",
  },
];

export function useChat(settings: ChatSettings) {
  const [messages, setMessages] = useState<ChatMessage[]>(initialMessages);
  const [contextDocs, setContextDocs] = useState<ContextDoc[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const evaluations = useEvaluations({ settings, setMessages, setErrorMessage });
  const sessions = useSessions({
    initialMessages,
    setMessages,
    setContextDocs,
    setIsLoading,
    setErrorMessage,
    setSelectedMessageId: evaluations.setSelectedMessageId,
    setEvaluationHistory: evaluations.setEvaluationHistory,
  });
  const stream = useChatStream({
    settings,
    isLoading,
    setMessages,
    setContextDocs,
    setIsLoading,
    setErrorMessage,
    sessionIdRef: sessions.sessionIdRef,
    setActiveSessionId: sessions.setActiveSessionId,
    selectedMessageIdRef: evaluations.selectedMessageIdRef,
    setSelectedMessageId: evaluations.setSelectedMessageId,
    refreshSessions: sessions.refreshSessions,
  });

  useScoreSubscription({
    setMessages,
    assistantIdAliasRef: stream.assistantIdAliasRef,
    selectedMessageIdRef: evaluations.selectedMessageIdRef,
    setSelectedMessageId: evaluations.setSelectedMessageId,
    loadEvaluationHistory: evaluations.loadEvaluationHistory,
  });

  const clearChat = useCallback(() => {
    stream.stopActiveStream();
    sessions.resetToInitialMessages();
    sessions.resetSessionState();
    stream.resetStreamState();
    setContextDocs([]);
    evaluations.resetEvaluations();
    setErrorMessage(null);
    setIsLoading(false);
  }, [evaluations, sessions, stream]);

  const loadSession = useCallback(
    async (sessionId: string) => {
      stream.stopActiveStream();
      await sessions.loadSession(sessionId);
    },
    [sessions, stream],
  );

  return {
    messages,
    contextDocs,
    isLoading,
    errorMessage,
    sessions: sessions.sessions,
    activeSessionId: sessions.activeSessionId,
    selectedMessageId: evaluations.selectedMessageId,
    evaluationHistory: evaluations.evaluationHistory,
    sendMessage: stream.sendMessage,
    clearChat,
    loadSession,
    refreshSessions: sessions.refreshSessions,
    selectAssistantMessage: evaluations.selectAssistantMessage,
    reevaluateAssistantMessage: evaluations.reevaluateAssistantMessage,
  };
}
