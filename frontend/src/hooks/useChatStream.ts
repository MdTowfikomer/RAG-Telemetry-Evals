import {
  useCallback,
  useRef,
  type Dispatch,
  type MutableRefObject,
  type SetStateAction,
} from "react";
import { api } from "../lib/api";
import type { ChatMessage, ChatSettings, ContextDoc } from "../types";

function createId(prefix: string): string {
  return `${prefix}-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

type UseChatStreamArgs = {
  settings: ChatSettings;
  isLoading: boolean;
  setMessages: Dispatch<SetStateAction<ChatMessage[]>>;
  setContextDocs: Dispatch<SetStateAction<ContextDoc[]>>;
  setIsLoading: Dispatch<SetStateAction<boolean>>;
  setErrorMessage: Dispatch<SetStateAction<string | null>>;
  sessionIdRef: MutableRefObject<string | null>;
  setActiveSessionId: Dispatch<SetStateAction<string | null>>;
  selectedMessageIdRef: MutableRefObject<string | null>;
  setSelectedMessageId: Dispatch<SetStateAction<string | null>>;
  refreshSessions: () => Promise<void>;
};

export function useChatStream({
  settings,
  isLoading,
  setMessages,
  setContextDocs,
  setIsLoading,
  setErrorMessage,
  sessionIdRef,
  setActiveSessionId,
  selectedMessageIdRef,
  setSelectedMessageId,
  refreshSessions,
}: UseChatStreamArgs) {
  const streamCleanupRef = useRef<(() => void) | null>(null);
  const assistantIdAliasRef = useRef<Record<string, string>>({});

  const stopActiveStream = useCallback(() => {
    streamCleanupRef.current?.();
    streamCleanupRef.current = null;
  }, []);

  const resetStreamState = useCallback(() => {
    assistantIdAliasRef.current = {};
  }, []);

  const sendMessage = useCallback(
    async (queryText: string) => {
      const trimmedQuery = queryText.trim();
      if (!trimmedQuery || isLoading) {
        return;
      }

      stopActiveStream();

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

      setMessages((prev) => [...prev, userMessage, assistantPlaceholder]);
      setErrorMessage(null);
      setIsLoading(true);

      try {
        setContextDocs(await api.fetchContext(trimmedQuery, settings));
        let activeUserMessageId = userMessage.id;
        let activeAssistantMessageId = assistantMessageId;

        const cleanup = api.streamChat(
          trimmedQuery,
          settings,
          (token) => {
            setMessages((prev) =>
              prev.map((message) =>
                message.id === activeAssistantMessageId ||
                message.id === assistantMessageId
                  ? { ...message, content: `${message.content}${token}` }
                  : message,
              ),
            );
          },
          () => {
            setIsLoading(false);
            if (streamCleanupRef.current === cleanup) streamCleanupRef.current = null;
          },
          (error) => {
            setErrorMessage(error.message);
            setIsLoading(false);
            setMessages((prev) =>
              prev.map((message) => {
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
                  content: "I could not stream a response this time. Please retry.",
                };
              }),
            );
            if (streamCleanupRef.current === cleanup) streamCleanupRef.current = null;
          },
          (meta) => {
            sessionIdRef.current = meta.session_id;
            setActiveSessionId(meta.session_id);
            assistantIdAliasRef.current[meta.assistant_message_id] =
              activeAssistantMessageId;
            setMessages((prev) =>
              prev.map((message) => {
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
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantMessageId
              ? {
                  ...m,
                  content: "I could not start the response stream. Please try again.",
                }
              : m,
          ),
        );
      }
    },
    [
      isLoading,
      refreshSessions,
      selectedMessageIdRef,
      sessionIdRef,
      setActiveSessionId,
      setContextDocs,
      setErrorMessage,
      setIsLoading,
      setMessages,
      setSelectedMessageId,
      settings,
      stopActiveStream,
    ],
  );

  return { sendMessage, stopActiveStream, resetStreamState, assistantIdAliasRef };
}
