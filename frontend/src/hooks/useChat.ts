import { useEffect, useRef, useState } from "react";
import { api } from "../lib/api";
import type { ChatMessage, ChatSettings, ContextDoc } from "../types";

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

export function useChat(settings: ChatSettings) {
  const [messages, setMessages] = useState<ChatMessage[]>(initialMessages);
  const [contextDocs, setContextDocs] = useState<ContextDoc[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const streamCleanupRef = useRef<(() => void) | null>(null);
  const sessionIdRef = useRef<string | null>(null);

  useEffect(() => {
    return () => {
      streamCleanupRef.current?.();
      streamCleanupRef.current = null;
    };
  }, []);

  const clearChat = () => {
    streamCleanupRef.current?.();
    streamCleanupRef.current = null;
    setMessages(initialMessages);
    sessionIdRef.current = null;
    setContextDocs([]);
    setErrorMessage(null);
    setIsLoading(false);
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
              message.id === activeAssistantMessageId
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
              if (message.id !== activeAssistantMessageId) {
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

          activeUserMessageId = meta.user_message_id;
          activeAssistantMessageId = meta.assistant_message_id;
        },
        sessionIdRef.current ?? undefined,
      );

      streamCleanupRef.current = cleanup;
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
    sendMessage,
    clearChat,
  };
}
