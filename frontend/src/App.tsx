import { useEffect, useRef, useState } from "react";
import ChatPane from "./components/ChatPane";
import LeftPane from "./components/LeftPane";
import RightPane from "./components/RightPane";
import type { ChatMessage, ContextDoc, ChatSettings } from "./types";

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

type ContextApiResponse = {
  query: string;
  source_documents: string[];
};

const initialMessages: ChatMessage[] = [
  {
    id: "msg-welcome",
    role: "assistant",
    content:
      "Hi! I am your RAG assistant. Ask me about your indexed documents and I will answer with retrieved context.",
  },
];

const defaultSettings: ChatSettings = {
  topK: 3,
  model: "google/gemini-2.0-flash-001",
  includeChunks: true,
};

function createId(prefix: string): string {
  return `${prefix}-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

function App() {
  const [messages, setMessages] = useState<ChatMessage[]>(initialMessages);
  const [contextDocs, setContextDocs] = useState<ContextDoc[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [query, setQuery] = useState("");
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [settings, setSettings] = useState<ChatSettings>(defaultSettings);

  const eventSourceRef = useRef<EventSource | null>(null);

  useEffect(() => {
    return () => {
      eventSourceRef.current?.close();
      eventSourceRef.current = null;
    };
  }, []);

  const handleClearChat = () => {
    eventSourceRef.current?.close();
    eventSourceRef.current = null;
    setMessages(initialMessages);
    setContextDocs([]);
    setQuery("");
    setErrorMessage(null);
    setIsLoading(false);
  };

  const handleSendQuery = async () => {
    const trimmedQuery = query.trim();
    if (!trimmedQuery || isLoading) {
      return;
    }

    eventSourceRef.current?.close();
    eventSourceRef.current = null;

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
    setQuery("");

    try {
      const contextResponse = await fetch(`${API_BASE_URL}/context`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ 
          query: trimmedQuery,
          k: settings.topK 
        }),
      });

      if (!contextResponse.ok) {
        throw new Error(
          `Context request failed with status ${contextResponse.status}`,
        );
      }

      const contextData = (await contextResponse.json()) as ContextApiResponse;

      const nextContextDocs: ContextDoc[] = contextData.source_documents.map(
        (content, index) => ({
          id: createId(`context-${index}`),
          title: `Context ${index + 1}`,
          content,
        }),
      );

      setContextDocs(nextContextDocs);

      const streamUrl = `${API_BASE_URL}/chat/stream?query=${encodeURIComponent(trimmedQuery)}&k=${settings.topK}&model=${encodeURIComponent(settings.model)}`;
      const source = new EventSource(streamUrl);
      eventSourceRef.current = source;

      source.onmessage = (event) => {
        if (event.data === "[DONE]") {
          source.close();
          if (eventSourceRef.current === source) {
            eventSourceRef.current = null;
          }
          setIsLoading(false);
          return;
        }

        try {
          const payload = JSON.parse(event.data) as { token?: string };

          if (typeof payload.token !== "string") {
            throw new Error("Malformed stream payload");
          }

          setMessages((previous) =>
            previous.map((message) =>
              message.id === assistantMessageId
                ? { ...message, content: `${message.content}${payload.token}` }
                : message,
            ),
          );
        } catch {
          setErrorMessage(
            "Could not parse streamed response data from the backend.",
          );
          setIsLoading(false);
          source.close();
          if (eventSourceRef.current === source) {
            eventSourceRef.current = null;
          }
        }
      };

      source.onerror = () => {
        setErrorMessage("Streaming connection failed. Please try again.");
        setIsLoading(false);
        source.close();
        if (eventSourceRef.current === source) {
          eventSourceRef.current = null;
        }

        setMessages((previous) =>
          previous.map((message) => {
            if (message.id !== assistantMessageId) {
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
      };
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

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      <div className="mx-auto grid min-h-screen max-w-400 grid-cols-1 lg:grid-cols-[280px_1fr_360px]">
        <LeftPane settings={settings} onSettingsChange={setSettings} />
        <ChatPane
          messages={messages}
          isLoading={isLoading}
          query={query}
          onQueryChange={setQuery}
          onSendQuery={handleSendQuery}
          onClearChat={handleClearChat}
          errorMessage={errorMessage}
        />
        <RightPane contextDocs={contextDocs} />
      </div>
    </div>
  );
}

export default App;
