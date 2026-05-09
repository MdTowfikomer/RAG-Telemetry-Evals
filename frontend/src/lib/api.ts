import type {
  ChatSettings,
  ContextDoc,
  SessionMessage,
  SessionSummary,
} from "../types";

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

type ContextApiResponse = {
  query: string;
  source_documents: string[];
};

function createId(prefix: string): string {
  return `${prefix}-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

type StreamMeta = {
  session_id: string;
  user_message_id: string;
  assistant_message_id: string;
};

type ScoreEvent = {
  type: "score";
  message_id: string;
  faithfulness?: number | null;
  answer_relevancy?: number | null;
  reasoning?: string | null;
  status?: "pending" | "completed" | "failed";
  version?: number | null;
  latency_ms?: number | null;
  token_count?: number | null;
};

export type MessageEvaluationVersion = {
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
};

export const api = {
  async fetchSessions(): Promise<SessionSummary[]> {
    const response = await fetch(`${API_BASE_URL}/sessions`);

    if (!response.ok) {
      throw new Error(`Sessions request failed with status ${response.status}`);
    }

    return (await response.json()) as SessionSummary[];
  },

  async fetchSessionMessages(sessionId: string): Promise<SessionMessage[]> {
    const response = await fetch(
      `${API_BASE_URL}/sessions/${encodeURIComponent(sessionId)}/messages`,
    );

    if (!response.ok) {
      throw new Error(
        `Session messages request failed with status ${response.status}`,
      );
    }

    return (await response.json()) as SessionMessage[];
  },

  async fetchMessageEvaluations(
    messageId: string,
  ): Promise<MessageEvaluationVersion[]> {
    const response = await fetch(
      `${API_BASE_URL}/messages/${encodeURIComponent(messageId)}/evaluations`,
    );

    if (!response.ok) {
      throw new Error(
        `Message evaluations request failed with status ${response.status}`,
      );
    }

    return (await response.json()) as MessageEvaluationVersion[];
  },

  async fetchContext(
    query: string,
    settings: ChatSettings,
  ): Promise<ContextDoc[]> {
    const contextResponse = await fetch(`${API_BASE_URL}/context`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        query,
        k: settings.topK,
        model: settings.model,
      }),
    });

    if (!contextResponse.ok) {
      throw new Error(
        `Context request failed with status ${contextResponse.status}`,
      );
    }

    const contextData = (await contextResponse.json()) as ContextApiResponse;

    return contextData.source_documents.map((content, index) => ({
      id: createId(`context-${index}`),
      title: `Context ${index + 1}`,
      content,
    }));
  },

  streamScores(
    onScore: (event: ScoreEvent) => void,
    onError: (error: Error) => void,
  ): () => void {
    const source = new EventSource(`${API_BASE_URL}/scores/stream`);

    source.onmessage = (event) => {
      try {
        const payload = JSON.parse(event.data) as ScoreEvent;

        if (
          payload.type !== "score" ||
          typeof payload.message_id !== "string"
        ) {
          throw new Error("Malformed score payload");
        }

        onScore(payload);
      } catch {
        onError(
          new Error("Could not parse score stream data from the backend."),
        );
        source.close();
      }
    };

    source.onerror = () => {
      onError(new Error("Score stream connection failed. Please try again."));
      source.close();
    };

    return () => source.close();
  },

  streamChat(
    query: string,
    settings: ChatSettings,
    onToken: (token: string) => void,
    onComplete: () => void,
    onError: (error: Error) => void,
    onMeta?: (meta: StreamMeta) => void,
    sessionId?: string,
  ): () => void {
    const sessionParam = sessionId
      ? `&session_id=${encodeURIComponent(sessionId)}`
      : "";
    const streamUrl = `${API_BASE_URL}/chat/stream?query=${encodeURIComponent(query)}&k=${settings.topK}&model=${encodeURIComponent(settings.model)}${sessionParam}`;
    const source = new EventSource(streamUrl);

    source.onmessage = (event) => {
      if (event.data === "[DONE]") {
        onComplete();
        source.close();
        return;
      }

      try {
        const payload = JSON.parse(event.data) as {
          type?: string;
          token?: string;
          session_id?: string;
          user_message_id?: string;
          assistant_message_id?: string;
        };

        if (payload.type === "meta") {
          if (
            typeof payload.session_id !== "string" ||
            typeof payload.user_message_id !== "string" ||
            typeof payload.assistant_message_id !== "string"
          ) {
            throw new Error("Malformed stream metadata payload");
          }

          onMeta?.({
            session_id: payload.session_id,
            user_message_id: payload.user_message_id,
            assistant_message_id: payload.assistant_message_id,
          });
          return;
        }

        if (typeof payload.token !== "string") {
          throw new Error("Malformed stream payload");
        }

        onToken(payload.token);
      } catch {
        onError(
          new Error("Could not parse streamed response data from the backend."),
        );
        source.close();
      }
    };

    source.onerror = () => {
      onError(new Error("Streaming connection failed. Please try again."));
      source.close();
    };

    return () => source.close();
  },
};
