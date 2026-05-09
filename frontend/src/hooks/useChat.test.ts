import { act, renderHook } from "@testing-library/react";
import { api } from "../lib/api";
import { useChat } from "./useChat";
import type { ChatSettings } from "../types";

vi.mock("../lib/api", () => ({
  api: {
    fetchContext: vi.fn(),
    fetchSessions: vi.fn(),
    fetchSessionMessages: vi.fn(),
    fetchMessageEvaluations: vi.fn(),
    streamScores: vi.fn(),
    streamChat: vi.fn(),
  },
}));

const settings: ChatSettings = {
  topK: 3,
  model: "google/gemini-2.0-flash-001",
  includeChunks: true,
};

type StreamHandlers = {
  onToken: (token: string) => void;
  onComplete: () => void;
  onError: (error: Error) => void;
};

describe("useChat", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(api.fetchSessions).mockResolvedValue([]);
    vi.mocked(api.streamScores).mockImplementation(() => vi.fn());
  });

  it("sends message, loads context, and accumulates streamed tokens", async () => {
    let handlers: StreamHandlers | null = null;

    vi.mocked(api.fetchContext).mockResolvedValue([
      { id: "ctx-1", title: "Context 1", content: "Doc one" },
    ]);

    vi.mocked(api.streamChat).mockImplementation(
      (_, __, onToken, onComplete, onError, onMeta) => {
        handlers = { onToken, onComplete, onError };
        onMeta?.({
          session_id: "session-1",
          user_message_id: "user-1",
          assistant_message_id: "assistant-1",
        });
        return vi.fn();
      },
    );

    const { result } = renderHook(() => useChat(settings));

    await act(async () => {
      await result.current.sendMessage("What is RAG?");
    });

    expect(result.current.isLoading).toBe(true);
    expect(result.current.contextDocs).toHaveLength(1);
    expect(result.current.messages.at(-2)?.role).toBe("user");
    expect(result.current.messages.at(-2)?.content).toBe("What is RAG?");

    await act(async () => {
      handlers?.onToken("RAG ");
      handlers?.onToken("works");
    });

    expect(result.current.messages.at(-1)?.content).toBe("RAG works");

    await act(async () => {
      handlers?.onComplete();
    });

    expect(result.current.isLoading).toBe(false);
    expect(result.current.errorMessage).toBeNull();
  });

  it("sets error state when context fetch fails", async () => {
    vi.mocked(api.fetchContext).mockRejectedValue(new Error("Context failed"));
    vi.mocked(api.streamChat).mockImplementation(() => vi.fn());

    const { result } = renderHook(() => useChat(settings));

    await act(async () => {
      await result.current.sendMessage("Hello");
    });

    expect(result.current.isLoading).toBe(false);
    expect(result.current.errorMessage).toBe("Context failed");
    expect(result.current.messages.at(-1)?.content).toContain(
      "could not start the response stream",
    );
  });

  it("loads a selected session history", async () => {
    vi.mocked(api.fetchSessionMessages).mockResolvedValue([
      {
        id: "user-1",
        session_id: "session-1",
        role: "user",
        content: "What is RAG?",
        created_at: "2026-01-01T00:00:00Z",
      },
      {
        id: "assistant-1",
        session_id: "session-1",
        role: "assistant",
        content: "RAG combines retrieval and generation.",
        created_at: "2026-01-01T00:00:01Z",
      },
    ]);

    const { result } = renderHook(() => useChat(settings));

    await act(async () => {
      await result.current.loadSession("session-1");
    });

    expect(result.current.messages).toHaveLength(2);
    expect(result.current.messages[0].content).toBe("What is RAG?");
    expect(result.current.messages[1].content).toContain("combines retrieval");
    expect(result.current.activeSessionId).toBe("session-1");
  });

  it("applies score updates to the matching assistant message", async () => {
    let emitScore:
      | ((event: {
          type: "score";
          message_id: string;
          faithfulness?: number;
          answer_relevancy?: number;
          status?: "pending" | "completed" | "failed";
          version?: number;
          reasoning?: string;
          latency_ms?: number;
          token_count?: number;
        }) => void)
      | null = null;

    vi.mocked(api.fetchContext).mockResolvedValue([]);

    vi.mocked(api.streamScores).mockImplementation((onScore) => {
      emitScore = onScore;
      return vi.fn();
    });

    vi.mocked(api.streamChat).mockImplementation(
      (_, __, onToken, _onComplete, _onError, onMeta) => {
        onMeta?.({
          session_id: "session-1",
          user_message_id: "user-1",
          assistant_message_id: "assistant-1",
        });
        onToken("hello");
        return vi.fn();
      },
    );

    const { result } = renderHook(() => useChat(settings));

    await act(async () => {
      await result.current.sendMessage("score me");
    });

    await act(async () => {
      emitScore?.({
        type: "score",
        message_id: "assistant-1",
        faithfulness: 0.92,
        answer_relevancy: 0.89,
        status: "completed",
        version: 2,
        reasoning: "The answer aligns with retrieved chunks.",
        latency_ms: 150,
        token_count: 12,
      });
    });

    const assistant = [...result.current.messages]
      .reverse()
      .find((message) => message.role === "assistant");
    expect(assistant?.faithfulness).toBe(0.92);
    expect(assistant?.answerRelevancy).toBe(0.89);
    expect(assistant?.evaluationStatus).toBe("completed");
    expect(assistant?.evaluationVersion).toBe(2);
    expect(assistant?.reasoning).toContain("aligns");
    expect(assistant?.latencyMs).toBe(150);
    expect(assistant?.tokenCount).toBe(12);
  });

  it("loads evaluation version history for selected assistant message", async () => {
    vi.mocked(api.fetchMessageEvaluations).mockResolvedValue([
      {
        id: "eval-2",
        message_id: "assistant-1",
        version: 2,
        status: "completed",
        faithfulness: 0.93,
        answer_relevancy: 0.9,
        reasoning: "Latest reasoning",
        error_message: null,
        created_at: "2026-01-01T00:00:01Z",
        updated_at: "2026-01-01T00:00:01Z",
      },
      {
        id: "eval-1",
        message_id: "assistant-1",
        version: 1,
        status: "failed",
        faithfulness: null,
        answer_relevancy: null,
        reasoning: null,
        error_message: "Judge timeout",
        created_at: "2026-01-01T00:00:00Z",
        updated_at: "2026-01-01T00:00:00Z",
      },
    ]);

    const { result } = renderHook(() => useChat(settings));

    await act(async () => {
      await result.current.selectAssistantMessage("assistant-1");
    });

    expect(result.current.selectedMessageId).toBe("assistant-1");
    expect(result.current.evaluationHistory).toHaveLength(2);
    expect(result.current.evaluationHistory[0].version).toBe(2);
    expect(result.current.evaluationHistory[1].status).toBe("failed");
  });

  it("clears chat and stops active stream", async () => {
    const cleanup = vi.fn();

    vi.mocked(api.fetchContext).mockResolvedValue([]);
    vi.mocked(api.streamChat).mockImplementation(() => cleanup);

    const { result } = renderHook(() => useChat(settings));

    await act(async () => {
      await result.current.sendMessage("clear me");
    });

    await act(async () => {
      result.current.clearChat();
    });

    expect(cleanup).toHaveBeenCalledTimes(1);
    expect(result.current.isLoading).toBe(false);
    expect(result.current.errorMessage).toBeNull();
    expect(result.current.contextDocs).toHaveLength(0);
    expect(result.current.messages).toHaveLength(1);
    expect(result.current.messages[0].role).toBe("assistant");
  });
});
