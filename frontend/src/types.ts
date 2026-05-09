export interface MessageEvaluationVersion {
  id: string;
  messageId: string;
  version: number;
  status: "pending" | "completed" | "failed";
  faithfulness?: number;
  answerRelevancy?: number;
  reasoning?: string;
  errorMessage?: string;
  createdAt: string;
  updatedAt: string;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  latencyMs?: number;
  tokenCount?: number;
  faithfulness?: number;
  answerRelevancy?: number;
  reasoning?: string;
  evaluationStatus?: "pending" | "completed" | "failed";
  evaluationVersion?: number;
}

export interface ContextDoc {
  id: string;
  title: string;
  content: string;
}

export interface HistoryItem {
  id: string;
  title: string;
  preview: string;
  timestamp: string;
}

export interface SessionSummary {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
}

export interface SessionMessage {
  id: string;
  session_id: string;
  role: "user" | "assistant";
  content: string;
  latency_ms?: number | null;
  token_count?: number | null;
  faithfulness?: number | null;
  answer_relevancy?: number | null;
  reasoning?: string | null;
  evaluation_status?: "pending" | "completed" | "failed" | null;
  evaluation_version?: number | null;
  created_at: string;
}

export interface ChatSettings {
  topK: number;
  model: string;
  includeChunks: boolean;
}
