export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
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
  created_at: string;
}

export interface ChatSettings {
  topK: number;
  model: string;
  includeChunks: boolean;
}
