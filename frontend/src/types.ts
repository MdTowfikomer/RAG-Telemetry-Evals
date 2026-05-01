export interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
}

export interface ContextDoc {
  id: string
  title: string
  content: string
}

export interface HistoryItem {
  id: string
  title: string
  preview: string
  timestamp: string
}

export interface ChatSettings {
  topK: number
  model: string
  includeChunks: boolean
}
