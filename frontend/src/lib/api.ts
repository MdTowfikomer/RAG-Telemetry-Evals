import type { ChatSettings, ContextDoc } from '../types'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000'

type ContextApiResponse = {
  query: string
  source_documents: string[]
}

function createId(prefix: string): string {
  return `${prefix}-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`
}

export const api = {
  async fetchContext(query: string, settings: ChatSettings): Promise<ContextDoc[]> {
    const contextResponse = await fetch(`${API_BASE_URL}/context`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        query,
        k: settings.topK,
        model: settings.model,
      }),
    })

    if (!contextResponse.ok) {
      throw new Error(`Context request failed with status ${contextResponse.status}`)
    }

    const contextData = (await contextResponse.json()) as ContextApiResponse

    return contextData.source_documents.map((content, index) => ({
      id: createId(`context-${index}`),
      title: `Context ${index + 1}`,
      content,
    }))
  },

  streamChat(
    query: string,
    settings: ChatSettings,
    onToken: (token: string) => void,
    onComplete: () => void,
    onError: (error: Error) => void,
  ): () => void {
    const streamUrl = `${API_BASE_URL}/chat/stream?query=${encodeURIComponent(query)}&k=${settings.topK}&model=${encodeURIComponent(settings.model)}`
    const source = new EventSource(streamUrl)

    source.onmessage = (event) => {
      if (event.data === '[DONE]') {
        onComplete()
        source.close()
        return
      }

      try {
        const payload = JSON.parse(event.data) as { token?: string }

        if (typeof payload.token !== 'string') {
          throw new Error('Malformed stream payload')
        }

        onToken(payload.token)
      } catch {
        onError(new Error('Could not parse streamed response data from the backend.'))
        source.close()
      }
    }

    source.onerror = () => {
      onError(new Error('Streaming connection failed. Please try again.'))
      source.close()
    }

    return () => source.close()
  },
}
