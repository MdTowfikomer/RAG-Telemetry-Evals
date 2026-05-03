import { act, renderHook } from '@testing-library/react'
import { api } from '../lib/api'
import { useChat } from './useChat'
import type { ChatSettings } from '../types'

vi.mock('../lib/api', () => ({
  api: {
    fetchContext: vi.fn(),
    streamChat: vi.fn(),
  },
}))

const settings: ChatSettings = {
  topK: 3,
  model: 'google/gemini-2.0-flash-001',
  includeChunks: true,
}

type StreamHandlers = {
  onToken: (token: string) => void
  onComplete: () => void
  onError: (error: Error) => void
}

describe('useChat', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('sends message, loads context, and accumulates streamed tokens', async () => {
    let handlers: StreamHandlers | null = null

    vi.mocked(api.fetchContext).mockResolvedValue([
      { id: 'ctx-1', title: 'Context 1', content: 'Doc one' },
    ])

    vi.mocked(api.streamChat).mockImplementation((_, __, onToken, onComplete, onError) => {
      handlers = { onToken, onComplete, onError }
      return vi.fn()
    })

    const { result } = renderHook(() => useChat(settings))

    await act(async () => {
      await result.current.sendMessage('What is RAG?')
    })

    expect(result.current.isLoading).toBe(true)
    expect(result.current.contextDocs).toHaveLength(1)
    expect(result.current.messages.at(-2)?.role).toBe('user')
    expect(result.current.messages.at(-2)?.content).toBe('What is RAG?')

    await act(async () => {
      handlers?.onToken('RAG ')
      handlers?.onToken('works')
    })

    expect(result.current.messages.at(-1)?.content).toBe('RAG works')

    await act(async () => {
      handlers?.onComplete()
    })

    expect(result.current.isLoading).toBe(false)
    expect(result.current.errorMessage).toBeNull()
  })

  it('sets error state when context fetch fails', async () => {
    vi.mocked(api.fetchContext).mockRejectedValue(new Error('Context failed'))
    vi.mocked(api.streamChat).mockImplementation(() => vi.fn())

    const { result } = renderHook(() => useChat(settings))

    await act(async () => {
      await result.current.sendMessage('Hello')
    })

    expect(result.current.isLoading).toBe(false)
    expect(result.current.errorMessage).toBe('Context failed')
    expect(result.current.messages.at(-1)?.content).toContain('could not start the response stream')
  })

  it('clears chat and stops active stream', async () => {
    const cleanup = vi.fn()

    vi.mocked(api.fetchContext).mockResolvedValue([])
    vi.mocked(api.streamChat).mockImplementation(() => cleanup)

    const { result } = renderHook(() => useChat(settings))

    await act(async () => {
      await result.current.sendMessage('clear me')
    })

    await act(async () => {
      result.current.clearChat()
    })

    expect(cleanup).toHaveBeenCalledTimes(1)
    expect(result.current.isLoading).toBe(false)
    expect(result.current.errorMessage).toBeNull()
    expect(result.current.contextDocs).toHaveLength(0)
    expect(result.current.messages).toHaveLength(1)
    expect(result.current.messages[0].role).toBe('assistant')
  })
})
