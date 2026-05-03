import { api } from './api'
import type { ChatSettings } from '../types'

const settings: ChatSettings = {
  topK: 3,
  model: 'google/gemini-2.0-flash-001',
  includeChunks: true,
}

class MockEventSource {
  static latest: MockEventSource | null = null

  url: string
  onmessage: ((event: MessageEvent) => void) | null = null
  onerror: ((event: Event) => void) | null = null
  closed = false

  constructor(url: string) {
    this.url = url
    MockEventSource.latest = this
  }

  close() {
    this.closed = true
  }
}

describe('api adapter', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    MockEventSource.latest = null
    vi.stubGlobal('EventSource', MockEventSource)
  })

  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('fetchContext maps backend response into ContextDoc items', async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        query: 'q',
        source_documents: ['Doc A', 'Doc B'],
      }),
    })
    vi.stubGlobal('fetch', fetchMock)

    const docs = await api.fetchContext('q', settings)

    expect(fetchMock).toHaveBeenCalledTimes(1)
    expect(docs).toHaveLength(2)
    expect(docs[0].title).toBe('Context 1')
    expect(docs[0].content).toBe('Doc A')
    expect(docs[1].title).toBe('Context 2')
  })

  it('fetchContext throws on non-ok response', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: false, status: 500 }))

    await expect(api.fetchContext('q', settings)).rejects.toThrow(
      'Context request failed with status 500',
    )
  })

  it('streamChat emits tokens, completes on [DONE], and supports cleanup', () => {
    const onToken = vi.fn()
    const onComplete = vi.fn()
    const onError = vi.fn()

    const cleanup = api.streamChat('query', settings, onToken, onComplete, onError)
    const source = MockEventSource.latest

    expect(source).not.toBeNull()
    expect(source?.url).toContain('/chat/stream?query=query')

    source?.onmessage?.({ data: JSON.stringify({ token: 'hello ' }) } as MessageEvent)
    source?.onmessage?.({ data: JSON.stringify({ token: 'world' }) } as MessageEvent)

    expect(onToken).toHaveBeenNthCalledWith(1, 'hello ')
    expect(onToken).toHaveBeenNthCalledWith(2, 'world')

    source?.onmessage?.({ data: '[DONE]' } as MessageEvent)
    expect(onComplete).toHaveBeenCalledTimes(1)
    expect(source?.closed).toBe(true)

    cleanup()
    expect(source?.closed).toBe(true)
  })

  it('streamChat reports parse and transport errors', () => {
    const onToken = vi.fn()
    const onComplete = vi.fn()
    const onError = vi.fn()

    api.streamChat('query', settings, onToken, onComplete, onError)
    const source = MockEventSource.latest

    source?.onmessage?.({ data: '{bad-json' } as MessageEvent)
    expect(onError).toHaveBeenCalledTimes(1)
    expect(source?.closed).toBe(true)

    const source2Cleanup = api.streamChat('query', settings, onToken, onComplete, onError)
    const source2 = MockEventSource.latest
    source2?.onerror?.({} as Event)

    expect(onError).toHaveBeenCalledTimes(2)
    expect(source2?.closed).toBe(true)

    source2Cleanup()
  })
})
