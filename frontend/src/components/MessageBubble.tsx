import { Bot, User } from 'lucide-react'
import type { ChatMessage } from '../types'

interface MessageBubbleProps {
  message: ChatMessage
}

function MessageBubble({ message }: MessageBubbleProps) {
  const isUser = message.role === 'user'

  return (
    <div className={`flex gap-3 ${isUser ? 'justify-end' : 'justify-start'}`}>
      {!isUser && (
        <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-cyan-500/15 text-cyan-300">
          <Bot className="h-4 w-4" />
        </div>
      )}

      <div
        className={`max-w-[85%] rounded-2xl px-4 py-3 text-sm leading-relaxed ${
          isUser
            ? 'bg-cyan-500 text-slate-950'
            : 'border border-slate-800 bg-slate-900 text-slate-100'
        }`}
      >
        {message.content || '...'}
      </div>

      {isUser && (
        <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-slate-800 text-slate-200">
          <User className="h-4 w-4" />
        </div>
      )}
    </div>
  )
}

export default MessageBubble
