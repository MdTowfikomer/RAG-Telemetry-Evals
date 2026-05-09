import { Bot, User } from "lucide-react";
import type { ChatMessage } from "../types";

interface MessageBubbleProps {
  message: ChatMessage;
}

function formatScore(score: number | undefined): string {
  if (typeof score !== "number") {
    return "--";
  }

  return `${Math.round(score * 100)}%`;
}

function MessageBubble({ message }: MessageBubbleProps) {
  const isUser = message.role === "user";

  return (
    <div className={`flex gap-3 ${isUser ? "justify-end" : "justify-start"}`}>
      {!isUser && (
        <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-cyan-500/15 text-cyan-300">
          <Bot className="h-4 w-4" />
        </div>
      )}

      <div
        className={`max-w-[85%] rounded-2xl px-4 py-3 text-sm leading-relaxed ${
          isUser
            ? "bg-cyan-500 text-slate-950"
            : "border border-slate-800 bg-slate-900 text-slate-100"
        }`}
      >
        {message.content || "..."}

        {!isUser && (
          <div className="mt-3 flex flex-wrap gap-2 text-[11px]">
            <span className="rounded-full border border-slate-700 bg-slate-950 px-2 py-1 text-slate-300">
              Faithfulness: {formatScore(message.faithfulness)}
            </span>
            <span className="rounded-full border border-slate-700 bg-slate-950 px-2 py-1 text-slate-300">
              Relevancy: {formatScore(message.answerRelevancy)}
            </span>
            <span className="rounded-full border border-slate-700 bg-slate-950 px-2 py-1 text-slate-300">
              Version: {message.evaluationVersion ?? "--"}
            </span>
            {typeof message.latencyMs === "number" && (
              <span className="rounded-full border border-cyan-500/40 bg-cyan-500/10 px-2 py-1 text-cyan-200">
                {message.latencyMs} ms
              </span>
            )}
            {typeof message.tokenCount === "number" && (
              <span className="rounded-full border border-slate-700 bg-slate-950 px-2 py-1 text-slate-300">
                Tokens: {message.tokenCount}
              </span>
            )}
          </div>
        )}
      </div>

      {isUser && (
        <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-slate-800 text-slate-200">
          <User className="h-4 w-4" />
        </div>
      )}
    </div>
  );
}

export default MessageBubble;
