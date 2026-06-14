import { useEffect, useRef, type SubmitEvent } from "react";
import { LoaderCircle, Send, Trash2, PanelLeftOpen, PanelRightOpen, SlidersHorizontal } from "lucide-react";
import MessageBubble from "./MessageBubble";
import type { ChatMessage } from "../types";

interface ChatPaneProps {
  messages: ChatMessage[];
  isLoading: boolean;
  query: string;
  onQueryChange: (value: string) => void;
  onSendQuery: () => Promise<void>;
  onClearChat: () => void;
  onSelectAssistantMessage: (messageId: string) => void;
  onReevaluateAssistantMessage: (messageId: string) => void;
  selectedMessageId: string | null;
  errorMessage: string | null;
  isLeftCollapsed: boolean;
  isRightCollapsed: boolean;
  onToggleLeft: () => void;
  onToggleRight: () => void;
  onGoToSettings: () => void;
}

function ChatPane({
  messages,
  isLoading,
  query,
  onQueryChange,
  onSendQuery,
  onClearChat,
  onSelectAssistantMessage,
  onReevaluateAssistantMessage,
  selectedMessageId,
  errorMessage,
  isLeftCollapsed,
  isRightCollapsed,
  onToggleLeft,
  onToggleRight,
  onGoToSettings,
}: ChatPaneProps) {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (containerRef.current) {
      containerRef.current.scrollTop = containerRef.current.scrollHeight;
    }
  }, [messages]);

  const handleSubmit = (event: SubmitEvent<HTMLFormElement>) => {
    event.preventDefault();
    void onSendQuery();
  };

  return (
    <main className="flex flex-1 flex-col border-b border-slate-800 p-4 lg:border-b-0 lg:border-r bg-slate-950 min-h-0">
      <div className="mb-4 flex items-center justify-between gap-3 border-b border-slate-900 pb-2.5 shrink-0">
        <div className="flex items-center gap-2">
          {isLeftCollapsed && (
            <button
              type="button"
              onClick={onToggleLeft}
              className="p-1.5 text-slate-400 hover:text-slate-100 hover:bg-slate-900 rounded-lg transition cursor-pointer"
              title="Expand Sources"
            >
              <PanelLeftOpen className="h-4 w-4" />
            </button>
          )}
          <h1 className="text-sm font-bold text-slate-100">AI Assistant</h1>
        </div>
        
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={onGoToSettings}
            className="p-1.5 text-slate-400 hover:text-slate-100 hover:bg-slate-900 rounded-lg transition cursor-pointer"
            title="Open Settings"
          >
            <SlidersHorizontal className="h-4 w-4" />
          </button>
          
          <span className="rounded-full border border-slate-850 bg-slate-900/60 px-2.5 py-0.5 text-[10px] text-slate-450 font-medium select-none">
            {isLoading ? "Generating..." : "Ready"}
          </span>
          
          <button
            type="button"
            onClick={onClearChat}
            className="inline-flex items-center gap-1 rounded-md border border-slate-800 bg-slate-900 px-2 py-1 text-xs text-slate-300 transition hover:border-slate-600 hover:bg-slate-800 cursor-pointer"
            disabled={isLoading && messages.length <= 1}
          >
            <Trash2 className="h-3.5 w-3.5" />
            Clear
          </button>

          {isRightCollapsed && (
            <button
              type="button"
              onClick={onToggleRight}
              className="p-1.5 text-slate-400 hover:text-slate-100 hover:bg-slate-900 rounded-lg transition cursor-pointer ml-1"
              title="Expand Details"
            >
              <PanelRightOpen className="h-4 w-4" />
            </button>
          )}
        </div>
      </div>

      {errorMessage && (
        <div className="mb-3 rounded-lg border border-red-500/40 bg-red-500/10 px-3 py-2 text-sm text-red-200">
          {errorMessage}
        </div>
      )}

      {isLoading && (
        <div className="mb-3 inline-flex items-center gap-2 rounded-lg border border-cyan-500/30 bg-cyan-500/10 px-3 py-2 text-xs text-cyan-200">
          <LoaderCircle className="h-4 w-4 animate-spin" />
          Generating response...
        </div>
      )}

      <div
        ref={containerRef}
        className="flex-1 space-y-4 overflow-y-auto rounded-xl border border-slate-800 bg-slate-950/60 p-4"
      >
        {messages.map((message) => (
          <div
            key={message.id}
            onClick={() => {
              if (message.role === "assistant") {
                onSelectAssistantMessage(message.id);
              }
            }}
            className={
              message.role === "assistant"
                ? `cursor-pointer rounded-lg transition ${
                    selectedMessageId === message.id
                      ? "ring-1 ring-cyan-500/60"
                      : "hover:ring-1 hover:ring-slate-700"
                  }`
                : ""
            }
          >
            {message.role === "assistant" && !message.id.startsWith("msg-") && (
              <div className="mb-1 flex justify-end px-2">
                <button
                  type="button"
                  className="rounded-md border border-slate-700 bg-slate-900 px-2 py-1 text-xs text-slate-200 transition hover:border-cyan-500 hover:text-cyan-200"
                  onClick={(event) => {
                    event.stopPropagation();
                    onReevaluateAssistantMessage(message.id);
                  }}
                >
                  Re-evaluate
                </button>
              </div>
            )}
            <MessageBubble message={message} />
          </div>
        ))}
      </div>

      <form onSubmit={handleSubmit} className="mt-4 flex gap-2">
        <input
          type="text"
          value={query}
          onChange={(event) => onQueryChange(event.target.value)}
          placeholder="Ask a question..."
          className="flex-1 rounded-lg border border-slate-700 bg-slate-900 px-3 py-2 text-sm text-slate-100 placeholder:text-slate-500 focus:border-cyan-500 focus:outline-none"
        />
        <button
          type="submit"
          className="inline-flex items-center gap-2 rounded-lg bg-cyan-500 px-4 py-2 text-sm font-medium text-slate-950 transition hover:bg-cyan-400 disabled:cursor-not-allowed disabled:bg-slate-700 disabled:text-slate-300"
          disabled={isLoading || query.trim().length === 0}
        >
          <Send className="h-4 w-4" />
          Send
        </button>
      </form>
    </main>
  );
}

export default ChatPane;
