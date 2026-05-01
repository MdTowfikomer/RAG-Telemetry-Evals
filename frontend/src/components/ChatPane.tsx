import type { SubmitEvent } from "react";
import { LoaderCircle, Send, Trash2 } from "lucide-react";
import MessageBubble from "./MessageBubble";
import type { ChatMessage } from "../types";

interface ChatPaneProps {
  messages: ChatMessage[];
  isLoading: boolean;
  query: string;
  onQueryChange: (value: string) => void;
  onSendQuery: () => Promise<void>;
  onClearChat: () => void;
  errorMessage: string | null;
}

function ChatPane({
  messages,
  isLoading,
  query,
  onQueryChange,
  onSendQuery,
  onClearChat,
  errorMessage,
}: ChatPaneProps) {
  const handleSubmit = (event: SubmitEvent<HTMLFormElement>) => {
    event.preventDefault();
    void onSendQuery();
  };

  return (
    <main className="flex min-h-[60vh] flex-col border-b border-slate-800 p-4 lg:border-b-0 lg:border-r">
      <div className="mb-4 flex items-center justify-between gap-3">
        <h1 className="text-base font-semibold text-slate-100">Chat</h1>
        <div className="flex items-center gap-2">
          <span className="rounded-full border border-slate-700 bg-slate-900 px-2 py-1 text-xs text-slate-400">
            {isLoading ? "Loading..." : "Ready"}
          </span>
          <button
            type="button"
            onClick={onClearChat}
            className="inline-flex items-center gap-1 rounded-md border border-slate-700 bg-slate-900 px-2 py-1 text-xs text-slate-200 transition hover:border-slate-500 hover:bg-slate-800"
            disabled={isLoading && messages.length <= 2}
          >
            <Trash2 className="h-3.5 w-3.5" />
            Clear
          </button>
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

      <div className="flex-1 space-y-4 overflow-y-auto rounded-xl border border-slate-800 bg-slate-950/60 p-4">
        {messages.map((message) => (
          <MessageBubble key={message.id} message={message} />
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
