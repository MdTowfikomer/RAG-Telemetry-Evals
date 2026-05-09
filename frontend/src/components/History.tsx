import { History as HistoryIcon, MessageSquareText, Plus } from "lucide-react";
import type { HistoryItem } from "../types";

interface HistoryProps {
  items: HistoryItem[];
  activeSessionId: string | null;
  onSelectSession: (sessionId: string) => void;
  onNewChat: () => void;
}

function History({
  items,
  activeSessionId,
  onSelectSession,
  onNewChat,
}: HistoryProps) {
  return (
    <section className="rounded-xl border border-slate-800 bg-slate-900/60 p-4">
      <div className="mb-4 flex items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <HistoryIcon className="h-4 w-4 text-cyan-400" />
          <h2 className="text-sm font-semibold text-slate-100">History</h2>
        </div>

        <button
          type="button"
          onClick={onNewChat}
          className="inline-flex items-center gap-1 rounded-md border border-slate-700 bg-slate-900 px-2 py-1 text-xs text-slate-200 transition hover:border-slate-500 hover:bg-slate-800"
        >
          <Plus className="h-3.5 w-3.5" />
          New Chat
        </button>
      </div>

      {items.length === 0 ? (
        <p className="text-xs text-slate-500">No saved sessions yet.</p>
      ) : (
        <ul className="space-y-3">
          {items.map((item) => {
            const isActive = item.id === activeSessionId;

            return (
              <li key={item.id}>
                <button
                  type="button"
                  onClick={() => onSelectSession(item.id)}
                  className={`w-full rounded-lg border p-3 text-left transition ${
                    isActive
                      ? "border-cyan-500/70 bg-cyan-500/10"
                      : "border-slate-800 bg-slate-950/70 hover:border-slate-600"
                  }`}
                >
                  <div className="mb-1 flex items-start gap-2">
                    <MessageSquareText className="mt-0.5 h-4 w-4 shrink-0 text-slate-400" />
                    <p className="text-sm font-medium text-slate-200">
                      {item.title}
                    </p>
                  </div>
                  <p className="line-clamp-2 text-xs text-slate-400">
                    {item.preview}
                  </p>
                  <p className="mt-2 text-[11px] uppercase tracking-wide text-slate-500">
                    {item.timestamp}
                  </p>
                </button>
              </li>
            );
          })}
        </ul>
      )}
    </section>
  );
}

export default History;
