import History from "./History";
import Settings from "./Settings";
import type { ChatSettings, HistoryItem, SessionSummary } from "../types";

interface LeftPaneProps {
  settings: ChatSettings;
  onSettingsChange: (settings: ChatSettings) => void;
  sessions: SessionSummary[];
  activeSessionId: string | null;
  onSelectSession: (sessionId: string) => void;
  onNewChat: () => void;
}

function formatSessionTimestamp(isoDate: string): string {
  const date = new Date(isoDate);
  return date.toLocaleString();
}

function toHistoryItem(session: SessionSummary): HistoryItem {
  return {
    id: session.id,
    title: session.title,
    preview: "Open this session to view full conversation history.",
    timestamp: formatSessionTimestamp(session.created_at),
  };
}

function LeftPane({
  settings,
  onSettingsChange,
  sessions,
  activeSessionId,
  onSelectSession,
  onNewChat,
}: LeftPaneProps) {
  const historyItems = sessions.map(toHistoryItem);

  return (
    <aside className="border-b border-slate-800 p-4 lg:border-b-0 lg:border-r">
      <h1 className="mb-4 text-base font-semibold text-slate-100">Controls</h1>
      <div className="space-y-4">
        <Settings settings={settings} onSettingsChange={onSettingsChange} />
        <History
          items={historyItems}
          activeSessionId={activeSessionId}
          onSelectSession={onSelectSession}
          onNewChat={onNewChat}
        />
      </div>
    </aside>
  );
}

export default LeftPane;
