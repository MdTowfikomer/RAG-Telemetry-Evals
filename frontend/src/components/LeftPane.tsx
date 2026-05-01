import History from "./History";
import Settings from "./Settings";
import type { HistoryItem, ChatSettings } from "../types";

const historyItems: HistoryItem[] = [
  {
    id: "history-1",
    title: "What is retrieval-augmented generation?",
    preview: "A quick primer on combining retrieval with generation.",
    timestamp: "Today",
  },
  {
    id: "history-2",
    title: "Explain vector similarity search",
    preview: "How embeddings and nearest-neighbor lookup work in RAG.",
    timestamp: "Yesterday",
  }
];

interface LeftPaneProps {
  settings: ChatSettings;
  onSettingsChange: (settings: ChatSettings) => void;
}

function LeftPane({ settings, onSettingsChange }: LeftPaneProps) {
  return (
    <aside className="border-b border-slate-800 p-4 lg:border-b-0 lg:border-r">
      <h1 className="mb-4 text-base font-semibold text-slate-100">Controls</h1>
      <div className="space-y-4">
        <Settings settings={settings} onSettingsChange={onSettingsChange} />
        <History items={historyItems} />
      </div>
    </aside>
  );
}

export default LeftPane;
