import { useState } from "react";
import ChatPane from "./components/ChatPane";
import LeftPane from "./components/LeftPane";
import RightPane from "./components/RightPane";
import { useChat } from "./hooks/useChat";
import type { ChatSettings } from "./types";

const defaultSettings: ChatSettings = {
  topK: 3,
  model: "google/gemini-2.0-flash-001",
  includeChunks: true,
};

function App() {
  const [query, setQuery] = useState("");
  const [settings, setSettings] = useState<ChatSettings>(defaultSettings);

  const {
    messages,
    contextDocs,
    isLoading,
    errorMessage,
    sendMessage,
    clearChat,
  } = useChat(settings);

  const handleSendQuery = async () => {
    const snapshot = query;
    setQuery("");
    await sendMessage(snapshot);
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      <div className="mx-auto grid min-h-screen max-w-400 grid-cols-1 lg:grid-cols-[280px_1fr_360px]">
        <LeftPane settings={settings} onSettingsChange={setSettings} />
        <ChatPane
          messages={messages}
          isLoading={isLoading}
          query={query}
          onQueryChange={setQuery}
          onSendQuery={handleSendQuery}
          onClearChat={clearChat}
          errorMessage={errorMessage}
        />
        <RightPane contextDocs={contextDocs} />
      </div>
    </div>
  );
}

export default App;
