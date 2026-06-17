import { useState, useEffect } from "react";
import ChatPane from "./components/ChatPane";
import LeftPane from "./components/LeftPane";
import RightPane from "./components/RightPane";
import Settings from "./components/Settings";
import { useChat } from "./hooks/useChat";
import type { ChatSettings } from "./types";

const defaultSettings: ChatSettings = {
  topK: 3,
  model: "openrouter/free",
  includeChunks: true,
  apiKey: "",
  temperature: 0.7,
  systemPrompt: "Answer the following question based ONLY on the provided context. If the answer is not in the context, say that you don't know.",
};

function App() {
  const [query, setQuery] = useState("");
  const [showSettingsModal, setShowSettingsModal] = useState(() => {
    return window.location.hash === "#settings";
  });

  useEffect(() => {
    const handleHashChange = () => {
      setShowSettingsModal(window.location.hash === "#settings");
    };
    window.addEventListener("hashchange", handleHashChange);
    return () => {
      window.removeEventListener("hashchange", handleHashChange);
    };
  }, []);

  const handleOpenSettings = () => {
    window.location.hash = "settings";
  };

  const handleCloseSettings = () => {
    if (window.location.hash === "#settings") {
      history.pushState("", document.title, window.location.pathname + window.location.search);
    }
    setShowSettingsModal(false);
  };
  const [isLeftCollapsed, setIsLeftCollapsed] = useState(() => {
    return localStorage.getItem("sidebar_left_collapsed") === "true";
  });
  const [isRightCollapsed, setIsRightCollapsed] = useState(() => {
    return localStorage.getItem("sidebar_right_collapsed") === "true";
  });

  const [settings, setSettings] = useState<ChatSettings>(() => {
    const saved = localStorage.getItem("rag_workbench_settings");
    if (saved) {
      try {
        return JSON.parse(saved);
      } catch (e) {
        console.error("Failed to parse settings:", e);
      }
    }
    return defaultSettings;
  });

  const handleSettingsChange = (newSettings: ChatSettings) => {
    setSettings(newSettings);
    localStorage.setItem("rag_workbench_settings", JSON.stringify(newSettings));
  };

  const handleToggleLeft = () => {
    const nextVal = !isLeftCollapsed;
    setIsLeftCollapsed(nextVal);
    localStorage.setItem("sidebar_left_collapsed", String(nextVal));
  };

  const handleToggleRight = () => {
    const nextVal = !isRightCollapsed;
    setIsRightCollapsed(nextVal);
    localStorage.setItem("sidebar_right_collapsed", String(nextVal));
  };

  const {
    messages,
    contextDocs,
    isLoading,
    errorMessage,
    sessions,
    activeSessionId,
    selectedMessageId,
    evaluationHistory,
    sendMessage,
    clearChat,
    loadSession,
    deleteSession,
    selectAssistantMessage,
    reevaluateAssistantMessage,
  } = useChat(settings);

  const [currentPath, setCurrentPath] = useState(() => window.location.pathname);

  useEffect(() => {
    const handlePopState = () => {
      setCurrentPath(window.location.pathname);
    };
    window.addEventListener("popstate", handlePopState);
    return () => {
      window.removeEventListener("popstate", handlePopState);
    };
  }, []);

  // When activeSessionId changes, update url
  useEffect(() => {
    if (activeSessionId) {
      const expectedPath = `/c/${activeSessionId}`;
      if (window.location.pathname !== expectedPath) {
        window.history.pushState(null, "", expectedPath);
        setCurrentPath(expectedPath);
      }
    } else {
      if (window.location.pathname !== "/") {
        window.history.pushState(null, "", "/");
        setCurrentPath("/");
      }
    }
  }, [activeSessionId]);

  // When url path changes, load appropriate session
  useEffect(() => {
    const match = currentPath.match(/^\/c\/([a-fA-F0-9-]+)$/);
    if (match) {
      const sessionIdFromUrl = match[1];
      if (sessionIdFromUrl !== activeSessionId) {
        void loadSession(sessionIdFromUrl);
      }
    } else if (currentPath === "/" || currentPath === "") {
      clearChat();
    }
  }, [currentPath, loadSession, clearChat, activeSessionId]);

  const handleSendQuery = async () => {
    const snapshot = query;
    setQuery("");
    await sendMessage(snapshot);
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex justify-center overflow-hidden">
      <div className="w-full max-w-[1600px] min-h-screen flex relative">
        {/* Left Sidebar */}
        <div
          className={`transition-all duration-300 ease-in-out border-r border-slate-900 bg-slate-950 shrink-0 ${
            isLeftCollapsed ? "w-0 opacity-0 pointer-events-none" : "w-[280px] opacity-100"
          } overflow-hidden flex flex-col h-screen`}
        >
          <LeftPane
            onGoToSettings={handleOpenSettings}
            onCollapse={handleToggleLeft}
            sessions={sessions}
            activeSessionId={activeSessionId}
            onSelectSession={(sessionId) => {
              void loadSession(sessionId);
            }}
            onNewChat={clearChat}
            onDeleteSession={deleteSession}
          />
        </div>

        {/* Center Panel */}
        <div className="flex-1 flex flex-col min-w-0 h-screen overflow-hidden">
          <ChatPane
            messages={messages}
            isLoading={isLoading}
            query={query}
            onQueryChange={setQuery}
            onSendQuery={handleSendQuery}
            onClearChat={clearChat}
            onSelectAssistantMessage={(messageId) => {
              void selectAssistantMessage(messageId);
            }}
            onReevaluateAssistantMessage={(messageId) => {
              void reevaluateAssistantMessage(messageId);
            }}
            selectedMessageId={selectedMessageId}
            errorMessage={errorMessage}
            isLeftCollapsed={isLeftCollapsed}
            isRightCollapsed={isRightCollapsed}
            onToggleLeft={handleToggleLeft}
            onToggleRight={handleToggleRight}
            onGoToSettings={handleOpenSettings}
          />
        </div>

        {/* Right Sidebar */}
        <div
          className={`transition-all duration-300 ease-in-out bg-slate-950 shrink-0 ${
            isRightCollapsed ? "w-0 opacity-0 pointer-events-none" : "w-[360px] opacity-100"
          } overflow-hidden flex flex-col h-screen`}
        >
          <RightPane
            contextDocs={contextDocs}
            selectedMessage={
              messages.find((message) => message.id === selectedMessageId) ?? null
            }
            evaluationHistory={evaluationHistory}
            onCollapse={handleToggleRight}
          />
        </div>
      </div>

      {showSettingsModal && (
        <Settings
          settings={settings}
          onSettingsChange={handleSettingsChange}
          onClose={handleCloseSettings}
        />
      )}
    </div>
  );
}

export default App;
