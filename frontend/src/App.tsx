import { useState, useEffect, useRef, useCallback } from "react";
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

  const [currentPath, setCurrentPath] = useState(() => window.location.pathname);
  const navigateToPath = useCallback((path: string) => {
    if (window.location.pathname !== path) {
      window.history.pushState(null, "", path);
    }
    setCurrentPath(path);
  }, []);

  const {
    messages,
    contextDocs,
    isLoading,
    errorMessage,
    sessions,
    sessionIdRef,
    selectedMessageId,
    evaluationHistory,
    sendMessage,
    clearChat,
    loadSession,
    deleteSession,
    selectAssistantMessage,
    reevaluateAssistantMessage,
  } = useChat(settings, navigateToPath);

  // Track whether the first URL-based init has been handled
  const didInitRef = useRef(false);
  // Stable refs so the routing effect never re-runs just because a callback changed reference
  const loadSessionRef = useRef(loadSession);
  const clearChatRef = useRef(clearChat);
  useEffect(() => { loadSessionRef.current = loadSession; });
  useEffect(() => { clearChatRef.current = clearChat; });

  useEffect(() => {
    const handlePopState = () => {
      setCurrentPath(window.location.pathname);
    };
    window.addEventListener("popstate", handlePopState);
    return () => {
      window.removeEventListener("popstate", handlePopState);
    };
  }, []);

  // When url path changes (popstate / programmatic nav), load the appropriate session.
  // Deps are ONLY primitives — no callback refs — to prevent re-running on every render.
  useEffect(() => {
    const match = currentPath.match(/^\/c\/([a-fA-F0-9-]+)$/);
    if (match) {
      const sessionIdFromUrl = match[1];
      didInitRef.current = true;
      if (sessionIdFromUrl !== sessionIdRef.current) {
        void loadSessionRef.current(sessionIdFromUrl);
      }
    } else if (currentPath === "/" || currentPath === "") {
      if (!didInitRef.current) {
        // First render at '/' — no session to load, just mark init done
        didInitRef.current = true;
      } else if (sessionIdRef.current !== null) {
        // Only call clearChat when there IS an active session to clear;
        // prevents infinite loop when the active chat is already null
        clearChatRef.current();
      }
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [currentPath]);

  const activeChatPath = currentPath.startsWith("/c/") ? currentPath : null;

  const handleNewChat = () => {
    if (window.location.pathname === "/") {
      clearChat();
      return;
    }

    navigateToPath("/");
  };

  const handleSendQuery = async () => {
    const snapshot = query;
    setQuery("");
    await sendMessage(snapshot);
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex justify-center overflow-hidden">
      <div className="w-full max-w-400 min-h-screen flex relative">
        {/* Left Sidebar */}
        <div
          className={`transition-all duration-300 ease-in-out border-r border-slate-900 bg-slate-950 shrink-0 ${
            isLeftCollapsed ? "w-0 opacity-0 pointer-events-none" : "w-70 opacity-100"
          } overflow-hidden flex flex-col h-screen`}
        >
          <LeftPane
            onGoToSettings={handleOpenSettings}
            onCollapse={handleToggleLeft}
            sessions={sessions}
            activeChatId={activeChatPath}
            onSelectSession={(sessionId) => {
              navigateToPath(`/c/${sessionId}`);
            }}
            onNewChat={handleNewChat}
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
            onClearChat={handleNewChat}
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
            isRightCollapsed ? "w-0 opacity-0 pointer-events-none" : "w-90 opacity-100"
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
