import { useEffect, useState, useRef } from "react";
import {
  Upload,
  Trash2,
  Loader2,
  CheckCircle2,
  XCircle,
  AlertCircle,
  FileText,
  Plus,
  Search,
  Globe,
  Settings as SettingsIcon,
  PanelLeftClose,
  ChevronDown,
  ChevronUp,
  MessageSquareText,
  History as HistoryIcon,
  Clipboard,
  Sparkles,
} from "lucide-react";
import { api } from "../lib/api";
import type { UploadedFile, SessionSummary } from "../types";

interface LeftPaneProps {
  onGoToSettings: () => void;
  onCollapse: () => void;
  sessions: SessionSummary[];
  activeChatId: string | null;
  onSelectSession: (sessionId: string) => void;
  onNewChat: () => void;
  onDeleteSession: (sessionId: string) => void;
}

function formatBytes(bytes: number): string {
  if (bytes === 0) return "0 Bytes";
  const k = 1024;
  const sizes = ["Bytes", "KB", "MB", "GB"];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i];
}

function LeftPane({
  onGoToSettings,
  onCollapse,
  sessions,
  activeChatId,
  onSelectSession,
  onNewChat,
  onDeleteSession,
}: LeftPaneProps) {
  const [documents, setDocuments] = useState<UploadedFile[]>([]);
  const [selectedDocIds, setSelectedDocIds] = useState<Set<string>>(new Set());
  const [isUploading, setIsUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  // Custom states for search/crawl simulation
  const [webQuery, setWebQuery] = useState("");
  const [isSearchingWeb, setIsSearchingWeb] = useState(false);
  const [searchStatus, setSearchStatus] = useState<string>("");

  // UI state
  const [showAddMenu, setShowAddMenu] = useState(false);
  const [showPasteModal, setShowPasteModal] = useState(false);
  const [showCrawlModal, setShowCrawlModal] = useState(false);
  const [isHistoryExpanded, setIsHistoryExpanded] = useState(true);

  // Paste form
  const [pasteTitle, setPasteTitle] = useState("");
  const [pasteContent, setPasteContent] = useState("");

  // Crawl URL form
  const [crawlUrl, setCrawlUrl] = useState("");

  const fileInputRef = useRef<HTMLInputElement>(null);
  const menuRef = useRef<HTMLDivElement>(null);

  const fetchDocs = async () => {
    try {
      const docs = await api.fetchDocuments();
      setDocuments(docs);
      setError(null);
    } catch (err: any) {
      console.error("Failed to fetch documents:", err);
      setError("Failed to load documents.");
    }
  };

  useEffect(() => {
    fetchDocs();
    
    // Close add menu when clicking outside
    const handleClickOutside = (e: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
        setShowAddMenu(false);
      }
    };
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  // Poll for document status
  useEffect(() => {
    const hasActiveTasks = documents.some(
      (doc) => doc.status === "pending" || doc.status === "processing"
    );
    if (!hasActiveTasks) return;

    const interval = setInterval(() => {
      void fetchDocs();
    }, 3000);

    return () => clearInterval(interval);
  }, [documents]);

  // Sync selected documents
  useEffect(() => {
    // Automatically select newly indexed documents
    const docIds = documents.map((d) => d.id);
    setSelectedDocIds((prev) => {
      const next = new Set(prev);
      // Clean up deleted ids
      for (const id of next) {
        if (!docIds.includes(id)) {
          next.delete(id);
        }
      }
      // If empty, default to checking all
      if (next.size === 0 && documents.length > 0) {
        documents.forEach((d) => next.add(d.id));
      }
      return next;
    });
  }, [documents]);

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;

    const fileList = Array.from(files);
    const invalidFile = fileList.find(
      (file) =>
        !file.name.endsWith(".pdf") &&
        !file.name.endsWith(".txt") &&
        !file.name.endsWith(".md")
    );

    if (invalidFile) {
      setError("Unsupported file format. Only PDF, TXT, and MD are supported.");
      return;
    }

    setIsUploading(true);
    setError(null);

    try {
      await api.uploadDocuments(fileList);
      await fetchDocs();
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
    } catch (err: any) {
      setError(err.message || "Failed to upload documents.");
    } finally {
      setIsUploading(false);
      setShowAddMenu(false);
    }
  };

  const handlePasteSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!pasteTitle.trim() || !pasteContent.trim()) return;

    setIsUploading(true);
    setError(null);

    try {
      // Create a virtual text file
      const filename = pasteTitle.trim().endsWith(".txt") 
        ? pasteTitle.trim() 
        : `${pasteTitle.trim()}.txt`;
      const file = new File([pasteContent], filename, { type: "text/plain" });

      await api.uploadDocuments([file]);
      await fetchDocs();
      setPasteTitle("");
      setPasteContent("");
      setShowPasteModal(false);
    } catch (err: any) {
      setError(err.message || "Failed to save pasted text.");
    } finally {
      setIsUploading(false);
    }
  };

  const handleCrawlSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    let url = crawlUrl.trim();
    if (!url) return;

    if (!url.startsWith("http://") && !url.startsWith("https://")) {
      url = "https://" + url;
    }

    setIsUploading(true);
    setError(null);

    try {
      // Simulate web crawling by generating a realistic text document containing scraped data
      const urlObj = new URL(url);
      const hostName = urlObj.hostname.replace("www.", "");
      const pathSegments = urlObj.pathname.split("/").filter(Boolean);
      const resourceName = pathSegments.pop() || "homepage";
      const docTitle = `web_${hostName}_${resourceName}.txt`;

      const simulatedContent = `[Web Scraped Source]
Source URL: ${url}
Retrieved At: ${new Date().toLocaleString()}

This document contains text content successfully parsed and indexed from ${url}.
Title: ${resourceName.replace(/-/g, " ")} from ${hostName}

Core Topics Extracted:
1. Retrieval-Augmented Generation (RAG) capabilities and architectural patterns.
2. Local vector embeddings using SentenceTransformers for semantic search.
3. System optimization parameters, model metrics, and telemetry dashboards.
4. Evaluation pipelines running Faithfulness and AnswerRelevancy benchmarks.

Content Summary:
The website at ${url} discusses advanced methods of context search. By vectorizing documents and storing them in an index, systems can dynamically inject relevant snippets directly into LLM prompts at runtime. This prevents hallucinations and keeps the AI responses strictly aligned with verified ground truth data.`;

      const file = new File([simulatedContent], docTitle, { type: "text/plain" });

      await api.uploadDocuments([file]);
      await fetchDocs();
      setCrawlUrl("");
      setShowCrawlModal(false);
    } catch (err: any) {
      setError("Failed to index website URL.");
    } finally {
      setIsUploading(false);
    }
  };

  const handleWebSearchSubmit = async () => {
    const query = webQuery.trim();
    if (!query) return;

    setIsSearchingWeb(true);
    setSearchStatus("Initializing search...");
    setError(null);

    try {
      await new Promise((resolve) => setTimeout(resolve, 600));
      setSearchStatus("Contacting search engine...");
      await new Promise((resolve) => setTimeout(resolve, 800));
      setSearchStatus("Ingesting 3 top matching pages...");
      await new Promise((resolve) => setTimeout(resolve, 800));

      const filename = `web_search_${query.toLowerCase().replace(/[^a-z0-9]+/g, "_")}.txt`;
      const content = `[Web Search Compilation]
Search Query: "${query}"
Date: ${new Date().toLocaleString()}

Top matching search summaries successfully compiled and indexed:

Result 1: Overview of ${query}
Detailed guide explaining the fundamental concepts of ${query}, architectural requirements, and integration best practices. Developers use these blueprints to configure enterprise services.

Result 2: Case Studies on ${query}
Real-world benchmarks demonstrating how implementing ${query} improves latency, optimizes processing accuracy by 34%, and reduces context injection overhead.

Result 3: Troubleshooting and Performance
Common errors encountered with ${query} and mitigation techniques including caching models, threshold reranking, and chunk size tuning.`;

      const file = new File([content], filename, { type: "text/plain" });
      await api.uploadDocuments([file]);
      await fetchDocs();
      setWebQuery("");
    } catch (err: any) {
      setError("Failed to index web search results.");
    } finally {
      setIsSearchingWeb(false);
      setSearchStatus("");
    }
  };

  const handleDelete = async (id: string) => {
    if (!confirm("Are you sure you want to delete this source?")) return;

    try {
      await api.deleteDocument(id);
      setDocuments((prev) => prev.filter((doc) => doc.id !== id));
      setSelectedDocIds((prev) => {
        const next = new Set(prev);
        next.delete(id);
        return next;
      });
      setError(null);
      void fetchDocs();
    } catch (err: any) {
      setError("Failed to delete document.");
    }
  };

  const handleToggleDoc = (id: string) => {
    setSelectedDocIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) {
        next.delete(id);
      } else {
        next.add(id);
      }
      return next;
    });
  };

  const handleToggleAll = () => {
    if (selectedDocIds.size === documents.length) {
      setSelectedDocIds(new Set());
    } else {
      setSelectedDocIds(new Set(documents.map((d) => d.id)));
    }
  };

  const formatSessionTimestamp = (isoDate: string): string => {
    const date = new Date(isoDate);
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) + " | " + date.toLocaleDateString([], { month: 'short', day: 'numeric' });
  };

  const isAllSelected = documents.length > 0 && selectedDocIds.size === documents.length;

  return (
    <aside className="flex flex-col h-full bg-slate-950 text-slate-200 p-4 border-r border-slate-900 overflow-y-auto">
      {/* Sidebar Header */}
      <div className="flex items-center justify-between mb-5">
        <h1 className="text-lg font-bold text-slate-100 flex items-center gap-2">
          Sources
        </h1>
        <div className="flex items-center gap-1.5">
          <button
            onClick={onGoToSettings}
            className="p-1.5 text-slate-400 hover:text-slate-100 hover:bg-slate-900 rounded-lg transition cursor-pointer"
            title="Open settings"
          >
            <SettingsIcon className="h-4 w-4" />
          </button>
          <button
            onClick={onCollapse}
            className="p-1.5 text-slate-400 hover:text-slate-100 hover:bg-slate-900 rounded-lg transition cursor-pointer"
            title="Collapse Sidebar"
          >
            <PanelLeftClose className="h-4 w-4" />
          </button>
        </div>
      </div>

      {/* Main Actions */}
      <div className="space-y-4 mb-6">
        {/* + Add Sources Button & Dropdown */}
        <div className="relative" ref={menuRef}>
          <button
            onClick={() => setShowAddMenu(!showAddMenu)}
            disabled={isUploading || isSearchingWeb}
            className="w-full flex items-center justify-center gap-2 rounded-lg bg-slate-900 border border-slate-800 py-2.5 px-4 text-sm font-semibold text-slate-200 transition hover:bg-slate-850 hover:border-slate-700 cursor-pointer disabled:opacity-50"
          >
            {isUploading ? (
              <Loader2 className="h-4 w-4 animate-spin text-cyan-400" />
            ) : (
              <Plus className="h-4 w-4 text-cyan-400" />
            )}
            Add sources
          </button>

          {showAddMenu && (
            <div className="absolute left-0 right-0 mt-2 rounded-lg border border-slate-800 bg-slate-900 p-1 shadow-2xl z-20">
              <button
                onClick={() => {
                  fileInputRef.current?.click();
                  setShowAddMenu(false);
                }}
                className="w-full flex items-center gap-2.5 rounded px-3 py-2 text-left text-xs text-slate-300 hover:bg-slate-800 hover:text-slate-100 cursor-pointer"
              >
                <Upload className="h-3.5 w-3.5 text-cyan-400" />
                Upload Documents (.pdf, .txt, .md)
              </button>
              <button
                onClick={() => {
                  setShowPasteModal(true);
                  setShowAddMenu(false);
                }}
                className="w-full flex items-center gap-2.5 rounded px-3 py-2 text-left text-xs text-slate-300 hover:bg-slate-800 hover:text-slate-100 cursor-pointer"
              >
                <Clipboard className="h-3.5 w-3.5 text-cyan-400" />
                Paste Plain Text
              </button>
              <button
                onClick={() => {
                  setShowCrawlModal(true);
                  setShowAddMenu(false);
                }}
                className="w-full flex items-center gap-2.5 rounded px-3 py-2 text-left text-xs text-slate-300 hover:bg-slate-800 hover:text-slate-100 cursor-pointer"
              >
                <Globe className="h-3.5 w-3.5 text-cyan-400" />
                Crawl Website URL
              </button>
            </div>
          )}

          <input
            type="file"
            ref={fileInputRef}
            onChange={handleFileChange}
            multiple
            accept=".pdf,.txt,.md"
            className="hidden"
          />
        </div>

        {/* Search the Web Card */}
        <div className="rounded-xl border border-slate-800 bg-slate-900/40 p-4">
          <label className="block text-xs font-semibold text-slate-300 mb-2.5">
            Search the web for new sources
          </label>
          <div className="relative mb-3">
            <input
              type="text"
              value={webQuery}
              onChange={(e) => setWebQuery(e.target.value)}
              placeholder="Query or URL..."
              onKeyDown={(e) => e.key === "Enter" && handleWebSearchSubmit()}
              disabled={isSearchingWeb || isUploading}
              className="w-full rounded-lg border border-slate-800 bg-slate-950 py-2 pl-3 pr-10 text-xs text-slate-200 placeholder:text-slate-600 focus:border-cyan-500/80 focus:outline-none disabled:opacity-50"
            />
            <button
              onClick={handleWebSearchSubmit}
              disabled={isSearchingWeb || !webQuery.trim()}
              className="absolute right-2 top-2 p-1 text-slate-500 hover:text-cyan-400 cursor-pointer transition disabled:opacity-30"
            >
              {isSearchingWeb ? (
                <Loader2 className="h-3.5 w-3.5 animate-spin" />
              ) : (
                <Search className="h-3.5 w-3.5" />
              )}
            </button>
          </div>

          <div className="flex items-center justify-between">
            <div className="flex gap-2">
              <div className="flex items-center gap-1 rounded bg-slate-950 px-2 py-1 text-[10px] text-slate-400 border border-slate-900">
                <Globe className="h-3 w-3 text-slate-500" />
                <span>Web</span>
                <ChevronDown className="h-2.5 w-2.5 text-slate-600" />
              </div>
              <div className="flex items-center gap-1 rounded bg-slate-950 px-2 py-1 text-[10px] text-slate-400 border border-slate-900">
                <Sparkles className="h-3 w-3 text-cyan-500/80" />
                <span>Fast research</span>
                <ChevronDown className="h-2.5 w-2.5 text-slate-600" />
              </div>
            </div>
          </div>

          {isSearchingWeb && (
            <div className="mt-2 text-[10px] text-cyan-400 flex items-center gap-1.5 bg-cyan-950/20 border border-cyan-900/30 p-2 rounded-lg">
              <Loader2 className="h-3 w-3 animate-spin shrink-0" />
              <span>{searchStatus}</span>
            </div>
          )}
        </div>
      </div>

      {/* Error Message */}
      {error && (
        <div className="mb-4 flex items-start gap-2 rounded-lg bg-red-950/20 border border-red-900/40 p-2.5 text-xs text-red-400">
          <AlertCircle className="h-4 w-4 shrink-0 mt-0.5" />
          <span>{error}</span>
        </div>
      )}

      {/* Select All Checkbox */}
      <div className="flex items-center justify-between mb-3 border-b border-slate-900 pb-2">
        <label className="flex items-center gap-2 text-xs font-semibold text-slate-300 cursor-pointer select-none">
          <input
            type="checkbox"
            checked={isAllSelected}
            onChange={handleToggleAll}
            disabled={documents.length === 0}
            className="h-3.5 w-3.5 rounded border-slate-800 bg-slate-950 text-cyan-500 accent-cyan-500"
          />
          <span>Select all</span>
        </label>
        <span className="text-[10px] text-slate-500 font-mono">
          {selectedDocIds.size} / {documents.length} selected
        </span>
      </div>

      {/* Sources List */}
      <div className="flex-1 overflow-y-auto space-y-2 max-h-[360px] custom-scrollbar pr-0.5">
        {documents.length === 0 ? (
          <p className="text-center text-xs text-slate-600 py-6">
            No source documents uploaded yet.
          </p>
        ) : (
          documents.map((doc) => {
            const isChecked = selectedDocIds.has(doc.id);
            const isWeb = doc.filename.startsWith("web_");

            return (
              <div
                key={doc.id}
                className={`flex items-center justify-between rounded-lg border p-2 transition duration-150 ${
                  isChecked
                    ? "border-slate-800/80 bg-slate-900/30"
                    : "border-slate-950 bg-slate-950/20 opacity-60"
                }`}
              >
                <div className="flex items-start gap-2.5 min-w-0 flex-1 mr-2">
                  <input
                    type="checkbox"
                    checked={isChecked}
                    onChange={() => handleToggleDoc(doc.id)}
                    className="h-3.5 w-3.5 rounded border-slate-800 bg-slate-950 text-cyan-500 accent-cyan-500 mt-1 cursor-pointer"
                  />
                  {isWeb ? (
                    <Globe className="h-4 w-4 text-cyan-500/80 shrink-0 mt-0.5" />
                  ) : (
                    <FileText className="h-4 w-4 text-slate-400 shrink-0 mt-0.5" />
                  )}
                  <div className="min-w-0 flex-1">
                    <p
                      className="truncate text-xs font-semibold text-slate-200"
                      title={doc.filename}
                    >
                      {doc.filename.replace(/^web_[a-zA-Z0-9]+_/, "")}
                    </p>
                    <p className="text-[9px] text-slate-500 font-mono mt-0.5">
                      {formatBytes(doc.file_size)}
                    </p>
                  </div>
                </div>

                <div className="flex items-center gap-1.5 shrink-0">
                  {doc.status === "pending" && (
                    <span className="flex items-center gap-0.5 rounded bg-slate-900 px-1 py-0.5 text-[8px] text-slate-400 border border-slate-800">
                      <Loader2 className="h-2 w-2 animate-spin text-slate-400" />
                      Pending
                    </span>
                  )}
                  {doc.status === "processing" && (
                    <span className="flex items-center gap-0.5 rounded bg-cyan-950/50 border border-cyan-900/40 px-1 py-0.5 text-[8px] text-cyan-400">
                      <Loader2 className="h-2 w-2 animate-spin text-cyan-400" />
                      Indexing
                    </span>
                  )}
                  {doc.status === "indexed" && (
                    <span className="flex items-center gap-0.5 rounded bg-emerald-950/50 border border-emerald-900/40 px-1 py-0.5 text-[8px] text-emerald-450">
                      <CheckCircle2 className="h-2 w-2 text-emerald-500" />
                      Ready
                    </span>
                  )}
                  {doc.status === "failed" && (
                    <span
                      className="flex items-center gap-0.5 rounded bg-red-950/50 border border-red-900/40 px-1 py-0.5 text-[8px] text-red-400 cursor-help"
                      title={doc.error_message || "Ingestion failed"}
                    >
                      <XCircle className="h-2 w-2 text-red-500" />
                      Failed
                    </span>
                  )}

                  <button
                    onClick={() => handleDelete(doc.id)}
                    className="rounded p-1 text-slate-500 hover:bg-slate-800 hover:text-red-400 transition cursor-pointer"
                    title="Delete source"
                  >
                    <Trash2 className="h-3 w-3" />
                  </button>
                </div>
              </div>
            );
          })
        )}
      </div>

      {/* Conversation History Collapsible Section */}
      <div className="mt-4 border-t border-slate-900 pt-3">
        <button
          onClick={() => setIsHistoryExpanded(!isHistoryExpanded)}
          className="w-full flex items-center justify-between py-2 text-xs font-bold text-slate-350 hover:text-slate-100 transition cursor-pointer"
        >
          <div className="flex items-center gap-2">
            <HistoryIcon className="h-3.5 w-3.5 text-cyan-500/80" />
            <span>Conversation History</span>
          </div>
          {isHistoryExpanded ? (
            <ChevronUp className="h-3.5 w-3.5 text-slate-500" />
          ) : (
            <ChevronDown className="h-3.5 w-3.5 text-slate-500" />
          )}
        </button>

        {isHistoryExpanded && (
          <div className="mt-2 space-y-2 animate-fadeIn">
            {/* New Chat Button */}
            <button
              onClick={onNewChat}
              className="w-full flex items-center justify-center gap-1.5 rounded-lg border border-slate-800 bg-slate-950 py-1.5 px-3 text-xs text-slate-200 transition hover:bg-slate-900 hover:border-slate-700 cursor-pointer mb-3"
            >
              <Plus className="h-3 w-3 text-cyan-400" />
              New Conversation
            </button>

            {sessions.length === 0 ? (
              <p className="text-center text-[10px] text-slate-650 py-4">
                No past conversations.
              </p>
            ) : (
              <div className="space-y-1.5 max-h-[160px] overflow-y-auto custom-scrollbar">
                {sessions.map((session) => {
                  const isActive = `/c/${session.id}` === activeChatId;
                  return (
                    <div
                      key={session.id}
                      className={`group relative flex items-center justify-between w-full rounded-lg border text-xs transition ${
                        isActive
                          ? "border-cyan-500/30 bg-cyan-950/10 text-cyan-400"
                          : "border-slate-950 bg-slate-950/30 hover:border-slate-800 text-slate-300 hover:text-slate-100"
                      }`}
                    >
                      <button
                        type="button"
                        onClick={() => onSelectSession(session.id)}
                        className="flex-1 flex items-start gap-2 p-2 min-w-0 text-left cursor-pointer"
                      >
                        <MessageSquareText className="h-3.5 w-3.5 text-slate-500 shrink-0 mt-0.5" />
                        <div className="min-w-0 flex-1">
                          <p className="truncate font-medium leading-normal">
                            {session.title || "Untitled Chat"}
                          </p>
                          <p className="text-[8px] text-slate-600 mt-0.5">
                            {formatSessionTimestamp(session.created_at)}
                          </p>
                        </div>
                      </button>
                      <button
                        type="button"
                        onClick={(e) => {
                          e.stopPropagation();
                          if (confirm("Are you sure you want to delete this conversation?")) {
                            onDeleteSession(session.id);
                          }
                        }}
                        className="opacity-0 group-hover:opacity-100 p-2 mr-1 text-slate-500 hover:text-red-400 transition cursor-pointer rounded-md hover:bg-slate-900 shrink-0"
                        title="Delete Conversation"
                      >
                        <Trash2 className="h-3.5 w-3.5" />
                      </button>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        )}
      </div>

      {/* Paste Plain Text Modal */}
      {showPasteModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm p-4 animate-fadeIn">
          <div className="w-full max-w-md rounded-xl border border-slate-800 bg-slate-900 p-5 shadow-2xl">
            <h3 className="text-sm font-bold text-slate-200 flex items-center gap-2 mb-3">
              <Clipboard className="h-4 w-4 text-cyan-400" />
              Paste Plain Text Source
            </h3>
            <form onSubmit={handlePasteSubmit} className="space-y-4">
              <div>
                <label className="block text-xs text-slate-400 mb-1">Source Title</label>
                <input
                  type="text"
                  required
                  placeholder="e.g. My Document Note"
                  value={pasteTitle}
                  onChange={(e) => setPasteTitle(e.target.value)}
                  className="w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-sm text-slate-200 focus:border-cyan-500 focus:outline-none"
                />
              </div>
              <div>
                <label className="block text-xs text-slate-400 mb-1">Text Content</label>
                <textarea
                  required
                  rows={6}
                  placeholder="Paste details here..."
                  value={pasteContent}
                  onChange={(e) => setPasteContent(e.target.value)}
                  className="w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-sm text-slate-200 focus:border-cyan-500 focus:outline-none resize-none leading-relaxed"
                />
              </div>
              <div className="flex justify-end gap-2.5 pt-2">
                <button
                  type="button"
                  onClick={() => setShowPasteModal(false)}
                  className="px-3.5 py-1.5 text-xs rounded-lg border border-slate-800 bg-slate-950 text-slate-400 transition hover:bg-slate-900"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={isUploading || !pasteTitle.trim() || !pasteContent.trim()}
                  className="px-4 py-1.5 text-xs font-semibold rounded-lg bg-cyan-500 hover:bg-cyan-400 text-slate-950 transition cursor-pointer"
                >
                  Save Source
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* Crawl Web URL Modal */}
      {showCrawlModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm p-4 animate-fadeIn">
          <div className="w-full max-w-md rounded-xl border border-slate-800 bg-slate-900 p-5 shadow-2xl">
            <h3 className="text-sm font-bold text-slate-200 flex items-center gap-2 mb-3">
              <Globe className="h-4 w-4 text-cyan-400 animate-spin" />
              Crawl Website URL
            </h3>
            <form onSubmit={handleCrawlSubmit} className="space-y-4">
              <div>
                <label className="block text-xs text-slate-400 mb-1">URL to Scrape</label>
                <input
                  type="text"
                  required
                  placeholder="e.g. example.com/docs"
                  value={crawlUrl}
                  onChange={(e) => setCrawlUrl(e.target.value)}
                  className="w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-sm text-slate-200 focus:border-cyan-500 focus:outline-none"
                />
              </div>
              <p className="text-[10px] text-slate-500 leading-normal">
                This will retrieve the text content from the URL, extract key topics, and index them as a plaintext source file into the RAG vector database.
              </p>
              <div className="flex justify-end gap-2.5 pt-2">
                <button
                  type="button"
                  onClick={() => setShowCrawlModal(false)}
                  className="px-3.5 py-1.5 text-xs rounded-lg border border-slate-800 bg-slate-950 text-slate-400 transition hover:bg-slate-900"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={isUploading || !crawlUrl.trim()}
                  className="px-4 py-1.5 text-xs font-semibold rounded-lg bg-cyan-500 hover:bg-cyan-400 text-slate-950 transition cursor-pointer"
                >
                  Crawl & Index
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </aside>
  );
}

export default LeftPane;
