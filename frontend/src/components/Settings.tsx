import { useState, useEffect } from "react";
import {
  Eye,
  EyeOff,
  CheckCircle2,
  XCircle,
  Loader2,
  X,
} from "lucide-react";
import type { ChatSettings } from "../types";

interface SettingsProps {
  settings: ChatSettings;
  onSettingsChange: (settings: ChatSettings) => void;
  onClose: () => void;
}

type TabType = "api" | "model" | "instructions";

const PREDEFINED_MODELS = [
  { label: "Gemini 2.0 Flash (Recommended)", value: "google/gemini-2.0-flash-001" },
  { label: "GPT-4o Mini", value: "openai/gpt-4o-mini" },
  { label: "Claude 3.5 Haiku", value: "anthropic/claude-3.5-haiku" },
];

function Settings({ settings, onClose, onSettingsChange }: SettingsProps) {
  const [activeTab, setActiveTab] = useState<TabType>("api");
  const [isCustom, setIsCustom] = useState(false);
  const [customModel, setCustomModel] = useState("");
  const [showKey, setShowKey] = useState(false);

  // Temporary local states so we only save on clicking "Save"
  const [apiKey, setApiKey] = useState(settings.apiKey || "");
  const [topK, setTopK] = useState(settings.topK);
  const [model, setModel] = useState(settings.model);
  const [includeChunks, setIncludeChunks] = useState(settings.includeChunks);
  const [temperature, setTemperature] = useState(settings.temperature ?? 0.7);
  const [systemPrompt, setSystemPrompt] = useState(
    settings.systemPrompt ?? "Answer the following question based ONLY on the provided context. If the answer is not in the context, say that you don't know."
  );

  // Testing connection state
  const [isTesting, setIsTesting] = useState(false);
  const [testResult, setTestResult] = useState<{
    success: boolean;
    message: string;
  } | null>(null);

  useEffect(() => {
    const isPredefined = PREDEFINED_MODELS.some((m) => m.value === settings.model);
    if (!isPredefined && settings.model) {
      setIsCustom(true);
      setCustomModel(settings.model);
    }
  }, [settings.model]);

  const handleModelChange = (val: string) => {
    if (val === "custom") {
      setIsCustom(true);
    } else {
      setIsCustom(false);
      setModel(val);
    }
  };

  const handleTestConnection = async () => {
    if (!apiKey) {
      setTestResult({
        success: false,
        message: "Please enter an API key to test.",
      });
      return;
    }

    setIsTesting(true);
    setTestResult(null);

    try {
      const response = await fetch("https://openrouter.ai/api/v1/auth/key", {
        method: "GET",
        headers: {
          Authorization: `Bearer ${apiKey}`,
        },
      });

      if (response.ok) {
        const data = await response.json();
        const label = data.data?.label || "Workspace Key";
        const limit = data.data?.limit ? `$${data.data.limit.toFixed(2)}` : "No limit";
        const usage = data.data?.usage ? `$${data.data.usage.toFixed(4)}` : "$0.00";
        setTestResult({
          success: true,
          message: `Connected! Label: "${label}" | Credits Used: ${usage} of ${limit}`,
        });
      } else {
        const errorData = await response.json().catch(() => ({}));
        setTestResult({
          success: false,
          message: errorData.error?.message || "Invalid API key. Please check your credentials.",
        });
      }
    } catch (err: any) {
      setTestResult({
        success: false,
        message: err.message || "Failed to contact OpenRouter. Check your network connection.",
      });
    } finally {
      setIsTesting(false);
    }
  };

  const handleSave = () => {
    const finalModel = isCustom ? customModel.trim() : model;
    onSettingsChange({
      apiKey: apiKey.trim(),
      model: finalModel,
      topK,
      includeChunks,
      temperature,
      systemPrompt: systemPrompt.trim(),
    });
    onClose();
  };

  return (
    <div 
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/75 backdrop-blur-sm p-4 animate-fadeIn"
      onClick={onClose}
    >
      {/* Modal Dialog */}
      <div 
        className="w-full max-w-[760px] h-[480px] bg-slate-900 border border-slate-800 rounded-xl flex overflow-hidden shadow-2xl animate-scaleUp"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Left Sidebar inside Modal */}
        <div className="w-56 bg-slate-950/40 border-r border-slate-850 p-3 flex flex-col justify-between shrink-0">
          <div className="space-y-4">
            {/* Close (X) button at the top left of modal sidebar */}
            <div className="flex items-center gap-2 px-1">
              <button
                onClick={onClose}
                className="p-1.5 text-slate-400 hover:text-slate-100 hover:bg-slate-905 border border-slate-800 rounded-lg transition cursor-pointer"
                title="Close settings"
              >
                <X className="h-4 w-4" />
              </button>
              <span className="text-xs font-bold text-slate-350 uppercase tracking-wide">Settings</span>
            </div>

            {/* Navigation Tabs */}
            <nav className="space-y-1">
              <button
                type="button"
                onClick={() => setActiveTab("api")}
                className={`w-full flex items-center gap-2.5 rounded-lg px-3 py-2 text-left text-xs font-semibold transition ${
                  activeTab === "api"
                    ? "bg-slate-800 text-slate-100"
                    : "text-slate-400 hover:bg-slate-950/40 hover:text-slate-200"
                }`}
              >
                API Access
              </button>
              <button
                type="button"
                onClick={() => setActiveTab("model")}
                className={`w-full flex items-center gap-2.5 rounded-lg px-3 py-2 text-left text-xs font-semibold transition ${
                  activeTab === "model"
                    ? "bg-slate-800 text-slate-100"
                    : "text-slate-400 hover:bg-slate-950/40 hover:text-slate-200"
                }`}
              >
                Model Config
              </button>
              <button
                type="button"
                onClick={() => setActiveTab("instructions")}
                className={`w-full flex items-center gap-2.5 rounded-lg px-3 py-2 text-left text-xs font-semibold transition ${
                  activeTab === "instructions"
                    ? "bg-slate-800 text-slate-100"
                    : "text-slate-400 hover:bg-slate-950/40 hover:text-slate-200"
                }`}
              >
                Instructions
              </button>
            </nav>
          </div>

          <div className="px-2 py-1 text-[10px] text-slate-500 font-mono text-center border-t border-slate-850 pt-2">
            RAG v1.0.0
          </div>
        </div>

        {/* Right Settings Content */}
        <div className="flex-1 flex flex-col justify-between h-full bg-slate-900">
          <div className="p-6 overflow-y-auto flex-1 space-y-4">
            {/* Tab 1: API Config */}
            {activeTab === "api" && (
              <div className="space-y-4 animate-fadeIn">
                <div>
                  <h3 className="text-sm font-bold text-slate-100">
                    API Key Configuration
                  </h3>
                  <p className="text-[11px] text-slate-400 mt-1">
                    Your API credentials are stored securely inside your browser's local storage.
                  </p>
                </div>

                <div className="space-y-2.5">
                  <label className="block text-xs text-slate-300 font-medium">OpenRouter API Key</label>
                  <div className="flex gap-2">
                    <div className="relative flex-1">
                      <input
                        type={showKey ? "text" : "password"}
                        value={apiKey}
                        onChange={(e) => setApiKey(e.target.value)}
                        placeholder="sk-or-v1-..."
                        className="w-full rounded-lg border border-slate-700 bg-slate-950/80 pl-3 pr-10 py-1.5 text-xs text-slate-100 placeholder:text-slate-600 focus:border-cyan-500 focus:outline-none focus:ring-1 focus:ring-cyan-500/40 font-mono"
                      />
                      <button
                        type="button"
                        onClick={() => setShowKey(!showKey)}
                        className="absolute right-3 top-2.5 text-slate-500 hover:text-slate-300 transition"
                      >
                        {showKey ? <EyeOff className="h-3.5 w-3.5" /> : <Eye className="h-3.5 w-3.5" />}
                      </button>
                    </div>
                    <button
                      type="button"
                      onClick={handleTestConnection}
                      disabled={isTesting}
                      className="px-3.5 py-1.5 text-xs font-semibold rounded-lg bg-slate-800 text-slate-200 border border-slate-750 hover:border-slate-500 hover:bg-slate-700 disabled:opacity-50 cursor-pointer transition flex items-center gap-1"
                    >
                      {isTesting ? <Loader2 className="h-3 w-3 animate-spin text-cyan-400" /> : null}
                      Test Connection
                    </button>
                  </div>

                  {testResult && (
                    <div
                      className={`mt-2 flex items-start gap-2 rounded-lg border p-2.5 text-[11px] ${
                        testResult.success
                          ? "bg-emerald-950/20 border-emerald-900/40 text-emerald-450"
                          : "bg-red-950/20 border-red-900/40 text-red-400"
                      }`}
                    >
                      {testResult.success ? (
                        <CheckCircle2 className="h-3.5 w-3.5 shrink-0 mt-0.5" />
                      ) : (
                        <XCircle className="h-3.5 w-3.5 shrink-0 mt-0.5" />
                      )}
                      <span>{testResult.message}</span>
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* Tab 2: Model Config */}
            {activeTab === "model" && (
              <div className="space-y-4 animate-fadeIn">
                <div>
                  <h3 className="text-sm font-bold text-slate-100">
                    Model & Inference Config
                  </h3>
                  <p className="text-[11px] text-slate-400 mt-1">
                    Select the target LLM and optimize generation parameters.
                  </p>
                </div>

                <div className="space-y-3">
                  <div className="space-y-1.5">
                    <label className="block text-xs text-slate-350 font-medium">Target LLM</label>
                    {!isCustom ? (
                      <select
                        value={model}
                        onChange={(e) => handleModelChange(e.target.value)}
                        className="w-full rounded-lg border border-slate-700 bg-slate-950 px-2.5 py-1.5 text-xs text-slate-200 focus:border-cyan-500 focus:outline-none"
                      >
                        {PREDEFINED_MODELS.map((m) => (
                          <option key={m.value} value={m.value}>
                            {m.label}
                          </option>
                        ))}
                        <option value="custom">Custom Model...</option>
                      </select>
                    ) : (
                      <div className="flex gap-2">
                        <input
                          type="text"
                          value={customModel}
                          placeholder="e.g. meta-llama/llama-3-8b-instruct:free"
                          onChange={(e) => setCustomModel(e.target.value)}
                          className="flex-1 rounded-lg border border-slate-700 bg-slate-950 px-2.5 py-1.5 text-xs text-slate-200 focus:border-cyan-500 focus:outline-none placeholder:text-slate-600 font-mono"
                        />
                        <button
                          type="button"
                          onClick={() => setIsCustom(false)}
                          className="px-2 py-1 text-[10px] text-slate-400 hover:text-slate-200 border border-slate-700 rounded-md"
                        >
                          Cancel
                        </button>
                      </div>
                    )}
                  </div>

                  <div className="grid grid-cols-2 gap-4 pt-2">
                    <div className="space-y-1.5">
                      <div className="flex justify-between items-center text-xs">
                        <span className="text-slate-350">Temperature</span>
                        <span className="text-cyan-455 font-mono font-bold">{temperature}</span>
                      </div>
                      <input
                        type="range"
                        min={0.0}
                        max={1.0}
                        step={0.1}
                        value={temperature}
                        onChange={(e) => setTemperature(parseFloat(e.target.value))}
                        className="w-full accent-cyan-500 h-1 bg-slate-950 rounded-lg cursor-pointer"
                      />
                    </div>

                    <div className="space-y-1.5">
                      <div className="flex justify-between items-center text-xs">
                        <span className="text-slate-350">Top-K Retrieval Chunks</span>
                        <span className="text-cyan-455 font-mono font-bold">{topK}</span>
                      </div>
                      <input
                        type="range"
                        min={1}
                        max={10}
                        value={topK}
                        onChange={(e) => setTopK(parseInt(e.target.value))}
                        className="w-full accent-cyan-500 h-1 bg-slate-950 rounded-lg cursor-pointer"
                      />
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Tab 3: Instructions */}
            {activeTab === "instructions" && (
              <div className="space-y-4 animate-fadeIn">
                <div>
                  <h3 className="text-sm font-bold text-slate-100">
                    Instructions & Context
                  </h3>
                  <p className="text-[11px] text-slate-400 mt-1">
                    Fine-tune target system prompts and specify context output parameters.
                  </p>
                </div>

                <div className="space-y-3.5">
                  <div className="space-y-1.5">
                    <label className="block text-xs text-slate-350 font-medium">System Instructions</label>
                    <textarea
                      rows={4}
                      value={systemPrompt}
                      onChange={(e) => setSystemPrompt(e.target.value)}
                      placeholder="Instruct the model how to output..."
                      className="w-full rounded-lg border border-slate-700 bg-slate-950/80 px-2.5 py-1.5 text-xs text-slate-200 placeholder:text-slate-600 focus:border-cyan-500 focus:outline-none resize-none leading-relaxed"
                    />
                  </div>

                  <div className="flex items-center gap-2 text-xs text-slate-300 font-medium cursor-pointer pt-1 select-none">
                    <input
                      type="checkbox"
                      id="chunksToggle"
                      checked={includeChunks}
                      onChange={(e) => setIncludeChunks(e.target.checked)}
                      className="h-3.5 w-3.5 rounded border-slate-800 bg-slate-950 text-cyan-500 accent-cyan-500"
                    />
                    <label htmlFor="chunksToggle" className="cursor-pointer">
                      Include source text chunks in evaluation sidebar
                    </label>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Bottom Footer Actions */}
          <div className="p-4 border-t border-slate-850 flex justify-end gap-2 shrink-0 bg-slate-950/20">
            <button
              onClick={onClose}
              className="px-3.5 py-1.5 text-xs rounded-lg border border-slate-800 bg-slate-950 text-slate-400 transition hover:bg-slate-900 cursor-pointer"
            >
              Cancel
            </button>
            <button
              onClick={handleSave}
              className="px-4 py-1.5 text-xs font-semibold rounded-lg bg-cyan-500 hover:bg-cyan-450 text-slate-950 transition cursor-pointer shadow-lg shadow-cyan-500/10"
            >
              Save Changes
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

export default Settings;
