import { useState } from "react";
import ContextCard from "./ContextCard";
import type {
  ChatMessage,
  ContextDoc,
  MessageEvaluationVersion,
} from "../types";

interface RightPaneProps {
  contextDocs: ContextDoc[];
  selectedMessage: ChatMessage | null;
  evaluationHistory: MessageEvaluationVersion[];
}

function formatScore(score: number | undefined): string {
  if (typeof score !== "number") {
    return "--";
  }

  return `${Math.round(score * 100)}%`;
}

function RightPane({
  contextDocs,
  selectedMessage,
  evaluationHistory,
}: RightPaneProps) {
  const [activeTab, setActiveTab] = useState<"context" | "metrics">("context");

  return (
    <aside className="p-4">
      <div className="mb-4 flex items-center gap-2">
        <button
          type="button"
          onClick={() => setActiveTab("context")}
          className={`rounded-md px-3 py-1.5 text-sm ${
            activeTab === "context"
              ? "bg-cyan-500 text-slate-950"
              : "border border-slate-700 bg-slate-900 text-slate-200"
          }`}
        >
          Retrieved Context
        </button>
        <button
          type="button"
          onClick={() => setActiveTab("metrics")}
          className={`rounded-md px-3 py-1.5 text-sm ${
            activeTab === "metrics"
              ? "bg-cyan-500 text-slate-950"
              : "border border-slate-700 bg-slate-900 text-slate-200"
          }`}
        >
          Metrics
        </button>
      </div>

      {activeTab === "context" ? (
        <div className="space-y-3">
          {contextDocs.map((doc) => (
            <ContextCard key={doc.id} doc={doc} />
          ))}
        </div>
      ) : (
        <div className="space-y-3">
          {!selectedMessage ? (
            <p className="text-sm text-slate-400">
              Select an assistant message to view detailed metrics.
            </p>
          ) : (
            <>
              <article className="rounded-xl border border-slate-800 bg-slate-900/70 p-4">
                <h3 className="mb-2 text-sm font-semibold text-slate-200">
                  Latest Metrics (v{selectedMessage.evaluationVersion ?? "--"})
                </h3>
                <div className="grid grid-cols-2 gap-2 text-xs text-slate-300">
                  <p>
                    Faithfulness: {formatScore(selectedMessage.faithfulness)}
                  </p>
                  <p>
                    Relevancy: {formatScore(selectedMessage.answerRelevancy)}
                  </p>
                  <p>Latency: {selectedMessage.latencyMs ?? "--"} ms</p>
                  <p>Token Count: {selectedMessage.tokenCount ?? "--"}</p>
                </div>
                <p className="mt-3 text-xs leading-relaxed text-slate-400">
                  {selectedMessage.reasoning ?? "Reasoning not available yet."}
                </p>
              </article>

              <article className="rounded-xl border border-slate-800 bg-slate-900/70 p-4">
                <h3 className="mb-2 text-sm font-semibold text-slate-200">
                  Version History
                </h3>
                {evaluationHistory.length === 0 ? (
                  <p className="text-xs text-slate-400">
                    No evaluation history found.
                  </p>
                ) : (
                  <ul className="space-y-2 text-xs text-slate-300">
                    {evaluationHistory.map((entry) => (
                      <li
                        key={entry.id}
                        className="rounded-md border border-slate-800 bg-slate-950/70 p-2"
                      >
                        <p className="font-medium text-slate-200">
                          v{entry.version} • {entry.status}
                        </p>
                        <p>Faithfulness: {formatScore(entry.faithfulness)}</p>
                        <p>Relevancy: {formatScore(entry.answerRelevancy)}</p>
                        <p className="mt-1 text-slate-400">
                          {entry.reasoning ??
                            entry.errorMessage ??
                            "No reasoning recorded."}
                        </p>
                      </li>
                    ))}
                  </ul>
                )}
              </article>
            </>
          )}
        </div>
      )}
    </aside>
  );
}

export default RightPane;
