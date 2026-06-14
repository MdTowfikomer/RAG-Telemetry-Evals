import { useEffect, useState, useRef } from "react";
import {
  Upload,
  Trash2,
  Loader2,
  CheckCircle2,
  XCircle,
  AlertCircle,
  FileText,
} from "lucide-react";
import { api } from "../lib/api";
import type { UploadedFile } from "../types";

function formatBytes(bytes: number): string {
  if (bytes === 0) return "0 Bytes";
  const k = 1024;
  const sizes = ["Bytes", "KB", "MB", "GB"];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i];
}

function DocumentUpload() {
  const [documents, setDocuments] = useState<UploadedFile[]>([]);
  const [isUploading, setIsUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

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

  // Poll for document status updates while any document is pending/processing
  useEffect(() => {
    fetchDocs();
  }, []);

  useEffect(() => {
    const hasActiveTasks = documents.some(
      (doc) => doc.status === "pending" || doc.status === "processing",
    );

    if (!hasActiveTasks) return;

    const interval = setInterval(() => {
      void fetchDocs();
    }, 3000);

    return () => clearInterval(interval);
  }, [documents]);

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;

    const fileList = Array.from(files);
    // Client-side extension validation
    const invalidFile = fileList.find(
      (file) =>
        !file.name.endsWith(".pdf") &&
        !file.name.endsWith(".txt") &&
        !file.name.endsWith(".md"),
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
    }
  };

  const handleDelete = async (id: string) => {
    if (!confirm("Are you sure you want to delete this document?")) return;

    try {
      await api.deleteDocument(id);
      setDocuments((prev) => prev.filter((doc) => doc.id !== id));
      setError(null);
      // Fetch documents to trigger status updates if a rebuild is running
      void fetchDocs();
    } catch (err: any) {
      setError("Failed to delete document.");
    }
  };

  return (
    <section className="rounded-xl border border-slate-800 bg-slate-900/60 p-4">
      <div className="mb-4 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Upload className="h-4 w-4 text-cyan-400" />
          <h2 className="text-sm font-semibold text-slate-100">Documents</h2>
        </div>
        <button
          onClick={() => fileInputRef.current?.click()}
          disabled={isUploading}
          className="flex items-center gap-1 rounded bg-cyan-600 px-2.5 py-1 text-xs font-semibold text-white transition hover:bg-cyan-500 disabled:opacity-50 cursor-pointer"
        >
          {isUploading ? (
            <Loader2 className="h-3 w-3 animate-spin" />
          ) : (
            <FileText className="h-3 w-3" />
          )}
          Upload
        </button>
        <input
          type="file"
          ref={fileInputRef}
          onChange={(e) => {
            void handleFileChange(e);
          }}
          multiple
          accept=".pdf,.txt,.md"
          className="hidden"
        />
      </div>

      {error && (
        <div className="mb-3 flex items-start gap-2 rounded-md bg-red-950/40 border border-red-900/50 p-2 text-xs text-red-400">
          <AlertCircle className="h-4 w-4 shrink-0 mt-0.5" />
          <span>{error}</span>
        </div>
      )}

      <div className="max-h-60 overflow-y-auto space-y-2 pr-1 custom-scrollbar">
        {documents.length === 0 ? (
          <p className="text-center text-xs text-slate-500 py-4">
            No uploaded documents yet.
          </p>
        ) : (
          documents.map((doc) => (
            <div
              key={doc.id}
              className="flex items-center justify-between rounded-lg border border-slate-800 bg-slate-950/40 p-2.5 transition hover:border-slate-700/80"
            >
              <div className="flex items-start gap-2 min-w-0 flex-1 mr-2">
                <FileText className="h-4 w-4 text-slate-400 shrink-0 mt-0.5" />
                <div className="min-w-0 flex-1">
                  <p className="truncate text-xs font-medium text-slate-200" title={doc.filename}>
                    {doc.filename}
                  </p>
                  <p className="text-[10px] text-slate-500">
                    {formatBytes(doc.file_size)}
                  </p>
                </div>
              </div>

              <div className="flex items-center gap-2 shrink-0">
                {/* Status indicator */}
                {doc.status === "pending" && (
                  <span className="flex items-center gap-1 rounded bg-slate-800 px-1.5 py-0.5 text-[9px] text-slate-400">
                    <Loader2 className="h-2.5 w-2.5 animate-spin" />
                    Pending
                  </span>
                )}
                {doc.status === "processing" && (
                  <span className="flex items-center gap-1 rounded bg-cyan-950/60 border border-cyan-800/40 px-1.5 py-0.5 text-[9px] text-cyan-400">
                    <Loader2 className="h-2.5 w-2.5 animate-spin" />
                    Ingesting
                  </span>
                )}
                {doc.status === "indexed" && (
                  <span className="flex items-center gap-1 rounded bg-emerald-950/60 border border-emerald-800/40 px-1.5 py-0.5 text-[9px] text-emerald-400">
                    <CheckCircle2 className="h-2.5 w-2.5" />
                    Indexed
                  </span>
                )}
                {doc.status === "failed" && (
                  <span
                    className="flex items-center gap-1 rounded bg-red-950/60 border border-red-800/40 px-1.5 py-0.5 text-[9px] text-red-400 cursor-help"
                    title={doc.error_message || "Ingestion failed"}
                  >
                    <XCircle className="h-2.5 w-2.5" />
                    Failed
                  </span>
                )}

                <button
                  onClick={() => {
                    void handleDelete(doc.id);
                  }}
                  className="rounded p-1 text-slate-500 hover:bg-slate-800 hover:text-red-400 transition cursor-pointer"
                  title="Delete Document"
                >
                  <Trash2 className="h-3.5 w-3.5" />
                </button>
              </div>
            </div>
          ))
        )}
      </div>
    </section>
  );
}

export default DocumentUpload;
