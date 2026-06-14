import { FileText } from "lucide-react";
import type { ContextDoc } from "../types";

interface ContextCardProps {
  doc: ContextDoc;
}

function ContextCard({ doc }: ContextCardProps) {
  const parentId = doc.metadata?.parent_id;

  return (
    <article className="rounded-xl border border-slate-800 bg-slate-900/50 p-4 hover:border-slate-700 transition duration-200">
      <div className="mb-2 flex flex-wrap items-center gap-1.5">
        <span className="rounded-full bg-cyan-500/10 px-2 py-0.5 text-[10px] font-semibold text-cyan-400 border border-cyan-500/20">
          Hybrid Fused
        </span>
        {parentId && (
          <span className="rounded-full bg-purple-500/10 px-2 py-0.5 text-[10px] font-semibold text-purple-400 border border-purple-500/20">
            Parent: {parentId.split("_p").pop() ? `Chunk p${parentId.split("_p").pop()}` : parentId}
          </span>
        )}
      </div>

      <div className="mb-1.5 flex items-center gap-1.5 break-all">
        <FileText className="h-3.5 w-3.5 flex-shrink-0 text-cyan-400" />
        <h3 className="text-xs font-semibold text-slate-300">{doc.title}</h3>
      </div>
      <p className="text-xs leading-relaxed text-slate-400 line-clamp-4 select-all">
        {doc.content}
      </p>
    </article>
  );
}

export default ContextCard;
