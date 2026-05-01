import ContextCard from "./ContextCard";
import type { ContextDoc } from "../types";

interface RightPaneProps {
  contextDocs: ContextDoc[];
}

function RightPane({ contextDocs }: RightPaneProps) {
  return (
    <aside className="p-4">
      <h1 className="mb-4 text-base font-semibold text-slate-100">
        Retrieved Context
      </h1>
      <div className="space-y-3">
        {contextDocs.map((doc) => (
          <ContextCard key={doc.id} doc={doc} />
        ))}
      </div>
    </aside>
  );
}

export default RightPane;
