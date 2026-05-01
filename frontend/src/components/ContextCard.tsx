import { FileText } from 'lucide-react'
import type { ContextDoc } from '../types'

interface ContextCardProps {
  doc: ContextDoc
}

function ContextCard({ doc }: ContextCardProps) {
  return (
    <article className="rounded-xl border border-slate-800 bg-slate-900/70 p-4">
      <div className="mb-2 flex items-center gap-2">
        <FileText className="h-4 w-4 text-cyan-400" />
        <h3 className="text-sm font-semibold text-slate-200">{doc.title}</h3>
      </div>
      <p className="text-xs leading-relaxed text-slate-400">{doc.content}</p>
    </article>
  )
}

export default ContextCard
