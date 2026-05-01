import { History as HistoryIcon, MessageSquareText } from 'lucide-react'
import type { HistoryItem } from '../types'

interface HistoryProps {
  items: HistoryItem[]
}

function History({ items }: HistoryProps) {
  return (
    <section className="rounded-xl border border-slate-800 bg-slate-900/60 p-4">
      <div className="mb-4 flex items-center gap-2">
        <HistoryIcon className="h-4 w-4 text-cyan-400" />
        <h2 className="text-sm font-semibold text-slate-100">History</h2>
      </div>

      <ul className="space-y-3">
        {items.map((item) => (
          <li key={item.id} className="rounded-lg border border-slate-800 bg-slate-950/70 p-3">
            <div className="mb-1 flex items-start gap-2">
              <MessageSquareText className="mt-0.5 h-4 w-4 shrink-0 text-slate-400" />
              <p className="text-sm font-medium text-slate-200">{item.title}</p>
            </div>
            <p className="line-clamp-2 text-xs text-slate-400">{item.preview}</p>
            <p className="mt-2 text-[11px] uppercase tracking-wide text-slate-500">{item.timestamp}</p>
          </li>
        ))}
      </ul>
    </section>
  )
}

export default History
