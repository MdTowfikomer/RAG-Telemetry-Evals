import { SlidersHorizontal, Plus } from 'lucide-react'
import { useState, useEffect } from 'react'
import type { ChatSettings } from '../types'

interface SettingsProps {
  settings: ChatSettings
  onSettingsChange: (settings: ChatSettings) => void
}

const PREDEFINED_MODELS = [
  { label: 'Gemini Flash', value: 'google/gemini-2.0-flash-001' },
  { label: 'GPT-4o Mini', value: 'openai/gpt-4o-mini' },
  { label: 'Claude 3.5 Haiku', value: 'anthropic/claude-3.5-haiku' },
]

function Settings({ settings, onSettingsChange }: SettingsProps) {
  const [isCustom, setIsCustom] = useState(false)
  const [customModel, setCustomModel] = useState('')

  // Check if current setting is a custom model on mount
  useEffect(() => {
    const isPredefined = PREDEFINED_MODELS.some(m => m.value === settings.model)
    if (!isPredefined && settings.model) {
      setIsCustom(true)
      setCustomModel(settings.model)
    }
  }, [])

  const handleChange = (key: keyof ChatSettings, value: any) => {
    onSettingsChange({ ...settings, [key]: value })
  }

  const handleModelChange = (val: string) => {
    if (val === 'custom') {
      setIsCustom(true)
    } else {
      setIsCustom(false)
      handleChange('model', val)
    }
  }

  const handleCustomSubmit = () => {
    if (customModel.trim()) {
      handleChange('model', customModel.trim())
    }
  }

  return (
    <section className="rounded-xl border border-slate-800 bg-slate-900/60 p-4">
      <div className="mb-4 flex items-center gap-2">
        <SlidersHorizontal className="h-4 w-4 text-cyan-400" />
        <h2 className="text-sm font-semibold text-slate-100">Settings</h2>
      </div>

      <div className="space-y-4 text-sm">
        <label className="block text-slate-300">
          Top-K Retrieval: <span className="text-cyan-400 font-mono">{settings.topK}</span>
          <input
            type="range"
            min={1}
            max={10}
            value={settings.topK}
            onChange={(e) => handleChange('topK', parseInt(e.target.value))}
            className="mt-2 w-full accent-cyan-500"
          />
        </label>

        <div className="space-y-2">
          <label className="block text-slate-300">Model</label>
          {!isCustom ? (
            <select
              value={settings.model}
              onChange={(e) => handleModelChange(e.target.value)}
              className="w-full rounded-md border border-slate-700 bg-slate-950 px-2 py-1 text-slate-200 focus:border-cyan-500 focus:outline-none"
            >
              {PREDEFINED_MODELS.map(m => (
                <option key={m.value} value={m.value}>{m.label}</option>
              ))}
              <option value="custom">Custom Model...</option>
            </select>
          ) : (
            <div className="flex gap-2">
              <input
                type="text"
                value={customModel}
                placeholder="e.g. tencent/hy3-preview:free"
                onChange={(e) => setCustomModel(e.target.value)}
                onBlur={handleCustomSubmit}
                onKeyDown={(e) => e.key === 'Enter' && handleCustomSubmit()}
                className="flex-1 rounded-md border border-slate-700 bg-slate-950 px-2 py-1 text-slate-200 focus:border-cyan-400 focus:outline-none placeholder:text-slate-600"
              />
              <button 
                onClick={() => setIsCustom(false)}
                className="text-xs text-slate-500 hover:text-slate-300"
              >
                Cancel
              </button>
            </div>
          )}
        </div>

        <label className="flex items-center gap-2 text-slate-300 cursor-pointer">
          <input 
            type="checkbox" 
            checked={settings.includeChunks} 
            onChange={(e) => handleChange('includeChunks', e.target.checked)}
            className="accent-cyan-500" 
          />
          Include source chunks
        </label>
      </div>
    </section>
  )
}

export default Settings
