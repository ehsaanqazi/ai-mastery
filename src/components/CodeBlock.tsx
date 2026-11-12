import { useState, type ReactNode } from 'react'
import { Check, Copy } from 'lucide-react'

const LABELS: Record<string, string> = {
  python: 'Python',
  py: 'Python',
  bash: 'Terminal',
  sh: 'Terminal',
  shell: 'Terminal',
  json: 'JSON',
  csv: 'CSV',
  output: 'Output',
  text: 'Text',
  toml: 'TOML',
  sql: 'SQL',
}

export default function CodeBlock({ lang, text, children }: { lang: string; text: string; children: ReactNode }) {
  const [copied, setCopied] = useState(false)
  const isOutput = lang === 'output'

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(text)
      setCopied(true)
      setTimeout(() => setCopied(false), 1500)
    } catch {
      /* clipboard unavailable */
    }
  }

  return (
    <figure
      className={`group overflow-hidden rounded-xl border border-line ${isOutput ? 'bg-sunken' : 'bg-code'}`}
    >
      <figcaption className="flex h-9 items-center justify-between border-b border-line px-4">
        <span className="flex items-center gap-2 font-mono text-[11px] font-medium uppercase tracking-wider text-faint">
          {isOutput && <span className="h-1.5 w-1.5 rounded-full bg-accent" />}
          {LABELS[lang] ?? (lang || 'Code')}
        </span>
        {!isOutput && (
          <button
            onClick={copy}
            className="flex items-center gap-1.5 rounded px-1.5 py-0.5 text-xs text-faint transition-colors hover:text-fg"
          >
            {copied ? <Check size={13} className="text-accent" /> : <Copy size={13} />}
            {copied ? 'Copied' : 'Copy'}
          </button>
        )}
      </figcaption>
      <pre className="thin-scroll overflow-x-auto px-4 py-3.5 font-mono text-[13.5px] leading-[1.7]">
        <code>{children}</code>
      </pre>
    </figure>
  )
}
