import { useEffect, useMemo, useRef, useState } from 'react'
import { useNavigate } from 'react-router'
import { CornerDownLeft, Search } from 'lucide-react'
import { modules, pad } from '../data/curriculum'
import { isPublished, lessons, loadLesson, searchableText } from '../lib/content'

interface Entry {
  title: string
  context: string
  path: string
  published: boolean
  body?: string
}

const base: Entry[] = modules.flatMap((m) => [
  { title: m.title, context: `Module ${pad(m.n)}`, path: `/learn/${m.slug}`, published: true },
  ...m.topics.map((t) => {
    const published = isPublished(m.slug, t.slug)
    return {
      title: t.title,
      context: m.title,
      path: published ? `/learn/${m.slug}/${t.slug}` : `/learn/${m.slug}`,
      published,
    }
  }),
])

let bodies: Promise<Record<string, string>> | null = null
const loadBodies = () =>
  (bodies ??= Promise.all(
    lessons.map(async (l) => [l.path, searchableText(await loadLesson(l.module.slug, l.slug)).toLowerCase()] as const),
  ).then(Object.fromEntries))

interface Hit extends Entry {
  snippet?: string
}

function search(q: string, text: Record<string, string>): Hit[] {
  const terms = q.toLowerCase().split(/\s+/).filter(Boolean)
  if (!terms.length) return []
  const scored: [number, Hit][] = []
  for (const e of base) {
    const title = e.title.toLowerCase()
    const ctx = e.context.toLowerCase()
    const body = text[e.path] ?? ''
    let score = 0
    let ok = true
    let snippet: string | undefined
    for (const t of terms) {
      if (title.startsWith(t)) score += 12
      else if (title.includes(t)) score += 8
      else if (ctx.includes(t)) score += 3
      else if (e.published && body.includes(t)) {
        score += 1
        if (!snippet) {
          const at = body.indexOf(t)
          snippet = '…' + body.slice(Math.max(0, at - 40), at + 70).trim() + '…'
        }
      } else {
        ok = false
        break
      }
    }
    if (ok) scored.push([score + (e.published ? 0.5 : 0), { ...e, snippet }])
  }
  return scored
    .sort((a, b) => b[0] - a[0])
    .slice(0, 12)
    .map(([, h]) => h)
}

export default function SearchDialog({ onClose }: { onClose: () => void }) {
  const [q, setQ] = useState('')
  const [text, setText] = useState<Record<string, string>>({})
  const [sel, setSel] = useState(0)
  const navigate = useNavigate()
  const listRef = useRef<HTMLUListElement>(null)

  useEffect(() => {
    loadBodies().then(setText)
  }, [])

  const hits = useMemo(() => search(q, text), [q, text])

  useEffect(() => setSel(0), [q])

  useEffect(() => {
    listRef.current?.children[sel]?.scrollIntoView({ block: 'nearest' })
  }, [sel])

  const go = (h: Hit) => {
    navigate(h.path)
    onClose()
  }

  return (
    <div className="fixed inset-0 z-50 flex items-start justify-center px-4 pt-[12vh]" role="dialog" aria-modal="true">
      <div className="absolute inset-0 bg-black/45 backdrop-blur-[2px]" onClick={onClose} />
      <div className="relative w-full max-w-xl overflow-hidden rounded-2xl border border-line bg-surface shadow-2xl">
        <div className="flex items-center gap-3 border-b border-line px-4">
          <Search size={17} className="text-faint" />
          <input
            autoFocus
            value={q}
            onChange={(e) => setQ(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Escape') onClose()
              else if (e.key === 'ArrowDown') {
                e.preventDefault()
                setSel((s) => Math.min(s + 1, hits.length - 1))
              } else if (e.key === 'ArrowUp') {
                e.preventDefault()
                setSel((s) => Math.max(s - 1, 0))
              } else if (e.key === 'Enter' && hits[sel]) go(hits[sel])
            }}
            placeholder="Search topics, modules and lesson text…"
            className="h-13 flex-1 bg-transparent text-[15px] outline-none placeholder:text-faint"
          />
          <kbd className="rounded border border-line bg-sunken px-1.5 font-mono text-[11px] text-faint">esc</kbd>
        </div>

        {q.trim() === '' ? (
          <p className="px-5 py-8 text-center text-sm text-muted">
            Search all 46 modules. Try <span className="font-medium text-fg">decorators</span>,{' '}
            <span className="font-medium text-fg">gradient descent</span> or{' '}
            <span className="font-medium text-fg">RAG</span>.
          </p>
        ) : hits.length === 0 ? (
          <p className="px-5 py-8 text-center text-sm text-muted">No results for “{q}”.</p>
        ) : (
          <ul ref={listRef} className="thin-scroll max-h-[52vh] overflow-y-auto p-2">
            {hits.map((h, i) => (
              <li key={h.path + h.title}>
                <button
                  onClick={() => go(h)}
                  onMouseMove={() => setSel(i)}
                  className={`flex w-full items-center gap-3 rounded-lg px-3 py-2 text-left ${
                    i === sel ? 'bg-accent-soft' : ''
                  }`}
                >
                  <span className="min-w-0 flex-1">
                    <span className="flex items-center gap-2 text-sm font-medium">
                      <span className="truncate">{h.title}</span>
                      {!h.published && (
                        <span className="shrink-0 rounded-full border border-line px-1.5 text-[10px] text-faint">soon</span>
                      )}
                    </span>
                    <span className="block truncate text-xs text-muted">{h.snippet ?? h.context}</span>
                  </span>
                  {i === sel && <CornerDownLeft size={14} className="shrink-0 text-accent" />}
                </button>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  )
}
