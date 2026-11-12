import { useEffect, useRef, useState } from 'react'
import { Link, NavLink, useParams } from 'react-router'
import { Bookmark, Check, ChevronRight, Home, Map } from 'lucide-react'
import Logo from './Logo'
import { pad, phases, type Module } from '../data/curriculum'
import { ASSESSMENT, hasAssessment, isPublished, lessonId, modulePublished, moduleLessons } from '../lib/content'
import { useStore } from '../lib/store'

const navItem = ({ isActive }: { isActive: boolean }) =>
  `flex items-center gap-2.5 rounded-md px-2.5 py-1.5 text-sm transition-colors ${
    isActive ? 'bg-accent-soft font-medium text-accent' : 'text-muted hover:bg-sunken hover:text-fg'
  }`

export default function Sidebar() {
  const { moduleSlug } = useParams()
  const [open, setOpen] = useState<Record<string, boolean>>(() => (moduleSlug ? { [moduleSlug]: true } : {}))

  useEffect(() => {
    if (moduleSlug) setOpen((o) => (o[moduleSlug] ? o : { ...o, [moduleSlug]: true }))
  }, [moduleSlug])

  return (
    <div className="flex h-full flex-col">
      <Link to="/" className="flex h-14 shrink-0 items-center border-b border-line px-5">
        <Logo />
      </Link>

      <div className="thin-scroll flex-1 overflow-y-auto px-3 pb-10 pt-4">
        <nav className="space-y-0.5">
          <NavLink to="/" end className={navItem}>
            <Home size={15} /> Home
          </NavLink>
          <NavLink to="/roadmap" className={navItem}>
            <Map size={15} /> Roadmap
          </NavLink>
          <NavLink to="/library" className={navItem}>
            <Bookmark size={15} /> My library
          </NavLink>
        </nav>

        {phases.map((phase) => (
          <section key={phase.n} className="mt-6">
            <h3 className="flex items-baseline gap-2 px-2.5 pb-1.5">
              <span className="shrink-0 whitespace-nowrap font-mono text-[10.5px] font-semibold tracking-wider text-accent">
                PHASE {pad(phase.n)}
              </span>
              <span className="truncate text-[11.5px] font-semibold uppercase tracking-wider text-faint">
                {phase.title}
              </span>
            </h3>
            <ul className="space-y-px">
              {phase.modules.map((m) => (
                <ModuleItem
                  key={m.slug}
                  module={m}
                  open={!!open[m.slug]}
                  onToggle={() => setOpen((o) => ({ ...o, [m.slug]: !o[m.slug] }))}
                />
              ))}
            </ul>
          </section>
        ))}
      </div>
    </div>
  )
}

function ModuleItem({ module: m, open, onToggle }: { module: Module; open: boolean; onToggle: () => void }) {
  const { moduleSlug, topicSlug } = useParams()
  const { done } = useStore()
  const active = moduleSlug === m.slug
  const published = modulePublished(m)
  const activeRef = useRef<HTMLAnchorElement>(null)

  const total = moduleLessons(m).length
  const completed = moduleLessons(m).filter((l) => done[l.id]).length

  useEffect(() => {
    if (active) activeRef.current?.scrollIntoView({ block: 'nearest' })
  }, [active, topicSlug])

  return (
    <li>
      <div
        className={`group flex items-center rounded-md transition-colors ${
          active ? 'bg-sunken' : 'hover:bg-sunken'
        }`}
      >
        <Link
          to={`/learn/${m.slug}`}
          className={`flex min-w-0 flex-1 items-baseline gap-2.5 py-1.5 pl-2.5 text-[13.5px] ${
            active ? 'font-medium text-fg' : published ? 'text-fg' : 'text-muted'
          }`}
        >
          <span className="w-4 shrink-0 text-right font-mono text-[11px] text-faint">{m.n}</span>
          <span className="truncate">{m.title}</span>
        </Link>
        {published ? (
          total > 0 &&
          completed > 0 && (
            <span className="shrink-0 font-mono text-[10.5px] text-faint">
              {completed}/{total}
            </span>
          )
        ) : (
          <span className="shrink-0 rounded-full border border-line px-1.5 text-[10px] font-medium text-faint">soon</span>
        )}
        <button
          onClick={onToggle}
          aria-label={open ? `Collapse ${m.title}` : `Expand ${m.title}`}
          aria-expanded={open}
          className="shrink-0 p-2 text-faint hover:text-fg"
        >
          <ChevronRight size={14} className={`transition-transform ${open ? 'rotate-90' : ''}`} />
        </button>
      </div>

      {open && (
        <ul className="relative my-1 ml-[1.15rem] border-l border-line pl-2.5">
          {m.topics.map((t) => {
            const pub = isPublished(m.slug, t.slug)
            const isActive = active && topicSlug === t.slug
            const isDone = done[lessonId(m.slug, t.slug)]
            if (!pub)
              return (
                <li key={t.slug} className="px-2.5 py-1 text-[13px] leading-snug text-faint">
                  {t.title}
                </li>
              )
            return (
              <li key={t.slug}>
                <Link
                  ref={isActive ? activeRef : undefined}
                  to={`/learn/${m.slug}/${t.slug}`}
                  className={`relative flex items-start gap-2 rounded-md px-2.5 py-1 text-[13px] leading-snug transition-colors ${
                    isActive
                      ? 'bg-accent-soft font-medium text-accent'
                      : 'text-muted hover:bg-sunken hover:text-fg'
                  }`}
                >
                  {isActive && <span className="absolute -left-[11.5px] top-1 h-5 w-px bg-accent" />}
                  <span className="flex-1">{t.title}</span>
                  {isDone && <Check size={13} className="mt-0.5 shrink-0 text-accent" />}
                </Link>
              </li>
            )
          })}
          {hasAssessment(m) && (
            <li>
              <Link
                ref={active && topicSlug === ASSESSMENT ? activeRef : undefined}
                to={`/learn/${m.slug}/${ASSESSMENT}`}
                className={`mt-1 flex items-center gap-2 rounded-md px-2.5 py-1 text-[13px] font-medium transition-colors ${
                  active && topicSlug === ASSESSMENT
                    ? 'bg-accent-soft text-accent'
                    : 'text-fg hover:bg-sunken'
                }`}
              >
                <span className="flex-1">Module quiz</span>
                {done[lessonId(m.slug, ASSESSMENT)] && <Check size={13} className="shrink-0 text-accent" />}
              </Link>
            </li>
          )}
        </ul>
      )}
    </li>
  )
}
