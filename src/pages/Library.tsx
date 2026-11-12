import { Link } from 'react-router'
import { Bookmark, NotebookPen, X } from 'lucide-react'
import { lessons } from '../lib/content'
import { actions, useStore } from '../lib/store'

export default function Library() {
  const { bookmarks, notes, done } = useStore()
  const saved = lessons.filter((l) => bookmarks[l.id])
  const noted = lessons.filter((l) => notes[l.id])
  const completed = lessons.filter((l) => done[l.id]).length

  return (
    <div className="mx-auto max-w-3xl px-5 pb-24 pt-10 sm:px-8 sm:pt-14">
      <p className="font-mono text-xs font-semibold uppercase tracking-[0.14em] text-accent">My library</p>
      <h1 className="mt-3 font-serif text-4xl font-medium tracking-tight sm:text-5xl">Bookmarks and notes</h1>
      <p className="mt-4 text-lg leading-relaxed text-muted">
        Everything here is stored in this browser only. You've completed{' '}
        <span className="font-medium text-fg">
          {completed} of {lessons.length}
        </span>{' '}
        published lessons.
      </p>

      <h2 className="mt-12 flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-faint">
        <Bookmark size={14} /> Bookmarks
      </h2>
      {saved.length === 0 ? (
        <Empty>Bookmark a lesson to find it here later.</Empty>
      ) : (
        <ul className="mt-3 overflow-hidden rounded-xl border border-line bg-surface">
          {saved.map((l) => (
            <li key={l.id} className="flex items-center border-b border-line last:border-0">
              <Link to={l.path} className="min-w-0 flex-1 px-4 py-3 hover:bg-sunken">
                <span className="block truncate font-medium">{l.title}</span>
                <span className="block text-sm text-muted">{l.module.title}</span>
              </Link>
              <button
                onClick={() => actions.toggleBookmark(l.id)}
                aria-label={`Remove bookmark for ${l.title}`}
                className="p-3 text-faint hover:text-danger"
              >
                <X size={16} />
              </button>
            </li>
          ))}
        </ul>
      )}

      <h2 className="mt-12 flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-faint">
        <NotebookPen size={14} /> Notes
      </h2>
      {noted.length === 0 ? (
        <Empty>Notes you write at the bottom of a lesson are collected here.</Empty>
      ) : (
        <ul className="mt-3 space-y-3">
          {noted.map((l) => (
            <li key={l.id} className="rounded-xl border border-line bg-surface p-4">
              <Link to={l.path} className="font-medium hover:text-accent">
                {l.title}
              </Link>
              <span className="ml-2 text-sm text-faint">{l.module.title}</span>
              <p className="mt-2 whitespace-pre-wrap text-[15px] leading-relaxed text-muted">{notes[l.id]}</p>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}

function Empty({ children }: { children: React.ReactNode }) {
  return (
    <p className="mt-3 rounded-xl border border-dashed border-line px-4 py-8 text-center text-sm text-muted">
      {children}
    </p>
  )
}
