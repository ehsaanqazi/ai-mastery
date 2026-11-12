import { useEffect, useMemo, useState } from 'react'
import { Link, useParams } from 'react-router'
import { ArrowLeft, ArrowRight, Bookmark, Check, ChevronRight, Clock, NotebookPen } from 'lucide-react'
import Markdown from '../components/Markdown'
import { findModule, pad, phaseOf } from '../data/curriculum'
import { ASSESSMENT, extractHeadings, lessons, loadLesson, moduleLessons, readingMinutes, type Heading } from '../lib/content'
import { actions, useStore } from '../lib/store'
import NotFound from './NotFound'

export default function LessonPage() {
  const { moduleSlug = '', topicSlug = '' } = useParams()
  const index = lessons.findIndex((l) => l.module.slug === moduleSlug && l.slug === topicSlug)
  const lesson = lessons[index]
  const [loaded, setLoaded] = useState<{ id: string; source: string } | null>(null)
  const { done, bookmarks } = useStore()

  useEffect(() => {
    if (!lesson) return
    let cancelled = false
    actions.setLast(lesson.id)
    loadLesson(lesson.module.slug, lesson.slug).then((source) => {
      if (!cancelled) setLoaded({ id: lesson.id, source })
    })
    return () => {
      cancelled = true
    }
  }, [lesson])

  const source = loaded && lesson && loaded.id === lesson.id ? loaded.source : null
  const headings = useMemo(() => (source ? extractHeadings(source) : []), [source])

  if (!lesson) return findModule(moduleSlug) ? <Unpublished moduleSlug={moduleSlug} /> : <NotFound />

  const m = lesson.module
  const phase = phaseOf(m)
  const siblings = moduleLessons(m)
  const position = siblings.indexOf(lesson) + 1
  const prev = lessons[index - 1]
  const next = lessons[index + 1]
  const isDone = !!done[lesson.id]
  const isBookmarked = !!bookmarks[lesson.id]
  const isQuiz = lesson.slug === ASSESSMENT

  return (
    <div className="mx-auto flex max-w-6xl justify-center gap-12 px-5 pb-24 pt-8 sm:px-8 sm:pt-10">
      <article className="min-w-0 max-w-[46rem] flex-1">
        <nav className="flex flex-wrap items-center gap-1.5 text-[13px] text-muted">
          <span className="font-mono text-xs text-faint">PHASE {pad(phase.n)}</span>
          <ChevronRight size={13} className="text-faint" />
          <Link to={`/learn/${m.slug}`} className="hover:text-accent">
            {m.title}
          </Link>
        </nav>

        <header className="mt-4">
          <h1 className="font-serif text-[2.4rem] font-medium leading-[1.12] tracking-tight sm:text-[2.9rem]">
            {isQuiz ? `${m.title}: module quiz` : lesson.title}
          </h1>
          <div className="mt-4 flex flex-wrap items-center gap-x-5 gap-y-2 text-sm text-muted">
            <span>
              Lesson {position} of {siblings.length}
            </span>
            {source && !isQuiz && (
              <span className="flex items-center gap-1.5">
                <Clock size={14} /> {readingMinutes(source)} min read
              </span>
            )}
            {isDone && (
              <span className="flex items-center gap-1.5 text-accent">
                <Check size={14} /> Completed
              </span>
            )}
            <button
              onClick={() => actions.toggleBookmark(lesson.id)}
              aria-pressed={isBookmarked}
              className={`ml-auto flex items-center gap-1.5 rounded-md border px-2.5 py-1 text-[13px] transition-colors ${
                isBookmarked ? 'border-accent bg-accent-soft text-accent' : 'border-line hover:border-faint hover:text-fg'
              }`}
            >
              <Bookmark size={14} fill={isBookmarked ? 'currentColor' : 'none'} />
              {isBookmarked ? 'Bookmarked' : 'Bookmark'}
            </button>
          </div>
        </header>

        <div className="mt-8 min-h-[50vh] border-t border-line pt-8">
          {source ? <Markdown source={source} id={lesson.id} /> : <Skeleton />}
        </div>

        {source && (
          <>
            <Notes id={lesson.id} />

            <div className="mt-8 flex flex-wrap items-center justify-between gap-4 rounded-xl border border-line bg-surface p-5">
              <div>
                <p className="font-medium">{isDone ? 'Lesson completed' : 'Finished this lesson?'}</p>
                <p className="text-sm text-muted">Your progress is saved in this browser.</p>
              </div>
              <button
                onClick={() => actions.toggleDone(lesson.id)}
                className={`inline-flex items-center gap-2 rounded-lg px-4 py-2 text-sm font-semibold transition-colors ${
                  isDone
                    ? 'border border-accent bg-accent-soft text-accent'
                    : 'bg-accent text-accent-fg hover:opacity-90'
                }`}
              >
                <Check size={15} /> {isDone ? 'Completed' : 'Mark as complete'}
              </button>
            </div>

            <nav className="mt-4 grid gap-4 sm:grid-cols-2">
              {prev ? (
                <Link to={prev.path} className="group rounded-xl border border-line bg-surface p-4 hover:border-accent">
                  <span className="flex items-center gap-1.5 text-xs text-faint">
                    <ArrowLeft size={13} /> Previous
                  </span>
                  <span className="mt-1 block font-medium group-hover:text-accent">{prev.title}</span>
                </Link>
              ) : (
                <span />
              )}
              {next && (
                <Link
                  to={next.path}
                  onClick={() => actions.markDone(lesson.id)}
                  className="group rounded-xl border border-line bg-surface p-4 text-right hover:border-accent"
                >
                  <span className="flex items-center justify-end gap-1.5 text-xs text-faint">
                    Next <ArrowRight size={13} />
                  </span>
                  <span className="mt-1 block font-medium group-hover:text-accent">{next.title}</span>
                </Link>
              )}
            </nav>
          </>
        )}
      </article>

      <aside className="hidden w-56 shrink-0 xl:block">
        {headings.length > 1 && <Toc headings={headings} />}
      </aside>
    </div>
  )
}

function Toc({ headings }: { headings: Heading[] }) {
  const [active, setActive] = useState(headings[0]?.id)

  useEffect(() => {
    const onScroll = () => {
      let current = headings[0]?.id
      for (const h of headings) {
        const el = document.getElementById(h.id)
        if (el && el.getBoundingClientRect().top < 120) current = h.id
      }
      setActive(current)
    }
    onScroll()
    window.addEventListener('scroll', onScroll, { passive: true })
    return () => window.removeEventListener('scroll', onScroll)
  }, [headings])

  return (
    <nav className="thin-scroll sticky top-24 max-h-[calc(100vh-8rem)] overflow-y-auto">
      <p className="mb-3 text-[11px] font-semibold uppercase tracking-wider text-faint">On this page</p>
      <ul className="border-l border-line text-[13px]">
        {headings.map((h) => (
          <li key={h.id}>
            <a
              href={`#${h.id}`}
              className={`-ml-px block border-l py-1 leading-snug transition-colors ${h.depth === 3 ? 'pl-7' : 'pl-4'} ${
                active === h.id
                  ? 'border-accent font-medium text-accent'
                  : 'border-transparent text-muted hover:text-fg'
              }`}
            >
              {h.text}
            </a>
          </li>
        ))}
      </ul>
    </nav>
  )
}

function Notes({ id }: { id: string }) {
  const saved = useStore().notes[id] ?? ''
  const [open, setOpen] = useState(!!saved)

  useEffect(() => setOpen(!!saved), [id]) // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <section className="mt-14 rounded-xl border border-line bg-surface">
      <button
        onClick={() => setOpen((o) => !o)}
        className="flex w-full items-center gap-2.5 px-5 py-3.5 text-left text-sm font-medium"
        aria-expanded={open}
      >
        <NotebookPen size={16} className="text-accent" />
        My notes
        {saved && !open && <span className="text-xs font-normal text-faint">· saved</span>}
        <ChevronRight size={15} className={`ml-auto text-faint transition-transform ${open ? 'rotate-90' : ''}`} />
      </button>
      {open && (
        <div className="border-t border-line p-3">
          <textarea
            key={id}
            defaultValue={saved}
            onChange={(e) => actions.setNote(id, e.target.value)}
            placeholder="Write down what clicked, what confused you, or questions to revisit. Saved automatically."
            rows={5}
            className="block w-full resize-y rounded-lg bg-sunken p-3 text-[15px] leading-relaxed outline-none placeholder:text-faint focus:ring-1 focus:ring-accent"
          />
        </div>
      )}
    </section>
  )
}

function Skeleton() {
  return (
    <div className="animate-pulse space-y-4" aria-label="Loading lesson">
      {[100, 92, 96, 60].map((w, i) => (
        <div key={i} className="h-4 rounded bg-sunken" style={{ width: `${w}%` }} />
      ))}
      <div className="h-40 rounded-xl bg-sunken" />
    </div>
  )
}

function Unpublished({ moduleSlug }: { moduleSlug: string }) {
  return (
    <div className="mx-auto max-w-xl px-5 py-24 text-center">
      <h1 className="font-serif text-3xl font-medium">This lesson isn't published yet</h1>
      <p className="mt-3 text-muted">It's on the syllabus and will appear here once it's written.</p>
      <Link to={`/learn/${moduleSlug}`} className="mt-6 inline-block font-medium text-accent hover:underline">
        Back to the module
      </Link>
    </div>
  )
}
