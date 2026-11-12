import { Link, useParams } from 'react-router'
import { ArrowRight, Check, ClipboardCheck } from 'lucide-react'
import { findModule, modules, pad, phaseOf } from '../data/curriculum'
import { ASSESSMENT, hasAssessment, isPublished, lessonId, moduleLessons, modulePublished } from '../lib/content'
import { useStore } from '../lib/store'
import NotFound from './NotFound'

export default function ModulePage() {
  const { moduleSlug } = useParams()
  const m = findModule(moduleSlug)
  const { done, quiz } = useStore()
  if (!m) return <NotFound />

  const phase = phaseOf(m)
  const published = modulePublished(m)
  const mine = moduleLessons(m)
  const completed = mine.filter((l) => done[l.id]).length
  const next = mine.find((l) => !done[l.id]) ?? mine[0]
  const nextModule = modules[m.n]
  const quizScore = quiz[lessonId(m.slug, ASSESSMENT)]

  return (
    <div className="mx-auto max-w-3xl px-5 pb-24 pt-10 sm:px-8 sm:pt-14">
      <p className="font-mono text-xs font-semibold tracking-wider text-accent">
        PHASE {pad(phase.n)} <span className="text-faint">· {phase.title} · {phase.months}</span>
      </p>
      <h1 className="mt-3 font-serif text-4xl font-medium leading-tight tracking-tight sm:text-5xl">
        <span className="mr-3 text-faint">{pad(m.n)}</span>
        {m.title}
      </h1>
      <p className="mt-4 text-lg leading-relaxed text-muted">{m.blurb}</p>

      {published ? (
        <div className="mt-8 flex flex-wrap items-center gap-x-6 gap-y-4 rounded-xl border border-line bg-surface p-5">
          <div className="min-w-0 flex-1 basis-52">
            <div className="flex justify-between text-sm">
              <span className="text-muted">Your progress</span>
              <span className="font-mono text-muted">
                {completed}/{mine.length}
              </span>
            </div>
            <div className="mt-2 h-1.5 overflow-hidden rounded-full bg-sunken">
              <div
                className="h-full rounded-full bg-accent transition-all"
                style={{ width: `${(completed / mine.length) * 100}%` }}
              />
            </div>
          </div>
          {next && (
            <Link
              to={next.path}
              className="inline-flex items-center gap-2 rounded-lg bg-accent px-4 py-2 text-sm font-semibold text-accent-fg hover:opacity-90"
            >
              {completed === 0 ? 'Start module' : completed === mine.length ? 'Review' : 'Continue'}
              <ArrowRight size={15} />
            </Link>
          )}
        </div>
      ) : (
        <div className="mt-8 rounded-xl border border-dashed border-line bg-surface p-5 text-[15px] leading-relaxed text-muted">
          <span className="font-semibold text-fg">Lessons for this module are being written.</span> The syllabus below
          shows what it will cover.
        </div>
      )}

      <h2 className="mt-12 text-xs font-semibold uppercase tracking-wider text-faint">
        {published ? 'Lessons' : 'Syllabus'}
      </h2>
      <ol className="mt-3 overflow-hidden rounded-xl border border-line bg-surface">
        {m.topics.map((t, i) => {
          const pub = isPublished(m.slug, t.slug)
          const isDone = done[lessonId(m.slug, t.slug)]
          const row = (
            <>
              <span
                className={`grid h-6 w-6 shrink-0 place-items-center rounded-full border font-mono text-[11px] ${
                  isDone ? 'border-accent bg-accent text-accent-fg' : 'border-line text-faint'
                }`}
              >
                {isDone ? <Check size={13} /> : i + 1}
              </span>
              <span className={`flex-1 ${pub ? '' : 'text-muted'}`}>{t.title}</span>
              {pub && <ArrowRight size={15} className="text-faint transition-transform group-hover:translate-x-0.5 group-hover:text-accent" />}
            </>
          )
          return (
            <li key={t.slug} className="border-b border-line last:border-0">
              {pub ? (
                <Link to={`/learn/${m.slug}/${t.slug}`} className="group flex items-center gap-3.5 px-4 py-3 hover:bg-sunken">
                  {row}
                </Link>
              ) : (
                <div className="flex items-center gap-3.5 px-4 py-3">{row}</div>
              )}
            </li>
          )
        })}
      </ol>

      {hasAssessment(m) && (
        <Link
          to={`/learn/${m.slug}/${ASSESSMENT}`}
          className="group mt-4 flex items-center gap-4 rounded-xl border border-line bg-surface p-4 hover:border-accent"
        >
          <span className="grid h-10 w-10 place-items-center rounded-lg bg-accent-soft text-accent">
            <ClipboardCheck size={19} />
          </span>
          <span className="flex-1">
            <span className="block font-medium group-hover:text-accent">Module quiz</span>
            <span className="block text-sm text-muted">
              {quizScore ? `Best score: ${quizScore.score}/${quizScore.total}` : 'Test everything from this module.'}
            </span>
          </span>
          <ArrowRight size={16} className="text-faint group-hover:text-accent" />
        </Link>
      )}

      {nextModule && (
        <p className="mt-12 border-t border-line pt-6 text-sm text-muted">
          Up next:{' '}
          <Link to={`/learn/${nextModule.slug}`} className="font-medium text-accent hover:underline">
            {pad(nextModule.n)} · {nextModule.title}
          </Link>
        </p>
      )}
    </div>
  )
}
