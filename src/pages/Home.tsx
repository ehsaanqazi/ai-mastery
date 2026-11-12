import { Link } from 'react-router'
import { ArrowRight, Check } from 'lucide-react'
import { modules, pad, phases, totalTopics } from '../data/curriculum'
import { lessons, modulePublished } from '../lib/content'
import { useStore } from '../lib/store'

export default function Home() {
  const { done, last } = useStore()
  const completed = lessons.filter((l) => done[l.id]).length
  const resume = lessons.find((l) => l.id === last) ?? lessons.find((l) => !done[l.id]) ?? lessons[0]
  const started = completed > 0 || !!last

  return (
    <div className="mx-auto max-w-5xl px-5 pb-24 sm:px-8">
      <section className="pb-14 pt-14 sm:pt-20">
        <p className="font-mono text-xs font-semibold uppercase tracking-[0.14em] text-accent">
          The complete AI engineering curriculum
        </p>
        <h1 className="mt-4 max-w-3xl font-serif text-[2.6rem] font-medium leading-[1.08] tracking-tight sm:text-[3.6rem]">
          From your first line of Python to production AI systems.
        </h1>
        <p className="mt-5 max-w-2xl text-lg leading-relaxed text-muted">
          A structured, textbook-style path through programming, mathematics, machine learning, LLMs, RAG, agents and
          deployment. Read, run the code, test yourself, move on.
        </p>

        <div className="mt-8 flex flex-wrap items-center gap-3">
          {resume && (
            <Link
              to={resume.path}
              className="inline-flex items-center gap-2 rounded-lg bg-accent px-5 py-2.5 text-[15px] font-semibold text-accent-fg transition-opacity hover:opacity-90"
            >
              {started ? 'Continue learning' : 'Start learning'} <ArrowRight size={16} />
            </Link>
          )}
          <Link
            to="/roadmap"
            className="inline-flex items-center gap-2 rounded-lg border border-line bg-surface px-5 py-2.5 text-[15px] font-medium hover:border-faint"
          >
            View the roadmap
          </Link>
        </div>

        <dl className="mt-12 grid max-w-2xl grid-cols-2 gap-px overflow-hidden rounded-xl border border-line bg-line sm:grid-cols-4">
          {[
            [phases.length, 'phases'],
            [modules.length, 'modules'],
            [totalTopics, 'topics'],
            ['18', 'months'],
          ].map(([n, label]) => (
            <div key={label} className="bg-surface px-5 py-4">
              <dt className="font-serif text-3xl font-medium">{n}</dt>
              <dd className="text-sm text-muted">{label}</dd>
            </div>
          ))}
        </dl>
      </section>

      {started && resume && (
        <section className="mb-14 flex flex-wrap items-center gap-x-8 gap-y-4 rounded-xl border border-line bg-surface p-5">
          <div className="min-w-0 flex-1 basis-64">
            <p className="text-xs font-semibold uppercase tracking-wider text-faint">Pick up where you left off</p>
            <Link to={resume.path} className="mt-1 block truncate font-serif text-xl font-medium hover:text-accent">
              {resume.title}
            </Link>
            <p className="text-sm text-muted">{resume.module.title}</p>
          </div>
          <div className="basis-56">
            <div className="flex justify-between text-xs text-muted">
              <span>Published lessons completed</span>
              <span className="font-mono">
                {completed}/{lessons.length}
              </span>
            </div>
            <div className="mt-2 h-1.5 overflow-hidden rounded-full bg-sunken">
              <div
                className="h-full rounded-full bg-accent transition-all"
                style={{ width: `${lessons.length ? (completed / lessons.length) * 100 : 0}%` }}
              />
            </div>
          </div>
        </section>
      )}

      <section>
        <h2 className="font-serif text-2xl font-medium">The curriculum</h2>
        <p className="mt-1 text-muted">Twelve phases, taken in order. Each one builds on the last.</p>

        <div className="mt-8 space-y-10">
          {phases.map((phase) => (
            <div key={phase.n} className="grid gap-x-10 gap-y-4 border-t border-line pt-6 md:grid-cols-[15rem_1fr]">
              <div>
                <p className="font-mono text-xs font-semibold tracking-wider text-accent">
                  PHASE {pad(phase.n)} <span className="text-faint">· {phase.months}</span>
                </p>
                <h3 className="mt-1.5 font-serif text-xl font-medium leading-snug">{phase.title}</h3>
                <p className="mt-1.5 text-sm leading-relaxed text-muted">{phase.blurb}</p>
              </div>
              <ul className="grid content-start gap-2.5 sm:grid-cols-2">
                {phase.modules.map((m) => {
                  const published = modulePublished(m)
                  const mine = lessons.filter((l) => l.module === m)
                  const allDone = mine.length > 0 && mine.every((l) => done[l.id])
                  return (
                    <li key={m.slug}>
                      <Link
                        to={`/learn/${m.slug}`}
                        className="group flex h-full gap-3 rounded-xl border border-line bg-surface p-4 transition-colors hover:border-accent"
                      >
                        <span className="font-mono text-xs text-faint">{pad(m.n)}</span>
                        <span className="min-w-0 flex-1">
                          <span className="flex items-center gap-2">
                            <span className="font-medium leading-snug group-hover:text-accent">{m.title}</span>
                            {allDone && <Check size={14} className="shrink-0 text-accent" />}
                          </span>
                          <span className="mt-1 block text-[13px] leading-snug text-muted">{m.blurb}</span>
                          <span className="mt-2 block text-xs text-faint">
                            {m.topics.length} topics{!published && ' · coming soon'}
                          </span>
                        </span>
                      </Link>
                    </li>
                  )
                })}
              </ul>
            </div>
          ))}
        </div>
      </section>
    </div>
  )
}
