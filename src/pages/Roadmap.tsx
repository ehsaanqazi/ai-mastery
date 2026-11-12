import { Link } from 'react-router'
import { pad, phases } from '../data/curriculum'
import { lessons, modulePublished } from '../lib/content'
import { useStore } from '../lib/store'

export default function Roadmap() {
  const { done } = useStore()

  return (
    <div className="mx-auto max-w-3xl px-5 pb-24 pt-10 sm:px-8 sm:pt-14">
      <p className="font-mono text-xs font-semibold uppercase tracking-[0.14em] text-accent">Learning roadmap</p>
      <h1 className="mt-3 font-serif text-4xl font-medium tracking-tight sm:text-5xl">The 18-month path</h1>
      <p className="mt-4 text-lg leading-relaxed text-muted">
        Twelve phases that take you from beginner programmer to AI systems engineer. Phases overlap slightly on purpose:
        you start the next one while finishing the last.
      </p>

      <ol className="mt-12">
        {phases.map((phase, i) => {
          const mine = lessons.filter((l) => l.module.phase === phase.n)
          const complete = mine.length > 0 && mine.every((l) => done[l.id])
          const started = mine.some((l) => done[l.id])
          return (
            <li key={phase.n} className="relative grid grid-cols-[2.5rem_1fr] gap-x-5 pb-10">
              {i < phases.length - 1 && <span className="absolute bottom-0 left-5 top-10 w-px -translate-x-1/2 bg-line" />}
              <span
                className={`relative grid h-10 w-10 place-items-center rounded-full border font-mono text-sm font-semibold ${
                  complete
                    ? 'border-accent bg-accent text-accent-fg'
                    : started
                      ? 'border-accent bg-accent-soft text-accent'
                      : 'border-line bg-surface text-muted'
                }`}
              >
                {pad(phase.n)}
              </span>
              <div className="pt-1.5">
                <p className="font-mono text-xs text-faint">{phase.months}</p>
                <h2 className="mt-0.5 font-serif text-2xl font-medium leading-snug">{phase.title}</h2>
                <p className="mt-1.5 leading-relaxed text-muted">{phase.blurb}</p>
                <ul className="mt-4 flex flex-wrap gap-2">
                  {phase.modules.map((m) => (
                    <li key={m.slug}>
                      <Link
                        to={`/learn/${m.slug}`}
                        className={`inline-flex items-center gap-2 rounded-full border px-3 py-1 text-[13px] transition-colors hover:border-accent hover:text-accent ${
                          modulePublished(m) ? 'border-line bg-surface' : 'border-dashed border-line text-muted'
                        }`}
                      >
                        <span className="font-mono text-[11px] text-faint">{m.n}</span>
                        {m.title}
                      </Link>
                    </li>
                  ))}
                </ul>
              </div>
            </li>
          )
        })}
      </ol>
    </div>
  )
}
