import { useState } from 'react'
import { ChevronLeft, ChevronRight, RotateCcw } from 'lucide-react'

export interface TraceData {
  title?: string
  code: string[]
  steps: { line: number; vars?: Record<string, string>; out?: string[]; note?: string }[]
}

/** Interactive visualizer: step through code line by line and watch variables change. */
export default function Trace({ data }: { data: TraceData }) {
  const [i, setI] = useState(0)
  const step = data.steps[i]
  const prev = i > 0 ? data.steps[i - 1] : undefined
  const vars = Object.entries(step.vars ?? {})
  const last = data.steps.length - 1

  return (
    <section className="overflow-hidden rounded-xl border border-line bg-surface">
      <header className="flex items-center justify-between border-b border-line bg-sunken px-5 py-2.5">
        <span className="text-[11px] font-bold uppercase tracking-[0.08em] text-accent">
          Step through it{data.title ? ` · ${data.title}` : ''}
        </span>
        <span className="font-mono text-xs text-muted">
          step {i + 1}/{data.steps.length}
        </span>
      </header>

      <div className="grid md:grid-cols-[1.25fr_1fr]">
        <pre className="thin-scroll overflow-x-auto bg-code py-3 font-mono text-[13.5px] leading-[1.9]">
          {data.code.map((line, n) => {
            const active = n + 1 === step.line
            return (
              <div
                key={n}
                className={`flex border-l-2 pr-4 transition-colors ${
                  active ? 'border-accent bg-accent-soft' : 'border-transparent'
                }`}
              >
                <span className={`w-9 shrink-0 select-none pr-3 text-right text-[11px] ${active ? 'text-accent' : 'text-faint'}`}>
                  {n + 1}
                </span>
                <span className="whitespace-pre">{line || ' '}</span>
              </div>
            )
          })}
        </pre>

        <div className="flex flex-col gap-4 border-t border-line p-4 md:border-l md:border-t-0">
          <div>
            <h4 className="mb-1.5 text-[10.5px] font-bold uppercase tracking-wider text-faint">Variables</h4>
            {vars.length === 0 ? (
              <p className="text-sm text-faint">Nothing defined yet.</p>
            ) : (
              <dl className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-1 font-mono text-[13px]">
                {vars.map(([k, v]) => {
                  const changed = prev?.vars?.[k] !== v
                  return (
                    <div key={k} className="contents">
                      <dt className="text-muted">{k}</dt>
                      <dd className={`break-all rounded px-1.5 ${changed ? 'bg-accent-soft text-accent' : ''}`}>{v}</dd>
                    </div>
                  )
                })}
              </dl>
            )}
          </div>
          <div>
            <h4 className="mb-1.5 text-[10.5px] font-bold uppercase tracking-wider text-faint">Output</h4>
            <pre className="min-h-9 rounded-md bg-sunken px-2.5 py-1.5 font-mono text-[13px] leading-relaxed">
              {step.out?.length ? step.out.join('\n') : <span className="text-faint">(nothing printed yet)</span>}
            </pre>
          </div>
        </div>
      </div>

      <footer className="flex flex-wrap items-center gap-3 border-t border-line px-4 py-3">
        <div className="flex gap-1.5">
          <button
            onClick={() => setI(0)}
            disabled={i === 0}
            aria-label="Restart"
            className="rounded-md border border-line p-1.5 text-muted hover:text-fg disabled:opacity-40"
          >
            <RotateCcw size={15} />
          </button>
          <button
            onClick={() => setI(i - 1)}
            disabled={i === 0}
            aria-label="Previous step"
            className="rounded-md border border-line p-1.5 text-muted hover:text-fg disabled:opacity-40"
          >
            <ChevronLeft size={15} />
          </button>
          <button
            onClick={() => setI(i + 1)}
            disabled={i === last}
            className="flex items-center gap-1 rounded-md bg-accent px-3 py-1.5 text-sm font-medium text-accent-fg disabled:opacity-40"
          >
            Next <ChevronRight size={15} />
          </button>
        </div>
        <p className="min-w-0 flex-1 basis-56 text-[0.92rem] leading-snug text-muted">{step.note}</p>
      </footer>
    </section>
  )
}
