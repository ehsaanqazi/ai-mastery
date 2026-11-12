import { useContext, useEffect, useState } from 'react'
import { Check, RotateCcw, X } from 'lucide-react'
import { LessonContext } from '../lib/lessonContext'
import { actions, useStore } from '../lib/store'

export interface QuizQuestion {
  q: string
  options: string[]
  answer: number
  why?: string
}

/** Renders `inline code` inside quiz text. */
function Inline({ text }: { text: string }) {
  return (
    <>
      {text.split(/(`[^`]+`)/g).map((part, i) =>
        part.startsWith('`') ? (
          <code key={i} className="rounded border border-line bg-sunken px-1 font-mono text-[0.86em]">
            {part.slice(1, -1)}
          </code>
        ) : (
          part
        ),
      )}
    </>
  )
}

export default function Quiz({ questions }: { questions: QuizQuestion[] }) {
  const id = useContext(LessonContext)
  const best = useStore().quiz[id]
  const [picked, setPicked] = useState<(number | null)[]>(() => questions.map(() => null))

  const answered = picked.filter((p) => p !== null).length
  const score = picked.filter((p, i) => p === questions[i].answer).length
  const finished = answered === questions.length

  useEffect(() => {
    if (finished && id && (!best || score > best.score)) actions.setQuiz(id, score, questions.length)
  }, [finished, score, id, best, questions.length])

  return (
    <section className="overflow-hidden rounded-xl border border-line bg-surface">
      <header className="flex items-center justify-between border-b border-line bg-sunken px-5 py-2.5">
        <span className="text-[11px] font-bold uppercase tracking-[0.08em] text-accent">Check your understanding</span>
        <span className="font-mono text-xs text-muted">
          {finished ? `${score}/${questions.length} correct` : `${answered}/${questions.length} answered`}
        </span>
      </header>

      <ol className="divide-y divide-line">
        {questions.map((question, qi) => {
          const choice = picked[qi]
          const done = choice !== null
          return (
            <li key={qi} className="px-5 py-5">
              <p className="flex gap-3 text-[1rem] font-medium leading-snug">
                <span className="font-mono text-sm text-faint">{qi + 1}.</span>
                <span>
                  <Inline text={question.q} />
                </span>
              </p>
              <div className="mt-3 grid gap-2 pl-7">
                {question.options.map((opt, oi) => {
                  const correct = oi === question.answer
                  const chosen = choice === oi
                  let cls = 'border-line hover:border-faint hover:bg-sunken'
                  if (done && correct) cls = 'border-accent bg-accent-soft'
                  else if (done && chosen) cls = 'border-danger bg-danger-soft'
                  else if (done) cls = 'border-line opacity-55'
                  return (
                    <button
                      key={oi}
                      disabled={done}
                      onClick={() => setPicked((p) => p.map((v, i) => (i === qi ? oi : v)))}
                      className={`flex items-center gap-3 rounded-lg border px-3.5 py-2 text-left text-[0.95rem] leading-snug transition-colors disabled:cursor-default ${cls}`}
                    >
                      <span className="flex-1">
                        <Inline text={opt} />
                      </span>
                      {done && correct && <Check size={16} className="shrink-0 text-accent" />}
                      {done && chosen && !correct && <X size={16} className="shrink-0 text-danger" />}
                    </button>
                  )
                })}
              </div>
              {done && question.why && (
                <p className="mt-3 pl-7 text-[0.92rem] leading-relaxed text-muted">
                  <Inline text={question.why} />
                </p>
              )}
            </li>
          )
        })}
      </ol>

      {finished && (
        <footer className="flex items-center justify-between border-t border-line bg-sunken px-5 py-3 text-sm">
          <span className="text-muted">
            {score === questions.length
              ? 'Perfect score.'
              : score / questions.length >= 0.7
                ? 'Solid. Review the ones you missed.'
                : 'Worth another read of the lesson, then try again.'}
          </span>
          <button
            onClick={() => setPicked(questions.map(() => null))}
            className="flex items-center gap-1.5 font-medium text-accent hover:underline"
          >
            <RotateCcw size={13} /> Retry
          </button>
        </footer>
      )}
    </section>
  )
}
