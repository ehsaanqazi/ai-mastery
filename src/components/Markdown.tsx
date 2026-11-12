import { isValidElement, memo, type ReactNode } from 'react'
import ReactMarkdown from 'react-markdown'
import { Link } from 'react-router'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'
import rehypeRaw from 'rehype-raw'
import rehypeHighlight from 'rehype-highlight'
import rehypeSlug from 'rehype-slug'
import CodeBlock from './CodeBlock'
import Mermaid from './Mermaid'
import Quiz, { type QuizQuestion } from './Quiz'
import Trace, { type TraceData } from './Trace'

import { LessonContext } from '../lib/lessonContext'

const DATA_BLOCKS = ['mermaid', 'quiz', 'trace', 'output', 'text', 'csv']

const CALLOUT_LABELS: Record<string, string> = {
  note: 'Note',
  tip: 'Tip',
  warning: 'Watch out',
  key: 'Key idea',
}

/** Turns `> [!TIP]` blockquotes into styled callouts. */
function remarkCallouts() {
  return (tree: any) => {
    const walk = (node: any) => {
      if (node.type === 'blockquote') {
        const first = node.children?.[0]?.children?.[0]
        const m = first?.type === 'text' && /^\[!(\w+)\]\s*/.exec(first.value)
        if (m) {
          const kind = m[1].toLowerCase()
          first.value = first.value.slice(m[0].length)
          node.data = {
            hProperties: {
              className: ['callout'],
              'data-callout': kind,
              'data-label': CALLOUT_LABELS[kind] ?? kind,
            },
          }
        }
      }
      node.children?.forEach(walk)
    }
    walk(tree)
  }
}

function textOf(node: ReactNode): string {
  if (node == null || typeof node === 'boolean') return ''
  if (typeof node === 'string' || typeof node === 'number') return String(node)
  if (Array.isArray(node)) return node.map(textOf).join('')
  if (isValidElement(node)) return textOf((node.props as { children?: ReactNode }).children)
  return ''
}

function parse<T>(src: string): T | null {
  try {
    return JSON.parse(src) as T
  } catch (e) {
    console.error('Invalid JSON in lesson block', e)
    return null
  }
}

function Pre({ children }: { children?: ReactNode }) {
  const child = Array.isArray(children) ? children[0] : children
  if (!isValidElement(child)) return <pre>{children}</pre>
  const props = child.props as { className?: string; children?: ReactNode }
  const lang = /language-([\w-]+)/.exec(props.className ?? '')?.[1] ?? ''
  const text = textOf(props.children).replace(/\n$/, '')

  if (lang === 'mermaid') return <Mermaid code={text} />
  if (lang === 'quiz') {
    const data = parse<QuizQuestion[]>(text)
    return data ? <Quiz questions={data} /> : null
  }
  if (lang === 'trace') {
    const data = parse<TraceData>(text)
    return data ? <Trace data={data} /> : null
  }
  return (
    <CodeBlock lang={lang} text={text}>
      {props.children}
    </CodeBlock>
  )
}

function Markdown({ source, id }: { source: string; id: string }) {
  return (
    <LessonContext.Provider value={id}>
      <div className="prose">
        <ReactMarkdown
          remarkPlugins={[remarkGfm, remarkMath, remarkCallouts]}
          rehypePlugins={[rehypeRaw, rehypeSlug, rehypeKatex, [rehypeHighlight, { plainText: DATA_BLOCKS }]]}
          components={{
            pre: Pre,
            table: ({ children }) => (
              <div className="table-wrap">
                <table>{children}</table>
              </div>
            ),
            a: ({ href = '', children }) =>
              href.startsWith('/') ? (
                <Link to={href}>{children}</Link>
              ) : (
                <a href={href} target="_blank" rel="noreferrer">
                  {children}
                </a>
              ),
          }}
        >
          {source}
        </ReactMarkdown>
      </div>
    </LessonContext.Provider>
  )
}

export default memo(Markdown)
