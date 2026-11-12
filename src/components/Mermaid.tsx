import { useEffect, useId, useState } from 'react'
import { useTheme } from '../lib/store'

const palettes = {
  light: {
    primaryColor: '#ffffff',
    primaryTextColor: '#1c1b18',
    primaryBorderColor: '#0d7a5f',
    lineColor: '#68645b',
    secondaryColor: '#e2f1eb',
    tertiaryColor: '#f4f2ec',
    background: '#fbfaf7',
    mainBkg: '#ffffff',
    clusterBkg: '#f4f2ec',
    clusterBorder: '#e6e2d8',
    edgeLabelBackground: '#fbfaf7',
    noteBkgColor: '#fbf0d9',
    noteTextColor: '#1c1b18',
    noteBorderColor: '#e6e2d8',
    git0: '#0d7a5f',
    git1: '#2b5fb3',
    git2: '#a15c07',
    git3: '#a3296b',
    gitBranchLabel0: '#ffffff',
    gitBranchLabel1: '#ffffff',
    gitBranchLabel2: '#ffffff',
    gitBranchLabel3: '#ffffff',
    commitLabelColor: '#1c1b18',
    commitLabelBackground: '#f4f2ec',
  },
  dark: {
    primaryColor: '#191b18',
    primaryTextColor: '#ecebe6',
    primaryBorderColor: '#4fd1a5',
    lineColor: '#a09e95',
    secondaryColor: '#14302a',
    tertiaryColor: '#0b0c0a',
    background: '#111210',
    mainBkg: '#191b18',
    clusterBkg: '#151714',
    clusterBorder: '#2a2d28',
    edgeLabelBackground: '#111210',
    noteBkgColor: '#33270f',
    noteTextColor: '#ecebe6',
    noteBorderColor: '#2a2d28',
    git0: '#4fd1a5',
    git1: '#7fb0f5',
    git2: '#e8b04b',
    git3: '#f08fc0',
    gitBranchLabel0: '#06261c',
    gitBranchLabel1: '#0b1b33',
    gitBranchLabel2: '#2b1d05',
    gitBranchLabel3: '#330f22',
    commitLabelColor: '#ecebe6',
    commitLabelBackground: '#191b18',
  },
}

// mermaid keeps global state, so diagrams must render one at a time
let queue: Promise<unknown> = Promise.resolve()

function renderDiagram(id: string, code: string, theme: 'light' | 'dark'): Promise<string> {
  const job = queue.then(async () => {
    const { default: mermaid } = await import('mermaid')
    mermaid.initialize({
      startOnLoad: false,
      theme: 'base',
      look: 'classic',
      fontFamily: "'Inter Variable', system-ui, sans-serif",
      themeVariables: { ...palettes[theme], fontSize: '14px' },
    })
    const { svg } = await mermaid.render(id, code)
    return svg
  })
  queue = job.catch(() => {})
  return job
}

export default function Mermaid({ code }: { code: string }) {
  const theme = useTheme()
  const id = 'm' + useId().replace(/[^a-zA-Z0-9]/g, '')
  const [svg, setSvg] = useState('')
  const [error, setError] = useState(false)

  useEffect(() => {
    let cancelled = false
    renderDiagram(`${id}-${theme}`, code, theme)
      .then((svg) => {
        if (!cancelled) {
          setSvg(svg)
          setError(false)
        }
      })
      .catch((e) => {
        console.error('Mermaid render failed', e)
        if (!cancelled) setError(true)
      })
    return () => {
      cancelled = true
    }
  }, [code, theme, id])

  if (error)
    return (
      <pre data-mermaid-error className="overflow-x-auto rounded-xl border border-line bg-code p-4 font-mono text-xs text-muted">{code}</pre>
    )

  return (
    <figure
      className="mermaid-figure flex min-h-24 justify-center overflow-x-auto rounded-xl border border-line bg-surface px-4 py-6"
      dangerouslySetInnerHTML={{ __html: svg }}
    />
  )
}
