import GithubSlugger from 'github-slugger'
import { modules, type Module } from '../data/curriculum'

const files = import.meta.glob('../content/**/*.md', {
  query: '?raw',
  import: 'default',
}) as Record<string, () => Promise<string>>

const key = (moduleSlug: string, topicSlug: string) => `../content/${moduleSlug}/${topicSlug}.md`

export const ASSESSMENT = 'assessment'

export const isPublished = (moduleSlug: string, topicSlug: string) => key(moduleSlug, topicSlug) in files

export const modulePublished = (m: Module) => m.topics.some((t) => isPublished(m.slug, t.slug))

export const hasAssessment = (m: Module) => isPublished(m.slug, ASSESSMENT)

export function loadLesson(moduleSlug: string, topicSlug: string): Promise<string> {
  const loader = files[key(moduleSlug, topicSlug)]
  return loader ? loader() : Promise.reject(new Error('Lesson not found'))
}

export const lessonId = (moduleSlug: string, topicSlug: string) => `${moduleSlug}/${topicSlug}`

export interface LessonRef {
  id: string
  module: Module
  slug: string
  title: string
  path: string
}

/** Every published lesson, in curriculum order (module quiz last in each module). */
export const lessons: LessonRef[] = modules.flatMap((m) => {
  const list: LessonRef[] = m.topics
    .filter((t) => isPublished(m.slug, t.slug))
    .map((t) => ({
      id: lessonId(m.slug, t.slug),
      module: m,
      slug: t.slug,
      title: t.title,
      path: `/learn/${m.slug}/${t.slug}`,
    }))
  if (hasAssessment(m)) {
    list.push({
      id: lessonId(m.slug, ASSESSMENT),
      module: m,
      slug: ASSESSMENT,
      title: 'Module quiz',
      path: `/learn/${m.slug}/${ASSESSMENT}`,
    })
  }
  return list
})

export const moduleLessons = (m: Module) => lessons.filter((l) => l.module === m)

export interface Heading {
  id: string
  text: string
  depth: number
}

/** Extract h2/h3 headings using the same slug rules as rehype-slug. */
export function extractHeadings(md: string): Heading[] {
  const slugger = new GithubSlugger()
  const out: Heading[] = []
  let fence: string | null = null
  for (const line of md.split('\n')) {
    const marker = /^(```|~~~)/.exec(line)?.[1]
    if (marker && (fence === null || marker === fence)) {
      fence = fence === null ? marker : null
      continue
    }
    if (fence) continue
    const m = /^(#{2,3})\s+(.+?)\s*$/.exec(line)
    if (!m) continue
    const text = m[2].replace(/[`*_]/g, '')
    out.push({ id: slugger.slug(text), text, depth: m[1].length })
  }
  return out
}

export function readingMinutes(md: string): number {
  const words = md.split(/\s+/).length
  return Math.max(3, Math.round(words / 190))
}

/** Plain text of a lesson for search (code fences with data blocks removed). */
export function searchableText(md: string): string {
  return md
    .replace(/```(quiz|trace|mermaid)[\s\S]*?```/g, ' ')
    .replace(/```\w*\n?/g, ' ')
    .replace(/<[^>]+>/g, ' ')
    .replace(/[#>*_`$|\\[\]]/g, ' ')
    .replace(/\s+/g, ' ')
}
