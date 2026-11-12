import { useSyncExternalStore } from 'react'

interface State {
  done: Record<string, true>
  bookmarks: Record<string, true>
  notes: Record<string, string>
  quiz: Record<string, { score: number; total: number }>
  last?: string
}

const KEY = 'ai-mastery:v1'
const empty: State = { done: {}, bookmarks: {}, notes: {}, quiz: {} }

function read(): State {
  try {
    const raw = localStorage.getItem(KEY)
    return raw ? { ...empty, ...JSON.parse(raw) } : empty
  } catch {
    return empty
  }
}

let state = read()
const listeners = new Set<() => void>()

function set(next: State) {
  state = next
  try {
    localStorage.setItem(KEY, JSON.stringify(state))
  } catch {
    /* storage unavailable: keep in memory */
  }
  listeners.forEach((l) => l())
}

const subscribe = (l: () => void) => {
  listeners.add(l)
  return () => listeners.delete(l)
}

export function useStore(): State {
  return useSyncExternalStore(subscribe, () => state)
}

function toggle(map: Record<string, true>, id: string): Record<string, true> {
  const next = { ...map }
  if (next[id]) delete next[id]
  else next[id] = true
  return next
}

export const actions = {
  toggleDone: (id: string) => set({ ...state, done: toggle(state.done, id) }),
  markDone: (id: string) => {
    if (!state.done[id]) set({ ...state, done: { ...state.done, [id]: true } })
  },
  toggleBookmark: (id: string) => set({ ...state, bookmarks: toggle(state.bookmarks, id) }),
  setNote: (id: string, text: string) => {
    const notes = { ...state.notes }
    if (text.trim()) notes[id] = text
    else delete notes[id]
    set({ ...state, notes })
  },
  setQuiz: (id: string, score: number, total: number) =>
    set({ ...state, quiz: { ...state.quiz, [id]: { score, total } } }),
  setLast: (id: string) => {
    if (state.last !== id) set({ ...state, last: id })
  },
  reset: () => set(empty),
}

/* ---------- theme ---------- */

type Theme = 'light' | 'dark'
const THEME_KEY = 'ai-mastery:theme'
let theme: Theme = document.documentElement.classList.contains('dark') ? 'dark' : 'light'
const themeListeners = new Set<() => void>()

export function useTheme(): Theme {
  return useSyncExternalStore(
    (l) => {
      themeListeners.add(l)
      return () => themeListeners.delete(l)
    },
    () => theme,
  )
}

export function toggleTheme() {
  theme = theme === 'dark' ? 'light' : 'dark'
  document.documentElement.classList.toggle('dark', theme === 'dark')
  try {
    localStorage.setItem(THEME_KEY, theme)
  } catch {
    /* ignore */
  }
  themeListeners.forEach((l) => l())
}
