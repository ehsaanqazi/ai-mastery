import { createContext } from 'react'

/** Id of the lesson currently being rendered (used to key quiz scores). */
export const LessonContext = createContext<string>('')
