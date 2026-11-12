import { lazy, Suspense } from 'react'
import { Route, Routes } from 'react-router'
import Layout from './components/Layout'
import Home from './pages/Home'
import ModulePage from './pages/ModulePage'
import Roadmap from './pages/Roadmap'
import Library from './pages/Library'
import NotFound from './pages/NotFound'

const LessonPage = lazy(() => import('./pages/LessonPage'))

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route index element={<Home />} />
        <Route path="roadmap" element={<Roadmap />} />
        <Route path="library" element={<Library />} />
        <Route path="learn/:moduleSlug" element={<ModulePage />} />
        <Route path="learn/:moduleSlug/:topicSlug" element={<Suspense fallback={null}><LessonPage /></Suspense>} />
        <Route path="*" element={<NotFound />} />
      </Route>
    </Routes>
  )
}
