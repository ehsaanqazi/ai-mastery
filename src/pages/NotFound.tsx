import { Link } from 'react-router'

export default function NotFound() {
  return (
    <div className="mx-auto max-w-xl px-5 py-24 text-center">
      <p className="font-mono text-sm text-accent">404</p>
      <h1 className="mt-2 font-serif text-3xl font-medium">Page not found</h1>
      <p className="mt-3 text-muted">That page doesn't exist. The curriculum is in the sidebar.</p>
      <Link to="/" className="mt-6 inline-block font-medium text-accent hover:underline">
        Back to home
      </Link>
    </div>
  )
}
