export default function Logo() {
  return (
    <span className="flex items-center gap-2.5">
      <span className="grid h-7 w-7 place-items-center rounded-lg bg-accent text-accent-fg">
        <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round">
          <path d="M5 19 12 5l7 14M8.5 13.5h7" />
        </svg>
      </span>
      <span className="font-serif text-[1.2rem] font-semibold tracking-tight text-fg">AI Mastery</span>
    </span>
  )
}
