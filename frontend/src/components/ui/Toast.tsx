import { AlertCircle, CheckCircle2, X } from 'lucide-react'
import { useEffect } from 'react'
import { useWorkspace } from '../../hooks/useWorkspace'

const AUTO_DISMISS_MS = 5000

export function Toast() {
  const { toast, dismissToast } = useWorkspace()

  useEffect(() => {
    if (!toast) return
    const timer = setTimeout(dismissToast, AUTO_DISMISS_MS)
    return () => clearTimeout(timer)
  }, [toast, dismissToast])

  if (!toast) return null

  const isError = toast.kind === 'error'

  return (
    <div
      role="alert"
      className={`fixed right-5 bottom-5 z-50 flex max-w-sm items-start gap-2.5 rounded-lg border px-4 py-3 text-sm shadow-[var(--shadow-panel)]
        ${isError ? 'border-danger/20 bg-danger-soft text-danger' : 'border-success/20 bg-success-soft text-success'}`}
    >
      {isError ? (
        <AlertCircle size={18} className="mt-0.5 shrink-0" />
      ) : (
        <CheckCircle2 size={18} className="mt-0.5 shrink-0" />
      )}
      <p className="flex-1">{toast.message}</p>
      <button onClick={dismissToast} className="shrink-0 opacity-70 hover:opacity-100">
        <X size={16} />
      </button>
    </div>
  )
}
