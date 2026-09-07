import type { JobStatus } from '../../api/types'

const STATUS_STYLES: Record<JobStatus, string> = {
  pending: 'bg-surface-sunken text-ink-muted',
  running: 'bg-surface-raised text-ink',
  done: 'bg-success-soft text-success',
  failed: 'bg-danger-soft text-danger',
}

const STATUS_LABELS: Record<JobStatus, string> = {
  pending: 'Pending',
  running: 'Running',
  done: 'Done',
  failed: 'Failed',
}

export function StatusPill({ status }: { status: JobStatus }) {
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium ${STATUS_STYLES[status]}`}
    >
      {(status === 'pending' || status === 'running') && (
        <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-current" />
      )}
      {STATUS_LABELS[status]}
    </span>
  )
}
