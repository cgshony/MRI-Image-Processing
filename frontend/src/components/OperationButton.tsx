import { Play } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import { useJob, useProcessImage } from '../hooks/useJob'
import { useWorkspace } from '../hooks/useWorkspace'
import type { OperationMeta } from '../operations'
import { Button } from './ui/Button'
import { Slider } from './ui/Slider'
import { Spinner } from './ui/Spinner'
import { StatusPill } from './ui/StatusPill'

function defaultParamValues(meta: OperationMeta): Record<string, number> {
  return Object.fromEntries(meta.params.map((spec) => [spec.key, spec.default]))
}

interface OperationButtonProps {
  operation: OperationMeta
  disabled: boolean
}

/** One toolbar icon button + its params popover, replacing the old dedicated
 * side panel. Owns its own param-values and open/closed state, and reuses
 * the same job hooks the side panel used to. */
export function OperationButton({ operation, disabled }: OperationButtonProps) {
  const { selectedImageId, activeJobByImageAndOperation, setActiveJob, showToast } =
    useWorkspace()
  const processImage = useProcessImage()
  const [isOpen, setIsOpen] = useState(false)
  const [paramValues, setParamValues] = useState<Record<string, number>>(() =>
    defaultParamValues(operation),
  )
  const containerRef = useRef<HTMLDivElement>(null)

  const jobId = selectedImageId
    ? (activeJobByImageAndOperation[selectedImageId]?.[operation.id] ?? null)
    : null
  const { data: job } = useJob(jobId)
  const isRunning = job?.status === 'pending' || job?.status === 'running'

  // Close the popover when the button becomes disabled (image deselected),
  // without a useEffect: adjust state directly during render.
  const [wasDisabled, setWasDisabled] = useState(disabled)
  if (disabled !== wasDisabled) {
    setWasDisabled(disabled)
    if (disabled) setIsOpen(false)
  }

  useEffect(() => {
    if (!isOpen) return
    function handleClickOutside(event: MouseEvent) {
      if (containerRef.current && !containerRef.current.contains(event.target as Node)) {
        setIsOpen(false)
      }
    }
    document.addEventListener('mousedown', handleClickOutside)
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [isOpen])

  function handleRun() {
    if (!selectedImageId) return
    processImage.mutate(
      { imageId: selectedImageId, operation: operation.id, params: paramValues },
      {
        onSuccess: (createdJob) => setActiveJob(selectedImageId, operation.id, createdJob.id),
        onError: (error) => showToast({ kind: 'error', message: error.message }),
      },
    )
  }

  const Icon = operation.icon

  return (
    <div ref={containerRef} className="relative">
      <button
        onClick={() => setIsOpen((prev) => !prev)}
        disabled={disabled}
        title={operation.label}
        className={`flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-sm font-medium transition-colors
          disabled:cursor-not-allowed disabled:opacity-40
          ${isOpen ? 'bg-accent-soft text-accent-hover' : 'text-ink-muted hover:bg-surface-sunken hover:text-ink'}`}
      >
        <Icon size={16} strokeWidth={2} />
        {operation.label}
        {job && !isOpen && <StatusPill status={job.status} />}
      </button>

      {isOpen && (
        <div className="absolute top-full left-0 z-20 mt-2 w-72 rounded-xl border border-border bg-surface p-4 shadow-[var(--shadow-panel)]">
          <div className="mb-1 flex items-center gap-2">
            <Icon size={16} strokeWidth={2} />
            <h3 className="text-sm font-semibold text-ink">{operation.label}</h3>
          </div>
          <p className="mb-4 text-xs text-ink-muted">{operation.description}</p>

          {operation.params.length > 0 && (
            <div className="mb-4 flex flex-col gap-3">
              {operation.params.map((spec) => (
                <Slider
                  key={spec.key}
                  label={spec.label}
                  value={paramValues[spec.key] ?? spec.default}
                  min={spec.min}
                  max={spec.max}
                  step={spec.step}
                  disabled={isRunning}
                  onChange={(value) => setParamValues((prev) => ({ ...prev, [spec.key]: value }))}
                />
              ))}
            </div>
          )}

          <Button
            variant="primary"
            className="w-full"
            onClick={handleRun}
            disabled={!selectedImageId || isRunning || processImage.isPending}
          >
            {isRunning || processImage.isPending ? <Spinner size={16} /> : <Play size={16} />}
            Run
          </Button>

          {job && (
            <div className="mt-3 flex items-center justify-between">
              <span className="text-xs font-semibold tracking-wide text-ink-subtle uppercase">
                Job status
              </span>
              <StatusPill status={job.status} />
            </div>
          )}
          {job?.status === 'failed' && job.error && (
            <p className="mt-1 text-xs text-danger">{job.error}</p>
          )}
        </div>
      )}
    </div>
  )
}
