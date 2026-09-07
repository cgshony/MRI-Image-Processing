import { Play } from 'lucide-react'
import { useEffect, useState } from 'react'
import type { ImageOut, OperationName } from '../api/types'
import { useJob, useProcessImage } from '../hooks/useJob'
import { useWorkspace } from '../hooks/useWorkspace'
import { OPERATIONS, defaultParamValues, getOperationMeta } from '../operations'
import { OperationViewport } from './OperationViewport'
import { Button } from './ui/Button'
import { Slider } from './ui/Slider'
import { Spinner } from './ui/Spinner'
import { StatusPill } from './ui/StatusPill'

interface ProcessedPaneProps {
  sourceImage: ImageOut
}

/**
 * Right-hand pane of the comparison view: a 4-tab switcher between the
 * operations, a permanently-visible control strip (params + Run + status -
 * the old toolbar popover's contents, un-popover'd) for whichever tab is
 * active, and that operation's result viewport.
 */
export function ProcessedPane({ sourceImage }: ProcessedPaneProps) {
  const {
    activeJobByImageAndOperation,
    activeOperationByImage,
    setActiveOperation,
    setActiveJob,
    showToast,
  } = useWorkspace()
  const processImage = useProcessImage()

  const jobIdFor = (operationId: OperationName) =>
    activeJobByImageAndOperation[sourceImage.id]?.[operationId] ?? null

  // Unrolled rather than looped over OPERATIONS so hooks are always called
  // the same fixed number of times, in the same order, every render.
  const scaleJob = useJob(jobIdFor('scale_image'))
  const upsampleJob = useJob(jobIdFor('bicubic_upsample'))
  const enhanceJob = useJob(jobIdFor('wavelet_enhance'))
  const colourizeJob = useJob(jobIdFor('colourize'))
  const jobByOperation: Record<OperationName, typeof scaleJob> = {
    scale_image: scaleJob,
    bicubic_upsample: upsampleJob,
    wavelet_enhance: enhanceJob,
    colourize: colourizeJob,
  }

  const explicitOperationId = activeOperationByImage[sourceImage.id] ?? null

  // No explicit tab chosen yet for this image: default to whichever operation
  // most recently finished, or the first operation if none has ever run.
  let defaultOperationId: OperationName = OPERATIONS[0].id
  let latestCompletedAt: string | null = null
  for (const operation of OPERATIONS) {
    const job = jobByOperation[operation.id].data
    if (job?.status === 'done' && job.completed_at) {
      if (!latestCompletedAt || job.completed_at > latestCompletedAt) {
        latestCompletedAt = job.completed_at
        defaultOperationId = operation.id
      }
    }
  }

  const activeOperationId = explicitOperationId ?? defaultOperationId

  // Persist the resolved default once, so it stays "stuck" as the reference
  // point for this image even as job data settles or the user navigates away
  // and back, instead of recomputing (and potentially flipping) every render.
  useEffect(() => {
    if (!explicitOperationId) {
      setActiveOperation(sourceImage.id, defaultOperationId)
    }
  }, [sourceImage.id, explicitOperationId, defaultOperationId, setActiveOperation])

  const meta = getOperationMeta(activeOperationId)

  // Per-active-operation param values, reset to that operation's defaults
  // when the tab changes - adjusted directly during render rather than via
  // an effect, matching the pattern the old OperationButton popover used.
  const [paramValues, setParamValues] = useState<Record<string, number>>(() =>
    defaultParamValues(meta),
  )
  const [paramsForOperationId, setParamsForOperationId] = useState(activeOperationId)
  if (paramsForOperationId !== activeOperationId) {
    setParamsForOperationId(activeOperationId)
    setParamValues(defaultParamValues(meta))
  }

  const activeJob = jobByOperation[activeOperationId].data
  const isRunning = activeJob?.status === 'pending' || activeJob?.status === 'running'

  function handleRun() {
    processImage.mutate(
      { imageId: sourceImage.id, operation: activeOperationId, params: paramValues },
      {
        onSuccess: (createdJob) => setActiveJob(sourceImage.id, activeOperationId, createdJob.id),
        onError: (error) => showToast({ kind: 'error', message: error.message }),
      },
    )
  }

  return (
    <div className="flex flex-1 flex-col gap-2 overflow-hidden">
      <div className="flex shrink-0 items-center gap-1 rounded-lg border border-border bg-surface p-1">
        {OPERATIONS.map((operation) => {
          const Icon = operation.icon
          const isActive = operation.id === activeOperationId
          return (
            <button
              key={operation.id}
              onClick={() => setActiveOperation(sourceImage.id, operation.id)}
              className={`flex flex-1 items-center justify-center gap-1.5 rounded-md px-2 py-1.5 text-xs font-medium transition-colors
                ${isActive ? 'bg-selected text-selected-ink' : 'text-ink-muted hover:bg-surface-sunken hover:text-ink'}`}
            >
              <Icon size={14} strokeWidth={2} />
              {operation.label}
            </button>
          )
        })}
      </div>

      <div className="flex shrink-0 flex-col gap-3 rounded-lg border border-border bg-surface p-3">
        <div className="flex items-start justify-between gap-3">
          <p className="text-xs text-ink-muted">{meta.description}</p>
          {activeJob && <StatusPill status={activeJob.status} />}
        </div>

        {meta.params.length > 0 && (
          <div className="flex flex-col gap-3">
            {meta.params.map((spec) => (
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

        <Button variant="primary" onClick={handleRun} disabled={isRunning || processImage.isPending}>
          {isRunning || processImage.isPending ? <Spinner size={16} /> : <Play size={16} />}
          Run {meta.label}
        </Button>

        {activeJob?.status === 'failed' && activeJob.error && (
          <p className="text-xs text-danger">{activeJob.error}</p>
        )}
      </div>

      <div className="flex-1 overflow-hidden">
        <OperationViewport operationId={activeOperationId} sourceImageId={sourceImage.id} />
      </div>
    </div>
  )
}
