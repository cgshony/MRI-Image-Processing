import { useEffect } from 'react'
import type { ImageOut, OperationName } from '../api/types'
import { useJob } from '../hooks/useJob'
import { useWorkspace } from '../hooks/useWorkspace'
import { OPERATIONS } from '../operations'
import { OperationViewport } from './OperationViewport'

interface ProcessedPaneProps {
  sourceImage: ImageOut
}

/**
 * Right-hand side of the comparison view. Resolves which operation's result
 * to show (switching operations and running them both happen from the
 * toolbar now, via OperationButton) and renders just that result - a single
 * Viewport, matching the Original pane 1:1 with no extra chrome in between.
 */
export function ProcessedPane({ sourceImage }: ProcessedPaneProps) {
  const { activeJobByImageAndOperation, activeOperationByImage, setActiveOperation } =
    useWorkspace()

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

  return <OperationViewport operationId={activeOperationId} sourceImageId={sourceImage.id} />
}
