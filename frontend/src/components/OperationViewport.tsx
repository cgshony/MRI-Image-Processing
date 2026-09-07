import type { ImageId, OperationName } from '../api/types'
import { useImages } from '../hooks/useImages'
import { useJob } from '../hooks/useJob'
import { useWorkspace } from '../hooks/useWorkspace'
import { getOperationMeta } from '../operations'
import { Viewport, type ChannelSliderConfig } from './Viewport'

interface OperationViewportProps {
  operationId: OperationName
  sourceImageId: ImageId
}

/** Resolves one operation's pane for the selected original: looks up its
 * tracked job (independently polled via useJob), and once done, the result
 * image's metadata - `ImageOut` doesn't carry which operation produced it,
 * so the job map is what ties a derived image back to its operation.
 *
 * A job can carry several named result channels (wavelet_enhance's
 * reconstructed image plus its Haar sub-bands) - when it does, this also
 * drives the pane's bottom channel slider and resolves whichever channel is
 * currently selected instead of always the first. */
export function OperationViewport({ operationId, sourceImageId }: OperationViewportProps) {
  const meta = getOperationMeta(operationId)
  const { activeJobByImageAndOperation, activeChannelIndexByJob, setActiveChannelIndex } =
    useWorkspace()
  const { data: images } = useImages()

  const jobId = activeJobByImageAndOperation[sourceImageId]?.[operationId] ?? null
  const { data: job } = useJob(jobId)

  const channels = job?.status === 'done' ? (job.channels ?? null) : null
  const rawChannelIndex = jobId ? (activeChannelIndexByJob[jobId] ?? 0) : 0
  const channelIndex = channels ? Math.min(rawChannelIndex, channels.length - 1) : 0

  const resultImageId = channels
    ? (channels[channelIndex]?.image_id ?? null)
    : job?.status === 'done'
      ? job.result_image_id
      : null
  const resultImage = resultImageId
    ? (images?.find((image) => image.id === resultImageId) ?? null)
    : null

  const channelSlider: ChannelSliderConfig | null =
    channels && jobId
      ? {
          index: channelIndex,
          labels: channels.map((channel) => channel.label),
          onChange: (index) => setActiveChannelIndex(jobId, index),
        }
      : null

  const paramSummary =
    job?.status === 'done' && meta.params.length > 0
      ? meta.params
          .map((spec) => `${spec.label}: ${Number(job.params[spec.key] ?? spec.default).toFixed(1)}`)
          .join(' · ')
      : null

  const timestamp = job?.completed_at
    ? new Date(job.completed_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    : null

  return (
    <Viewport
      label={meta.label}
      imageId={resultImage?.id ?? null}
      filename={resultImage?.filename ?? null}
      width={resultImage?.width ?? null}
      height={resultImage?.height ?? null}
      paramSummary={paramSummary}
      timestamp={timestamp}
      status={job?.status ?? null}
      errorMessage={job?.status === 'failed' ? job.error : null}
      placeholderText={`${meta.label} not yet run`}
      channelSlider={channelSlider}
    />
  )
}
