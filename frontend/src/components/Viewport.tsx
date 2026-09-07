import { ImageOff, Minus, Square, X } from 'lucide-react'
import { getImageFileUrl } from '../api/client'
import type { ImageId, JobStatus } from '../api/types'
import { Spinner } from './ui/Spinner'
import { StatusPill } from './ui/StatusPill'

interface ViewportProps {
  /** Pane title, shown both in the title bar and as the top-left overlay. */
  label: string
  imageId: ImageId | null
  filename: string | null
  width: number | null
  height: number | null
  /** Bottom-left overlay - a param summary for the run, blank when not applicable. */
  paramSummary: string | null
  /** Bottom-right overlay text, alongside the status pill. */
  timestamp: string | null
  status: JobStatus | null
  errorMessage: string | null
  placeholderText: string
}

/** One tiled viewport pane: title bar, black image area, four-corner metadata
 * overlay - shared by the original pane and all four operation panes. */
export function Viewport({
  label,
  imageId,
  filename,
  width,
  height,
  paramSummary,
  timestamp,
  status,
  errorMessage,
  placeholderText,
}: ViewportProps) {
  const isBusy = status === 'pending' || status === 'running'

  return (
    <div className="flex flex-col overflow-hidden rounded-md border border-border bg-black">
      <div className="flex h-6 shrink-0 items-center justify-between border-b border-border bg-surface px-2">
        <span className="text-[11px] font-semibold tracking-wide text-ink-muted uppercase">
          {label}
        </span>
        <div className="flex items-center gap-1 text-ink-subtle">
          <Minus size={10} />
          <Square size={9} />
          <X size={10} />
        </div>
      </div>

      <div className="relative flex flex-1 items-center justify-center overflow-hidden bg-black">
        {imageId ? (
          <img
            key={imageId}
            src={getImageFileUrl(imageId)}
            alt={filename ?? label}
            className="max-h-full max-w-full object-contain"
          />
        ) : (
          <div className="flex flex-col items-center gap-2 text-ink-subtle">
            <ImageOff size={28} strokeWidth={1.5} />
            <p className="text-xs">{placeholderText}</p>
          </div>
        )}

        {isBusy && (
          <div className="absolute inset-0 flex items-center justify-center bg-black/70">
            <Spinner size={24} />
          </div>
        )}

        {status === 'failed' && (
          <div className="absolute inset-x-2 bottom-8 rounded bg-danger-soft px-2 py-1 text-center text-xs text-danger">
            {errorMessage ?? 'Processing failed'}
          </div>
        )}

        {imageId && (
          <>
            <span className="viewport-overlay pointer-events-none absolute top-1.5 left-1.5 text-[10px]">
              {label}
            </span>
            {filename && (
              <span className="viewport-overlay pointer-events-none absolute top-1.5 right-1.5 text-[10px]">
                {filename} · {width}×{height}
              </span>
            )}
            {paramSummary && (
              <span className="viewport-overlay pointer-events-none absolute bottom-1.5 left-1.5 text-[10px]">
                {paramSummary}
              </span>
            )}
            {status && (
              <span className="pointer-events-none absolute right-1.5 bottom-1.5 flex items-center gap-1.5">
                {timestamp && <span className="viewport-overlay text-[10px]">{timestamp}</span>}
                <StatusPill status={status} />
              </span>
            )}
          </>
        )}
      </div>
    </div>
  )
}
