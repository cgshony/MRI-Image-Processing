import { Trash2 } from 'lucide-react'
import { getImageFileUrl } from '../../api/client'
import type { ImageOut } from '../../api/types'

interface LibraryEntryProps {
  image: ImageOut
  resultCount: number
  selected: boolean
  onSelect: () => void
  onDelete: () => void
  deleteDisabled?: boolean
}

/** One rail entry per uploaded original, with a RadiAnt-style count badge for
 * its derived results. */
export function LibraryEntry({
  image,
  resultCount,
  selected,
  onSelect,
  onDelete,
  deleteDisabled,
}: LibraryEntryProps) {
  return (
    <div className="group relative">
      <button
        onClick={onSelect}
        title={image.filename}
        className={`flex w-full flex-col items-center gap-1 rounded-lg border-2 p-1.5 text-center transition-colors
          ${selected ? 'border-border-strong bg-selected' : 'border-transparent hover:border-border'}`}
      >
        <span className="relative h-14 w-full shrink-0 overflow-hidden rounded-md bg-surface-sunken">
          <img
            src={getImageFileUrl(image.id)}
            alt={image.filename}
            className="h-full w-full object-cover"
            loading="lazy"
          />
          {resultCount > 0 && (
            <span className="absolute right-0.5 bottom-0.5 flex h-4 min-w-[16px] items-center justify-center rounded-full bg-ink px-1 text-[9px] font-semibold text-canvas">
              {resultCount}
            </span>
          )}
        </span>
        <span className="w-full min-w-0 truncate text-[11px] font-medium text-ink">
          {image.filename}
        </span>
      </button>
      <button
        onClick={(event) => {
          event.stopPropagation()
          onDelete()
        }}
        disabled={deleteDisabled}
        title="Delete image and its results"
        className="absolute top-1 right-1 rounded-md bg-black/60 p-1 text-white opacity-0
          transition-opacity group-hover:opacity-100 hover:bg-danger disabled:cursor-not-allowed disabled:opacity-60"
      >
        <Trash2 size={12} />
      </button>
    </div>
  )
}
