import { ImageOff } from 'lucide-react'
import { useImages } from '../hooks/useImages'
import { useWorkspace } from '../hooks/useWorkspace'
import { OPERATIONS } from '../operations'
import { OperationViewport } from './OperationViewport'
import { Viewport } from './Viewport'

/**
 * Fixed 5-pane grid: the original on the left spanning both rows, and the
 * four operation results auto-placed into the remaining 2x2 block in
 * operation order (Scale, Upsample, Enhance, Colourize).
 */
export function ViewportGrid() {
  const { selectedImageId } = useWorkspace()
  const { data: images } = useImages()
  const selectedImage = images?.find((image) => image.id === selectedImageId) ?? null

  if (!selectedImage) {
    return (
      <div className="flex flex-1 flex-col items-center justify-center gap-3 bg-canvas text-ink-subtle">
        <ImageOff size={40} strokeWidth={1.5} />
        <p className="text-sm">Select an image from the library, or upload one to begin.</p>
      </div>
    )
  }

  return (
    <div className="grid flex-1 grid-cols-[minmax(220px,1fr)_1.4fr_1.4fr] grid-rows-2 gap-2 overflow-hidden bg-canvas p-2">
      <div className="row-span-2">
        <Viewport
          label="Original"
          imageId={selectedImage.id}
          filename={selectedImage.filename}
          width={selectedImage.width}
          height={selectedImage.height}
          paramSummary={null}
          timestamp={null}
          status={null}
          errorMessage={null}
          placeholderText="No image selected"
        />
      </div>

      {OPERATIONS.map((operation) => (
        <OperationViewport
          key={operation.id}
          operationId={operation.id}
          sourceImageId={selectedImage.id}
        />
      ))}
    </div>
  )
}
