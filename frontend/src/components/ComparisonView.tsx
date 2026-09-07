import { ImageOff } from 'lucide-react'
import { useImages } from '../hooks/useImages'
import { useWorkspace } from '../hooks/useWorkspace'
import { ProcessedPane } from './ProcessedPane'
import { Viewport } from './Viewport'

/**
 * Two-pane comparison, both panes the same size: the Original pinned on the
 * left at all times, and the Processed pane on the right, showing whichever
 * operation's result is active (switched from the toolbar) while the
 * Original never changes.
 */
export function ComparisonView() {
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
    <div className="grid flex-1 grid-cols-2 gap-2 overflow-hidden bg-canvas p-2">
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

      <ProcessedPane sourceImage={selectedImage} />
    </div>
  )
}
