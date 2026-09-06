import { useMutation, useQueryClient } from '@tanstack/react-query'
import { UploadCloud } from 'lucide-react'
import { useRef, useState } from 'react'
import type { DragEvent } from 'react'
import { deleteImage } from '../api/client'
import type { ImageId } from '../api/types'
import { IMAGES_QUERY_KEY, useUploadImage } from '../hooks/useImages'
import { useImageGroups } from '../hooks/useImageGroups'
import { useWorkspace } from '../hooks/useWorkspace'
import { LibraryEntry } from './ui/LibraryEntry'
import { Spinner } from './ui/Spinner'

/**
 * Deletes a whole group client-side: derivatives first, then the original.
 * `images.parent_image_id` has no `ON DELETE CASCADE` (backend untouched),
 * so deleting a referenced original first would 500.
 */
function useDeleteGroup() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: async ({
      originalId,
      derivativeIds,
    }: {
      originalId: ImageId
      derivativeIds: ImageId[]
    }) => {
      for (const derivativeId of derivativeIds) {
        await deleteImage(derivativeId)
      }
      await deleteImage(originalId)
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: IMAGES_QUERY_KEY })
    },
  })
}

export function LibraryRail() {
  const { data: groups, isLoading } = useImageGroups()
  const uploadImage = useUploadImage()
  const deleteGroup = useDeleteGroup()
  const { selectedImageId, selectImage, showToast } = useWorkspace()
  const [isDraggingOver, setIsDraggingOver] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)

  function handleFiles(files: FileList | null) {
    const file = files?.[0]
    if (!file) return
    uploadImage.mutate(file, {
      onSuccess: (image) => selectImage(image.id),
      onError: (error) => showToast({ kind: 'error', message: error.message }),
    })
  }

  function handleDrop(event: DragEvent<HTMLButtonElement>) {
    event.preventDefault()
    setIsDraggingOver(false)
    handleFiles(event.dataTransfer.files)
  }

  function handleDeleteGroup(originalId: ImageId, derivativeIds: ImageId[]) {
    const message =
      derivativeIds.length > 0
        ? `Delete this image and its ${derivativeIds.length} result${derivativeIds.length === 1 ? '' : 's'}?`
        : 'Delete this image?'
    if (!window.confirm(message)) return

    deleteGroup.mutate(
      { originalId, derivativeIds },
      {
        onSuccess: () => {
          if (selectedImageId === originalId) selectImage(null)
        },
        onError: (error) => showToast({ kind: 'error', message: error.message }),
      },
    )
  }

  return (
    <aside className="flex w-52 shrink-0 flex-col gap-3 overflow-y-auto border-r border-border bg-surface p-3">
      <button
        onClick={() => fileInputRef.current?.click()}
        onDragOver={(event) => {
          event.preventDefault()
          setIsDraggingOver(true)
        }}
        onDragLeave={() => setIsDraggingOver(false)}
        onDrop={handleDrop}
        disabled={uploadImage.isPending}
        className={`flex w-full flex-col items-center gap-1.5 rounded-lg border-2 border-dashed px-3 py-4 text-center transition-colors
          ${isDraggingOver ? 'border-accent bg-accent-soft' : 'border-border hover:border-ink-subtle'}`}
      >
        {uploadImage.isPending ? (
          <Spinner />
        ) : (
          <UploadCloud size={20} className="text-ink-muted" />
        )}
        <span className="text-xs text-ink-muted">
          {uploadImage.isPending ? 'Uploading…' : 'Drop or click to upload'}
        </span>
      </button>
      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        className="hidden"
        onChange={(event) => {
          handleFiles(event.target.files)
          event.target.value = ''
        }}
      />

      <div className="flex flex-col gap-2">
        {isLoading && <p className="text-xs text-ink-subtle">Loading…</p>}
        {!isLoading && groups.length === 0 && (
          <p className="text-xs text-ink-subtle">No images yet — upload one to get started.</p>
        )}
        {groups.map((group) => (
          <LibraryEntry
            key={group.original.id}
            image={group.original}
            resultCount={group.derivatives.length}
            selected={group.original.id === selectedImageId}
            onSelect={() => selectImage(group.original.id)}
            onDelete={() =>
              handleDeleteGroup(
                group.original.id,
                group.derivatives.map((derivative) => derivative.id),
              )
            }
            deleteDisabled={deleteGroup.isPending}
          />
        ))}
      </div>
    </aside>
  )
}
