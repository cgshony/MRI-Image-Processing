import { useCallback, useMemo, useState } from 'react'
import type { ReactNode } from 'react'
import type { ImageId, JobId, OperationName } from '../api/types'
import {
  type ActiveJobMap,
  type Toast,
  type WorkspaceState,
  WorkspaceContext,
} from './workspace-context'

export function WorkspaceProvider({ children }: { children: ReactNode }) {
  const [selectedImageId, setSelectedImageId] = useState<ImageId | null>(null)
  const [activeJobByImageAndOperation, setActiveJobByImageAndOperation] = useState<ActiveJobMap>(
    {},
  )
  const [toast, setToast] = useState<Toast | null>(null)

  const selectImage = useCallback((imageId: ImageId | null) => {
    setSelectedImageId(imageId)
  }, [])

  const setActiveJob = useCallback(
    (imageId: ImageId, operation: OperationName, jobId: JobId) => {
      setActiveJobByImageAndOperation((prev) => ({
        ...prev,
        [imageId]: { ...prev[imageId], [operation]: jobId },
      }))
    },
    [],
  )

  const showToast = useCallback((next: Toast) => setToast(next), [])
  const dismissToast = useCallback(() => setToast(null), [])

  const value = useMemo<WorkspaceState>(
    () => ({
      selectedImageId,
      selectImage,
      activeJobByImageAndOperation,
      setActiveJob,
      toast,
      showToast,
      dismissToast,
    }),
    [
      selectedImageId,
      selectImage,
      activeJobByImageAndOperation,
      setActiveJob,
      toast,
      showToast,
      dismissToast,
    ],
  )

  return <WorkspaceContext.Provider value={value}>{children}</WorkspaceContext.Provider>
}
