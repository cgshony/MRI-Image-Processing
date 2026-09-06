import { createContext } from 'react'
import type { ImageId, JobId, OperationName } from '../api/types'

export interface Toast {
  kind: 'error' | 'success'
  message: string
}

/** Most recently started job id per (image, operation) pair - all 4 operations
 * are tracked simultaneously per selected original, so this is keyed by both. */
export type ActiveJobMap = Record<ImageId, Partial<Record<OperationName, JobId>>>

export interface WorkspaceState {
  /** The selected original/group - resolves to all 5 panes in the viewport grid. */
  selectedImageId: ImageId | null
  selectImage: (imageId: ImageId | null) => void

  activeJobByImageAndOperation: ActiveJobMap
  setActiveJob: (imageId: ImageId, operation: OperationName, jobId: JobId) => void

  toast: Toast | null
  showToast: (toast: Toast) => void
  dismissToast: () => void
}

export const WorkspaceContext = createContext<WorkspaceState | null>(null)
