import { createContext } from 'react'
import type { ImageId, JobId, OperationName } from '../api/types'

export interface Toast {
  kind: 'error' | 'success'
  message: string
}

/** Most recently started job id per (image, operation) pair - all 4 operations
 * are tracked simultaneously per selected original, so this is keyed by both. */
export type ActiveJobMap = Record<ImageId, Partial<Record<OperationName, JobId>>>

/** Basic clinical context for an image, entered by hand in the UI. Frontend-only
 * - not persisted to the backend, lost on reload, same as the rest of this
 * context's state. */
export interface PatientInfo {
  patientName?: string
  patientId?: string
  dateOfBirth?: string
  studyDate?: string
  modality?: string
}

export type PatientInfoMap = Record<ImageId, PatientInfo>

export interface WorkspaceState {
  /** The selected original/group - resolves both panes in the comparison view. */
  selectedImageId: ImageId | null
  selectImage: (imageId: ImageId | null) => void

  activeJobByImageAndOperation: ActiveJobMap
  setActiveJob: (imageId: ImageId, operation: OperationName, jobId: JobId) => void

  /** Which of the 4 operations the Processed pane's tab is currently showing,
   * per source image. */
  activeOperationByImage: Record<ImageId, OperationName>
  setActiveOperation: (imageId: ImageId, operation: OperationName) => void

  patientInfoByImage: PatientInfoMap
  setPatientInfo: (imageId: ImageId, info: PatientInfo) => void

  toast: Toast | null
  showToast: (toast: Toast) => void
  dismissToast: () => void
}

export const WorkspaceContext = createContext<WorkspaceState | null>(null)
