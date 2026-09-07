/**
 * Types mirroring the backend's Pydantic schemas exactly
 * (backend/app/schemas/image.py, backend/app/schemas/processing.py).
 */

export type ImageId = string
export type JobId = string

export interface ImageOut {
  id: ImageId
  filename: string
  content_type: string
  width: number
  height: number
  parent_image_id: ImageId | null
  created_at: string
}

export type JobStatus = 'pending' | 'running' | 'done' | 'failed'

/** Keys of `OPERATION_NAMES` in backend/app/services/processing_service.py. */
export type OperationName = 'bicubic_upsample' | 'scale_image' | 'wavelet_enhance' | 'colourize'

/** One named, viewable result of a job - one entry for most operations,
 * several for wavelet_enhance (the reconstructed image plus each Haar
 * sub-band on its own). */
export interface JobChannel {
  key: string
  label: string
  image_id: ImageId
}

export interface ProcessingJobOut {
  id: JobId
  image_id: ImageId
  operation: string
  params: Record<string, unknown>
  status: JobStatus
  result_image_id: ImageId | null
  channels: JobChannel[] | null
  error: string | null
  created_at: string
  completed_at: string | null
}
