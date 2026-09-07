import type { ImageId, ImageOut, JobId, OperationName, ProcessingJobOut } from './types'

export const API_BASE_URL: string =
  import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000/api/v1'

/** Raised for any non-2xx response, carrying the backend's `detail` message when present. */
export class ApiError extends Error {
  readonly status: number

  constructor(status: number, message: string) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, init)

  if (!response.ok) {
    const message = await extractErrorMessage(response)
    throw new ApiError(response.status, message)
  }

  if (response.status === 204) {
    return undefined as T
  }
  return (await response.json()) as T
}

async function extractErrorMessage(response: Response): Promise<string> {
  try {
    const body = (await response.json()) as { detail?: string }
    if (body.detail) return body.detail
  } catch {
    // Response body wasn't JSON (or was empty) - fall through to the status text.
  }
  return response.statusText || `Request failed with status ${response.status}`
}

export function getImageFileUrl(imageId: ImageId): string {
  return `${API_BASE_URL}/images/${imageId}/file`
}

export async function uploadImage(file: File): Promise<ImageOut> {
  const formData = new FormData()
  formData.append('file', file)
  return request<ImageOut>('/images', { method: 'POST', body: formData })
}

export async function listImages(): Promise<ImageOut[]> {
  return request<ImageOut[]>('/images')
}

export async function getImage(imageId: ImageId): Promise<ImageOut> {
  return request<ImageOut>(`/images/${imageId}`)
}

export async function deleteImage(imageId: ImageId): Promise<void> {
  return request<void>(`/images/${imageId}`, { method: 'DELETE' })
}

export async function startProcessingJob(
  imageId: ImageId,
  operation: OperationName,
  params: Record<string, unknown>,
): Promise<ProcessingJobOut> {
  return request<ProcessingJobOut>(`/images/${imageId}/process`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ operation, params }),
  })
}

export async function getJob(jobId: JobId): Promise<ProcessingJobOut> {
  return request<ProcessingJobOut>(`/jobs/${jobId}`)
}
