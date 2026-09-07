import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useEffect } from 'react'
import { getJob, startProcessingJob } from '../api/client'
import type { ImageId, JobId, OperationName } from '../api/types'
import { IMAGES_QUERY_KEY } from './useImages'

const ACTIVE_STATUSES = new Set(['pending', 'running'])
const POLL_INTERVAL_MS = 1000

export function useProcessImage() {
  return useMutation({
    mutationFn: ({
      imageId,
      operation,
      params,
    }: {
      imageId: ImageId
      operation: OperationName
      params: Record<string, unknown>
    }) => startProcessingJob(imageId, operation, params),
  })
}

/**
 * Polls a processing job while it's pending/running and stops once it settles.
 * On completion, invalidates the images list so the result image (and its
 * new library thumbnail) shows up.
 */
export function useJob(jobId: JobId | null) {
  const queryClient = useQueryClient()

  const query = useQuery({
    queryKey: ['job', jobId],
    queryFn: () => getJob(jobId as JobId),
    enabled: jobId !== null,
    refetchInterval: (latest) =>
      ACTIVE_STATUSES.has(latest.state.data?.status ?? '') ? POLL_INTERVAL_MS : false,
  })

  const status = query.data?.status
  useEffect(() => {
    if (status === 'done') {
      queryClient.invalidateQueries({ queryKey: IMAGES_QUERY_KEY })
    }
  }, [status, queryClient])

  return query
}
