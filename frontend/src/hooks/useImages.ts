import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { deleteImage, listImages, uploadImage } from '../api/client'
import type { ImageId } from '../api/types'

export const IMAGES_QUERY_KEY = ['images'] as const

export function useImages() {
  return useQuery({
    queryKey: IMAGES_QUERY_KEY,
    queryFn: listImages,
  })
}

export function useUploadImage() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (file: File) => uploadImage(file),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: IMAGES_QUERY_KEY })
    },
  })
}

export function useDeleteImage() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (imageId: ImageId) => deleteImage(imageId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: IMAGES_QUERY_KEY })
    },
  })
}
