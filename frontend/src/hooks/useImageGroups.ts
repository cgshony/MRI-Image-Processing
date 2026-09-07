import { useMemo } from 'react'
import type { ImageOut } from '../api/types'
import { useImages } from './useImages'

export interface ImageGroup {
  original: ImageOut
  derivatives: ImageOut[]
}

/**
 * Groups the flat image list into one entry per uploaded original plus the
 * images derived from it, mirroring RadiAnt's "one rail entry per series".
 * Used by the library rail (for the count badge) and the viewport grid
 * (resolving which derived image belongs to the selected original).
 */
export function useImageGroups(): { data: ImageGroup[]; isLoading: boolean } {
  const { data: images, isLoading } = useImages()

  const data = useMemo<ImageGroup[]>(() => {
    if (!images) return []
    const originals = images.filter((image) => image.parent_image_id === null)
    return originals.map((original) => ({
      original,
      derivatives: images.filter((image) => image.parent_image_id === original.id),
    }))
  }, [images])

  return { data, isLoading }
}
