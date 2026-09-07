import { Maximize2, Palette, Waves, ZoomIn } from 'lucide-react'
import type { ComponentType } from 'react'
import type { OperationName } from './api/types'

export interface OperationParamSpec {
  key: string
  label: string
  min: number
  max: number
  step: number
  default: number
}

export interface OperationMeta {
  id: OperationName
  label: string
  description: string
  icon: ComponentType<{ size?: number | string; strokeWidth?: number | string }>
  params: OperationParamSpec[]
}

/**
 * Presentation + params metadata for each backend operation
 * (backend/app/services/processing_service.py OPERATIONS / OPERATION_NAMES).
 * Keep in sync with that dispatch table - this is the only place the frontend
 * hardcodes per-operation shape.
 */
export const OPERATIONS: OperationMeta[] = [
  {
    id: 'scale_image',
    label: 'Scale',
    description: 'Resize using nearest-neighbor interpolation - fast, blocky at large factors.',
    icon: Maximize2,
    params: [
      { key: 'scale_factor', label: 'Scale factor', min: 0.5, max: 4, step: 0.1, default: 2 },
    ],
  },
  {
    id: 'bicubic_upsample',
    label: 'Upsample',
    description: 'Resize using bicubic interpolation - smoother results for enlarging an image.',
    icon: ZoomIn,
    params: [
      { key: 'scale_factor', label: 'Scale factor', min: 0.5, max: 4, step: 0.1, default: 2 },
    ],
  },
  {
    id: 'wavelet_enhance',
    label: 'Enhance',
    description: 'Boost fine detail via a 2D Haar wavelet transform on the high-frequency bands.',
    icon: Waves,
    params: [{ key: 'factor', label: 'Enhance factor', min: 0.5, max: 3, step: 0.1, default: 1.5 }],
  },
  {
    id: 'colourize',
    label: 'Colourize',
    description: 'Map grayscale intensity to a pseudo-colour HSV ramp. No parameters to tune.',
    icon: Palette,
    params: [],
  },
]

export function getOperationMeta(id: OperationName): OperationMeta {
  const meta = OPERATIONS.find((operation) => operation.id === id)
  if (!meta) throw new Error(`Unknown operation: ${id}`)
  return meta
}

/** Each param's slider defaulted to its spec's `default`, keyed by param name. */
export function defaultParamValues(meta: OperationMeta): Record<string, number> {
  return Object.fromEntries(meta.params.map((spec) => [spec.key, spec.default]))
}
