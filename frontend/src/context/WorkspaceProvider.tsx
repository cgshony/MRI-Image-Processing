import { useCallback, useMemo, useState } from 'react'
import type { ReactNode } from 'react'
import type { ImageId, JobId, OperationName } from '../api/types'
import {
  type ActiveJobMap,
  type PatientInfo,
  type PatientInfoMap,
  type Toast,
  type WorkspaceState,
  WorkspaceContext,
} from './workspace-context'

export function WorkspaceProvider({ children }: { children: ReactNode }) {
  const [selectedImageId, setSelectedImageId] = useState<ImageId | null>(null)
  const [activeJobByImageAndOperation, setActiveJobByImageAndOperation] = useState<ActiveJobMap>(
    {},
  )
  const [activeOperationByImage, setActiveOperationByImage] = useState<
    Record<ImageId, OperationName>
  >({})
  const [activeChannelIndexByJob, setActiveChannelIndexByJob] = useState<Record<JobId, number>>(
    {},
  )
  const [patientInfoByImage, setPatientInfoByImage] = useState<PatientInfoMap>({})
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

  const setActiveOperation = useCallback((imageId: ImageId, operation: OperationName) => {
    setActiveOperationByImage((prev) => ({ ...prev, [imageId]: operation }))
  }, [])

  const setActiveChannelIndex = useCallback((jobId: JobId, index: number) => {
    setActiveChannelIndexByJob((prev) => ({ ...prev, [jobId]: index }))
  }, [])

  const setPatientInfo = useCallback((imageId: ImageId, info: PatientInfo) => {
    setPatientInfoByImage((prev) => ({ ...prev, [imageId]: info }))
  }, [])

  const showToast = useCallback((next: Toast) => setToast(next), [])
  const dismissToast = useCallback(() => setToast(null), [])

  const value = useMemo<WorkspaceState>(
    () => ({
      selectedImageId,
      selectImage,
      activeJobByImageAndOperation,
      setActiveJob,
      activeOperationByImage,
      setActiveOperation,
      activeChannelIndexByJob,
      setActiveChannelIndex,
      patientInfoByImage,
      setPatientInfo,
      toast,
      showToast,
      dismissToast,
    }),
    [
      selectedImageId,
      selectImage,
      activeJobByImageAndOperation,
      setActiveJob,
      activeOperationByImage,
      setActiveOperation,
      activeChannelIndexByJob,
      setActiveChannelIndex,
      patientInfoByImage,
      setPatientInfo,
      toast,
      showToast,
      dismissToast,
    ],
  )

  return <WorkspaceContext.Provider value={value}>{children}</WorkspaceContext.Provider>
}
