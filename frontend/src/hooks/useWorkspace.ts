import { useContext } from 'react'
import { WorkspaceContext, type WorkspaceState } from '../context/workspace-context'

export function useWorkspace(): WorkspaceState {
  const context = useContext(WorkspaceContext)
  if (!context) {
    throw new Error('useWorkspace must be used within a WorkspaceProvider')
  }
  return context
}
