import { ScanLine } from 'lucide-react'
import { useWorkspace } from '../hooks/useWorkspace'
import { OPERATIONS } from '../operations'
import { OperationButton } from './OperationButton'

export function Toolbar() {
  const { selectedImageId } = useWorkspace()

  return (
    <header className="flex h-14 shrink-0 items-center gap-4 border-b border-border bg-surface px-4">
      <div className="flex items-center gap-2">
        <ScanLine size={20} className="text-accent" strokeWidth={2.25} />
        <span className="text-sm font-semibold tracking-tight text-ink">MRI Workspace</span>
      </div>

      <div className="h-6 w-px bg-border" />

      <div className="flex items-center gap-1.5">
        {OPERATIONS.map((operation) => (
          <OperationButton
            key={operation.id}
            operation={operation}
            disabled={!selectedImageId}
          />
        ))}
      </div>
    </header>
  )
}
