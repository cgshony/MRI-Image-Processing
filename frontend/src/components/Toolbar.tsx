import { Scan } from 'lucide-react'
import { useWorkspace } from '../hooks/useWorkspace'
import { OPERATIONS } from '../operations'

export function Toolbar() {
  const { selectedImageId, activeOperationByImage, setActiveOperation } = useWorkspace()
  const activeOperationId = selectedImageId ? activeOperationByImage[selectedImageId] : undefined

  return (
    <header className="flex h-14 shrink-0 items-center gap-4 border-b border-border bg-surface px-4">
      <div className="flex items-center gap-2">
        <Scan size={20} className="text-ink" strokeWidth={2.25} />
        <span className="text-sm font-semibold tracking-tight text-ink">MRI Workspace</span>
      </div>

      <div className="h-6 w-px bg-border" />

      <div className="flex items-center gap-1.5">
        {OPERATIONS.map((operation) => {
          const Icon = operation.icon
          const isActive = !!selectedImageId && operation.id === activeOperationId
          return (
            <button
              key={operation.id}
              onClick={() => selectedImageId && setActiveOperation(selectedImageId, operation.id)}
              disabled={!selectedImageId}
              title={`Show ${operation.label}`}
              className={`flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-sm font-medium transition-colors
                disabled:cursor-not-allowed disabled:opacity-40
                ${isActive ? 'bg-selected text-selected-ink' : 'text-ink-muted hover:bg-surface-sunken hover:text-ink'}`}
            >
              <Icon size={16} strokeWidth={2} />
              {operation.label}
            </button>
          )
        })}
      </div>
    </header>
  )
}
