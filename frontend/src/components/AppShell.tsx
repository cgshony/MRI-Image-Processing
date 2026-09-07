import { ComparisonView } from './ComparisonView'
import { LibraryRail } from './LibraryRail'
import { Toolbar } from './Toolbar'
import { Toast } from './ui/Toast'

export function AppShell() {
  return (
    <div className="flex h-screen flex-col bg-canvas">
      <Toolbar />

      <div className="flex flex-1 overflow-hidden">
        <LibraryRail />
        <ComparisonView />
      </div>

      <Toast />
    </div>
  )
}
