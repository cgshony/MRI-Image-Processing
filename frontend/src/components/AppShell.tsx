import { LibraryRail } from './LibraryRail'
import { Toolbar } from './Toolbar'
import { Toast } from './ui/Toast'
import { ViewportGrid } from './ViewportGrid'

export function AppShell() {
  return (
    <div className="flex h-screen flex-col bg-canvas">
      <Toolbar />

      <div className="flex flex-1 overflow-hidden">
        <LibraryRail />
        <ViewportGrid />
      </div>

      <Toast />
    </div>
  )
}
