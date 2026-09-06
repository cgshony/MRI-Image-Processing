import { AppShell } from './components/AppShell'
import { WorkspaceProvider } from './context/WorkspaceProvider'

function App() {
  return (
    <WorkspaceProvider>
      <AppShell />
    </WorkspaceProvider>
  )
}

export default App
