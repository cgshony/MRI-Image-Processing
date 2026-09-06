# MRI Workspace (frontend)

A React + TypeScript workspace UI for the MRI Image Processing API: upload an image, run one of the
four processing operations, and view the result — Canva-style tool rail + canvas + properties panel.

## Stack

- **Vite + React + TypeScript**
- **Tailwind CSS v4** (CSS-first config via `@theme` in [src/index.css](src/index.css) — design tokens
  for the palette live there as plain CSS variables)
- **TanStack Query** for data fetching, mutations, and job-status polling
- **lucide-react** for icons

## Setup

```bash
npm install
cp .env.example .env   # defaults to http://localhost:8000/api/v1
npm run dev
```

This expects the backend (`backend/`) running and reachable at the URL in `VITE_API_BASE_URL`
(see the root [README](../README.md) for backend setup). CORS is open (`CORS_ORIGINS=["*"]`) by
default on the backend, so the Vite dev origin is allowed out of the box.

## Scripts

- `npm run dev` — start the Vite dev server
- `npm run build` — type-check (`tsc -b`) and build for production
- `npm run lint` — oxlint
- `npm run format` — Prettier, writes in place

## Structure

```
src/
  api/            typed fetch client + types mirroring the backend schemas
  hooks/          TanStack Query hooks (images, upload/delete, job polling)
  context/        WorkspaceProvider — shared selection state (selected image/operation,
                  active job per image, toasts)
  components/     Sidebar (upload/library/operations), Canvas (viewport), OperationPanel
                  (params + run + status), AppShell (layout), ui/ (shared primitives)
  operations.ts   per-operation metadata (label, icon, param sliders) - the one place
                  that hardcodes the four backend operations' shape
```
