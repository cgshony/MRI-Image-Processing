import { CalendarDays, Fingerprint, Pencil, UserRound } from 'lucide-react'
import { useState } from 'react'
import type { ImageOut } from '../api/types'
import type { PatientInfo } from '../context/workspace-context'
import { useWorkspace } from '../hooks/useWorkspace'

interface PatientInfoPanelProps {
  image: ImageOut | null
}

const EMPTY_FORM: PatientInfo = {
  patientName: '',
  patientId: '',
  dateOfBirth: '',
  studyDate: '',
  modality: '',
}

const FIELD_CLASSES =
  'w-full rounded border border-border bg-surface px-1.5 py-1 text-[11px] text-ink placeholder:text-ink-subtle'

/**
 * RadiAnt-style patient block above the thumbnail rail. Frontend-only: the
 * fields are held in workspace context (keyed by image id), never sent to
 * the backend, and reset on reload like the rest of that context's state.
 */
export function PatientInfoPanel({ image }: PatientInfoPanelProps) {
  const { patientInfoByImage, setPatientInfo } = useWorkspace()
  const [isEditing, setIsEditing] = useState(false)
  const [form, setForm] = useState<PatientInfo>(EMPTY_FORM)

  // Close the editor whenever the selected image changes, without a useEffect:
  // adjust state directly during render. Normalize to `null` on both sides -
  // comparing the raw `image?.id` (which is `undefined`, not `null`, when
  // there's no image) against a `null`-seeded state never settles.
  const normalizedImageId = image?.id ?? null
  const [editingForImageId, setEditingForImageId] = useState(normalizedImageId)
  if (normalizedImageId !== editingForImageId) {
    setEditingForImageId(normalizedImageId)
    setIsEditing(false)
  }

  if (!image) {
    return (
      <div className="rounded-lg border border-border bg-surface-sunken p-2 text-[11px] text-ink-subtle">
        No image selected
      </div>
    )
  }

  const info = patientInfoByImage[image.id]
  const hasInfo = !!info && Object.values(info).some((value) => value)

  function startEditing() {
    setForm(info ?? EMPTY_FORM)
    setIsEditing(true)
  }

  function handleSave() {
    if (!image) return
    setPatientInfo(image.id, form)
    setIsEditing(false)
  }

  if (isEditing) {
    return (
      <div className="flex flex-col gap-1.5 rounded-lg border border-border bg-surface-sunken p-2">
        <input
          value={form.patientName ?? ''}
          onChange={(event) => setForm((prev) => ({ ...prev, patientName: event.target.value }))}
          placeholder="Patient name"
          className={FIELD_CLASSES}
        />
        <input
          value={form.patientId ?? ''}
          onChange={(event) => setForm((prev) => ({ ...prev, patientId: event.target.value }))}
          placeholder="Patient ID"
          className={FIELD_CLASSES}
        />
        <input
          type="date"
          value={form.dateOfBirth ?? ''}
          onChange={(event) => setForm((prev) => ({ ...prev, dateOfBirth: event.target.value }))}
          className={FIELD_CLASSES}
        />
        <input
          type="date"
          value={form.studyDate ?? ''}
          onChange={(event) => setForm((prev) => ({ ...prev, studyDate: event.target.value }))}
          className={FIELD_CLASSES}
        />
        <input
          value={form.modality ?? ''}
          onChange={(event) => setForm((prev) => ({ ...prev, modality: event.target.value }))}
          placeholder="Modality (e.g. MRI)"
          className={FIELD_CLASSES}
        />
        <div className="mt-1 flex gap-1.5">
          <button
            onClick={handleSave}
            className="flex-1 rounded bg-ink py-1 text-[11px] font-medium text-canvas hover:bg-ink-muted"
          >
            Save
          </button>
          <button
            onClick={() => setIsEditing(false)}
            className="flex-1 rounded border border-border py-1 text-[11px] font-medium text-ink-muted hover:bg-surface"
          >
            Cancel
          </button>
        </div>
      </div>
    )
  }

  return (
    <div className="group relative rounded-lg border border-border bg-surface-sunken p-2">
      <button
        onClick={startEditing}
        title="Edit patient details"
        className="absolute top-1.5 right-1.5 rounded-md p-1 text-ink-subtle opacity-0
          transition-opacity hover:bg-surface hover:text-ink group-hover:opacity-100"
      >
        <Pencil size={11} />
      </button>

      {!hasInfo ? (
        <button
          onClick={startEditing}
          className="flex items-center gap-1.5 text-left text-[11px] text-ink-subtle hover:text-ink-muted"
        >
          <UserRound size={12} className="shrink-0" />
          No patient data — add details
        </button>
      ) : (
        <div className="flex flex-col gap-1 pr-4">
          {info?.patientName && (
            <div className="flex items-center gap-1.5 text-xs font-semibold text-ink">
              <UserRound size={12} className="shrink-0 text-ink-subtle" />
              <span className="truncate">{info.patientName}</span>
            </div>
          )}
          {info?.patientId && (
            <div className="flex items-center gap-1.5 text-[10px] text-ink-muted">
              <Fingerprint size={11} className="shrink-0 text-ink-subtle" />
              <span className="truncate">{info.patientId}</span>
            </div>
          )}
          {info?.dateOfBirth && (
            <div className="flex items-center gap-1.5 text-[10px] text-ink-muted">
              <CalendarDays size={11} className="shrink-0 text-ink-subtle" />
              <span className="truncate">DOB {info.dateOfBirth}</span>
            </div>
          )}
          {info?.studyDate && (
            <div className="flex items-center gap-1.5 text-[10px] text-ink-muted">
              <CalendarDays size={11} className="shrink-0 text-ink-subtle" />
              <span className="truncate">Study {info.studyDate}</span>
            </div>
          )}
          {info?.modality && (
            <span className="mt-0.5 inline-block w-fit rounded bg-surface px-1.5 py-0.5 text-[9px] font-semibold tracking-wide text-ink-muted uppercase">
              {info.modality}
            </span>
          )}
        </div>
      )}
    </div>
  )
}
