# MRI Image Processing

**Live demo:** _(add the deployed Vercel URL here once available)_

An async FastAPI backend for uploading MRI-style images and running classical image-processing operations on them as background jobs — built as a hands-on way to learn image processing and modern FastAPI/SQLAlchemy architecture, and evolving toward a small set of real, non-diagnostic problems in medical imaging.

## Motivation

This project started as a learning exercise: implement classic image-processing algorithms (wavelet transforms, interpolation-based resampling, pseudocolour mapping) from scratch, and pair that with a properly async, production-shaped FastAPI backend rather than a notebook. That part has worked — the processing math and the service architecture are both solid.

What the project didn't have was a *problem*. Wavelet transforms and bicubic upsampling are techniques, not products. So alongside continuing to build out the backend, I researched where those techniques (and their more modern deep-learning equivalents) actually create value in medical imaging today, and where the market has real, addressable gaps versus where it's a mature space with incumbents and heavy regulation. That research is summarized below and drives the roadmap.

## Current state

**Architecture**

- FastAPI app (`app/main.py`) with a versioned API under `/api/v1`.
- Async SQLAlchemy (`asyncpg`) + Alembic migrations; Postgres in Docker, SQLite (`aiosqlite`) for the test suite.
- A `StorageBackend` interface (`app/storage/`) with a local-disk implementation — swappable for object storage without touching services.
- Two persisted entities: `Image` (uploaded/derived files with basic raster metadata) and `ProcessingJob` (an operation applied to an image, tracked through `pending → running → done/failed`).
- Processing runs as a FastAPI `BackgroundTask`: a job is created and returned immediately (`202 Accepted`), then executed against its own DB session once the request has completed.

**API surface**

| Endpoint | Purpose |
|---|---|
| `POST /api/v1/images` | Upload an image |
| `GET /api/v1/images` / `GET /api/v1/images/{id}` | List / fetch image metadata |
| `GET /api/v1/images/{id}/file` | Fetch the raw file bytes |
| `DELETE /api/v1/images/{id}` | Delete an image |
| `POST /api/v1/images/{id}/process` | Queue a processing job (`operation`, `params`) |
| `GET /api/v1/jobs/{id}` | Poll job status / result |

**Processing operations** (`app/processing/`, dispatched via `OPERATIONS` in `processing_service.py`)

- `scale_image` — nearest-neighbour resampling
- `bicubic_upsample` — bicubic interpolation
- `wavelet_enhance` — 2D Haar wavelet transform with high-frequency band enhancement
- `colourize` — pseudocolour (HSV-ramp) mapping of grayscale intensity

**Tests** — `backend/tests/` covers app boot, the images API, the processing API, and the storage layer.

**Known gaps** (see roadmap): no DICOM support yet — images are handled as generic raster files (width/height, content-type), not as studies/series with clinical metadata. This is the main blocker for most of the directions below.

## Business research: what the market actually needs

Medical imaging software is a mature, well-funded market (the AI-enhanced MRI system market alone is estimated at ~$11B in 2026, growing to ~$17B by 2032), so the opportunity for a small project isn't "build a PACS" — it's finding the seams where established tools structurally strain:

- **Open-source viewers/toolkits have real gaps.** OHIF (browser-based, DICOMweb-native) has limited 3D/extensibility and no IT-independent deployment story; 3D Slicer (deep desktop toolkit) has no built-in anonymization or collaboration layer for handling sensitive data at scale.
- **PACS vendor lock-in is a named, recurring complaint.** Small imaging centers want cloud-native, API-first, DICOM/HL7-standard tooling they can point at whatever system they already run, instead of custom-coding against a closed PACS.
- **De-identification is only half-solved.** Plenty of tools scrub DICOM metadata tags; far fewer handle pixel-level PHI (burned-in patient names/IDs), which is exactly what blocks labs from legally sharing imaging data for research or AI training.
- **Low-field / portable MRI has a quality tax.** Portable scanners trade signal-to-noise and resolution for cost and access. Deep-learning enhancement (denoising, super-resolution) of these images is an active 2026 research area, not yet a packaged, easy-to-deploy tool — and it's the direction closest to what this codebase already does.
- **Access is a genuine crisis, not just a workflow one.** India has under 1 MRI scanner per million people; eleven African countries have none at all. Teleradiology and software that make a scarce radiologist's time go further are recognized as part of the solution — but "decision support" (QC, triage, flagging scan quality) is a very different regulatory animal from "diagnosis."

**The regulatory line matters more than anything else here.** The moment software output is used to diagnose, characterize, or rule out disease in a specific patient, it becomes Class II Software as a Medical Device under the FDA (510(k), ISO 14971 risk management, a documented QMS, clinical validation — a multi-year, well-funded undertaking). Image enhancement, denoising, de-identification, format conversion, teaching tools, veterinary imaging, and research-only pipelines sit outside that wall, as long as outputs are never marketed or used for diagnosis. That distinction is why the roadmap below deliberately stays on the research/infrastructure/veterinary side for now.

## Key areas being targeted

In rough order of how directly they build on the current codebase:

1. **DICOM de-identification** — tag *and* pixel-level PHI removal with an audit trail, for research labs and AI teams that need to legally share or train on imaging data. No diagnostic claim, fast to build once DICOM support lands.
2. **Low-field / portable MRI enhancement** — denoising and super-resolution for the image-quality gap in portable and resource-limited-setting scanners. Directly extends the wavelet/interpolation work already in `app/processing/`, framed for research and veterinary use rather than diagnosis.
3. **Vendor-neutral processing microservice** — a DICOMweb-speaking API that any existing PACS/viewer can call for enhancement or conversion, aimed at small imaging centers that explicitly want to avoid being locked into one vendor's tooling.
4. **Veterinary imaging** — same enhancement pipeline, aimed at a market that runs older/secondhand human-market hardware with almost no vendor tooling built for it, and carries none of the human-subject regulatory weight.
5. **A teaching/visualization tool** — packaging the existing Haar transform, bicubic interpolation, and pseudocolour code as an interactive teaching aid for imaging/CS courses. Lowest effort, lowest ceiling, but genuinely close to shippable today.

Deliberately **not** a near-term target: anomaly/lesion flagging, worklist triage by clinical urgency, or AI-assisted report drafting. These map to real, cited market demand (radiologist shortage and burnout), but they're squarely Class II SaMD territory — a future direction if this project gains traction and dedicated resourcing, not a v1.

## Future development

- [ ] **DICOM ingestion** — replace generic raster metadata with `pydicom`-backed study/series/modality/pixel-spacing metadata on `Image`. Prerequisite for everything below.
- [ ] Pixel-level + metadata de-identification pipeline with an audit log.
- [ ] Replace/augment classical `scale_image`/`bicubic_upsample` with a trained denoising/super-resolution model for low-field MRI, evaluated against a public low-field dataset.
- [ ] DICOMweb-compatible API surface so the service can sit in front of an existing PACS/OHIF deployment.
- [ ] Batch/async job queue (Celery or similar) once processing moves beyond single-request `BackgroundTasks`.
- [ ] Provenance/consent tracking on the data model, in case a future direction ever needs to support a regulatory submission — far cheaper to build in now than retrofit later.

## Running locally

```bash
docker compose up --build
```

Or without Docker (Postgres running separately):

```bash
cd backend
cp ../.env.example .env   # adjust DATABASE_URL / STORAGE_DIR as needed
pip install -e ".[dev]"
alembic upgrade head
uvicorn app.main:app --reload
```

Run tests:

```bash
cd backend
pytest
```

## Deployment

Free-tier deployment across three providers:

- **Frontend → [Vercel](https://vercel.com)** — static Vite build, project root `frontend/`, env var `VITE_API_BASE_URL` pointed at the backend's `/api/v1`.
- **Backend → [Render](https://render.com)** (free Web Service) — deployed from `backend/Dockerfile` (or the [render.yaml](render.yaml) Blueprint in the repo root), env vars `DATABASE_URL` and `CORS_ORIGINS` set in the Render dashboard.
- **Database → [Neon](https://neon.tech)** (free Postgres) — used instead of Render's own free Postgres, which auto-deletes 30 days after creation; Neon's free tier doesn't expire.

Known free-tier trade-offs:

- Render's free instance spins down after ~15 min idle; the first request afterward takes ~30-60s to wake up.
- By default, uploaded/processed images live on the backend's local disk (`STORAGE_DIR`), which is **ephemeral** on Render's free tier — files (and their disk-backed bytes) are lost on restart/redeploy even though their database rows survive. Fix this by switching to the S3-compatible storage backend (below).

### Object storage (Cloudflare R2)

`app/storage/` defines a `StorageBackend` interface with two implementations: `LocalDiskStorage` (default) and `S3StorageBackend`, which works with any S3-compatible provider. To use Cloudflare R2's free tier (10GB storage, no egress fees):

1. Create an R2 bucket in the Cloudflare dashboard, then an API token scoped to it (Account → R2 → Manage API Tokens) to get an access key ID and secret.
2. Set these env vars (on Render, and in `backend/.env` for local testing):
   - `STORAGE_BACKEND=s3`
   - `S3_BUCKET=<your bucket name>`
   - `S3_ENDPOINT_URL=https://<account id>.r2.cloudflarestorage.com`
   - `S3_ACCESS_KEY_ID=<access key id>`
   - `S3_SECRET_ACCESS_KEY=<secret access key>`
3. Redeploy. New uploads/processed images now persist in R2 regardless of backend restarts. (Existing rows created under `local` storage before the switch won't have a matching R2 object — re-upload them.)

## Frontend

A React + TypeScript workspace UI lives in `frontend/` — upload an image, run one of the four
processing operations, watch the job poll to completion, and compare original vs. result. It's a
fully decoupled app (Vite + Tailwind CSS + TanStack Query) that talks to the API over HTTP; it
doesn't touch or get served by the backend.

```bash
cd frontend
npm install
cp .env.example .env   # VITE_API_BASE_URL, defaults to http://localhost:8000/api/v1
npm run dev
```

Expects the backend running at that URL (see above). See [frontend/README.md](frontend/README.md)
for structure and scripts.

## Disclaimer

This project processes images for research, educational, and infrastructure purposes. It is not a medical device, is not intended for clinical diagnosis, and has not been evaluated by the FDA or any equivalent regulatory body.
