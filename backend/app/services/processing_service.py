import io
import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

import numpy as np
from PIL import Image as PILImage
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import async_session_factory
from app.models.processing_job import JobStatus, ProcessingJob
from app.processing.bicubic_upsample import bicubic_upsample
from app.processing.colourize import create_pseudo_color_image, find_min_max
from app.processing.scale_image import scale_image
from app.processing.wavelet_haar_transform import (
    enhance_high_frequency_bands,
    haar_transform_2d,
    inverse_haar_transform_2d,
)
from app.services.image_service import ImageService
from app.storage.base import StorageBackend


class JobNotFoundError(Exception):
    """Raised when a requested processing job id has no matching row."""


def _op_bicubic_upsample(image: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    scale_factor = float(params.get("scale_factor", 2.0))
    return bicubic_upsample(image, scale_factor)


def _op_scale_image(image: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    scale_factor = float(params.get("scale_factor", 2.0))
    return scale_image(image, scale_factor)


def _op_wavelet_enhance(image: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    factor = float(params.get("factor", 1.5))
    transformed = haar_transform_2d(image)
    enhanced = enhance_high_frequency_bands(transformed, factor)
    return inverse_haar_transform_2d(enhanced)


def _op_colourize(image: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    height, width = image.shape
    minval, maxval = find_min_max(image)
    coloured = create_pseudo_color_image(image, width, height, minval, maxval)
    return np.array(coloured)


# Dispatch table used instead of a Processor abstraction (deferred to Phase 2).
# Each entry takes a grayscale (or, for colourize's output, RGB) numpy array
# plus the request's `params` dict and returns the resulting array.
OPERATIONS: dict[str, Callable[[np.ndarray, dict[str, Any]], np.ndarray]] = {
    "bicubic_upsample": _op_bicubic_upsample,
    "scale_image": _op_scale_image,
    "wavelet_enhance": _op_wavelet_enhance,
    "colourize": _op_colourize,
}

OPERATION_NAMES = tuple(OPERATIONS)


def _load_grayscale_array(data: bytes) -> np.ndarray:
    with PILImage.open(io.BytesIO(data)) as im:
        return np.array(im.convert("L"), dtype=np.float64)


def _array_to_png_bytes(array: np.ndarray) -> tuple[bytes, str]:
    pixels = np.clip(array, 0, 255).astype(np.uint8)
    if pixels.ndim == 2:
        mode = "L"
    elif pixels.ndim == 3 and pixels.shape[2] == 3:
        mode = "RGB"
    else:
        raise ValueError(f"Unsupported array shape for PNG encoding: {pixels.shape}")
    buffer = io.BytesIO()
    PILImage.fromarray(pixels, mode=mode).save(buffer, format="PNG")
    return buffer.getvalue(), "image/png"


class ProcessingService:
    """Creates processing jobs and runs them via the OPERATIONS dispatch table.

    `create_job`/`get_job` use the request-scoped session passed in. `run_job`
    is handed to `BackgroundTasks` and runs *after* the response is sent, by
    which point that request-scoped session is already closed - so it opens
    its own session instead of reusing `self._session`.
    """

    def __init__(self, session: AsyncSession, storage: StorageBackend):
        self._session = session
        self._storage = storage
        self._images = ImageService(session, storage)

    async def create_job(
        self, image_id: uuid.UUID, operation: str, params: dict[str, Any]
    ) -> ProcessingJob:
        await self._images.get(image_id)  # raises ImageNotFoundError if missing
        job = ProcessingJob(
            id=uuid.uuid4(),
            image_id=image_id,
            operation=operation,
            params=params,
            status=JobStatus.PENDING,
        )
        self._session.add(job)
        # Commit (not just flush) before returning: `run_job` is about to be
        # handed to BackgroundTasks and will look this row up through its own,
        # separate connection. That happens before this request's session-scoped
        # dependency gets to commit on our behalf, so under normal read-committed
        # isolation `run_job` would see no row at all - a job created but never
        # actually processed - unless we commit it ourselves right here.
        await self._session.commit()
        await self._session.refresh(job)
        return job

    async def get_job(self, job_id: uuid.UUID) -> ProcessingJob:
        job = await self._session.get(ProcessingJob, job_id)
        if job is None:
            raise JobNotFoundError(job_id)
        return job

    async def run_job(self, job_id: uuid.UUID) -> None:
        async with async_session_factory() as session:
            job = await session.get(ProcessingJob, job_id)
            if job is None:
                return

            images = ImageService(session, self._storage)
            try:
                job.status = JobStatus.RUNNING
                await session.flush()

                source = await images.get(job.image_id)
                data, _content_type = await images.get_bytes(job.image_id)
                array = _load_grayscale_array(data)

                operation = OPERATIONS[job.operation]
                result_array = operation(array, job.params)
                result_bytes, content_type = _array_to_png_bytes(result_array)

                result_image = await images.save_processed(
                    result_bytes,
                    filename=f"{job.operation}_{source.filename}",
                    content_type=content_type,
                    parent_image_id=source.id,
                )

                job.status = JobStatus.DONE
                job.result_image_id = result_image.id
            except Exception as exc:
                await session.rollback()
                job = await session.get(ProcessingJob, job_id)
                job.status = JobStatus.FAILED
                job.error = str(exc)

            job.completed_at = datetime.now(timezone.utc)
            await session.commit()
