import uuid

from fastapi import APIRouter, BackgroundTasks, HTTPException

from app.api.v1.deps import ProcessingServiceDep
from app.schemas.processing import ProcessingJobOut, ProcessRequest
from app.services.image_service import ImageNotFoundError
from app.services.processing_service import JobNotFoundError

router = APIRouter(tags=["processing"])


@router.post("/images/{image_id}/process", response_model=ProcessingJobOut, status_code=202)
async def process_image(
    image_id: uuid.UUID,
    body: ProcessRequest,
    background_tasks: BackgroundTasks,
    processing_service: ProcessingServiceDep,
):
    try:
        job = await processing_service.create_job(image_id, body.operation, body.params)
    except ImageNotFoundError:
        raise HTTPException(status_code=404, detail="Image not found")
    background_tasks.add_task(processing_service.run_job, job.id)
    return job


@router.get("/jobs/{job_id}", response_model=ProcessingJobOut)
async def get_job(job_id: uuid.UUID, processing_service: ProcessingServiceDep):
    try:
        return await processing_service.get_job(job_id)
    except JobNotFoundError:
        raise HTTPException(status_code=404, detail="Processing job not found")
