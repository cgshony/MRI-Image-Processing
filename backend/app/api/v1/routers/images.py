import uuid

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import Response

from app.api.v1.deps import ImageServiceDep
from app.schemas.image import ImageOut
from app.services.image_service import ImageNotFoundError

router = APIRouter(prefix="/images", tags=["images"])


@router.post("", response_model=ImageOut, status_code=201)
async def upload_image(image_service: ImageServiceDep, file: UploadFile = File(...)):
    data = await file.read()
    try:
        image = await image_service.save_upload(
            data, file.filename, file.content_type or "application/octet-stream"
        )
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not decode uploaded file as an image: {exc}")
    return image


@router.get("", response_model=list[ImageOut])
async def list_images(image_service: ImageServiceDep):
    return await image_service.list()


@router.get("/{image_id}", response_model=ImageOut)
async def get_image(image_id: uuid.UUID, image_service: ImageServiceDep):
    try:
        return await image_service.get(image_id)
    except ImageNotFoundError:
        raise HTTPException(status_code=404, detail="Image not found")


@router.get("/{image_id}/file")
async def get_image_file(image_id: uuid.UUID, image_service: ImageServiceDep):
    try:
        data, content_type = await image_service.get_bytes(image_id)
    except ImageNotFoundError:
        raise HTTPException(status_code=404, detail="Image not found")
    return Response(content=data, media_type=content_type)


@router.delete("/{image_id}", status_code=204)
async def delete_image(image_id: uuid.UUID, image_service: ImageServiceDep):
    try:
        await image_service.delete(image_id)
    except ImageNotFoundError:
        raise HTTPException(status_code=404, detail="Image not found")
