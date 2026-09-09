from typing import Annotated

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
from app.db.session import get_session
from app.services.image_service import ImageService
from app.services.processing_service import ProcessingService
from app.storage.base import StorageBackend
from app.storage.local_disk import LocalDiskStorage
from app.storage.s3 import S3StorageBackend

SessionDep = Annotated[AsyncSession, Depends(get_session)]
SettingsDep = Annotated[Settings, Depends(get_settings)]


def get_storage(settings: SettingsDep) -> StorageBackend:
    if settings.storage_backend == "s3":
        return S3StorageBackend(
            settings.s3_bucket,
            endpoint_url=settings.s3_endpoint_url,
            access_key_id=settings.s3_access_key_id,
            secret_access_key=settings.s3_secret_access_key,
            region_name=settings.s3_region,
        )
    return LocalDiskStorage(settings.storage_dir)


StorageDep = Annotated[StorageBackend, Depends(get_storage)]


def get_image_service(session: SessionDep, storage: StorageDep) -> ImageService:
    return ImageService(session, storage)


def get_processing_service(session: SessionDep, storage: StorageDep) -> ProcessingService:
    return ProcessingService(session, storage)


ImageServiceDep = Annotated[ImageService, Depends(get_image_service)]
ProcessingServiceDep = Annotated[ProcessingService, Depends(get_processing_service)]
