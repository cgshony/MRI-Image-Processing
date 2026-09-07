import uuid
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from app.models.processing_job import JobStatus
from app.services.processing_service import OPERATION_NAMES


class ProcessRequest(BaseModel):
    operation: Literal[OPERATION_NAMES]  # type: ignore[valid-type]
    params: dict[str, Any] = {}


class ChannelOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    key: str
    label: str
    image_id: uuid.UUID


class ProcessingJobOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    image_id: uuid.UUID
    operation: str
    params: dict[str, Any]
    status: JobStatus
    result_image_id: uuid.UUID | None
    channels: list[ChannelOut] | None
    error: str | None
    created_at: datetime
    completed_at: datetime | None
