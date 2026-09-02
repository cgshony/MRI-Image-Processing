import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict


class ImageOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    filename: str
    content_type: str
    width: int
    height: int
    parent_image_id: uuid.UUID | None
    created_at: datetime
