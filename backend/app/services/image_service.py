import io
import uuid

import numpy as np
from PIL import Image as PILImage
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.image import Image
from app.storage.base import StorageBackend


class ImageNotFoundError(Exception):
    pass


class ImageService:
    pass