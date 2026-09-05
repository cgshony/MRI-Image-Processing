import io
import uuid
from pathlib import Path

from PIL import Image as PILImage
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.image import Image
from app.storage.base import StorageBackend


class ImageNotFoundError(Exception):
    """Raised when a requested image id has no matching row."""


class ImageService:
    """CRUD + storage orchestration for `Image` rows.

    Owns the DB row and the blob together: every write here keeps both in
    sync, and every read goes through here so callers never touch `storage`
    directly.
    """

    def __init__(self, session: AsyncSession, storage: StorageBackend):
        self._session = session
        self._storage = storage

    async def save_upload(self, data: bytes, filename: str, content_type: str) -> Image:
        """Decode, store, and record a freshly uploaded image."""
        return await self._persist(data, filename, content_type, parent_image_id=None)

    async def save_processed(
        self,
        data: bytes,
        filename: str,
        content_type: str,
        parent_image_id: uuid.UUID,
    ) -> Image:
        """Persist the output of a processing operation as a new Image row."""
        return await self._persist(data, filename, content_type, parent_image_id=parent_image_id)

    async def list(self) -> list[Image]:
        result = await self._session.execute(select(Image).order_by(Image.created_at.desc()))
        return list(result.scalars().all())

    async def get(self, image_id: uuid.UUID) -> Image:
        image = await self._session.get(Image, image_id)
        if image is None:
            raise ImageNotFoundError(image_id)
        return image

    async def get_bytes(self, image_id: uuid.UUID) -> tuple[bytes, str]:
        image = await self.get(image_id)
        data = await self._storage.load(image.storage_path)
        return data, image.content_type

    async def delete(self, image_id: uuid.UUID) -> None:
        image = await self.get(image_id)
        await self._storage.delete(image.storage_path)
        await self._session.delete(image)
        await self._session.flush()

    async def _persist(
        self,
        data: bytes,
        filename: str,
        content_type: str,
        parent_image_id: uuid.UUID | None,
    ) -> Image:
        width, height = self._decode_dimensions(data)
        image = Image(
            id=uuid.uuid4(),
            filename=filename,
            content_type=content_type,
            width=width,
            height=height,
            parent_image_id=parent_image_id,
        )
        image.storage_path = await self._storage.save(self._key_for(image.id, filename), data)
        self._session.add(image)
        await self._session.flush()
        await self._session.refresh(image)
        return image

    @staticmethod
    def _key_for(image_id: uuid.UUID, filename: str) -> str:
        suffix = Path(filename).suffix or ".bin"
        return f"{image_id}{suffix}"

    @staticmethod
    def _decode_dimensions(data: bytes) -> tuple[int, int]:
        """Raises PIL.UnidentifiedImageError (a subclass of OSError) if `data`
        isn't a decodable image; the images router turns that into a 422."""
        with PILImage.open(io.BytesIO(data)) as im:
            return im.width, im.height
