from abc import ABC, abstractmethod


class StorageBackend(ABC):
    """Abstract interface for persisting image blobs.

    An implementation stores bytes under an opaque key and hands back a
    `storage_path` string that the caller (ImageService) records on the
    `Image` row and passes back unchanged on later load/delete calls. Callers
    never need to know whether that string is a local file path, an S3 key,
    or anything else.
    """

    @abstractmethod
    async def save(self, key: str, data: bytes) -> str:
        """Persist `data` under `key`, returning the storage_path to record."""

    @abstractmethod
    async def load(self, storage_path: str) -> bytes:
        """Return the bytes stored at `storage_path`."""

    @abstractmethod
    async def delete(self, storage_path: str) -> None:
        """Remove the blob at `storage_path`. No-op if it doesn't exist."""
