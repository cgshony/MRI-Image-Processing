from pathlib import Path

import anyio

from app.storage.base import StorageBackend


class LocalDiskStorage(StorageBackend):
    """Stores blobs as files under a root directory on local disk."""

    def __init__(self, root: Path):
        self._root = root

    async def save(self, key: str, data: bytes) -> str:
        path = self._root / key
        await anyio.Path(path.parent).mkdir(parents=True, exist_ok=True)
        await anyio.Path(path).write_bytes(data)
        return key

    async def load(self, storage_path: str) -> bytes:
        return await anyio.Path(self._resolve(storage_path)).read_bytes()

    async def delete(self, storage_path: str) -> None:
        await anyio.Path(self._resolve(storage_path)).unlink(missing_ok=True)

    def _resolve(self, storage_path: str) -> Path:
        """Resolve `storage_path` against the root, rejecting escapes."""
        root = self._root.resolve()
        path = (root / storage_path).resolve()
        if path != root and root not in path.parents:
            raise ValueError(f"Refusing to access path outside storage root: {storage_path!r}")
        return path
