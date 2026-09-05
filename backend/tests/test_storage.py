"""Unit tests for LocalDiskStorage -- no DB or FastAPI app involved."""

import pytest

from app.storage.local_disk import LocalDiskStorage


async def test_save_then_load_roundtrips_bytes(tmp_path):
    storage = LocalDiskStorage(tmp_path)

    storage_path = await storage.save("abc.png", b"hello world")

    assert storage_path == "abc.png"
    assert (tmp_path / "abc.png").read_bytes() == b"hello world"
    assert await storage.load(storage_path) == b"hello world"


async def test_save_creates_nested_directories(tmp_path):
    storage = LocalDiskStorage(tmp_path)

    await storage.save("nested/dir/file.png", b"data")

    assert (tmp_path / "nested" / "dir" / "file.png").read_bytes() == b"data"


async def test_delete_removes_the_blob(tmp_path):
    storage = LocalDiskStorage(tmp_path)
    storage_path = await storage.save("abc.png", b"hello")

    await storage.delete(storage_path)

    assert not (tmp_path / "abc.png").exists()
    with pytest.raises(FileNotFoundError):
        await storage.load(storage_path)


async def test_delete_missing_key_is_a_noop(tmp_path):
    storage = LocalDiskStorage(tmp_path)

    await storage.delete("never-existed.png")  # must not raise


async def test_rejects_paths_that_escape_the_storage_root(tmp_path):
    storage = LocalDiskStorage(tmp_path)

    with pytest.raises(ValueError):
        await storage.load("../outside.png")
    with pytest.raises(ValueError):
        await storage.delete("../../etc/passwd")
