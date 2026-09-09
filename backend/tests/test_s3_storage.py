"""Unit tests for S3StorageBackend, against a mocked S3 API (moto) -- no real
network calls or credentials involved. Mirrors test_storage.py's coverage of
LocalDiskStorage so both backends are held to the same contract."""

import boto3
import pytest
from moto import mock_aws

from app.storage.s3 import S3StorageBackend

BUCKET = "test-bucket"


@pytest.fixture
def storage():
    with mock_aws():
        boto3.client("s3", region_name="us-east-1").create_bucket(Bucket=BUCKET)
        yield S3StorageBackend(BUCKET, region_name="us-east-1")


async def test_save_then_load_roundtrips_bytes(storage):
    storage_path = await storage.save("abc.png", b"hello world")

    assert storage_path == "abc.png"
    assert await storage.load(storage_path) == b"hello world"


async def test_save_accepts_nested_keys(storage):
    await storage.save("nested/dir/file.png", b"data")

    assert await storage.load("nested/dir/file.png") == b"data"


async def test_delete_removes_the_blob(storage):
    storage_path = await storage.save("abc.png", b"hello")

    await storage.delete(storage_path)

    with pytest.raises(FileNotFoundError):
        await storage.load(storage_path)


async def test_delete_missing_key_is_a_noop(storage):
    await storage.delete("never-existed.png")  # must not raise


async def test_load_missing_key_raises_file_not_found(storage):
    with pytest.raises(FileNotFoundError):
        await storage.load("never-existed.png")
