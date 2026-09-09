from __future__ import annotations

import anyio
import botocore.exceptions
import boto3

from app.storage.base import StorageBackend


class S3StorageBackend(StorageBackend):
    """Stores blobs as objects in an S3-compatible bucket (AWS S3, Cloudflare R2,
    MinIO, ...). `boto3` is synchronous, so every call is pushed to a worker
    thread via `anyio.to_thread.run_sync` to keep the storage interface async,
    matching LocalDiskStorage.
    """

    def __init__(
        self,
        bucket: str,
        *,
        endpoint_url: str | None = None,
        access_key_id: str | None = None,
        secret_access_key: str | None = None,
        region_name: str = "auto",
    ):
        self._bucket = bucket
        self._client = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
            region_name=region_name,
        )

    async def save(self, key: str, data: bytes) -> str:
        await anyio.to_thread.run_sync(
            lambda: self._client.put_object(Bucket=self._bucket, Key=key, Body=data)
        )
        return key

    async def load(self, storage_path: str) -> bytes:
        try:
            response = await anyio.to_thread.run_sync(
                lambda: self._client.get_object(Bucket=self._bucket, Key=storage_path)
            )
        except botocore.exceptions.ClientError as exc:
            if exc.response.get("Error", {}).get("Code") in ("NoSuchKey", "404"):
                raise FileNotFoundError(storage_path) from exc
            raise
        return await anyio.to_thread.run_sync(response["Body"].read)

    async def delete(self, storage_path: str) -> None:
        # S3's DeleteObject is a no-op for a missing key, matching
        # LocalDiskStorage.delete's missing_ok behaviour for free.
        await anyio.to_thread.run_sync(
            lambda: self._client.delete_object(Bucket=self._bucket, Key=storage_path)
        )
