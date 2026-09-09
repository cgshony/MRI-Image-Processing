from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration, sourced from environment variables / .env."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "MRI Image Processing API"
    api_v1_prefix: str = "/api/v1"

    database_url: str = "postgresql+asyncpg://mri:mri@localhost:5432/mri"

    # Which StorageBackend to use -- "local" (default, disk under storage_dir,
    # ephemeral on hosts without a persistent volume) or "s3" (any S3-compatible
    # object store, e.g. Cloudflare R2 -- see the s3_* settings below).
    storage_backend: Literal["local", "s3"] = "local"

    # Directory (inside the container/host) where uploaded and processed images are
    # stored. Only used when storage_backend == "local".
    storage_dir: Path = Path("storage")

    # Only used when storage_backend == "s3". s3_endpoint_url is required for
    # S3-compatible providers (Cloudflare R2, MinIO, ...); leave unset for real AWS S3.
    s3_bucket: str | None = None
    s3_endpoint_url: str | None = None
    s3_access_key_id: str | None = None
    s3_secret_access_key: str | None = None
    s3_region: str = "auto"

    cors_origins: list[str] = ["*"]

    def model_post_init(self, __context) -> None:
        if self.storage_backend == "local":
            self.storage_dir.mkdir(parents=True, exist_ok=True)
        elif self.storage_backend == "s3" and not self.s3_bucket:
            raise ValueError("S3_BUCKET is required when STORAGE_BACKEND=s3")


@lru_cache
def get_settings() -> Settings:
    return Settings()
