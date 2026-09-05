from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration, sourced from environment variables / .env."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "MRI Image Processing API"
    api_v1_prefix: str = "/api/v1"

    database_url: str = "postgresql+asyncpg://mri:mri@localhost:5432/mri"

    # Directory (inside the container/host) where uploaded and processed images are stored.
    storage_dir: Path = Path("storage")

    cors_origins: list[str] = ["*"]

    def model_post_init(self, __context) -> None:
        self.storage_dir.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    return Settings()
