from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Video Processing Service"
    api_prefix: str = "/api"
    mongo_uri: str = Field(default="mongodb://localhost:27017")
    mongo_db: str = Field(default="video_service")
    redis_url: str = Field(default="redis://localhost:6379/0")
    storage_dir: Path = Field(default=Path("storage"))
    video_storage_dir: Path = Field(default=Path("storage/videos"))
    csv_storage_dir: Path = Field(default=Path("storage/csv"))
    ml_models_dir: Path = Field(default=Path("ai_pipeline"))

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    @property
    def celery_broker_url(self) -> str:
        return self.redis_url

    @property
    def celery_result_backend(self) -> str:
        return self.redis_url


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.storage_dir.mkdir(parents=True, exist_ok=True)
    settings.video_storage_dir.mkdir(parents=True, exist_ok=True)
    settings.csv_storage_dir.mkdir(parents=True, exist_ok=True)
    return settings
