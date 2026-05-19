from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class VideoStatus(StrEnum):
    queued = "queued"
    processing = "processing"
    completed = "completed"
    failed = "failed"


class VideoUploadResponse(BaseModel):
    video_id: str
    status: VideoStatus


class VideoStatusResponse(BaseModel):
    video_id: str
    status: VideoStatus
    original_name: str
    created_at: datetime
    updated_at: datetime
    error: str | None = None
    csv_ready: bool = False


class VideoRecord(BaseModel):
    video_id: str = Field(..., alias="_id")
    original_name: str
    content_type: str | None = None
    file_path: str
    csv_path: str | None = None
    status: VideoStatus
    error: str | None = None
    created_at: datetime
    updated_at: datetime
