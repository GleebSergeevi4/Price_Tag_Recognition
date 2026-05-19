from pathlib import Path

from fastapi import APIRouter, Depends, File, UploadFile
from fastapi.responses import FileResponse
from motor.motor_asyncio import AsyncIOMotorDatabase

from app.config.settings import Settings, get_settings
from app.config.database import get_database_dependency
from app.job.schemas.models import VideoStatusResponse, VideoUploadResponse
from app.job.services.video import VideoService

router = APIRouter(prefix="/videos", tags=["videos"])


def get_video_service(
    database: AsyncIOMotorDatabase = Depends(get_database_dependency),
    settings: Settings = Depends(get_settings),
) -> VideoService:
    return VideoService(database, settings)


@router.post("/upload", response_model=VideoUploadResponse)
async def upload_video(
    uploaded_file: UploadFile = File(...),
    service: VideoService = Depends(get_video_service),
) -> VideoUploadResponse:
    response, _ = await service.create_video(uploaded_file)
    # Import task lazily to avoid circular imports at module import time
    from app.worker.tasks import process_video

    process_video.delay(response.video_id)
    return response


@router.get("/{video_id}/status", response_model=VideoStatusResponse)
async def get_video_status(
    video_id: str,
    service: VideoService = Depends(get_video_service),
) -> VideoStatusResponse:
    return await service.get_video_status(video_id)


@router.get("/{video_id}/csv")
async def download_video_csv(
    video_id: str,
    service: VideoService = Depends(get_video_service),
) -> FileResponse:
    csv_path = await service.get_video_csv_path(video_id)
    return FileResponse(
        path=csv_path,
        media_type="text/csv",
        filename=f"{video_id}.csv",
    )
