from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import aiofiles
from fastapi import HTTPException, UploadFile, status
from motor.motor_asyncio import AsyncIOMotorDatabase

from app.config.settings import Settings
from app.job.schemas.models import VideoStatus, VideoStatusResponse, VideoUploadResponse


class VideoService:
    def __init__(self, database: AsyncIOMotorDatabase, settings: Settings) -> None:
        self._database = database
        self._settings = settings
        self._collection = database["videos"]

    async def create_video(self, uploaded_file: UploadFile) -> tuple[VideoUploadResponse, Path]:
        video_id = uuid4().hex
        file_suffix = Path(uploaded_file.filename or "video").suffix or ".mp4"
        file_path = self._settings.video_storage_dir / f"{video_id}{file_suffix}"

        await self._save_file(uploaded_file, file_path)

        now = datetime.now(timezone.utc)
        await self._collection.insert_one(
            {
                "_id": video_id,
                "original_name": uploaded_file.filename or file_path.name,
                "content_type": uploaded_file.content_type,
                "file_path": str(file_path),
                "csv_path": None,
                "status": VideoStatus.queued.value,
                "error": None,
                "created_at": now,
                "updated_at": now,
            }
        )

        return VideoUploadResponse(video_id=video_id, status=VideoStatus.queued), file_path

    async def get_video_status(self, video_id: str) -> VideoStatusResponse:
        document = await self._collection.find_one({"_id": video_id})
        if document is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Video not found")

        return VideoStatusResponse(
            video_id=document["_id"],
            status=VideoStatus(document["status"]),
            original_name=document["original_name"],
            created_at=document["created_at"],
            updated_at=document["updated_at"],
            error=document.get("error"),
            csv_ready=document.get("csv_path") is not None and document["status"] == VideoStatus.completed.value,
        )

    async def get_video_csv_path(self, video_id: str) -> Path:
        document = await self._collection.find_one({"_id": video_id})
        if document is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Video not found")

        if document["status"] != VideoStatus.completed.value:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Video is not processed yet")

        csv_path = document.get("csv_path")
        if not csv_path:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="CSV file not found")

        return Path(csv_path)

    async def mark_processing(self, video_id: str) -> None:
        await self._update_status(video_id, VideoStatus.processing)

    async def mark_completed(self, video_id: str, csv_path: Path) -> None:
        await self._collection.update_one(
            {"_id": video_id},
            {
                "$set": {
                    "status": VideoStatus.completed.value,
                    "csv_path": str(csv_path),
                    "error": None,
                    "updated_at": datetime.now(timezone.utc),
                }
            },
        )

    async def mark_failed(self, video_id: str, error_message: str) -> None:
        await self._collection.update_one(
            {"_id": video_id},
            {
                "$set": {
                    "status": VideoStatus.failed.value,
                    "error": error_message,
                    "updated_at": datetime.now(timezone.utc),
                }
            },
        )

    async def _update_status(self, video_id: str, status_value: VideoStatus) -> None:
        await self._collection.update_one(
            {"_id": video_id},
            {"$set": {"status": status_value.value, "updated_at": datetime.now(timezone.utc)}},
        )

    @staticmethod
    async def _save_file(uploaded_file: UploadFile, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        await uploaded_file.seek(0)
        async with aiofiles.open(destination, mode="wb") as file_handle:
            while chunk := await uploaded_file.read(1024 * 1024):
                await file_handle.write(chunk)
