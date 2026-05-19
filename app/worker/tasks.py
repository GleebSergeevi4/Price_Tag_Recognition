from __future__ import annotations

import asyncio
from pathlib import Path

from motor.motor_asyncio import AsyncIOMotorClient

from app.config.celery_app import celery_app
from app.config.settings import get_settings
from app.job.schemas.models import VideoStatus
from app.job.services.video import VideoService
from app.worker.pipeline import run_pipeline, set_models_dir


def _process_video_sync(video_id: str) -> None:
    settings = get_settings()
    # Initialize models directory
    set_models_dir(settings.ml_models_dir)
    asyncio.run(_process_video_async(video_id))

# Register the task under the new module path
process_video = celery_app.task(name="app.worker.tasks.process_video")(_process_video_sync)
# Also register under the old name to support queued tasks from previous deployments
process_video_legacy = celery_app.task(name="app.tasks.video_tasks.process_video")(_process_video_sync)


async def _process_video_async(video_id: str) -> None:
    settings = get_settings()
    client = AsyncIOMotorClient(settings.mongo_uri)
    try:
        database = client[settings.mongo_db]
        service = VideoService(database, settings)
        await service.mark_processing(video_id)

        document = await database["videos"].find_one({"_id": video_id})
        if document is None:
            await service.mark_failed(video_id, "Video record disappeared")
            return

        video_path = Path(document["file_path"])
        csv_path = settings.csv_storage_dir / f"{video_id}.csv"

        try:
            await run_pipeline(video_path, csv_path)
            await service.mark_completed(video_id, csv_path)
        except Exception as exc:  # noqa: BLE001
            await service.mark_failed(video_id, str(exc))
    finally:
        client.close()
