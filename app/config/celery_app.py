from celery import Celery

from app.config.settings import get_settings

settings = get_settings()

celery_app = Celery(
    "video_processing_service",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=["app.worker.tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
)

# Import worker tasks so they are registered when the Celery app is imported
# This ensures tasks defined in `app.worker.tasks` are available to the worker
try:
    import app.worker.tasks as _  # noqa: F401
except Exception:
    # Import errors should not prevent Celery app creation; they will surface
    # later when the worker starts and the module is imported in the container.
    pass
