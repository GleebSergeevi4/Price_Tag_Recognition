from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.config.celery_app import celery_app
from app.config.database import close_mongo, init_mongo
from app.config.settings import get_settings
from app.job.router.videos import router as job_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    await init_mongo(settings)
    app.state.settings = settings
    app.state.celery_app = celery_app
    yield
    await close_mongo()


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title=settings.app_name, lifespan=lifespan)
    
    # Include job router
    app.include_router(job_router, prefix=settings.api_prefix)
    
    return app


app = create_app()
