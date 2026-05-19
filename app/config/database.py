from collections.abc import AsyncIterator

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

from app.config.settings import Settings

_client: AsyncIOMotorClient | None = None
_database: AsyncIOMotorDatabase | None = None


async def init_mongo(settings: Settings) -> None:
    global _client, _database
    _client = AsyncIOMotorClient(settings.mongo_uri)
    _database = _client[settings.mongo_db]


async def close_mongo() -> None:
    global _client, _database
    if _client is not None:
        _client.close()
    _client = None
    _database = None


def get_database() -> AsyncIOMotorDatabase:
    if _database is None:
        raise RuntimeError("MongoDB is not initialized")
    return _database


async def get_database_dependency() -> AsyncIterator[AsyncIOMotorDatabase]:
    yield get_database()
