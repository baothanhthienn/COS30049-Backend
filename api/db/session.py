from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from api.core.config import settings


def _normalise_url(url: str) -> str:
    for prefix in ("postgresql://", "postgres://"):
        if url.startswith(prefix):
            return "postgresql+asyncpg://" + url[len(prefix):]
    return url


def _build_engine():
    connect_args: dict = {}
    if settings.database_ssl:
        connect_args["ssl"] = "require"
    return create_async_engine(
        _normalise_url(settings.database_url),
        pool_pre_ping=True,
        connect_args=connect_args,
    )


engine = _build_engine()
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


async def get_db():
    async with AsyncSessionLocal() as session:
        yield session


async def dispose_engine():
    await engine.dispose()
