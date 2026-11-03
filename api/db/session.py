import ssl

from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from api.core.config import settings


def _normalise_url(url: str) -> str:
    for prefix in ("postgresql://", "postgres://"):
        if url.startswith(prefix):
            return "postgresql+asyncpg://" + url[len(prefix):]
    return url


def _build_engine():
    connect_args = {}
    if settings.database_ssl:
        # Railway PostgreSQL requires SSL; permissive context (Railway cert is valid)
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        connect_args["ssl"] = ctx
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
    """Call during lifespan shutdown so Railway SIGTERM drains the connection pool cleanly."""
    await engine.dispose()
