"""Shared fixtures for the API test suite.

Every test hits the real FastAPI app (real routers, real DI wiring, real
service/storage classes) so these tests double as a check that the app boots
end-to-end. Only the two genuinely external dependencies are swapped for
per-test fakes: Postgres -> an in-memory SQLite DB, and the storage directory
-> a pytest `tmp_path`.
"""

import io

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from PIL import Image as PILImage
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.pool import StaticPool

# SQLite (this test suite only) has no native JSONB type; render it as plain
# JSON instead. Production always talks to Postgres, where JSONB is used
# as-is -- this shim never runs outside tests.
@compiles(JSONB, "sqlite")
def _jsonb_as_json_on_sqlite(element, compiler, **kw):
    return "JSON"

import app.db.session as db_session
import app.services.processing_service as processing_service_module
from app.core.config import Settings, get_settings
from app.db.target_metadata import target_metadata
from app.main import app as fastapi_app


@pytest_asyncio.fixture
async def test_engine():
    """A fresh in-memory SQLite DB per test, schema created from the ORM models."""
    engine = create_async_engine(
        "sqlite+aiosqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    async with engine.begin() as conn:
        await conn.run_sync(target_metadata.create_all)
    yield engine
    await engine.dispose()


@pytest_asyncio.fixture
async def client(test_engine, tmp_path, monkeypatch):
    """An async client wired to the real app, with the DB and storage dir
    swapped for the fakes above.

    Two things read the session factory: the `get_session` FastAPI dependency
    (app/db/session.py) and `ProcessingService.run_job`, which runs as a
    BackgroundTask *after* the request's own session is closed and so opens
    its own via a module-level `async_session_factory` it imported directly.
    Both names are patched -- patching only `db_session`'s copy would leave
    the background job's copy pointed at the real (unreachable-in-tests)
    Postgres factory, since `from ... import async_session_factory` bound a
    separate reference into that module's namespace at import time.
    """
    session_factory = async_sessionmaker(
        test_engine, expire_on_commit=False, autoflush=False, class_=AsyncSession
    )
    monkeypatch.setattr(db_session, "async_session_factory", session_factory)
    monkeypatch.setattr(processing_service_module, "async_session_factory", session_factory)

    test_settings = Settings(storage_dir=tmp_path)
    fastapi_app.dependency_overrides[get_settings] = lambda: test_settings

    transport = ASGITransport(app=fastapi_app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac

    fastapi_app.dependency_overrides.clear()


@pytest.fixture
def png_bytes() -> bytes:
    """A tiny valid grayscale PNG, generated on the fly (no binary fixture file needed)."""
    buffer = io.BytesIO()
    PILImage.new("L", (4, 3), color=128).save(buffer, format="PNG")
    return buffer.getvalue()
