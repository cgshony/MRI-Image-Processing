"""Phase 1 smoke tests: does the app boot at all?

These exercise the import chain from `app.main` through the routers, deps,
services, storage and DB modules (see the module docstring in
app/db/base.py and app/db/target_metadata.py for why that chain is shaped
the way it is), and confirm FastAPI can actually build the app + schema from
it, independent of any single endpoint's behaviour.
"""

from app.core.config import get_settings
from app.main import app


def test_app_imports_and_exposes_expected_routes():
    # FastAPI wraps `include_router`-ed routers lazily (as `_IncludedRouter`),
    # so `app.routes` alone won't show their paths; walking the built
    # OpenAPI schema is what actually forces everything to resolve.
    paths = app.openapi()["paths"]
    assert "/health" in paths
    assert "/api/v1/images" in paths
    assert "/api/v1/images/{image_id}" in paths
    assert "/api/v1/images/{image_id}/process" in paths
    assert "/api/v1/jobs/{job_id}" in paths


def test_settings_load_with_defaults():
    settings = get_settings()
    assert settings.app_name
    assert settings.api_v1_prefix == "/api/v1"
    assert settings.storage_dir.exists()  # created by Settings.model_post_init


async def test_health_endpoint(client):
    response = await client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


async def test_openapi_schema_builds():
    """A stronger check than /health: this forces FastAPI to walk every
    router, dependency and Pydantic schema to build the OpenAPI document, so
    it fails if any of that wiring is broken even if no request is made."""
    schema = app.openapi()
    assert schema["paths"]["/api/v1/images"]["post"]["operationId"]
    assert "ImageOut" in schema["components"]["schemas"]
    assert "ProcessingJobOut" in schema["components"]["schemas"]
