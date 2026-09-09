import logging

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.v1.routers import images, processing
from app.core.config import get_settings

settings = get_settings()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title=settings.app_name)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(images.router, prefix=settings.api_v1_prefix)
app.include_router(processing.router, prefix=settings.api_v1_prefix)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Turn an unhandled exception into a normal JSON response.

    Starlette's default behaviour lets unhandled exceptions escape past
    CORSMiddleware entirely, so the browser never sees CORS headers on the
    resulting 500 and reports a generic "Failed to fetch" instead of the
    real error. Routing it through a registered handler keeps it inside the
    normal response path (headers included) while still surfacing as a 500.
    """
    logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


@app.get("/health", tags=["health"])
async def health():
    return {"status": "ok"}
