import logging
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from src.api.routes import costs, health, images, jobs, posts, suggestions
from src.config import settings

logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)

app = FastAPI(title="AI Image Understanding & Content Matching Engine")

app.include_router(images.router)
app.include_router(jobs.router)
app.include_router(costs.router)
app.include_router(health.router)
app.include_router(posts.router)
app.include_router(suggestions.router)


@app.middleware("http")
async def correlation_id_middleware(request: Request, call_next):
    request.state.correlation_id = str(uuid.uuid4())
    return await call_next(request)


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    correlation_id = getattr(request.state, "correlation_id", None)
    logger.exception("unhandled exception", extra={"correlation_id": correlation_id})
    return JSONResponse(
        status_code=500,
        content={"error": "internal server error", "correlation_id": correlation_id},
    )
