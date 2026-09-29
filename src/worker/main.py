import logging
import socket
import time

from src.config import settings
from src.db.repos import JobRepository
from src.db.session import SessionLocal
from src.providers.embedding import GeminiEmbedding
from src.providers.vision import GeminiFlashVision
from src.services.embedding_service import embed_pending_images
from src.services.vision_service import process_images_job

logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)

WORKER_ID = f"{socket.gethostname()}-{__name__}"
POLL_INTERVAL_SECONDS = 5


def process_job(db, job) -> None:
    if job.type == "process_images":
        provider = GeminiFlashVision(api_key=settings.gemini_api_key, model=settings.vision_model)
        corpus_path = job.payload.get("corpus_path", "corpus/images")
        progress = process_images_job(db, provider, job.id, corpus_path, job.tenant_id)

        embedding_provider = GeminiEmbedding(api_key=settings.gemini_api_key, model=settings.embedding_model, dim=settings.embedding_dim)
        progress["embedding"] = embed_pending_images(db, embedding_provider, job.tenant_id, job_id=job.id)

        JobRepository(db).mark_completed(job, progress=progress)
        logger.info("job %s completed: %s", job.id, progress)
    else:
        raise ValueError(f"unknown job type: {job.type}")


def worker_loop_once() -> bool:
    """Claim and run a single job. Returns True if a job was claimed."""
    db = SessionLocal()
    try:
        jobs = JobRepository(db)
        job = jobs.claim_next(WORKER_ID)
        if job is None:
            return False

        try:
            process_job(db, job)
        except Exception as e:  # noqa: BLE001 - any failure retries/fails the job, never crashes the worker
            logger.exception("job %s failed", job.id)
            jobs.mark_failed_or_retry(job, str(e))
        return True
    finally:
        db.close()


def worker_loop() -> None:
    logger.info("worker %s started, polling every %ss", WORKER_ID, POLL_INTERVAL_SECONDS)
    while True:
        claimed = worker_loop_once()
        if not claimed:
            time.sleep(POLL_INTERVAL_SECONDS)

if __name__ == "__main__":
    worker_loop()
