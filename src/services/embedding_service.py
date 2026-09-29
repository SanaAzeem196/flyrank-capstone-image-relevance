import logging
import time

from sqlalchemy.orm import Session

from src.config import estimate_cost, settings
from src.db.models import Post
from src.db.repos import (
    CostLogRepository,
    ImageMetadataRepository,
    ImageVectorRepository,
    PostVectorRepository,
)
from src.providers.embedding import EmbeddingProvider, ProviderError

logger = logging.getLogger(__name__)


def _log_and_cost(cost_log: CostLogRepository, job_id: int | None, tenant_id: str, ref_id: str, usage: dict, latency_ms: int, status: str) -> float:
    cost = estimate_cost(settings.embedding_model, usage["input_tokens"], usage["output_tokens"])
    cost_log.create(
        job_id=job_id,
        tenant_id=tenant_id,
        call_type="embedding",
        model=settings.embedding_model,
        input_tokens=usage["input_tokens"],
        output_tokens=usage["output_tokens"],
        est_cost_usd=cost,
        latency_ms=latency_ms,
        status=status,
        ref_id=ref_id,
    )
    return cost


def embed_pending_images(db: Session, provider: EmbeddingProvider, tenant_id: str, job_id: int | None = None) -> dict:
    """Embed every tagged/flagged image that has no vector yet. Idempotent."""
    meta_repo = ImageMetadataRepository(db)
    vector_repo = ImageVectorRepository(db)
    cost_log = CostLogRepository(db)

    embedded = 0
    errors: list[str] = []

    for image_id in vector_repo.images_missing_vector(tenant_id):
        meta = meta_repo.get_by_image_id(image_id)
        if meta is None:
            continue

        text = f"{meta.subject}. {meta.caption} Attributes: {', '.join(meta.attributes or [])}"
        start = time.monotonic()
        try:
            values, usage = provider.embed(text, task_type="RETRIEVAL_DOCUMENT")
        except ProviderError as e:
            errors.append(f"image {image_id}: {e}")
            continue
        latency_ms = int((time.monotonic() - start) * 1000)

        _log_and_cost(cost_log, job_id, tenant_id, str(image_id), usage, latency_ms, "success")
        vector_repo.upsert(image_id=image_id, tenant_id=tenant_id, embedding=values, model=settings.embedding_model)
        embedded += 1

    return {"embedded": embedded, "errors": errors}


def embed_post(db: Session, provider: EmbeddingProvider, post: Post, tenant_id: str) -> None:
    """Embed a post's title+body as the retrieval query side. Raises ProviderError on failure."""
    post_vector_repo = PostVectorRepository(db)
    cost_log = CostLogRepository(db)

    text = f"{post.title}. {post.body}"
    start = time.monotonic()
    try:
        values, usage = provider.embed(text, task_type="RETRIEVAL_QUERY")
    except ProviderError as e:
        latency_ms = int((time.monotonic() - start) * 1000)
        _log_and_cost(cost_log, None, tenant_id, str(post.id), {"input_tokens": 0, "output_tokens": 0}, latency_ms, "error")
        raise ProviderError(f"post {post.id}: {e}") from e
    latency_ms = int((time.monotonic() - start) * 1000)

    _log_and_cost(cost_log, None, tenant_id, str(post.id), usage, latency_ms, "success")
    post_vector_repo.upsert(post_id=post.id, tenant_id=tenant_id, embedding=values, model=settings.embedding_model)
