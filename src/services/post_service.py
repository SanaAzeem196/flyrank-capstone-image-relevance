import logging
import time

from sqlalchemy.orm import Session

from src.config import estimate_cost, settings
from src.db.models import Post
from src.db.repos import CostLogRepository, PostRepository
from src.providers.embedding import EmbeddingProvider
from src.providers.vision import ProviderError, VisionProvider
from src.services.embedding_service import embed_post

logger = logging.getLogger(__name__)


def create_post(
    db: Session,
    vision_provider: VisionProvider,
    embedding_provider: EmbeddingProvider,
    title: str,
    body: str,
    tenant_id: str,
) -> Post:
    """Create a post, extract its subject/taxon/category, and embed it for matching."""
    post_repo = PostRepository(db)
    cost_log = CostLogRepository(db)

    post = post_repo.create(tenant_id=tenant_id, title=title, body=body)

    start = time.monotonic()
    try:
        meta, usage = vision_provider.analyze_post(title, body)
        status = "success"
    except ProviderError as e:
        logger.warning("post %s analysis failed: %s", post.id, e)
        meta = None
        usage = {"input_tokens": 0, "output_tokens": 0}
        status = "error"
    latency_ms = int((time.monotonic() - start) * 1000)

    cost = estimate_cost(settings.vision_model, usage["input_tokens"], usage["output_tokens"])
    cost_log.create(
        job_id=None,
        tenant_id=tenant_id,
        call_type="post_analysis",
        model=settings.vision_model,
        input_tokens=usage["input_tokens"],
        output_tokens=usage["output_tokens"],
        est_cost_usd=cost,
        latency_ms=latency_ms,
        status=status,
        ref_id=str(post.id),
    )

    if meta is not None:
        post_repo.set_analysis(post, subject=meta.subject, taxon=meta.taxon, category=meta.category, subject_confidence=meta.confidence)

    try:
        embed_post(db, embedding_provider, post, tenant_id)
    except ProviderError as e:
        logger.warning("post %s embedding failed: %s", post.id, e)

    return post
