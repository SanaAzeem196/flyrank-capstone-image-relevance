import hashlib
import logging
import mimetypes
import time
from pathlib import Path

from sqlalchemy.orm import Session

from src.config import estimate_cost, guard_config, settings
from src.db.repos import CostLogRepository, ImageMetadataRepository, ImageRepository
from src.providers.vision import ProviderError, VisionProvider

logger = logging.getLogger(__name__)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def ingest_corpus(db: Session, corpus_path: str, tenant_id: str) -> list[int]:
    """Scan corpus_path for images, insert new ones (deduped by content hash) as status=pending.

    Returns the ids of images still pending after ingest (new + previously unprocessed).
    """
    images = ImageRepository(db)
    pending_ids: list[int] = []

    corpus_dir = Path(corpus_path)
    for path in sorted(corpus_dir.glob("*")):
        if not path.is_file() or path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue

        content_hash = _sha256(path.read_bytes())
        existing = images.get_by_content_hash(content_hash)
        if existing is None:
            image = images.create_pending(tenant_id=tenant_id, filename=path.name, content_hash=content_hash)
            db.commit()
            pending_ids.append(image.id)
        elif existing.status == "pending":
            pending_ids.append(existing.id)

    return pending_ids


def process_images_job(db: Session, provider: VisionProvider, job_id: int, corpus_path: str, tenant_id: str) -> dict:
    """Ingest the corpus, tag every pending image, log cost per call. Returns a progress summary."""
    images = ImageRepository(db)
    metadata_repo = ImageMetadataRepository(db)
    cost_log = CostLogRepository(db)

    pending_ids = ingest_corpus(db, corpus_path, tenant_id)

    tagged = 0
    flagged = 0
    errors: list[str] = []
    total_cost_usd = 0.0

    for image_id in pending_ids:
        image = images.get(image_id)
        if image is None or image.status != "pending":
            continue

        if total_cost_usd >= settings.max_est_usd:
            errors.append(f"image {image_id}: skipped, budget MAX_EST_USD={settings.max_est_usd} reached")
            break

        images.set_status(image, "processing")
        db.commit()

        image_path = Path(corpus_path) / image.filename
        mime_type = mimetypes.guess_type(image.filename)[0] or "image/jpeg"

        start = time.monotonic()
        try:
            metadata, usage = provider.tag_image(image_path.read_bytes(), mime_type)
            status = "success"
            error_message = None
        except ProviderError as e:
            images.set_status(image, "failed")
            db.commit()
            errors.append(f"image {image_id}: {e}")
            continue
        except Exception as e:  # noqa: BLE001 - retries exhausted (rate limit / server error)
            images.set_status(image, "failed")
            db.commit()
            errors.append(f"image {image_id}: {e}")
            continue
        finally:
            latency_ms = int((time.monotonic() - start) * 1000)

        cost = estimate_cost(settings.vision_model, usage["input_tokens"], usage["output_tokens"])
        total_cost_usd += cost
        cost_log.create(
            job_id=job_id,
            tenant_id=tenant_id,
            call_type="vision",
            model=settings.vision_model,
            input_tokens=usage["input_tokens"],
            output_tokens=usage["output_tokens"],
            est_cost_usd=cost,
            latency_ms=latency_ms,
            status=status,
        )

        needs_review = metadata.confidence < guard_config.conf_min
        metadata_repo.create(
            image_id=image.id,
            tenant_id=tenant_id,
            subject=metadata.subject,
            taxon=metadata.taxon,
            category=metadata.category,
            caption=metadata.caption,
            confidence=metadata.confidence,
            attributes=metadata.attributes,
            needs_review=needs_review,
            raw_response=metadata.model_dump(),
            model=settings.vision_model,
        )
        images.set_status(image, "flagged" if needs_review else "tagged")
        db.commit()

        if needs_review:
            flagged += 1
        else:
            tagged += 1

    return {
        "processed": tagged + flagged,
        "tagged": tagged,
        "flagged": flagged,
        "errors": errors,
        "cost_usd": round(total_cost_usd, 6),
    }
