from sqlalchemy.orm import Session

from src.config import guard_config
from src.db.repos import ImageMetadataRepository, ImageVectorRepository, PostRepository, PostVectorRepository, SuggestionRepository
from src.guard.check import GuardVerdict, run_guard


class MatchingError(Exception):
    """Post/image not found, or missing an embedding needed to match."""


def match_images_for_post(db: Session, post_id: int, tenant_id: str, limit: int = 20) -> dict:
    """Rank candidate images by similarity, run each through the guard, return the first ACCEPT
    or "no_confident_match" with the reasons every candidate failed for."""
    post_repo = PostRepository(db)
    post_vector_repo = PostVectorRepository(db)
    image_vector_repo = ImageVectorRepository(db)
    meta_repo = ImageMetadataRepository(db)
    suggestion_repo = SuggestionRepository(db)

    post = post_repo.get(post_id)
    if post is None:
        raise MatchingError(f"post {post_id} not found")

    post_vector = post_vector_repo.get_by_post_id(post_id)
    if post_vector is None:
        raise MatchingError(f"post {post_id} has no embedding yet")

    candidates = image_vector_repo.similarity_search(post_vector.embedding, tenant_id, limit)

    all_reasons = []
    for rank, (image_id, similarity) in enumerate(candidates, start=1):
        image_meta = meta_repo.get_by_image_id(image_id)
        if image_meta is None:
            continue

        verdict, checks = run_guard(image_meta, post, similarity, guard_config)
        checks_dict = [c.to_dict() for c in checks]
        suggestion_repo.upsert(
            tenant_id=tenant_id, post_id=post_id, image_id=image_id, rank=rank,
            similarity=similarity, verdict=verdict, guard_checks=checks_dict,
        )

        if verdict == GuardVerdict.ACCEPT:
            return {
                "status": "accepted",
                "rank": rank,
                "image_id": image_id,
                "similarity": similarity,
                "checks": checks_dict,
            }

        for check in checks_dict:
            if not check["passed"]:
                all_reasons.append({"image_id": image_id, "check": check["name"], "reason": check["reason"]})

    return {"status": "no_confident_match", "reasons": all_reasons}


def check_image_for_post(db: Session, post_id: int, image_id: int, tenant_id: str) -> dict:
    """Force a specific image through the guard against a post, regardless of rank."""
    post_repo = PostRepository(db)
    post_vector_repo = PostVectorRepository(db)
    image_vector_repo = ImageVectorRepository(db)
    meta_repo = ImageMetadataRepository(db)
    suggestion_repo = SuggestionRepository(db)

    post = post_repo.get(post_id)
    if post is None:
        raise MatchingError(f"post {post_id} not found")

    image_meta = meta_repo.get_by_image_id(image_id)
    if image_meta is None:
        raise MatchingError(f"image {image_id} not found")

    post_vector = post_vector_repo.get_by_post_id(post_id)
    if post_vector is None:
        raise MatchingError(f"post {post_id} has no embedding yet")

    similarity = image_vector_repo.similarity_to(post_vector.embedding, image_id)
    if similarity is None:
        raise MatchingError(f"image {image_id} has no embedding yet")

    verdict, checks = run_guard(image_meta, post, similarity, guard_config)
    checks_dict = [c.to_dict() for c in checks]
    suggestion_repo.upsert(
        tenant_id=tenant_id, post_id=post_id, image_id=image_id, rank=0,
        similarity=similarity, verdict=verdict, guard_checks=checks_dict,
    )

    return {
        "status": "accepted" if verdict == GuardVerdict.ACCEPT else "rejected",
        "image_id": image_id,
        "similarity": similarity,
        "checks": checks_dict,
    }
