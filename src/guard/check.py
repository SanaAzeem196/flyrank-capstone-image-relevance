from dataclasses import dataclass
from typing import Protocol

from src.guard.config import GuardConfig


class GuardVerdict:
    ACCEPT = "accept"
    REJECT = "reject"


@dataclass
class GuardCheck:
    name: str  # "confidence" | "category" | "taxon" | "similarity"
    passed: bool
    reason: str

    def to_dict(self) -> dict:
        return {"name": self.name, "passed": self.passed, "reason": self.reason}


class ImageMetaLike(Protocol):
    confidence: float
    category: str
    taxon: str


class PostMetaLike(Protocol):
    category: str
    taxon: str | None


def run_guard(
    image_meta: ImageMetaLike,
    post_meta: PostMetaLike,
    similarity: float,
    config: GuardConfig,
) -> tuple[str, list[GuardCheck]]:
    """Validate image against post. Runs all 4 checks, never short-circuits.

    ACCEPT only if every check passes.
    """
    checks = [
        _check_confidence(image_meta, config),
        _check_category(image_meta, post_meta),
        _check_taxon(image_meta, post_meta),
        _check_similarity(similarity, config),
    ]
    verdict = GuardVerdict.ACCEPT if all(c.passed for c in checks) else GuardVerdict.REJECT
    return verdict, checks


def _check_confidence(image_meta: ImageMetaLike, config: GuardConfig) -> GuardCheck:
    if image_meta.confidence < config.conf_min:
        return GuardCheck(
            "confidence", False,
            f"Image classification too uncertain ({image_meta.confidence:.2f} < {config.conf_min})",
        )
    return GuardCheck("confidence", True, "")


def _check_category(image_meta: ImageMetaLike, post_meta: PostMetaLike) -> GuardCheck:
    if image_meta.category != post_meta.category:
        return GuardCheck(
            "category", False,
            f"Category mismatch: expected {post_meta.category}, got {image_meta.category}",
        )
    return GuardCheck("category", True, "")


def _check_taxon(image_meta: ImageMetaLike, post_meta: PostMetaLike) -> GuardCheck:
    if not post_meta.taxon:
        # Post has no identifiable subject — nothing to compare.
        return GuardCheck("taxon", True, "")
    if image_meta.taxon != post_meta.taxon:
        label = (post_meta.category or "subject").title()
        return GuardCheck(
            "taxon", False,
            f"{label} mismatch: expected {post_meta.taxon}, got {image_meta.taxon}",
        )
    return GuardCheck("taxon", True, "")


def _check_similarity(similarity: float, config: GuardConfig) -> GuardCheck:
    if similarity < config.sim_min:
        return GuardCheck(
            "similarity", False,
            f"Similarity {similarity:.2f} below threshold {config.sim_min}",
        )
    return GuardCheck("similarity", True, "")
