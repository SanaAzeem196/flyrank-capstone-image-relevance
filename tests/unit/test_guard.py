from dataclasses import dataclass

from src.guard.check import GuardVerdict, run_guard
from src.guard.config import GuardConfig

config = GuardConfig(conf_min=0.70, sim_min=0.60)


@dataclass
class ImageMeta:
    confidence: float
    category: str
    taxon: str


@dataclass
class PostMeta:
    category: str
    taxon: str | None


def _reason(checks, name: str) -> str:
    return next(c.reason for c in checks if c.name == name)


def test_guard_accepts_fox_for_fox_post():
    image = ImageMeta(confidence=0.90, category="animal", taxon="fox")
    post = PostMeta(category="animal", taxon="fox")

    verdict, checks = run_guard(image, post, similarity=0.82, config=config)

    assert verdict == GuardVerdict.ACCEPT
    assert all(c.passed for c in checks)


def test_guard_rejects_wolf_for_fox_post():
    image = ImageMeta(confidence=0.95, category="animal", taxon="wolf")
    post = PostMeta(category="animal", taxon="fox")

    verdict, checks = run_guard(image, post, similarity=0.75, config=config)

    assert verdict == GuardVerdict.REJECT
    taxon_check = next(c for c in checks if c.name == "taxon")
    assert not taxon_check.passed
    assert "expected fox, got wolf" in taxon_check.reason


def test_guard_rejects_low_confidence():
    image = ImageMeta(confidence=0.45, category="animal", taxon="fox")
    post = PostMeta(category="animal", taxon="fox")

    verdict, checks = run_guard(image, post, similarity=0.80, config=config)

    assert verdict == GuardVerdict.REJECT
    assert "too uncertain" in _reason(checks, "confidence")


def test_guard_rejects_category_mismatch():
    image = ImageMeta(confidence=0.90, category="vehicle", taxon="car")
    post = PostMeta(category="animal", taxon="fox")

    verdict, checks = run_guard(image, post, similarity=0.80, config=config)

    assert verdict == GuardVerdict.REJECT
    assert "expected animal, got vehicle" in _reason(checks, "category")


def test_guard_rejects_low_similarity():
    image = ImageMeta(confidence=0.90, category="animal", taxon="fox")
    post = PostMeta(category="animal", taxon="fox")

    verdict, checks = run_guard(image, post, similarity=0.40, config=config)

    assert verdict == GuardVerdict.REJECT
    assert "below threshold" in _reason(checks, "similarity")


def test_guard_skips_taxon_when_post_has_no_subject():
    image = ImageMeta(confidence=0.90, category="architecture", taxon="bridge")
    post = PostMeta(category="architecture", taxon=None)

    verdict, checks = run_guard(image, post, similarity=0.80, config=config)

    assert verdict == GuardVerdict.ACCEPT
    taxon_check = next(c for c in checks if c.name == "taxon")
    assert taxon_check.passed


def test_guard_collects_all_failing_checks_not_short_circuit():
    image = ImageMeta(confidence=0.30, category="vehicle", taxon="car")
    post = PostMeta(category="animal", taxon="fox")

    verdict, checks = run_guard(image, post, similarity=0.10, config=config)

    assert verdict == GuardVerdict.REJECT
    failing = {c.name for c in checks if not c.passed}
    assert failing == {"confidence", "category", "taxon", "similarity"}
