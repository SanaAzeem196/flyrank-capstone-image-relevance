"""
Integration test: Low-confidence images are flagged (verdict REJECT).

Mocked database repos and similarity search. Tests that guard rejects low-confidence images.
No real Gemini calls, no SQLite schema issues. Deterministic, repeatable, fast.
"""

from unittest.mock import MagicMock, patch
import pytest

from src.config import guard_config
from src.guard.check import GuardVerdict
from src.services.matching_service import match_images_for_post


@pytest.fixture
def mock_db():
    """Mocked SQLAlchemy session with repo mocks."""
    return MagicMock()


@pytest.fixture
def test_tenant_id():
    return "test-tenant-123"


def test_low_confidence_image_rejected(mock_db, test_tenant_id: str):
    """
    Image with confidence 0.45 (< CONF_MIN=0.70) should:
    - Run through guard.run_guard()
    - Fail the confidence check
    - Return verdict REJECT with reason about low confidence
    - Upsert suggestion as REJECT
    """
    # Mock post
    post_id = 1
    post_meta = MagicMock()
    post_meta.category = "animal"
    post_meta.taxon = "fox"

    # Mock post vector repo
    post_vector = MagicMock()
    post_vector.embedding = [0.1] * 768
    mock_db.query.return_value.filter_by.return_value.first.return_value = post_vector

    # Mock post repo
    post_repo = MagicMock()
    post_repo.get.return_value = post_meta

    # Mock image metadata with LOW confidence (0.45 < 0.70)
    image_id = 100
    image_meta = MagicMock()
    image_meta.confidence = 0.45  # BELOW CONF_MIN=0.70
    image_meta.category = "animal"
    image_meta.taxon = "fox"

    # Mock similarity search: one candidate with high similarity (0.85)
    # so that only confidence check fails
    similarity = 0.85

    # Mock image metadata repo to return our low-confidence image
    meta_repo = MagicMock()
    meta_repo.get_by_image_id.return_value = image_meta

    # Mock image vector repo similarity search to return one candidate
    image_vector_repo = MagicMock()
    image_vector_repo.similarity_search.return_value = [(image_id, similarity)]

    # Mock suggestion repo upsert
    suggestion_repo = MagicMock()

    # Mock the repositories on the db session
    with (
        patch("src.services.matching_service.PostRepository") as mock_post_repo_class,
        patch("src.services.matching_service.PostVectorRepository") as mock_post_vec_repo_class,
        patch("src.services.matching_service.ImageVectorRepository") as mock_img_vec_repo_class,
        patch("src.services.matching_service.ImageMetadataRepository") as mock_meta_repo_class,
        patch("src.services.matching_service.SuggestionRepository") as mock_sugg_repo_class,
    ):
        mock_post_repo_class.return_value = post_repo
        mock_post_vec_repo_class.return_value = MagicMock(get_by_post_id=MagicMock(return_value=post_vector))
        mock_img_vec_repo_class.return_value = image_vector_repo
        mock_meta_repo_class.return_value = meta_repo
        mock_sugg_repo_class.return_value = suggestion_repo

        # Act: match images for post
        result = match_images_for_post(mock_db, post_id, test_tenant_id, limit=10)

    # Assert: no_confident_match (no ACCEPT), and reasons include confidence failure
    assert result["status"] == "no_confident_match", "Should not accept low-confidence image"
    assert len(result["reasons"]) > 0, "Should have rejection reasons"

    # Find the confidence check failure
    confidence_failures = [r for r in result["reasons"] if r["check"] == "confidence"]
    assert len(confidence_failures) == 1, "Should have exactly one confidence check failure"

    conf_failure = confidence_failures[0]
    assert conf_failure["image_id"] == image_id
    assert "too uncertain" in conf_failure["reason"]
    assert "0.45" in conf_failure["reason"]
    assert ("0.70" in conf_failure["reason"] or "0.7" in conf_failure["reason"])

    # Verify the suggestion was upserted as REJECT
    suggestion_repo.upsert.assert_called_once()
    call_kwargs = suggestion_repo.upsert.call_args[1]
    assert call_kwargs["verdict"] == GuardVerdict.REJECT
    assert call_kwargs["image_id"] == image_id
    assert len(call_kwargs["guard_checks"]) == 4  # confidence, category, taxon, similarity

    # Find confidence check in upserted checks
    conf_checks = [c for c in call_kwargs["guard_checks"] if c["name"] == "confidence"]
    assert len(conf_checks) == 1
    assert not conf_checks[0]["passed"], "Confidence check should fail"
    assert "too uncertain" in conf_checks[0]["reason"]


def test_multiple_candidates_first_rejected_on_confidence_second_accepted(mock_db, test_tenant_id: str):
    """
    Two candidates: first has low confidence (rejected), second has high confidence (accepted).
    Should return the second one as accepted.
    """
    # Mock post
    post_id = 1
    post_meta = MagicMock()
    post_meta.category = "animal"
    post_meta.taxon = "wolf"

    # Mock post vector repo
    post_vector = MagicMock()
    post_vector.embedding = [0.1] * 768

    # Mock repos
    post_repo = MagicMock()
    post_repo.get.return_value = post_meta

    post_vec_repo = MagicMock()
    post_vec_repo.get_by_post_id.return_value = post_vector

    # Image 1: LOW confidence (0.50), high similarity (0.99)
    image1_id = 100
    image1_meta = MagicMock()
    image1_meta.confidence = 0.50  # BELOW CONF_MIN
    image1_meta.category = "animal"
    image1_meta.taxon = "wolf"

    # Image 2: HIGH confidence (0.88), moderate similarity (0.75)
    image2_id = 101
    image2_meta = MagicMock()
    image2_meta.confidence = 0.88  # ABOVE CONF_MIN
    image2_meta.category = "animal"
    image2_meta.taxon = "wolf"

    # Mock image vector repo: return candidates ranked by similarity
    # image1 has higher similarity but low confidence
    # image2 has lower similarity but passes confidence
    image_vector_repo = MagicMock()
    image_vector_repo.similarity_search.return_value = [
        (image1_id, 0.99),  # highest similarity, but will fail confidence
        (image2_id, 0.75),  # lower similarity, but will pass confidence
    ]

    # Mock image metadata repo
    meta_repo = MagicMock()
    meta_repo.get_by_image_id.side_effect = lambda img_id: {
        image1_id: image1_meta,
        image2_id: image2_meta,
    }[img_id]

    # Mock suggestion repo
    suggestion_repo = MagicMock()

    with (
        patch("src.services.matching_service.PostRepository") as mock_post_repo_class,
        patch("src.services.matching_service.PostVectorRepository") as mock_post_vec_repo_class,
        patch("src.services.matching_service.ImageVectorRepository") as mock_img_vec_repo_class,
        patch("src.services.matching_service.ImageMetadataRepository") as mock_meta_repo_class,
        patch("src.services.matching_service.SuggestionRepository") as mock_sugg_repo_class,
    ):
        mock_post_repo_class.return_value = post_repo
        mock_post_vec_repo_class.return_value = post_vec_repo
        mock_img_vec_repo_class.return_value = image_vector_repo
        mock_meta_repo_class.return_value = meta_repo
        mock_sugg_repo_class.return_value = suggestion_repo

        # Act: match images for post
        result = match_images_for_post(mock_db, post_id, test_tenant_id, limit=10)

    # Assert: should accept image2 (second candidate), skipping image1 due to low confidence
    assert result["status"] == "accepted", "Should accept second candidate despite first's higher similarity"
    assert result["image_id"] == image2_id
    assert result["rank"] == 2, "Second image should be at rank 2"

    # Verify confidence check passed for accepted image
    conf_checks = [c for c in result["checks"] if c["name"] == "confidence"]
    assert len(conf_checks) == 1
    assert conf_checks[0]["passed"]


def test_all_checks_collected_not_short_circuited(mock_db, test_tenant_id: str):
    """
    Image fails BOTH confidence and taxon checks.
    Guard should collect ALL failures, not short-circuit on first failure.
    """
    # Mock post expecting "cat"
    post_id = 1
    post_meta = MagicMock()
    post_meta.category = "animal"
    post_meta.taxon = "cat"

    # Mock post vector
    post_vector = MagicMock()
    post_vector.embedding = [0.2] * 768

    # Mock repos
    post_repo = MagicMock()
    post_repo.get.return_value = post_meta

    post_vec_repo = MagicMock()
    post_vec_repo.get_by_post_id.return_value = post_vector

    # Image: LOW confidence AND wrong taxon (dog instead of cat)
    image_id = 100
    image_meta = MagicMock()
    image_meta.confidence = 0.55  # BELOW CONF_MIN
    image_meta.category = "animal"
    image_meta.taxon = "dog"  # WRONG! post expects cat

    # Mock image vector repo
    image_vector_repo = MagicMock()
    image_vector_repo.similarity_search.return_value = [(image_id, 0.85)]

    # Mock image metadata repo
    meta_repo = MagicMock()
    meta_repo.get_by_image_id.return_value = image_meta

    # Mock suggestion repo
    suggestion_repo = MagicMock()

    with (
        patch("src.services.matching_service.PostRepository") as mock_post_repo_class,
        patch("src.services.matching_service.PostVectorRepository") as mock_post_vec_repo_class,
        patch("src.services.matching_service.ImageVectorRepository") as mock_img_vec_repo_class,
        patch("src.services.matching_service.ImageMetadataRepository") as mock_meta_repo_class,
        patch("src.services.matching_service.SuggestionRepository") as mock_sugg_repo_class,
    ):
        mock_post_repo_class.return_value = post_repo
        mock_post_vec_repo_class.return_value = post_vec_repo
        mock_img_vec_repo_class.return_value = image_vector_repo
        mock_meta_repo_class.return_value = meta_repo
        mock_sugg_repo_class.return_value = suggestion_repo

        # Act
        result = match_images_for_post(mock_db, post_id, test_tenant_id, limit=10)

    # Assert: should have reasons for BOTH confidence and taxon failures
    assert result["status"] == "no_confident_match"
    assert len(result["reasons"]) >= 2

    check_names = {r["check"] for r in result["reasons"]}
    assert "confidence" in check_names, "Should collect confidence failure"
    assert "taxon" in check_names, "Should collect taxon failure"


def test_guard_config_thresholds_respected(mock_db, test_tenant_id: str):
    """
    Test that the actual guard config thresholds (from src.config) are respected.
    CONF_MIN is typically 0.70, SIM_MIN typically 0.60.
    Image at confidence=0.70 (exactly at threshold) should PASS.
    """
    # Sanity check: verify thresholds
    assert guard_config.conf_min == 0.70, "Sanity check: CONF_MIN should be 0.70"
    assert guard_config.sim_min == 0.60, "Sanity check: SIM_MIN should be 0.60"

    # Mock post
    post_id = 1
    post_meta = MagicMock()
    post_meta.category = "other"
    post_meta.taxon = "test"

    # Mock post vector
    post_vector = MagicMock()
    post_vector.embedding = [0.3] * 768

    # Mock repos
    post_repo = MagicMock()
    post_repo.get.return_value = post_meta

    post_vec_repo = MagicMock()
    post_vec_repo.get_by_post_id.return_value = post_vector

    # Image at the boundary: confidence = 0.70 (exactly at threshold, should PASS)
    image_id = 100
    image_meta = MagicMock()
    image_meta.confidence = 0.70  # EXACTLY at CONF_MIN
    image_meta.category = "other"
    image_meta.taxon = "test"

    # Mock image vector repo with good similarity (> 0.60)
    image_vector_repo = MagicMock()
    image_vector_repo.similarity_search.return_value = [(image_id, 0.75)]

    # Mock image metadata repo
    meta_repo = MagicMock()
    meta_repo.get_by_image_id.return_value = image_meta

    # Mock suggestion repo
    suggestion_repo = MagicMock()

    with (
        patch("src.services.matching_service.PostRepository") as mock_post_repo_class,
        patch("src.services.matching_service.PostVectorRepository") as mock_post_vec_repo_class,
        patch("src.services.matching_service.ImageVectorRepository") as mock_img_vec_repo_class,
        patch("src.services.matching_service.ImageMetadataRepository") as mock_meta_repo_class,
        patch("src.services.matching_service.SuggestionRepository") as mock_sugg_repo_class,
    ):
        mock_post_repo_class.return_value = post_repo
        mock_post_vec_repo_class.return_value = post_vec_repo
        mock_img_vec_repo_class.return_value = image_vector_repo
        mock_meta_repo_class.return_value = meta_repo
        mock_sugg_repo_class.return_value = suggestion_repo

        # Act
        result = match_images_for_post(mock_db, post_id, test_tenant_id, limit=10)

    # Assert: confidence=0.70 should PASS (not REJECT), all other checks should also pass
    assert result["status"] == "accepted", "Confidence at exactly CONF_MIN=0.70 should be accepted"
    conf_checks = [c for c in result["checks"] if c["name"] == "confidence"]
    assert conf_checks[0]["passed"], "Confidence check at threshold should pass"
