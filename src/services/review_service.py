from sqlalchemy.orm import Session

from src.db.models import Suggestion
from src.db.repos import ReviewRepository, SuggestionRepository


class ReviewError(Exception):
    """Suggestion not found."""


def _set_status(db: Session, suggestion_id: int, review_status: str, decision: str, note: str | None) -> Suggestion:
    suggestion_repo = SuggestionRepository(db)
    suggestion = suggestion_repo.get(suggestion_id)
    if suggestion is None:
        raise ReviewError(f"suggestion {suggestion_id} not found")

    if suggestion.review_status != review_status:
        suggestion.review_status = review_status
        ReviewRepository(db).create(
            tenant_id=suggestion.tenant_id, suggestion_id=suggestion.id, decision=decision, note=note,
        )
        db.commit()
        db.refresh(suggestion)

    return suggestion


def approve(db: Session, suggestion_id: int, note: str | None = None) -> Suggestion:
    return _set_status(db, suggestion_id, "approved", "human_approved", note)


def reject(db: Session, suggestion_id: int, note: str | None = None) -> Suggestion:
    return _set_status(db, suggestion_id, "rejected", "human_rejected", note)
