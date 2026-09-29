from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from src.api.schemas import SuggestionResponse
from src.db.repos import SuggestionRepository
from src.db.session import get_db
from src.services import review_service
from src.services.review_service import ReviewError

router = APIRouter()


@router.get("/suggestions/{suggestion_id}", response_model=SuggestionResponse)
def get_suggestion(suggestion_id: int, db: Session = Depends(get_db)):
    suggestion = SuggestionRepository(db).get(suggestion_id)
    if suggestion is None:
        raise HTTPException(status_code=404, detail="suggestion not found")
    return SuggestionResponse.model_validate(suggestion, from_attributes=True)


@router.post("/suggestions/{suggestion_id}/approve", response_model=SuggestionResponse)
def approve_suggestion(suggestion_id: int, db: Session = Depends(get_db)):
    try:
        suggestion = review_service.approve(db, suggestion_id)
    except ReviewError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return SuggestionResponse.model_validate(suggestion, from_attributes=True)


@router.post("/suggestions/{suggestion_id}/reject", response_model=SuggestionResponse)
def reject_suggestion(suggestion_id: int, db: Session = Depends(get_db)):
    try:
        suggestion = review_service.reject(db, suggestion_id)
    except ReviewError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return SuggestionResponse.model_validate(suggestion, from_attributes=True)
