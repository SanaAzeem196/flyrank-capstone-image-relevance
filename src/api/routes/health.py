from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.orm import Session

from src.api.schemas import HealthResponse
from src.db.session import get_db

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health(db: Session = Depends(get_db)):
    try:
        db.execute(text("SELECT 1"))
        database = "ok"
    except Exception:  # noqa: BLE001
        database = "unreachable"
    return HealthResponse(status="ok" if database == "ok" else "degraded", database=database)
