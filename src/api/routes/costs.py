from collections import defaultdict

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from src.api.schemas import CostEntry, CostsResponse
from src.db.repos import CostLogRepository
from src.db.session import get_db

router = APIRouter()


@router.get("/costs", response_model=CostsResponse)
def get_costs(db: Session = Depends(get_db)):
    entries = CostLogRepository(db).list(limit=1000)

    by_call_type: dict[str, dict] = defaultdict(lambda: {"count": 0, "cost_usd": 0.0})
    total_cost = 0.0
    for e in entries:
        bucket = by_call_type[e.call_type]
        bucket["count"] += 1
        bucket["cost_usd"] += e.est_cost_usd or 0.0
        total_cost += e.est_cost_usd or 0.0

    return CostsResponse(
        total_calls=len(entries),
        total_cost_usd=round(total_cost, 6),
        by_call_type=dict(by_call_type),
        recent_calls=[
            CostEntry(
                call_type=e.call_type,
                model=e.model,
                est_cost_usd=e.est_cost_usd or 0.0,
                status=e.status,
                created_at=e.created_at.isoformat() if e.created_at else "",
            )
            for e in entries[:20]
        ],
    )
