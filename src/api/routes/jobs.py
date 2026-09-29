from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from src.api.schemas import JobResponse
from src.db.repos import JobRepository
from src.db.session import get_db

router = APIRouter()


@router.get("/jobs/{job_id}", response_model=JobResponse)
def get_job(job_id: int, db: Session = Depends(get_db)):
    job = JobRepository(db).get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return JobResponse.model_validate(job, from_attributes=True)


@router.get("/jobs", response_model=list[JobResponse])
def list_jobs(status: str | None = None, db: Session = Depends(get_db)):
    jobs = JobRepository(db).list(status=status)
    return [JobResponse.model_validate(j, from_attributes=True) for j in jobs]
