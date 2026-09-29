from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from src.api.schemas import ImageMetadataResponse, JobCreateResponse, ProcessImagesRequest
from src.config import settings
from src.db.repos import ImageMetadataRepository, ImageRepository, JobRepository
from src.db.session import get_db

router = APIRouter()


@router.post("/jobs/process-images", status_code=202, response_model=JobCreateResponse)
def enqueue_process_images(req: ProcessImagesRequest, db: Session = Depends(get_db)):
    job = JobRepository(db).create(
        tenant_id=settings.default_tenant_id,
        type_="process_images",
        payload={"corpus_path": req.corpus_path},
    )
    return JobCreateResponse(job_id=job.id)


@router.get("/images", response_model=list[ImageMetadataResponse])
def list_images(status: str | None = None, db: Session = Depends(get_db)):
    images = ImageRepository(db).list(status=status)
    metadata_repo = ImageMetadataRepository(db)
    result = []
    for image in images:
        meta = metadata_repo.get_by_image_id(image.id)
        result.append(_to_response(image, meta))
    return result


@router.get("/images/{image_id}", response_model=ImageMetadataResponse)
def get_image(image_id: int, db: Session = Depends(get_db)):
    image = ImageRepository(db).get(image_id)
    if image is None:
        raise HTTPException(status_code=404, detail="image not found")
    meta = ImageMetadataRepository(db).get_by_image_id(image_id)
    return _to_response(image, meta)


def _to_response(image, meta) -> ImageMetadataResponse:
    return ImageMetadataResponse(
        id=image.id,
        filename=image.filename,
        status=image.status,
        subject=meta.subject if meta else None,
        taxon=meta.taxon if meta else None,
        category=meta.category if meta else None,
        caption=meta.caption if meta else None,
        confidence=meta.confidence if meta else None,
    )
