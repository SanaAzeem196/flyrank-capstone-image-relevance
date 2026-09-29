from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from src.api.schemas import CheckImageRequest, PostCreateRequest, PostCreateResponse, PostResponse
from src.config import settings
from src.db.repos import PostRepository
from src.db.session import get_db
from src.providers.embedding import GeminiEmbedding
from src.providers.vision import GeminiFlashVision
from src.services import matching_service, post_service
from src.services.matching_service import MatchingError

router = APIRouter()


def _providers():
    vision = GeminiFlashVision(api_key=settings.gemini_api_key, model=settings.vision_model)
    embedding = GeminiEmbedding(api_key=settings.gemini_api_key, model=settings.embedding_model, dim=settings.embedding_dim)
    return vision, embedding


@router.post("/posts", status_code=201, response_model=PostCreateResponse)
def create_post(req: PostCreateRequest, db: Session = Depends(get_db)):
    vision, embedding = _providers()
    post = post_service.create_post(db, vision, embedding, req.title, req.body, settings.default_tenant_id)
    return PostCreateResponse(post_id=post.id)


@router.get("/posts/{post_id}", response_model=PostResponse)
def get_post(post_id: int, db: Session = Depends(get_db)):
    post = PostRepository(db).get(post_id)
    if post is None:
        raise HTTPException(status_code=404, detail="post not found")
    return PostResponse.model_validate(post, from_attributes=True)


@router.get("/posts/{post_id}/images")
def get_post_images(post_id: int, db: Session = Depends(get_db)):
    try:
        return matching_service.match_images_for_post(db, post_id, settings.default_tenant_id)
    except MatchingError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/posts/{post_id}/check")
def check_post_image(post_id: int, req: CheckImageRequest, db: Session = Depends(get_db)):
    try:
        return matching_service.check_image_for_post(db, post_id, req.image_id, settings.default_tenant_id)
    except MatchingError as e:
        raise HTTPException(status_code=404, detail=str(e))
