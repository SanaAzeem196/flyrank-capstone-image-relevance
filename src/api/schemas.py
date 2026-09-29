from typing import Literal

from pydantic import BaseModel, Field

Category = Literal["animal", "vehicle", "food", "architecture", "landscape", "other"]


class ImageMetadata(BaseModel):
    """Vision model's structured output; also the response_schema sent to Gemini."""

    subject: str
    taxon: str
    category: Category
    attributes: list[str] = Field(default_factory=list)
    caption: str
    confidence: float = Field(ge=0.0, le=1.0)


class ImageMetadataResponse(BaseModel):
    id: int
    filename: str
    status: str
    subject: str | None = None
    taxon: str | None = None
    category: str | None = None
    caption: str | None = None
    confidence: float | None = None


class PostMetadata(BaseModel):
    """Post analyzer's structured output; also the response_schema sent to Gemini."""

    subject: str
    taxon: str = Field(description="empty string if the post has no identifiable subject")
    category: Category
    confidence: float = Field(ge=0.0, le=1.0)


class PostCreateRequest(BaseModel):
    title: str
    body: str


class PostCreateResponse(BaseModel):
    post_id: int


class PostResponse(BaseModel):
    id: int
    title: str
    body: str
    subject: str | None = None
    taxon: str | None = None
    category: str | None = None
    subject_confidence: float | None = None


class CheckImageRequest(BaseModel):
    image_id: int


class GuardCheckResponse(BaseModel):
    name: str
    passed: bool
    reason: str


class SuggestionResponse(BaseModel):
    id: int
    post_id: int
    image_id: int
    rank: int
    similarity: float
    verdict: str
    guard_checks: list[GuardCheckResponse]
    review_status: str


class JobResponse(BaseModel):
    id: int
    type: str
    status: str
    attempts: int
    max_attempts: int
    progress: dict | None = None
    last_error: str | None = None


class ProcessImagesRequest(BaseModel):
    corpus_path: str = "corpus/images"


class JobCreateResponse(BaseModel):
    job_id: int


class HealthResponse(BaseModel):
    status: str
    database: str


class CostEntry(BaseModel):
    call_type: str
    model: str | None
    est_cost_usd: float
    status: str | None
    created_at: str


class CostsResponse(BaseModel):
    total_calls: int
    total_cost_usd: float
    by_call_type: dict[str, dict]
    recent_calls: list[CostEntry]
