from datetime import datetime, timedelta

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.db.models import (
    CostLog,
    Image,
    ImageMetadata,
    ImageVector,
    Job,
    Post,
    PostVector,
    Review,
    Suggestion,
)


class ImageRepository:
    def __init__(self, db: Session):
        self.db = db

    def get_by_content_hash(self, content_hash: str) -> Image | None:
        return self.db.scalar(select(Image).where(Image.content_hash == content_hash))

    def create_pending(self, tenant_id: str, filename: str, content_hash: str) -> Image:
        image = Image(tenant_id=tenant_id, filename=filename, content_hash=content_hash, status="pending")
        self.db.add(image)
        self.db.flush()
        return image

    def get(self, image_id: int) -> Image | None:
        return self.db.get(Image, image_id)

    def list(self, status: str | None = None) -> list[Image]:
        stmt = select(Image)
        if status:
            stmt = stmt.where(Image.status == status)
        return list(self.db.scalars(stmt.order_by(Image.id)))

    def set_status(self, image: Image, status: str) -> None:
        image.status = status
        self.db.flush()


class ImageMetadataRepository:
    def __init__(self, db: Session):
        self.db = db

    def create(self, **kwargs) -> ImageMetadata:
        row = ImageMetadata(**kwargs)
        self.db.add(row)
        self.db.flush()
        return row

    def get_by_image_id(self, image_id: int) -> ImageMetadata | None:
        return self.db.scalar(select(ImageMetadata).where(ImageMetadata.image_id == image_id))


class JobRepository:
    def __init__(self, db: Session):
        self.db = db

    def create(self, tenant_id: str, type_: str, payload: dict) -> Job:
        job = Job(tenant_id=tenant_id, type=type_, payload=payload, status="queued")
        self.db.add(job)
        self.db.commit()
        self.db.refresh(job)
        return job

    def get(self, job_id: int) -> Job | None:
        return self.db.get(Job, job_id)

    def list(self, status: str | None = None) -> list[Job]:
        stmt = select(Job)
        if status:
            stmt = stmt.where(Job.status == status)
        return list(self.db.scalars(stmt.order_by(Job.created_at.desc())))

    def claim_next(self, worker_id: str) -> Job | None:
        stmt = (
            select(Job)
            .where(Job.status == "queued", Job.run_after <= datetime.utcnow())
            .order_by(Job.created_at)
            .limit(1)
            .with_for_update(skip_locked=True)
        )
        job = self.db.scalar(stmt)
        if job is None:
            return None
        job.status = "in_progress"
        job.locked_at = datetime.utcnow()
        job.locked_by = worker_id
        self.db.commit()
        self.db.refresh(job)
        return job

    def mark_completed(self, job: Job, progress: dict | None = None) -> None:
        job.status = "completed"
        if progress is not None:
            job.progress = progress
        job.updated_at = datetime.utcnow()
        self.db.commit()

    def mark_failed_or_retry(self, job: Job, error: str) -> None:
        job.attempts += 1
        if job.attempts >= job.max_attempts:
            job.status = "failed"
            job.last_error = error
        else:
            job.status = "queued"
            job.run_after = datetime.utcnow() + timedelta(seconds=2 ** job.attempts)
            job.last_error = error
        job.updated_at = datetime.utcnow()
        self.db.commit()


class ImageVectorRepository:
    def __init__(self, db: Session):
        self.db = db

    def upsert(self, image_id: int, tenant_id: str, embedding: list[float], model: str) -> ImageVector:
        row = self.db.scalar(select(ImageVector).where(ImageVector.image_id == image_id))
        if row is None:
            row = ImageVector(image_id=image_id, tenant_id=tenant_id, embedding=embedding, model=model)
            self.db.add(row)
        else:
            row.embedding = embedding
            row.model = model
        self.db.commit()
        self.db.refresh(row)
        return row

    def images_missing_vector(self, tenant_id: str) -> list[int]:
        """Ids of tagged/flagged images that have no embedding yet."""
        stmt = (
            select(Image.id)
            .outerjoin(ImageVector, ImageVector.image_id == Image.id)
            .where(
                Image.tenant_id == tenant_id,
                Image.status.in_(["tagged", "flagged"]),
                ImageVector.id.is_(None),
            )
        )
        return list(self.db.scalars(stmt))

    def similarity_search(self, embedding: list[float], tenant_id: str, limit: int) -> list[tuple[int, float]]:
        """Top-K (image_id, cosine_similarity) ordered by similarity desc."""
        distance = ImageVector.embedding.cosine_distance(embedding)
        stmt = (
            select(ImageVector.image_id, (1 - distance).label("similarity"))
            .where(ImageVector.tenant_id == tenant_id)
            .order_by(distance)
            .limit(limit)
        )
        return [(row.image_id, float(row.similarity)) for row in self.db.execute(stmt)]

    def similarity_to(self, embedding: list[float], image_id: int) -> float | None:
        distance = ImageVector.embedding.cosine_distance(embedding)
        stmt = select((1 - distance).label("similarity")).where(ImageVector.image_id == image_id)
        row = self.db.execute(stmt).first()
        return float(row.similarity) if row else None


class PostRepository:
    def __init__(self, db: Session):
        self.db = db

    def create(self, tenant_id: str, title: str, body: str) -> Post:
        post = Post(tenant_id=tenant_id, title=title, body=body)
        self.db.add(post)
        self.db.commit()
        self.db.refresh(post)
        return post

    def get(self, post_id: int) -> Post | None:
        return self.db.get(Post, post_id)

    def set_analysis(self, post: Post, subject: str, taxon: str, category: str, subject_confidence: float) -> None:
        post.subject = subject
        post.taxon = taxon or None
        post.category = category
        post.subject_confidence = subject_confidence
        self.db.commit()


class PostVectorRepository:
    def __init__(self, db: Session):
        self.db = db

    def upsert(self, post_id: int, tenant_id: str, embedding: list[float], model: str) -> PostVector:
        row = self.db.scalar(select(PostVector).where(PostVector.post_id == post_id))
        if row is None:
            row = PostVector(post_id=post_id, tenant_id=tenant_id, embedding=embedding, model=model)
            self.db.add(row)
        else:
            row.embedding = embedding
            row.model = model
        self.db.commit()
        self.db.refresh(row)
        return row

    def get_by_post_id(self, post_id: int) -> PostVector | None:
        return self.db.scalar(select(PostVector).where(PostVector.post_id == post_id))


class SuggestionRepository:
    def __init__(self, db: Session):
        self.db = db

    def upsert(
        self, tenant_id: str, post_id: int, image_id: int, rank: int,
        similarity: float, verdict: str, guard_checks: list[dict],
    ) -> Suggestion:
        row = self.db.scalar(
            select(Suggestion).where(Suggestion.post_id == post_id, Suggestion.image_id == image_id)
        )
        if row is None:
            row = Suggestion(
                tenant_id=tenant_id, post_id=post_id, image_id=image_id, rank=rank,
                similarity=similarity, verdict=verdict, guard_checks=guard_checks,
            )
            self.db.add(row)
        else:
            row.rank = rank
            row.similarity = similarity
            row.verdict = verdict
            row.guard_checks = guard_checks
        self.db.commit()
        self.db.refresh(row)
        return row

    def get(self, suggestion_id: int) -> Suggestion | None:
        return self.db.get(Suggestion, suggestion_id)


class ReviewRepository:
    def __init__(self, db: Session):
        self.db = db

    def create(self, **kwargs) -> Review:
        row = Review(**kwargs)
        self.db.add(row)
        self.db.flush()
        return row


class CostLogRepository:
    def __init__(self, db: Session):
        self.db = db

    def create(self, **kwargs) -> CostLog:
        row = CostLog(**kwargs)
        self.db.add(row)
        self.db.flush()
        return row

    def list(self, limit: int = 50) -> list[CostLog]:
        stmt = select(CostLog).order_by(CostLog.created_at.desc()).limit(limit)
        return list(self.db.scalars(stmt))
