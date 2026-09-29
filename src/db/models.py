from sqlalchemy import Column, Integer, String, Float, DateTime, Text, ARRAY, BIGINT, JSON, UniqueConstraint
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime
from pgvector.sqlalchemy import Vector

Base = declarative_base()

class Image(Base):
    __tablename__ = "images"
    
    id = Column(BIGINT, primary_key=True)
    tenant_id = Column(String, nullable=False, index=True)
    filename = Column(String, nullable=False)
    source_url = Column(String)
    license = Column(String)
    content_hash = Column(String, nullable=False, unique=True)
    status = Column(String, default="pending")  # pending|processing|tagged|flagged|failed
    created_at = Column(DateTime, default=datetime.utcnow)

class ImageMetadata(Base):
    __tablename__ = "image_metadata"
    
    id = Column(BIGINT, primary_key=True)
    image_id = Column(BIGINT, unique=True, nullable=False)
    tenant_id = Column(String, nullable=False, index=True)
    subject = Column(String)
    taxon = Column(String, index=True)
    category = Column(String, index=True)
    caption = Column(Text)
    confidence = Column(Float)
    attributes = Column(ARRAY(String))
    needs_review = Column(Integer, default=0)
    raw_response = Column(JSON)
    model = Column(String)

class Job(Base):
    __tablename__ = "jobs"
    
    id = Column(BIGINT, primary_key=True)
    tenant_id = Column(String, nullable=False, index=True)
    type = Column(String, nullable=False)  # process_images|embed_image|embed_post
    payload = Column(JSON)
    status = Column(String, default="queued", index=True)
    attempts = Column(Integer, default=0)
    max_attempts = Column(Integer, default=3)
    run_after = Column(DateTime, default=datetime.utcnow, index=True)
    locked_at = Column(DateTime)
    locked_by = Column(String)
    last_error = Column(Text)
    progress = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class ImageVector(Base):
    __tablename__ = "image_vectors"

    id = Column(BIGINT, primary_key=True)
    image_id = Column(BIGINT, unique=True, nullable=False)
    tenant_id = Column(String, nullable=False, index=True)
    embedding = Column(Vector(768))
    model = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)


class Post(Base):
    __tablename__ = "posts"

    id = Column(BIGINT, primary_key=True)
    tenant_id = Column(String, nullable=False, index=True)
    title = Column(String, nullable=False)
    body = Column(Text, nullable=False)
    subject = Column(String)
    taxon = Column(String, index=True)
    category = Column(String, index=True)
    subject_confidence = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)


class PostVector(Base):
    __tablename__ = "post_vectors"

    id = Column(BIGINT, primary_key=True)
    post_id = Column(BIGINT, unique=True, nullable=False)
    tenant_id = Column(String, nullable=False, index=True)
    embedding = Column(Vector(768))
    model = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)


class Suggestion(Base):
    __tablename__ = "suggestions"

    id = Column(BIGINT, primary_key=True)
    tenant_id = Column(String, nullable=False, index=True)
    post_id = Column(BIGINT, nullable=False, index=True)
    image_id = Column(BIGINT, nullable=False)
    rank = Column(Integer, nullable=False)
    similarity = Column(Float, nullable=False)
    verdict = Column(String, nullable=False)
    guard_checks = Column(JSON, nullable=False)
    review_status = Column(String, default="pending")
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (UniqueConstraint("post_id", "image_id"),)


class Review(Base):
    __tablename__ = "reviews"

    id = Column(BIGINT, primary_key=True)
    tenant_id = Column(String, nullable=False)
    suggestion_id = Column(BIGINT, nullable=False, index=True)
    decision = Column(String)
    note = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)


class CostLog(Base):
    __tablename__ = "cost_log"
    
    id = Column(BIGINT, primary_key=True)
    job_id = Column(BIGINT, index=True)
    tenant_id = Column(String, nullable=False)
    call_type = Column(String)  # vision|embedding|post_analysis
    model = Column(String)
    input_tokens = Column(Integer)
    output_tokens = Column(Integer)
    est_cost_usd = Column(Float)
    latency_ms = Column(Integer)
    status = Column(String)
    ref_id = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)