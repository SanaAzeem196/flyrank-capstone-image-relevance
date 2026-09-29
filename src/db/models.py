from sqlalchemy import Column, Integer, String, Float, DateTime, Text, ARRAY, BIGINT, JSON
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