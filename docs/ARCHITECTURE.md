# Architecture: AI Image Understanding & Content Matching Engine

## Overview

A production-grade system that tags an image library with a vision model, matches images to blog posts by semantic meaning, and guards against confident mistakes. The architecture separates concerns into layers: HTTP, orchestration, business logic, data access, and external providers. The mismatch guard is the core reliability feature—it refuses wrong recommendations with human-readable explanations.

**Tech stack:** Python 3.12 · FastAPI · Pydantic v2 · SQLAlchemy 2 + Alembic · PostgreSQL 16 + pgvector · google-genai SDK (Interactions API) · tenacity (retries) · Docker Compose.

---

## System Layers

```
┌──────────────────────────────────────┐
│       HTTP / FastAPI routers         │  Request/response, validation, 4xx/5xx
├──────────────────────────────────────┤
│   Services (orchestration)           │  Coordinate vision, embeddings, matching
├──────────────────────────────────────┤
│   Guard (pure functions)             │  Validate tags, check similarity, explain
├──────────────────────────────────────┤
│   Repos (SQLAlchemy models)          │  Queries, transactions, migrations
├──────────────────────────────────────┤
│   Providers (Gemini | Ollama)        │  Vision, embeddings, cost tracking
└──────────────────────────────────────┘
```

### 1. HTTP Layer (`api/`)

**Routers** (`routers/`):
- `images.py`: `POST /jobs/process-images`, `GET /images/{id}`
- `posts.py`: `POST /posts`, `GET /posts/{id}/images`, `POST /posts/{id}/check`
- `suggestions.py`: `GET /suggestions/{id}`, `POST /suggestions/{id}/approve/reject`
- `jobs.py`: `GET /jobs/{id}`, `GET /jobs?status=failed`
- `costs.py`: `GET /costs`
- `health.py`: `GET /health`

**Models** (`schemas/`): Pydantic request/response schemas. All user input is validated here; invalid input returns `422 Unprocessable Entity`, never a 500.

**Error handling:** Pydantic validation errors automatically produce 422 responses. Service errors are caught by exception handlers and returned as JSON errors with a correlation ID.

### 2. Services (`services/`)

**Purpose:** Orchestrate multiple repos and providers. No SQL or I/O inside the guard—only here.

**Key services:**
- `vision_service.py`: orchestrate image ingestion, call the vision provider, validate schema, flag/reject low-confidence results, store metadata
- `embedding_service.py`: orchestrate embedding generation for image captions and post text, store vectors, handle retries and cost tracking
- `matching_service.py`: query image vectors, rank candidates by similarity, run each through the guard, return suggestions or "no match"
- `post_service.py`: analyze posts (extract subject/taxon/category), enqueue post embedding jobs
- `review_service.py`: handle approval/rejection, idempotent updates

### 3. Guard (`guard/`)

**Pure functions, no I/O, fully unit-tested.** Takes validated tags, post metadata, similarity scores, and configuration, returns a verdict and per-check explanations.

```python
# guard/check.py
@dataclass
class GuardCheck:
    name: str  # "confidence" | "category" | "taxon" | "similarity"
    passed: bool
    reason: str  # human-readable failure reason

class GuardVerdict:
    ACCEPT = "accept"
    REJECT = "reject"

def run_guard(
    image_meta: ImageMetadata,
    post_meta: PostMetadata,
    similarity_score: float,
    config: GuardConfig
) -> tuple[str, list[GuardCheck]]:
    """
    Returns (verdict, checks).
    Checks include all results (passed and failed).
    Only ACCEPT if all checks pass.
    """
    checks = []
    
    # confidence check
    if image_meta.confidence < config.conf_min:
        checks.append(GuardCheck("confidence", False, 
            f"Image classification too uncertain ({image_meta.confidence:.2f} < {config.conf_min})"))
    else:
        checks.append(GuardCheck("confidence", True, ""))
    
    # category match
    if image_meta.category != post_meta.category:
        checks.append(GuardCheck("category", False,
            f"Category mismatch: expected {post_meta.category}, got {image_meta.category}"))
    else:
        checks.append(GuardCheck("category", True, ""))
    
    # taxon match (only if post has a subject)
    if post_meta.taxon and image_meta.taxon != post_meta.taxon:
        checks.append(GuardCheck("taxon", False,
            f"{post_meta.category.title()} category mismatch: expected {post_meta.taxon}, got {image_meta.taxon}"))
    else:
        checks.append(GuardCheck("taxon", True, ""))
    
    # similarity threshold
    if similarity_score < config.sim_min:
        checks.append(GuardCheck("similarity", False,
            f"Similarity {similarity_score:.2f} below threshold {config.sim_min}"))
    else:
        checks.append(GuardCheck("similarity", True, ""))
    
    verdict = GuardVerdict.ACCEPT if all(c.passed for c in checks) else GuardVerdict.REJECT
    return verdict, checks
```

---

## Data Flow

### Image Ingestion Pipeline

```
1. User calls: POST /jobs/process-images
   ├─> FastAPI validates request, creates a Job record (status="queued")
   └─> Returns 202 Accepted {job_id}

2. Worker picks up job (SELECT ... FOR UPDATE SKIP LOCKED)
   ├─> Load image files from corpus/images/ (local or download script)
   ├─> For each image:
   │   ├─> Skip if already processed (content_hash check)
   │   ├─> Load image bytes, encode as base64 for API
   │   ├─> Call vision provider with structured output schema (Pydantic model)
   │   │   ├─> Gemini Interactions API: client.interactions.create(
   │   │   │     model="gemini-3.5-flash-lite",
   │   │   │     input=[{"type":"text", "text":"..."}, {"type":"image", "data":b64}],
   │   │   │     response_format={
   │   │   │       "type":"text",
   │   │   │       "mime_type":"application/json",
   │   │   │       "schema": ImageMetadata.model_json_schema()
   │   │   │     }
   │   │   │   )
   │   │   ├─> Extract interaction.output_text (JSON string)
   │   │   ├─> Parse: ImageMetadata.model_validate_json(output_text)
   │   │   ├─> Log cost: interaction.usage.{total_input_tokens, total_output_tokens}
   │   │   └─> On validation failure or 429/5xx: retry with tenacity, then mark failed
   │   ├─> If confidence < CONF_MIN, mark image status="flagged", reason="low confidence"
   │   ├─> Store ImageMetadata, create tags
   │   └─> Job attempt incremented, run_after set to retry time
   └─> Job status = "completed" or "failed"

3. User polls: GET /jobs/{id}
   └─> Returns {id, status, progress, errors, cost}
```

### Embedding Pipeline

```
1. When image is tagged (status="tagged"), enqueue embedding job
   └─> async_to_sync or background worker picks up

2. Embed image:
   ├─> subject + caption + attributes → document
   └─> Call embedding provider:
       ├─> Gemini: client.models.embed_content(
       │     model="gemini-embedding-2",
       │     content="subject + caption + attributes",
       │     task_type="SEMANTIC_SIMILARITY"  // prefix added at call time
       │   )
       ├─> Extract response.embedding (list of 768 floats)
       ├─> Store vector + metadata
       └─> Log cost

3. When post is created (POST /posts):
   ├─> Extract post subject/category (text call to vision or simple pattern match)
   ├─> Embed post text (title + body)
   └─> Store vector
```

### Matching & Guard Flow

```
1. User calls: GET /posts/{id}/images

2. Matching service:
   ├─> Fetch post vector + post metadata (subject, category, taxon)
   ├─> Query pgvector: SELECT * FROM image_vectors
   │     WHERE tenant_id = ? ORDER BY embedding <=> post_vector LIMIT 20
   │   (cosine similarity in pgvector)
   ├─> Fetch image metadata for top 20
   ├─> For each image (rank 1 to 20):
   │   ├─> Compute cosine similarity (pgvector returned this)
   │   ├─> Run guard(image, post, similarity, config)
   │   ├─> If ACCEPT, return this as suggestion rank 1
   │   └─> Else, collect reasons and continue
   └─> If no ACCEPT found:
       └─> Return {"status":"no_confident_match", "reasons":[all checks from all images]}

3. Response: 200 {"status":"accepted", "rank":1, "image_id":123, "similarity":0.78, "checks":[...]}
   or 200 {"status":"no_confident_match", "reasons":[...]}
```

### Review Workflow

```
POST /suggestions/{id}/approve
├─> Idempotent: if already approved, return 200 (same decision)
├─> Update suggestions.review_status="approved", decision="human_approved"
└─> Return 200

POST /posts/{id}/check {image_id: 456}
├─> Force image through guard (Probe 3)
├─> Same flow as matching, but with only one candidate
└─> Return guard verdict + checks (for Probe 3)
```

---

## Database Schema

All tables carry `tenant_id` (string, defaults to `DEFAULT_TENANT_ID` from `.env`). All tenant-scoped queries filter by it.

### Images

```sql
CREATE TABLE images (
    id BIGSERIAL PRIMARY KEY,
    tenant_id TEXT NOT NULL,
    filename TEXT NOT NULL,
    source_url TEXT,
    license TEXT,  -- e.g., "CC0", "Unsplash", "Pexels"
    content_hash TEXT NOT NULL,  -- SHA256(file bytes)
    status TEXT NOT NULL DEFAULT 'pending',  -- pending|processing|tagged|flagged|failed
    created_at TIMESTAMP DEFAULT now(),
    UNIQUE(tenant_id, content_hash),
    INDEX(tenant_id, status)
);
```

### Image Metadata

```sql
CREATE TABLE image_metadata (
    id BIGSERIAL PRIMARY KEY,
    image_id BIGINT NOT NULL UNIQUE,
    tenant_id TEXT NOT NULL,
    subject TEXT,
    taxon TEXT,  -- broad common-name group: "fox", "wolf", "dog"
    category TEXT,  -- "animal", "vehicle", etc.
    caption TEXT,
    confidence FLOAT NOT NULL,  -- 0.0–1.0
    attributes TEXT[],  -- ARRAY of strings
    needs_review BOOLEAN DEFAULT FALSE,
    raw_response JSONB,  -- full API response for debugging
    model TEXT,  -- which vision model produced this
    created_at TIMESTAMP DEFAULT now(),
    FOREIGN KEY (image_id) REFERENCES images(id) ON DELETE CASCADE,
    INDEX(tenant_id, taxon),
    INDEX(tenant_id, category)
);
```

### Image Vectors

```sql
CREATE TABLE image_vectors (
    id BIGSERIAL PRIMARY KEY,
    image_id BIGINT NOT NULL UNIQUE,
    tenant_id TEXT NOT NULL,
    embedding vector(768),  -- pgvector column
    model TEXT,  -- which embedding model
    created_at TIMESTAMP DEFAULT now(),
    FOREIGN KEY (image_id) REFERENCES images(id) ON DELETE CASCADE,
    INDEX USING hnsw (embedding vector_cosine_ops)  -- fast similarity search
);
```

### Posts

```sql
CREATE TABLE posts (
    id BIGSERIAL PRIMARY KEY,
    tenant_id TEXT NOT NULL,
    title TEXT NOT NULL,
    body TEXT NOT NULL,
    subject TEXT,  -- e.g., "red fox behavior"
    taxon TEXT,    -- e.g., "fox"
    category TEXT,  -- e.g., "animal"
    subject_confidence FLOAT,  -- may be low or NULL if no clear subject
    created_at TIMESTAMP DEFAULT now(),
    INDEX(tenant_id, taxon),
    INDEX(tenant_id, category)
);
```

### Post Vectors

```sql
CREATE TABLE post_vectors (
    id BIGSERIAL PRIMARY KEY,
    post_id BIGINT NOT NULL UNIQUE,
    tenant_id TEXT NOT NULL,
    embedding vector(768),
    model TEXT,
    created_at TIMESTAMP DEFAULT now(),
    FOREIGN KEY (post_id) REFERENCES posts(id) ON DELETE CASCADE
);
```

### Suggestions

```sql
CREATE TABLE suggestions (
    id BIGSERIAL PRIMARY KEY,
    post_id BIGINT NOT NULL,
    image_id BIGINT NOT NULL,
    tenant_id TEXT NOT NULL,
    rank INT NOT NULL,  -- 1, 2, 3... (sorted by similarity)
    similarity FLOAT NOT NULL,
    verdict TEXT NOT NULL,  -- "accept" | "reject"
    guard_checks JSONB NOT NULL,  -- [{"name":"confidence", "passed":true, "reason":""}]
    review_status TEXT DEFAULT 'pending',  -- pending|approved|rejected|human_approved
    created_at TIMESTAMP DEFAULT now(),
    UNIQUE(post_id, image_id),
    INDEX(post_id, rank),
    FOREIGN KEY (post_id) REFERENCES posts(id) ON DELETE CASCADE,
    FOREIGN KEY (image_id) REFERENCES images(id) ON DELETE CASCADE
);
```

### Reviews

```sql
CREATE TABLE reviews (
    id BIGSERIAL PRIMARY KEY,
    suggestion_id BIGINT NOT NULL,
    tenant_id TEXT NOT NULL,
    decision TEXT,  -- "human_approved" | "human_rejected"
    note TEXT,
    created_at TIMESTAMP DEFAULT now(),
    FOREIGN KEY (suggestion_id) REFERENCES suggestions(id) ON DELETE CASCADE,
    INDEX(suggestion_id)
);
```

### Jobs (Background Processing)

```sql
CREATE TABLE jobs (
    id BIGSERIAL PRIMARY KEY,
    tenant_id TEXT NOT NULL,
    type TEXT NOT NULL,  -- "process_images" | "embed_images" | "embed_posts"
    payload JSONB NOT NULL,  -- {"corpus_path": "..."}
    status TEXT NOT NULL DEFAULT 'queued',  -- queued|in_progress|completed|failed
    attempts INT DEFAULT 0,
    max_attempts INT DEFAULT 3,
    run_after TIMESTAMP DEFAULT now(),
    locked_at TIMESTAMP,
    locked_by TEXT,  -- worker identifier
    last_error TEXT,
    progress JSONB,  -- {"processed": 10, "total": 50, "errors": []}
    created_at TIMESTAMP DEFAULT now(),
    updated_at TIMESTAMP DEFAULT now(),
    INDEX(status, run_after),
    INDEX(tenant_id, status)
);
```

### Cost Log

```sql
CREATE TABLE cost_log (
    id BIGSERIAL PRIMARY KEY,
    job_id BIGINT,
    tenant_id TEXT NOT NULL,
    call_type TEXT NOT NULL,  -- "vision" | "embedding" | "post_analysis"
    model TEXT,
    input_tokens INT,
    output_tokens INT,
    est_cost_usd DECIMAL(10, 6),  -- estimated, using paid-tier pricing
    latency_ms INT,
    status TEXT,  -- "success" | "rate_limit" | "error"
    ref_id TEXT,  -- interaction.id or similar, for debugging
    created_at TIMESTAMP DEFAULT now(),
    INDEX(created_at),
    INDEX(job_id)
);
```

**Migrations:** Alembic manages all DDL. Each `alembic revision --autogenerate` creates a new migration file in `alembic/versions/`. Developers run `alembic upgrade head` on startup (or in docker-entrypoint.sh).

---

## Providers (`providers/`)

### Vision Provider Interface

```python
# providers/vision.py
class VisionProvider(ABC):
    async def tag_image(
        self,
        image_bytes: bytes,
        mime_type: str
    ) -> ImageMetadata:
        """
        Returns validated ImageMetadata or raises:
        - ValidationError if response doesn't match schema
        - RateLimitError (429)
        - APIError (5xx)
        """

class GeminiFlashVision(VisionProvider):
    def __init__(self, api_key: str, model: str = "gemini-3.5-flash-lite"):
        self.client = genai.Client(api_key=api_key)
        self.model = model
    
    async def tag_image(self, image_bytes, mime_type):
        b64 = base64.b64encode(image_bytes).decode()
        prompt = """Analyze this image and return structured metadata.
        subject: What is the main subject? (e.g., "red fox")
        taxon: Broad common name, lowercase, singular (e.g., "fox")
        category: One of: animal, vehicle, food, architecture, landscape, other
        attributes: List of visual characteristics (max 5)
        caption: Short human-readable caption
        confidence: 0.0–1.0 confidence in your classification
        """
        interaction = self.client.interactions.create(
            model=self.model,
            input=[
                {"type": "text", "text": prompt},
                {"type": "image", "data": b64, "mime_type": mime_type}
            ],
            response_format={
                "type": "text",
                "mime_type": "application/json",
                "schema": ImageMetadata.model_json_schema()
            }
        )
        
        # Parse and validate
        metadata = ImageMetadata.model_validate_json(interaction.output_text)
        
        # Log cost
        self._log_cost(
            call_type="vision",
            tokens_in=interaction.usage.total_input_tokens,
            tokens_out=interaction.usage.total_output_tokens,
            model=self.model
        )
        
        return metadata
```

### Embedding Provider

```python
class EmbeddingProvider(ABC):
    async def embed(self, text: str, task: str = "SEMANTIC_SIMILARITY") -> list[float]:
        """Returns 768-dimensional embedding."""

class GeminiEmbedding(EmbeddingProvider):
    def __init__(self, api_key: str, model: str = "gemini-embedding-2", dim: int = 768):
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.dim = dim
    
    async def embed(self, text, task="SEMANTIC_SIMILARITY"):
        # Prepend task type to text (Gemini's way)
        text_with_task = f"document: {text}" if task == "SEMANTIC_SIMILARITY" else text
        
        response = self.client.models.embed_content(
            model=self.model,
            contents=text_with_task
        )
        
        embedding = response.embedding
        # Validate dimension
        assert len(embedding) == self.dim, f"Expected {self.dim} dims, got {len(embedding)}"
        
        return embedding
```

### Retry Strategy

All provider calls use `tenacity` with exponential backoff:

```python
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    RetryError
)

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((APIError, RateLimitError)),
    reraise=True
)
async def tag_image_with_retry(self, image_bytes, mime_type):
    return await self.tag_image(image_bytes, mime_type)
```

On final failure, the job is marked `failed`, logged at ERROR level, and visible via `GET /jobs?status=failed`.

---

## Worker (`worker/`)

A separate process (or thread pool) that pulls jobs from the queue and processes them.

```python
# worker/main.py
async def worker_loop():
    while True:
        # Claim a job (SELECT ... FOR UPDATE SKIP LOCKED)
        job = await claim_job(status="queued", limit=1)
        
        if not job:
            await asyncio.sleep(5)
            continue
        
        try:
            job.status = "in_progress"
            job.locked_at = now()
            await db.commit()
            
            if job.type == "process_images":
                await process_images_job(job)
            elif job.type == "embed_images":
                await embed_images_job(job)
            
            job.status = "completed"
            await db.commit()
        
        except Exception as e:
            job.attempts += 1
            if job.attempts >= job.max_attempts:
                job.status = "failed"
                job.last_error = str(e)
            else:
                job.status = "queued"
                job.run_after = now() + timedelta(seconds=2 ** job.attempts)
            
            await db.commit()
            logger.exception(f"Job {job.id} failed", extra={"job_id": job.id})
```

Idempotency: Jobs are idempotent by content hash (images) or post ID (embeddings). Retrying the same job skips already-processed items.

---

## Cost Tracking

Every API call is logged to `cost_log` with:
- `call_type`: "vision" | "embedding" | "post_analysis"
- `input_tokens`, `output_tokens`: from `interaction.usage`
- `est_cost_usd`: `(input_tokens * input_price + output_tokens * output_price)`
- `model`, `ref_id` (interaction ID for debugging)

```python
# config/pricing.py
PRICING = {
    "gemini-3.5-flash-lite": {
        "input": 0.075 / 1_000_000,    # per token
        "output": 0.30 / 1_000_000
    },
    "gemini-embedding-2": {
        "input": 0.02 / 1_000_000,
        "output": 0  # embeddings usually don't have output cost
    }
}

def estimate_cost(model, input_tokens, output_tokens):
    p = PRICING.get(model, {})
    return (
        input_tokens * p.get("input", 0) +
        output_tokens * p.get("output", 0)
    )
```

**Budget guard:** If cumulative cost in a job exceeds `MAX_EST_USD`, the worker pauses and logs a warning.

---

## Error Handling

### Request Validation

FastAPI + Pydantic automatically return `422 Unprocessable Entity` if request body doesn't match schema.

### API Errors

Provider errors are caught and categorized:
- `RateLimitError` (429): retry with backoff
- `APIError` (5xx): retry with backoff
- `ValidationError`: mark job failed (no retry)
- `ClientError` (4xx, non-429): mark job failed, log for investigation

### Database Errors

Transaction failures are retried at the service layer. Deadlocks trigger a rollback and re-run of the transaction.

### Exceptions

Global exception handler in FastAPI returns JSON:

```python
@app.exception_handler(Exception)
async def generic_exception_handler(request, exc):
    correlation_id = request.state.correlation_id  # set by middleware
    logger.exception("Unhandled exception", extra={"correlation_id": correlation_id})
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "correlation_id": correlation_id}
    )
```

---

## Testing Strategy

### Unit Tests (`tests/unit/`)

- Guard logic: fox vs wolf, no-match cases
- Schema validation: valid/invalid vision responses
- Cost estimation

### Integration Tests (`tests/integration/`)

- Vision provider with mock responses (vcr.py or responses library)
- Embedding + pgvector similarity search
- Job workflow (claim, process, mark complete)
- Review workflow (approve → check idempotency)

### End-to-End Tests (`tests/e2e/`)

- Docker Compose with real postgres + API + worker
- Upload 5-image corpus, verify tags
- Create posts, verify matching
- Force guard rejection, verify explanation

---

## Deployment (Docker Compose)

```yaml
version: "3.8"
services:
  db:
    image: postgres:16
    environment:
      POSTGRES_USER: capstone
      POSTGRES_PASSWORD: capstone
      POSTGRES_DB: capstone
    volumes:
      - db_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  api:
    build: .
    depends_on:
      - db
    environment:
      DATABASE_URL: postgresql+psycopg://capstone:capstone@db:5432/capstone
      GEMINI_API_KEY: ${GEMINI_API_KEY}
      # ... other env vars
    ports:
      - "8000:8000"
    command: uvicorn src.main:app --host 0.0.0.0 --port 8000

  worker:
    build: .
    depends_on:
      - db
    environment:
      DATABASE_URL: postgresql+psycopg://capstone:capstone@db:5432/capstone
      GEMINI_API_KEY: ${GEMINI_API_KEY}
    command: python -m src.worker

volumes:
  db_data:
```

`docker compose up --build` starts all services. The API is ready at `http://localhost:8000`; worker begins polling jobs immediately.

---

## File Structure

```
flyrank-capstone-imagerelevance/
├── docs/
│   ├── DESIGN.md
│   └── ARCHITECTURE.md  (this file)
├── src/
│   ├── main.py  (FastAPI app, middleware, exception handlers)
│   ├── config.py  (settings, pricing, guard thresholds)
│   ├── api/
│   │   ├── schemas.py  (Pydantic request/response models)
│   │   └── routers/
│   │       ├── images.py
│   │       ├── posts.py
│   │       ├── suggestions.py
│   │       ├── jobs.py
│   │       ├── costs.py
│   │       └── health.py
│   ├── services/
│   │   ├── vision_service.py
│   │   ├── embedding_service.py
│   │   ├── matching_service.py
│   │   ├── post_service.py
│   │   └── review_service.py
│   ├── guard/
│   │   ├── check.py  (GuardCheck, GuardVerdict, run_guard)
│   │   └── config.py  (GuardConfig with thresholds)
│   ├── db/
│   │   ├── models.py  (SQLAlchemy ORM models)
│   │   ├── session.py  (SessionLocal, async context manager)
│   │   └── repos.py  (Repository classes)
│   ├── providers/
│   │   ├── vision.py  (VisionProvider, GeminiFlashVision)
│   │   ├── embedding.py  (EmbeddingProvider, GeminiEmbedding)
│   │   └── fallback.py  (Ollama fallback, if implemented)
│   └── worker/
│       └── main.py  (worker loop, job processor)
├── alembic/
│   ├── versions/  (migration files)
│   └── env.py
├── scripts/
│   ├── download_corpus.py  (download from Unsplash/Pexels)
│   ├── seed.py  (create sample posts)
│   └── eval.py  (run eval set, print top-1 precision)
├── tests/
│   ├── unit/
│   ├── integration/
│   └── e2e/
├── corpus/
│   ├── manifest.csv  (filename, url, photographer, license)
│   └── images/  (gitignored, reproduced by download_corpus.py)
├── eval/
│   ├── labels.json  ({"post_id": image_id, ...})
│   └── posts.json  (12+ scene-specific posts)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
├── BUILDLOG.md
├── capstone.yaml
├── LICENSE
└── EVIDENCE.md
```

---

## How It Ends

The guard is the core of this system. Everything feeds it: validated image metadata, post analysis, vector similarity scores, and configurable thresholds. The guard's job is simple and testable: given all the evidence, is this recommendation good enough? If not, explain why in terms a human understands.

The rest—vision, embeddings, jobs, retries, cost tracking—is infrastructure to get the guard the data it needs and let humans review and refine the boundaries.