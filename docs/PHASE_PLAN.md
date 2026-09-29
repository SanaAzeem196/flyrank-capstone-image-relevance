# Phase-Wise Development Plan

AI Image Understanding & Content Matching Engine · FlyRank Backend Track Capstone

---

## Overview

Four phases, each with clear objectives, deliverables, and a gate (proof of completion). Work through them in order. Each phase builds on the previous one. Estimated total: 35–50 focused hours.

**Key principle:** End-to-end slices, not waterfall. By the end of Phase 2, you have a working vision pipeline. By Phase 3, you have matching + guard. Phase 4 hardens everything and proves it works.

---

## Phase 1: Design ✅ COMPLETE

**Objective:** Document the system end-to-end before writing code.

**Deliverables:**
- ✅ `docs/DESIGN.md`: problem, metadata schema, guard rules, data model, API, layers, corpus/eval plan, non-goals
- ✅ `docs/ARCHITECTURE.md`: system layers, data flows, database schema, provider interface, worker loop, error handling
- ✅ Repo skeleton: `.gitignore`, `.env.example`, `README.md` (stubbed), `BUILDLOG.md`, `LICENSE`
- ✅ Phase 1 entry in `BUILDLOG.md` with decisions made

**Gate:** Design doc + architecture doc committed to public GitHub repo (`main` branch).

**Time:** ~4–6 h (done).

---

## Phase 2: Image Understanding Pipeline

**Objective:** Build the vision batch job with schema validation, retries, and cost tracking.

**What you'll build:**
1. Python project structure: `src/`, `alembic/`, `tests/`, `scripts/`, `requirements.txt`
2. Database: Postgres + pgvector setup, Alembic migrations, ORM models
3. Vision provider: Gemini Interactions API wrapper, schema validation with Pydantic
4. Batch job: claim jobs from queue, process images, store metadata, handle retries
5. Cost tracking: every vision call logged with token counts and estimated USD
6. Config: guard thresholds, provider models, database connection

**New files to create:**
```
src/
├── main.py  (FastAPI app boilerplate)
├── config.py  (settings, guard thresholds, pricing)
├── api/
│   ├── schemas.py  (ImageMetadataResponse, JobResponse, error models)
│   └── routers/
│       ├── images.py  (POST /jobs/process-images, GET /images/{id})
│       ├── jobs.py  (GET /jobs/{id}, GET /jobs?status=failed)
│       └── health.py  (GET /health)
├── db/
│   ├── models.py  (Image, ImageMetadata, Job, CostLog SQLAlchemy models)
│   ├── session.py  (SessionLocal, async context manager)
│   └── repos.py  (ImageRepository, JobRepository classes)
├── services/
│   └── vision_service.py  (ingest images, call vision provider, validate, store)
├── providers/
│   └── vision.py  (VisionProvider ABC, GeminiFlashVision implementation)
└── worker/
    └── main.py  (worker loop, claim jobs, process images, retries)

alembic/
├── env.py
├── script.py.mako
└── versions/
    └── 001_create_tables.py  (images, image_metadata, jobs, cost_log)

scripts/
├── download_corpus.py  (fetch ~44 images from Unsplash/Pexels, save manifest.csv)
└── seed_images.py  (load corpus into DB for local testing)

requirements.txt  (fastapi, sqlalchemy, psycopg, alembic, google-genai, tenacity, pydantic, etc.)
```

**Database schema (first migration):**
- `images` (id, filename, content_hash, status, created_at, tenant_id)
- `image_metadata` (id, image_id, subject, taxon, category, caption, confidence, attributes, raw_response, model)
- `jobs` (id, type, payload, status, attempts, max_attempts, run_after, locked_at, last_error, progress)
- `cost_log` (id, job_id, call_type, model, input_tokens, output_tokens, est_cost_usd, latency_ms, status)

**API endpoints:**
- `POST /jobs/process-images` → enqueue batch job, return `202 {job_id}`
- `GET /jobs/{id}` → job status, progress, errors, cost
- `GET /images` → list images (all or filtered by status)
- `GET /images/{id}` → image metadata, tags
- `GET /health` → liveness check

**Pydantic models (schema.py):**
```python
class ImageMetadata(BaseModel):
    subject: str
    taxon: str
    category: str
    attributes: list[str]
    caption: str
    confidence: float
    # validation: confidence in [0, 1], category in enum, etc.

class ImageMetadataResponse(BaseModel):
    id: int
    subject: str
    taxon: str
    category: str
    confidence: float
    status: str  # tagged | flagged | failed

class JobResponse(BaseModel):
    id: int
    type: str
    status: str
    attempts: int
    progress: dict
    cost_usd: float
    errors: list[str]
```

**Guard config (config.py):**
```python
@dataclass
class GuardConfig:
    conf_min: float = 0.70  # flag if confidence < this
    sim_min: float = 0.60   # reject if similarity < this (tuned in Phase 4)
    max_calls_per_run: int = 300
    max_est_usd: float = 1.00
```

**Vision provider (providers/vision.py):**
- Accept image bytes + mime type
- Call Gemini Interactions API with structured output schema (Pydantic ImageMetadata)
- Parse and validate response
- Log cost (input_tokens, output_tokens)
- Retry on 429/5xx with tenacity, give up after 3 attempts
- Raise ValidationError if response doesn't match schema, CostExceededError if budget exceeded

**Worker loop (worker/main.py):**
```python
async def worker_loop():
    while True:
        job = await claim_job(status="queued", max_attempts=3)
        if not job:
            await sleep(5)
            continue
        
        try:
            job.status = "in_progress"
            if job.type == "process_images":
                await process_images_job(job)
            job.status = "completed"
        except CostExceededError:
            job.status = "paused"  # admin attention needed
        except Exception as e:
            job.attempts += 1
            if job.attempts >= job.max_attempts:
                job.status = "failed"
                job.last_error = str(e)
            else:
                job.status = "queued"
                job.run_after = now() + exponential_backoff(job.attempts)
        
        await db.commit()
```

**Tests (tests/unit/ and tests/integration/):**
- Unit: schema validation (valid/invalid JSON), cost estimation
- Integration: mock vision provider, claim job + process + mark complete, cost log entry created

**Local setup:**
- Create `.env` from `.env.example`, set `GEMINI_API_KEY` (get from aistudio.google.com/apikey)
- Run migrations: `alembic upgrade head`
- Seed corpus: `python scripts/download_corpus.py` → downloads ~44 images, creates `corpus/manifest.csv`
- Run API: `uvicorn src.main:app --reload`
- Run worker: `python -m src.worker` (in separate terminal)

**Docker Compose:**
Create `docker-compose.yml` with three services:
- `db`: postgres:16 with pgvector extension
- `api`: build from Dockerfile, expose 8000
- `worker`: build from Dockerfile, separate process

**Gate:** All images tagged and stored in the database. At least one flagged (low confidence). Costs visible in cost_log. The command `docker compose up --build && sleep 5 && docker compose exec api python scripts/seed_images.py` boots the system, loads corpus, and processes all images in one go.

**Proof for EVIDENCE.md:**
```
## Probe 1: Batch job runs, tags images, flags low-confidence

$ docker compose exec worker python -m src.worker  # runs once
Processed 44 images in 12.3s
Tagged: 42
Flagged (low confidence): 2
Errors: 0

$ psql -c "SELECT status, COUNT(*) FROM images GROUP BY status;"
 status | count
--------+-------
 tagged | 42
 flagged| 2
(2 rows)

$ psql -c "SELECT COUNT(*) FROM cost_log WHERE call_type='vision';"
 count
-------
 44
```

**BUILDLOG entry:**
Document where AI helped (drafted vision_service.py, provider interface), where it was wrong (initial retry logic was missing idempotency check), what you changed (added content_hash deduplication).

**Time estimate:** 10–14 h
- Database + Alembic: 2 h
- Vision provider: 2 h
- Batch job + worker: 2 h
- API endpoints: 1 h
- Cost tracking: 1 h
- Tests: 1 h
- Docker setup: 1 h
- Debugging/polish: 2 h

---

## Phase 3: Matching Engine

**Objective:** Build embeddings, similarity search, and the mismatch guard.

**What you'll build:**
1. Embedding provider: Gemini embeddings API, cost tracking
2. Embed images: captions stored as vectors, indexed in pgvector
3. Embed posts: post text (title + body) embedded, indexed
4. Guard module: pure functions to validate tags, check similarity, explain verdicts
5. Matching service: rank images by similarity, run each through guard, return top match or "no confident match"
6. Review API: endpoints to approve/reject and inspect suggestions

**New files:**
```
src/
├── guard/
│   ├── check.py  (GuardCheck, GuardVerdict, run_guard function)
│   └── config.py  (GuardConfig, guard thresholds)
├── providers/
│   └── embedding.py  (EmbeddingProvider ABC, GeminiEmbedding)
├── services/
│   ├── embedding_service.py  (call embedding provider, store vectors)
│   ├── matching_service.py  (rank images, run guard, return suggestions)
│   └── post_service.py  (analyze posts, extract subject/taxon/category)
├── db/
│   └── repos.py  (add ImageVectorRepository, PostRepository, SuggestionRepository)
└── api/
    ├── schemas.py  (add SuggestionResponse, GuardCheckResponse, NoMatchResponse)
    └── routers/
        ├── posts.py  (POST /posts, GET /posts/{id}/images)
        ├── suggestions.py  (POST /posts/{id}/check, GET /suggestions/{id})
        └── reviews.py  (POST /suggestions/{id}/approve, /reject)

tests/
├── unit/
│   └── test_guard.py  (fox vs wolf cases, no-match, all checks)
└── integration/
    └── test_matching.py  (post + images + guard + verify ranks)
```

**Database schema additions (second migration):**
- `posts` (id, tenant_id, title, body, subject, taxon, category, subject_confidence)
- `post_vectors` (id, post_id, embedding vector(768), model)
- `image_vectors` (id, image_id, embedding vector(768), model) — move from Phase 2 or create here
- `suggestions` (id, post_id, image_id, rank, similarity, verdict, guard_checks JSONB, review_status)
- `reviews` (id, suggestion_id, decision, note)

**Guard module (guard/check.py):**
```python
@dataclass
class GuardCheck:
    name: str  # "confidence" | "category" | "taxon" | "similarity"
    passed: bool
    reason: str  # human-readable explanation

class GuardVerdict:
    ACCEPT = "accept"
    REJECT = "reject"

def run_guard(
    image_meta: ImageMetadata,
    post_meta: PostMetadata,
    similarity: float,
    config: GuardConfig
) -> tuple[str, list[GuardCheck]]:
    """
    Validate image against post. Return verdict + all checks (passed and failed).
    Only ACCEPT if ALL checks pass.
    """
    checks = []
    
    # Confidence
    if image_meta.confidence < config.conf_min:
        checks.append(GuardCheck(
            "confidence",
            False,
            f"Image classification too uncertain ({image_meta.confidence:.2f} < {config.conf_min})"
        ))
    else:
        checks.append(GuardCheck("confidence", True, ""))
    
    # Category
    if image_meta.category != post_meta.category:
        checks.append(GuardCheck(
            "category",
            False,
            f"Category mismatch: expected {post_meta.category}, got {image_meta.category}"
        ))
    else:
        checks.append(GuardCheck("category", True, ""))
    
    # Taxon (only if post has a subject)
    if post_meta.taxon and image_meta.taxon != post_meta.taxon:
        checks.append(GuardCheck(
            "taxon",
            False,
            f"{post_meta.category.title()} mismatch: expected {post_meta.taxon}, got {image_meta.taxon}"
        ))
    else:
        checks.append(GuardCheck("taxon", True, ""))
    
    # Similarity
    if similarity < config.sim_min:
        checks.append(GuardCheck(
            "similarity",
            False,
            f"Similarity {similarity:.2f} below threshold {config.sim_min}"
        ))
    else:
        checks.append(GuardCheck("similarity", True, ""))
    
    verdict = GuardVerdict.ACCEPT if all(c.passed for c in checks) else GuardVerdict.REJECT
    return verdict, checks
```

**Embedding provider (providers/embedding.py):**
```python
class EmbeddingProvider(ABC):
    async def embed(self, text: str, task: str = "SEMANTIC_SIMILARITY") -> list[float]:
        """Return 768-dimensional embedding."""

class GeminiEmbedding(EmbeddingProvider):
    def __init__(self, api_key: str, model: str = "gemini-embedding-2", dim: int = 768):
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.dim = dim
    
    async def embed(self, text: str, task: str = "SEMANTIC_SIMILARITY") -> list[float]:
        # Prepend task prefix to text
        text_with_task = f"document: {text}" if task == "SEMANTIC_SIMILARITY" else text
        
        response = self.client.models.embed_content(
            model=self.model,
            contents=text_with_task
        )
        
        embedding = response.embedding
        assert len(embedding) == self.dim, f"Expected {self.dim} dims, got {len(embedding)}"
        
        # Log cost
        self._log_cost(
            call_type="embedding",
            tokens_in=response.usage.prompt_token_count,
            tokens_out=0,  # embeddings usually don't count output
            model=self.model
        )
        
        return embedding
```

**Matching service (services/matching_service.py):**
```python
async def match_images_for_post(post_id: int, limit: int = 20) -> dict:
    """
    1. Fetch post + its vector
    2. Query pgvector for top-K similar image vectors
    3. Fetch image metadata for top-K
    4. For each image, run guard
    5. Return first ACCEPT or "no_confident_match"
    """
    post = await post_repo.get_by_id(post_id)
    post_vector = await post_vector_repo.get_by_post_id(post_id)
    
    # pgvector similarity search (cosine)
    top_images = await image_vector_repo.similarity_search(
        vector=post_vector.embedding,
        limit=limit
    )
    
    suggestions = []
    for rank, (image_id, similarity) in enumerate(top_images, 1):
        image_meta = await image_metadata_repo.get_by_image_id(image_id)
        verdict, checks = guard.run_guard(
            image_meta=image_meta,
            post_meta=post,
            similarity=similarity,
            config=config.guard_config
        )
        
        suggestion = Suggestion(
            post_id=post_id,
            image_id=image_id,
            rank=rank,
            similarity=similarity,
            verdict=verdict,
            guard_checks=[c.to_dict() for c in checks]
        )
        await suggestion_repo.create(suggestion)
        suggestions.append(suggestion)
        
        if verdict == GuardVerdict.ACCEPT:
            return {
                "status": "accepted",
                "rank": rank,
                "image_id": image_id,
                "similarity": similarity,
                "checks": [c.to_dict() for c in checks]
            }
    
    # No ACCEPT found: collect reasons from all checks
    all_reasons = []
    for suggestion in suggestions:
        for check in suggestion.guard_checks:
            if not check["passed"]:
                all_reasons.append({
                    "image_id": suggestion.image_id,
                    "check": check["name"],
                    "reason": check["reason"]
                })
    
    return {
        "status": "no_confident_match",
        "reasons": all_reasons
    }
```

**API endpoints:**
- `POST /posts` (title, body) → extract subject, embed text, create post, return 201 {post_id}
- `GET /posts/{id}/images` → call matching_service, return suggestions or no_match
- `POST /posts/{id}/check` (image_id) → force image through guard (Probe 3), return verdict
- `GET /suggestions/{id}` → return suggestion with guard_checks, let caller inspect why
- `POST /suggestions/{id}/approve` → idempotent approval, update review_status
- `POST /suggestions/{id}/reject` → idempotent rejection, update review_status

**Tests (tests/unit/test_guard.py):**
```python
def test_guard_rejects_wolf_for_fox_post():
    image = ImageMetadata(subject="gray wolf", taxon="wolf", category="animal", confidence=0.95)
    post = PostMetadata(subject="red fox behavior", taxon="fox", category="animal")
    similarity = 0.75
    
    verdict, checks = guard.run_guard(image, post, similarity, config)
    
    assert verdict == GuardVerdict.REJECT
    assert any(c.name == "taxon" and not c.passed for c in checks)

def test_guard_accepts_fox_for_fox_post():
    image = ImageMetadata(subject="red fox", taxon="fox", category="animal", confidence=0.90)
    post = PostMetadata(subject="red fox behavior", taxon="fox", category="animal")
    similarity = 0.82
    
    verdict, checks = guard.run_guard(image, post, similarity, config)
    
    assert verdict == GuardVerdict.ACCEPT
    assert all(c.passed for c in checks)

def test_guard_rejects_low_confidence():
    image = ImageMetadata(subject="?", taxon="?", category="animal", confidence=0.45)
    post = PostMetadata(subject="...", taxon="fox", category="animal")
    similarity = 0.80
    
    verdict, checks = guard.run_guard(image, post, similarity, config)
    
    assert verdict == GuardVerdict.REJECT
    reason = [c.reason for c in checks if c.name == "confidence"][0]
    assert "too uncertain" in reason
```

**Seed data (scripts/seed.py):**
```python
# Create 12–15 test posts with exactly one labeled correct image
posts = [
    {"title": "Red Fox Behavior in Winter", "body": "...", "correct_image_id": 5},
    {"title": "The Gray Wolf Pack", "body": "...", "correct_image_id": 12},
    {"title": "Dog Breeds and Training", "body": "...", "correct_image_id": 18},
    # ...
]

for post_data in posts:
    post = await post_service.create_post(post_data)
    # Store eval label
    await eval_repo.create_label(post_id=post.id, correct_image_id=post_data["correct_image_id"])
```

**Gate:** Fox ranks first for fox post, wolf and dog rank lower. Force wolf onto fox post, guard rejects with "animal mismatch" reason. Query a post about something absent from corpus (e.g., "ancient castles"), system returns "no confident match" with reasons.

**Proof for EVIDENCE.md:**
```
## Probe 2: Fox ranks first, wolf and dog rank lower
$ curl http://localhost:8000/posts/1/images
{
  "status": "accepted",
  "rank": 1,
  "image_id": 5,
  "similarity": 0.82,
  "checks": [...]
}

## Probe 3: Wolf rejected for fox post
$ curl -X POST http://localhost:8000/posts/1/check -d '{"image_id": 12}'
{
  "status": "rejected",
  "checks": [
    {"name": "taxon", "passed": false, "reason": "Animal mismatch: expected fox, got wolf"}
  ]
}

## Probe 4: No suitable image, "no confident match" + reasons
$ curl http://localhost:8000/posts/15/images
{
  "status": "no_confident_match",
  "reasons": [
    {"image_id": 1, "check": "category", "reason": "Category mismatch: expected architecture, got animal"}
  ]
}
```

**BUILDLOG entry:**
Document the guard's design (why pure functions?), where the embedding provider was tricky (handling task prefixes), and what tests proved the fox/wolf distinction.

**Time estimate:** 12–16 h
- Guard module: 2 h
- Embedding provider: 1.5 h
- Matching service: 2 h
- Post service: 1 h
- API endpoints (posts, suggestions, reviews): 2 h
- Database migrations: 1.5 h
- Tests (guard logic, matching, edge cases): 2 h
- Seed data and eval labels: 1.5 h
- Debugging/polish: 2 h

---

## Phase 4: Production Layer & Evaluation

**Objective:** Harden the system, measure quality on the labeled eval set, and document everything.

**What you'll build:**
1. Eval script: load labeled posts, run matching for each, compute top-1 precision
2. Cost dashboard: total calls, total cost, cost per image, cost per post
3. Review UI (minimal): a table of suggestions + approve/reject buttons (or just curl endpoints)
4. README: architecture diagram, run + seed steps, limitations
5. Error handling hardening: better error messages, correlation IDs, health checks
6. Integration tests: end-to-end flows, all six probes

**New files:**
```
scripts/
├── eval.py  (load eval labels, run matching, compute precision, print results)
└── eval_report.py  (detailed breakdown: precision per category, confusion matrix)

eval/
├── labels.json  ({"1": 5, "2": 12, ...}  — post_id → correct_image_id)
└── posts.json  (full eval posts, including the 3 no-match cases)

docs/
├── ARCHITECTURE.md  (already drafted in Phase 2 prep)
├── GUIDE.md  (how to run locally, interpret results, add more eval posts)

README.md  (updated with: what it is, architecture diagram, run steps, eval number, limitations)

tests/
└── e2e/
    └── test_probes.py  (all six acceptance probes: job tags, fox ranks, wolf rejected, no-match, eval script, cost log)

src/
├── main.py  (add middleware for correlation IDs, structured logging)
└── api/
    └── routers/
        └── health.py  (enhanced: database connection check, worker heartbeat check)
```

**Eval script (scripts/eval.py):**
```python
async def eval_main():
    """
    Load eval labels: {post_id: correct_image_id}
    For each post:
        - Call matching_service
        - Check if rank-1 image == correct_image_id
        - Increment top-1-correct or top-1-wrong
    Print: top-1 precision = (correct / total)
    """
    labels = json.load(open("eval/labels.json"))
    
    correct = 0
    no_match_correct = 0
    no_match_total = 0
    results = []
    
    for post_id, correct_image_id in labels.items():
        post = await post_repo.get_by_id(int(post_id))
        
        # Is this a no-match case?
        if correct_image_id == -1:  # sentinel for "no suitable image"
            result = await matching_service.match_images_for_post(int(post_id))
            no_match_total += 1
            if result["status"] == "no_confident_match":
                no_match_correct += 1
            results.append({
                "post_id": post_id,
                "expected": "no_confident_match",
                "got": result["status"],
                "correct": result["status"] == "no_confident_match"
            })
        else:
            result = await matching_service.match_images_for_post(int(post_id))
            if result["status"] == "accepted" and result["image_id"] == correct_image_id:
                correct += 1
            results.append({
                "post_id": post_id,
                "expected_image_id": correct_image_id,
                "got_image_id": result.get("image_id"),
                "rank": result.get("rank"),
                "correct": result.get("image_id") == correct_image_id
            })
    
    total_posts = len(labels)
    top1_precision = correct / (total_posts - no_match_total) if total_posts > no_match_total else 0.0
    
    print(f"Top-1 Precision: {top1_precision:.2%} ({correct}/{total_posts - no_match_total})")
    print(f"No-match accuracy: {no_match_correct}/{no_match_total}")
    print(f"Total posts: {total_posts}")
    
    # Log results
    with open("eval/results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    return top1_precision
```

**README update:**

```markdown
# flyrank-capstone-imagerelevance

AI Image Understanding & Content Matching Engine (FlyRank Backend Track capstone).

Tags an image library with a vision model, matches images to blog posts by meaning, and uses a **mismatch guard** to reject wrong pairings (a wolf for a fox post) with a human-readable explanation.

## Architecture

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the complete design. In brief:

```
Images (batch job) ——→ Vision Model ——→ Validated Tags + Vectors
Posts ———————————————→ Embed Text ———→ Vectors
                       ↓
                    Cosine Similarity Ranking
                       ↓
                    Mismatch Guard (checks: confidence, category, taxon, similarity)
                       ↓
                    Suggestions or "No Confident Match"
```

## Quick Start

### Prerequisites
- Docker & Docker Compose
- Python 3.12+ (for local scripts)
- Gemini API key (free tier, get at aistudio.google.com/apikey)

### Run

```bash
# 1. Copy .env.example to .env and fill in GEMINI_API_KEY
cp .env.example .env
# Edit .env and add your key

# 2. Boot the system (postgres, api, worker)
docker compose up --build

# 3. In another terminal, seed the corpus and create test posts
docker compose exec api python scripts/seed_images.py
docker compose exec api python scripts/seed_posts.py

# 4. Query the API
curl http://localhost:8000/posts/1/images

# 5. Run the eval set
docker compose exec api python scripts/eval.py
```

## Evaluation

**Top-1 Precision:** 83% (10/12 eval posts ranked correctly)

Full results: `eval/results.json`

## API Endpoints

### Images
- `POST /jobs/process-images` — enqueue batch vision job
- `GET /jobs/{id}` — job status and progress
- `GET /images` — list images (filtered by status)
- `GET /images/{id}` — image metadata and tags

### Posts & Matching
- `POST /posts` — create post (title, body)
- `GET /posts/{id}/images` — get suggestions for post
- `POST /posts/{id}/check` — force an image through the guard

### Suggestions & Review
- `GET /suggestions/{id}` — inspect why an image was selected/rejected
- `POST /suggestions/{id}/approve` — approve suggestion
- `POST /suggestions/{id}/reject` — reject suggestion

### Monitoring
- `GET /health` — system status (db, worker)
- `GET /costs` — cost log and totals

## Limitations

- **No frontend:** API only; use curl or a REST client
- **Corpus size:** ~44 images, 5 categories. Scales to ~500–1000 with paid Gemini tier
- **Tenancy:** thin tenant isolation (demo only; not production-grade)
- **Free tier constraints:** rate limits ~10 RPM for vision; stays well under budget
- **Guard thresholds:** tuned on 12 posts; may need adjustment with larger eval set

## Repo

- **Main branch always runnable:** every commit is tested locally
- **Alembic migrations** in `alembic/versions/`
- **Tests:** unit (guard logic), integration (jobs, matching), e2e (all six probes)
- **Git history:** commits per working session; see [BUILDLOG.md](BUILDLOG.md)
```

**Cost dashboard (GET /costs):**
```json
{
  "total_calls": 50,
  "total_cost_usd": 0.023,
  "calls_by_type": {
    "vision": {"count": 44, "cost_usd": 0.018},
    "embedding": {"count": 60, "cost_usd": 0.005}
  },
  "by_model": {
    "gemini-3.5-flash-lite": {"calls": 44, "cost_usd": 0.018},
    "gemini-embedding-2": {"calls": 60, "cost_usd": 0.005}
  },
  "recent_calls": [
    {"call_type": "vision", "model": "gemini-3.5-flash-lite", "cost_usd": 0.00041, "timestamp": "2026-09-28T14:23:10Z"},
    ...
  ]
}
```

**E2E tests (tests/e2e/test_probes.py):**
```python
@pytest.mark.asyncio
async def test_probe_1_batch_job_tags_images():
    """Probe 1: Run batch job, verify images tagged and at least one flagged."""
    job = await jobs_repo.create(
        type="process_images",
        payload={"corpus_path": "corpus/images/"}
    )
    
    # Run worker (or call service directly in test)
    await worker_loop_once(job.id)
    
    images = await images_repo.filter(status="tagged")
    flagged = await images_repo.filter(status="flagged")
    
    assert len(images) >= 42
    assert len(flagged) >= 1

@pytest.mark.asyncio
async def test_probe_2_fox_ranks_first():
    """Probe 2: Fox ranks first for fox post."""
    result = await matching_service.match_images_for_post(fox_post_id)
    assert result["status"] == "accepted"
    assert result["image_id"] == fox_image_id
    assert result["rank"] == 1

@pytest.mark.asyncio
async def test_probe_3_guard_rejects_wolf():
    """Probe 3: Forcing wolf onto fox post gets rejected."""
    result = await matching_service.check_image_for_post(
        post_id=fox_post_id,
        image_id=wolf_image_id
    )
    assert result["status"] == "rejected"
    assert any("mismatch" in c["reason"].lower() for c in result["checks"])

@pytest.mark.asyncio
async def test_probe_4_no_match():
    """Probe 4: Post about castles (not in corpus) returns no_confident_match."""
    result = await matching_service.match_images_for_post(castle_post_id)
    assert result["status"] == "no_confident_match"
    assert "reasons" in result

@pytest.mark.asyncio
async def test_probe_5_eval_script():
    """Probe 5: Eval script runs and matches README number."""
    precision = await scripts.eval_main()
    assert precision >= 0.80  # placeholder threshold

@pytest.mark.asyncio
async def test_probe_6_cost_log():
    """Probe 6: Every vision/embedding call has a cost entry."""
    cost_entries = await cost_log_repo.filter(call_type="vision")
    vision_images = await images_repo.filter(status__in=["tagged", "flagged"])
    assert len(cost_entries) == len(vision_images)
```

**Evaluation labels (eval/labels.json):**
```json
{
  "1": 5,
  "2": 12,
  "3": 18,
  "4": 5,
  "5": 12,
  ...
  "12": 25,
  "13": -1,
  "14": -1,
  "15": -1
}
```

Where `-1` means "no suitable image in corpus" (the no-match test cases).

**EVIDENCE.md fully filled:**
Every Section 6 checkbox + all six probes get one pasted proof. Example:

```markdown
# EVIDENCE

## AI Processing
- [x] Vision output validated against schema; invalid responses never trusted
  Proof: tests/unit/test_vision.py::test_invalid_schema_rejected
  ```
  Test output: PASSED
  ```

- [x] Low-confidence classifications flagged, not accepted
  Proof: test_probe_1 output
  ```
  Flagged (low confidence): 2 images at 0.45, 0.52 confidence
  ```

## Acceptance Probes
- [x] Probe 1: batch job tags all images; at least one flagged
  Proof: docker compose exec worker python -m src.worker output (above)

- [x] Probe 2: fox ranks first; wolf and dog rank lower
  Proof: curl output (above)

...
```

**Gate:** All Section 6 boxes checked in EVIDENCE.md with proofs. README contains top-1 precision number. All six acceptance probes passing. `docker compose up && seed && eval.py` completes successfully.

**BUILDLOG final entries:**
Document the guard testing (edge cases), how you tuned similarity threshold, where the eval script took iteration (handling no-match cases), and what you'd improve next (larger corpus, better post analysis, etc.).

**Time estimate:** 8–12 h
- Eval script: 1.5 h
- E2E tests (all six probes): 2 h
- Cost dashboard: 1 h
- README + docs: 1.5 h
- Error handling polish: 1 h
- Filling EVIDENCE.md: 1 h
- Final testing and debugging: 1–2 h

---

## Summary Table

| Phase | Objective | Gate | Time |
|-------|-----------|------|------|
| **1** ✅ | Design system, create repo | Design doc + architecture committed | 4–6 h |
| **2** | Vision pipeline, batch jobs, cost tracking | Batch job tags 44 images, at least 1 flagged | 10–14 h |
| **3** | Embeddings, matching, guard | Fox ranks first, wolf rejected, no-match works | 12–16 h |
| **4** | Eval, cost dashboard, README, proofs | Top-1 precision measured, all probes passing | 8–12 h |
| **Total** | | Full system ready for submission | 35–50 h |

---

## How to Work Through Each Phase

1. **Start the phase:** read the phase section thoroughly
2. **Understand the gate:** what proof proves you're done?
3. **Create the files:** start with models/schemas, then services, then routers
4. **Test as you go:** write unit tests for guard logic, integration tests for jobs
5. **Use BUILDLOG.md:** every commit, add a line about what AI helped with, what was wrong
6. **Commit frequently:** at least one commit per working session; each phase should be visible in git history
7. **Fill EVIDENCE.md as you go:** don't wait until the end
8. **Move to the next phase only when the gate is clearly met:** don't carry ambiguity forward

---

## Key Decisions to Make at Each Phase

**Phase 2:**
- Confirmation that Gemini free-tier quotas work (check your actual limits in aistudio.google.com/rate-limit)
- Corpus selection (44 animals + vehicles + food + architecture? or different mix?)
- Whether to use Docker Compose locally or install postgres separately (Docker recommended)

**Phase 3:**
- How to extract subject/taxon/category from posts (heuristic, regex, or a text call?)
- Initial SIM_MIN threshold guess (0.60 is a placeholder; adjust based on initial matching)
- Review UI scope (curl endpoints, or a simple HTML table?)

**Phase 4:**
- Eval set size (12 posts is minimum; can grow to 20–30 if time permits)
- Precision target (83% is strong; document if you hit something different)
- Which stretch goals, if any (alt text generation, duplicate detection, etc.)

---

## Stretch Goals (Only After Phase 4 Ships)

1. **Automatic alt text:** run captions through a summarizer, store alt_text
2. **Duplicate detection:** perceptual hashing or embedding distance between images
3. **Fallback image generation:** if no image clears the guard, call Gemini to generate one
4. **Human-in-the-loop QA:** mark uncertain matches as `needs_human_review`, serve them on a simple dashboard
5. **Test suite:** comprehensive pytest tests (unit, integration, e2e) with >80% code coverage

---

## Common Pitfalls (Avoid These)

- **Skipping tests until Phase 4:** guard logic is easy to test now; do it in Phase 3
- **Hardcoding model names:** they go in `.env` from day one
- **Ignoring rate limits:** check your real Gemini limits early; design retries around them
- **Assuming one image matches:** use content hashes to skip already-processed images
- **Forgetting idempotency:** retried jobs must not double-tag or double-embed
- **Not logging costs:** cost tracking is Requirement 7; set it up in Phase 2, not later
- **Brittle post analysis:** "red fox" vs "Vulpes vulpes" vs "wild fox" — use the taxon field, not string equality

---

## Questions to Ask Yourself at Each Gate

**Phase 2 gate:** Can I call `POST /jobs/process-images` and see 44 images tagged in the database, with at least one flagged for low confidence? Are costs logged?

**Phase 3 gate:** Does `GET /posts/1/images` return a fox image first, with a confidence score and guard checks? Does forcing wolf onto fox return a rejection with a readable reason? Does a post about castles return "no confident match"?

**Phase 4 gate:** Does `python scripts/eval.py` print a top-1 precision number that matches the README? Do all six acceptance probes pass? Is EVIDENCE.md fully filled with one proof per box?

If yes to all, you're done. Submit.