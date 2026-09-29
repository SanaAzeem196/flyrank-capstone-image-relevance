# Design Doc: AI Image Understanding & Content Matching Engine

Phase 1 · draft v0.1 · repo: `flyrank-capstone-imagerelevance`

## 1. Problem

Given an image library and a set of blog posts, tag every image with a vision model, then suggest the right image for each post by meaning (not filenames). The core feature is the **mismatch guard**: when the best candidate is still wrong (a wolf for a fox post), refuse it and explain why. When nothing clears the bar, answer "no confident match" with reasons.

## 2. Stack (locked)

Python 3.12 · FastAPI · Pydantic v2 · SQLAlchemy 2 + Alembic · Postgres + pgvector · `google-genai` · tenacity · pytest · Docker Compose (`api`, `worker`, `db`).
Vision: `gemini-3.5-flash-lite` (upgrade path `gemini-3.8-flash`). Embeddings: `gemini-embedding-2`, 768 dims. Model names live in `.env`. Ollama is a fallback behind the provider interface.

## 3. Image metadata schema (Pydantic, validated on every response)

```json
{
  "subject": "red fox",
  "taxon": "fox",
  "category": "animal",
  "attributes": ["orange fur", "wild", "forest"],
  "caption": "A red fox standing in a forest",
  "confidence": 0.94
}
```

`taxon` is our one addition to the brief's schema: the broad common-name group (lowercase, singular), so "Vulpes vulpes", "red fox" and "wild fox species" all normalize to `fox`. The guard compares `taxon`, never raw subject strings. Invalid JSON or schema failures are retried, then the image is marked `failed`. `confidence < CONF_MIN` marks the image `flagged` (never silently accepted).

## 4. Matching strategy and mismatch guard

1. **Post analysis:** one text call extracts `{subject, taxon, category, confidence}` from the post (same validation path).
2. **Embeddings:** image side embeds `subject + caption + attributes` as a document; post side embeds title + body as a query (asymmetric retrieval format for `gemini-embedding-2`). Cosine similarity via pgvector.
3. **Rank** the top-K images by similarity, then run **each** candidate through the guard in rank order.
4. **Guard** (pure functions, no I/O, evaluates all checks and reports all of them):

| Check | Rule | Fails with |
|---|---|---|
| confidence | image not flagged (`>= CONF_MIN`, placeholder 0.70) | "Image classification too uncertain" |
| category | post.category == image.category | "Category mismatch: expected X, detected Y" |
| taxon | post.taxon == image.taxon (skipped if the post has no identifiable subject) | "Animal category mismatch: expected fox, detected wolf" |
| similarity | cosine >= `SIM_MIN` (placeholder 0.60, **tuned on the eval set in Phase 4**) | "Similarity 0.41 below threshold 0.60" |

Verdict is ACCEPTED only if all checks pass. If no candidate is accepted, the response is `no_confident_match` plus the per-candidate reasons.

## 5. Data model (all tables carry `tenant_id`; every query is tenant-scoped)

| Table | Key columns | Indexes / constraints |
|---|---|---|
| `images` | id, filename, source_url, license, content_hash, status (`pending/processing/tagged/flagged/failed`) | UNIQUE(tenant_id, content_hash), (tenant_id, status) |
| `image_metadata` | image_id, subject, taxon, category, caption, confidence, needs_review, raw_response, model | PK image_id, (tenant_id, taxon) |
| `image_tags` | image_id, tag, kind (`subject/attribute`) | (tenant_id, tag) |
| `image_vectors` | image_id, embedding vector(768), model | HNSW (cosine) |
| `posts` | id, title, body, subject, taxon, category | (tenant_id, taxon) |
| `post_vectors` | post_id, embedding vector(768), model | PK post_id |
| `suggestions` | id, post_id, image_id, rank, similarity, verdict, guard_checks (JSONB), review_status | UNIQUE(post_id, image_id), (post_id, rank) |
| `reviews` | id, suggestion_id, decision, note, created_at | (suggestion_id) |
| `jobs` | id, type, payload, status, attempts, max_attempts, run_after, last_error, locked_at | (status, run_after) |
| `cost_log` | id, job_id, call_type (`vision/embedding/post_analysis`), model, input_tokens, output_tokens, est_cost_usd, latency_ms, status, ref_id | (created_at), (job_id) |

Schema ships as Alembic migrations. `est_cost_usd` is computed from a price table in config (paid-tier list prices, so the log is meaningful even at $0).

## 6. API surface

| Endpoint | Purpose |
|---|---|
| `POST /jobs/process-images` | Enqueue the batch job, return `202 {job_id}` |
| `GET /jobs/{id}` · `GET /jobs?status=failed` | Progress, attempts, errors |
| `GET /images` · `GET /images/{id}` | Metadata, flagged state |
| `POST /posts` · `GET /posts/{id}` | Create post (enqueues analysis and embedding) |
| `GET /posts/{id}/images` | Ranked suggestions with scores and guard verdicts, or `no_confident_match` + reasons |
| `POST /posts/{id}/check` `{image_id}` | Force any image through the guard (Probe 3) |
| `GET /suggestions/{id}` | Inspect why an image was selected or refused |
| `POST /suggestions/{id}/approve` · `/reject` | Review workflow (idempotent) |
| `GET /costs` | Cost log and totals |
| `GET /health` | Liveness |

Bad input returns 4xx via Pydantic validation, never a 500.

## 7. Layers

```
api/        FastAPI routers, Pydantic request/response models (HTTP only)
services/   ingestion, matching, review (orchestration and business logic)
guard/      pure functions, no I/O, unit-tested against fox/wolf cases
repos/      SQLAlchemy queries (data only)
providers/  VisionProvider / EmbeddingProvider -> Gemini (default) | Ollama
worker/     separate process: claims jobs (SELECT ... FOR UPDATE SKIP LOCKED), calls services
```

## 8. Reliability, cost, secrets

- **Retries:** tenacity with exponential backoff on 429/5xx, `max_attempts=3`. After that the job is `failed`, logged at ERROR, and visible in `GET /jobs?status=failed` (the failure alert).
- **Rate limit:** `GEMINI_RPM` in `.env`, set from the real per-project limits in AI Studio.
- **Budget guard:** `MAX_CALLS_PER_RUN` and `MAX_EST_USD`. The worker pauses the job when either is exceeded.
- **Idempotency:** UNIQUE content hash, skip already-tagged images on retry, upsert suggestions, repeated approve/reject returns the same result.
- **Secrets:** `GEMINI_API_KEY` only from env, redacted in logs, `.env` gitignored, `.env.example` committed.

## 9. Corpus and eval plan

- **Corpus (~44 images, 5 categories):** animals (red fox 4, gray wolf 4, dog 4, bear 3, deer 3), vehicles 6, food 6, architecture 6, plants/landscape 6, plus 2 deliberately ambiguous images (blurry, distant fox-or-dog) to trigger low-confidence flagging (Probe 1). Sourced from Unsplash/Pexels via `scripts/download_corpus.py` and `corpus/manifest.csv` (license, photographer). Images are gitignored, not committed.
- **Eval set:** 12 scene-specific posts, each with exactly one labeled correct image (`eval/labels.json`), plus 3 posts about subjects absent from the corpus (Probe 4). `scripts/eval.py` prints top-1 precision and the no-match refusal rate. The README number is copied from its output.

## 10. Explicit non-goal

We are **not** building an image search engine or asset-management platform: no frontend, no upload UI, no free-text search, no image generation.

## 11. Open decisions

1. Include thin tenant scoping (`tenant_id` + `X-Tenant-ID`)? Recommended: yes.
2. Keep the extra `taxon` field in the schema? Recommended: yes.
3. Corpus mix and one-correct-image-per-post labeling as above?
4. Confirm the non-goal wording.