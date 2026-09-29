# flyrank-capstone-imagerelevance

AI Image Understanding & Content Matching Engine. Tags image library with vision model, matches images to blog posts by meaning, uses mismatch guard to refuse wrong pairings with human-readable reason.

Full docs: [docs/DESIGN.md](docs/DESIGN.md), [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md), [docs/PHASE_PLAN.md](docs/PHASE_PLAN.md).

## Stack

Python 3.12 · FastAPI · Pydantic v2 · SQLAlchemy 2 + Alembic · PostgreSQL 16 + pgvector · google-genai SDK (Interactions API) · tenacity · Docker Compose (`api`, `worker`, `db`).

Vision: `gemini-3.5-flash-lite`. Embeddings: `gemini-embedding-2`, 768 dims. Model names in `.env`, never hardcoded.

## Layers (strict, one direction)

```
api/        FastAPI routers, Pydantic schemas — HTTP only, no business logic
services/   orchestration: coordinate providers + repos
guard/      pure functions, no I/O, fully unit-tested
db/         SQLAlchemy models, repos, session — queries only
providers/  VisionProvider / EmbeddingProvider -> Gemini (default) | Ollama fallback
worker/     separate process, claims jobs (SELECT ... FOR UPDATE SKIP LOCKED)
```

`guard/` never touches the DB or network — pass it validated data, get back `(verdict, checks)`.

## The guard (core feature)

`src/guard/check.py::run_guard()` runs 4 checks in order, collects ALL results (not short-circuit): `confidence`, `category`, `taxon`, `similarity`. Verdict is `ACCEPT` only if every check passes; otherwise `REJECT` with a human-readable reason per failed check. `taxon` check is skipped only if the post has no identifiable subject.

`taxon` is the normalized field the guard actually compares (lowercase singular common name, e.g. "fox") — never compare raw `subject` strings, they don't normalize ("Vulpes vulpes" vs "red fox" vs "wild fox species").

## Key thresholds (`.env`, also `src/config.py::GuardConfig`)

- `CONF_MIN=0.70` — image confidence below this → `status=flagged`, never silently accepted
- `SIM_MIN=0.60` — cosine similarity floor, placeholder tuned against the eval set in Phase 4
- `MAX_CALLS_PER_RUN` / `MAX_EST_USD` — worker pauses a job if exceeded (budget guard)

## Conventions

- All tables carry `tenant_id`; every query is tenant-scoped.
- Idempotency: images deduped by `content_hash` (UNIQUE per tenant); jobs retry without double-processing; approve/reject endpoints return the same result on repeat calls.
- Bad request input → Pydantic validation → `422`, never a `500`.
- Provider calls wrapped in `tenacity` retry (3 attempts, exponential backoff) on `429`/`5xx`; schema `ValidationError` is not retried, job marked `failed` directly.
- Every vision/embedding call logged to `cost_log` (tokens, estimated USD, model, latency).
- Migrations via Alembic only — `alembic revision --autogenerate` then `alembic upgrade head`, never hand-edit schema.

## Running locally

```bash
cp .env.example .env   # fill GEMINI_API_KEY
docker compose up --build
docker compose exec api python scripts/seed_images.py
docker compose exec api python scripts/eval.py
```

## Testing

- `tests/unit/` — guard logic (fox/wolf cases), schema validation, cost math. No I/O.
- `tests/integration/` — mocked provider calls, job claim/process/complete, pgvector similarity.
- `tests/e2e/` — Docker Compose, the six acceptance probes (see `docs/PHASE_PLAN.md` Phase 4).

Guard changes always need a unit test asserting the specific failing check name and reason string, not just verdict.

## Explicit non-goals

Not an image search engine or asset-management platform: no frontend, no upload UI, no free-text search, no image generation.
