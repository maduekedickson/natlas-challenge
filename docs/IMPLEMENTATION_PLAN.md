# Implementation Plan — Health Intelligence Platform Expansion

**Status:** Phase 1 (inspection + plan) complete. Stopping here per instructions — implementation starts only after this plan is reviewed.

---

## 1. Current architecture (as it actually exists today)

```
Streamlit UI (streamlit_app/app.py, single file)
        │  requests.post(API_BASE_URL + "/api/v1/analyze")
        ▼
FastAPI (api/main.py)
        │  app.include_router(...)
        ▼
NAtlasService (api/services/natlas_service.py) — singleton, loaded once at
FastAPI startup via a `lifespan` context manager
        │
        ▼
AutoModelForCausalLM.from_pretrained(NCAIR1/N-ATLaS, bf16, device_map=auto)
  + optional PeftModel.from_pretrained(model, ADAPTER_PATH) if ADAPTER_PATH is set
```

### Existing files and responsibilities

| File | Responsibility |
|---|---|
| `api/main.py` | FastAPI app, lifespan hook that force-loads `NAtlasService()` at startup, mounts routers under `/api/v1` |
| `api/routes/health.py` | `GET /api/v1/health` — returns `NAtlasService.health()` (status, model name, device, adapter_loaded) |
| `api/routes/inference.py` | `POST /api/v1/analyze` — calls `NAtlasService.analyze()`, wraps exceptions as HTTP 500 |
| `api/schemas/requests.py` | `AnalyzeRequest` (just `report_text: str`) |
| `api/schemas/responses.py` | `AnalyzeResponse`, `HealthResponse`, `StockItem` |
| `api/services/natlas_service.py` | Singleton model wrapper. Builds a chat-template prompt with a fixed `EXTRACTION_INSTRUCTION`, runs greedy generation, parses JSON out of the raw text with a tolerant brace-extraction parser (`_safe_parse_json`), returns a dict. No persistence — every call is stateless. |
| `streamlit_app/app.py` | Single-page app. Sidebar health check, example-report dropdown, text area, "Analyze" button, renders result as metrics + table. Calls FastAPI directly via `requests`, no model logic. |
| `training/prepare_dataset.py` | Generates the synthetic dataset (seed=42) → `data/{raw,train,validation,benchmark}` |
| `training/train.py` | LoRA fine-tune via plain `transformers.Trainer` + `peft` (NOT `trl.SFTTrainer` — deliberately avoided, see `docs/ARCHITECTURE.md`). Manual prompt-masking verified correct against the real tokenizer. Saves adapter to `training/output/lora-adapter/final` (gitignored — a build artifact, not committed) |
| `evaluation/metrics.py` | Field-level precision/recall/F1 + exact-match scorer, pure Python, no model dependency |
| `evaluation/run_benchmark.py` | Loads base or adapted model, runs it over a benchmark JSONL file, scores it, writes a results JSON including all 520 raw per-example outputs |
| `evaluation/benchmark.py` | Reads two results JSONs, produces `comparison.csv` + `comparison.png` |
| `evaluation/results/{base,adapted}_results.json` | **Committed, frozen evidence** — 520-example held-out benchmark, base F1 0.677 / exact-match 1.35%, adapted F1 0.9998 / exact-match 99.81% |
| `scripts/hardware_check.py`, `scripts/test_natlas.py` | Phase 0 feasibility scripts (local CPU hardware detection; a guarded, real-inference smoke test that refuses to run if available RAM is insufficient) |

### Configuration / environment variables (actual, from `.env.example`)

```
HF_TOKEN=            # required, gated model
MODEL_NAME=NCAIR1/N-ATLaS
MODEL_REVISION=
DEVICE=auto
API_HOST=0.0.0.0
API_PORT=8000
```
Plus `ADAPTER_PATH` (read directly via `os.environ.get`, not yet in `.env.example` — oversight to fix) and `API_BASE_URL` (Streamlit side, defaults to `http://localhost:8000`).

### Database: **none exists.** No SQLAlchemy, no `.db` file, no ORM models anywhere in the repo. This is a genuinely new addition, not something to "reuse."

### Dependencies (`requirements.txt`): torch, transformers, accelerate, safetensors, sentencepiece, huggingface_hub, python-dotenv, psutil, peft, trl, datasets, fastapi, uvicorn, pydantic, pandas, matplotlib.

**Found one stale entry:** `trl` is listed but no longer used anywhere (train.py was rewritten to use plain `transformers.Trainer` after TRL's `SFTTrainer`/`DataCollatorForCompletionOnlyLM` API changed incompatibly mid-project). Will remove it as part of this work rather than leave a misleading dependency.

### Tests: **none exist.** `scripts/test_natlas.py` is a feasibility smoke-test script, not a test suite. `requirements-dev.txt` already lists `pytest` and `httpx` but nothing uses them yet.

### Benchmark integrity note: `data/benchmark/benchmark.jsonl` (520 examples) and `evaluation/results/*.json` are the frozen evaluation artifacts referenced throughout `docs/EVALUATION.md` and the README's headline numbers. Nothing in this plan touches them — new demo/report data is a completely separate code path (user-submitted reports → new `reports` DB table), never mixed with the benchmark files.

---

## 2. Proposed additions

### 2.1 Database — SQLite + SQLAlchemy

SQLite, per `CLAUDE.md`'s own stated default ("SQLite initially / PostgreSQL if required") and because it needs zero infrastructure — works identically on a laptop, in Colab, or later on Hugging Face Spaces.

**Schema** (normalizing stock items into their own table, per the spec's own suggestion, so trend queries like "ORS quantity at Asaba PHC over time" are plain SQL, not JSON-parsing in Python on every request):

```sql
reports
  id              INTEGER PRIMARY KEY
  facility_name   TEXT
  reporter_name   TEXT
  report_text     TEXT NOT NULL        -- original free text, always kept
  structured_data TEXT NOT NULL        -- full raw JSON string, for audit/display
  services_available TEXT              -- JSON-encoded list (not normalized — no
                                        -- trend feature asks for per-service history;
                                        -- normalizing it would be unused complexity)
  facility_status TEXT                 -- 'open' | 'closed' | null
  model_name      TEXT NOT NULL        -- e.g. "NCAIR1/N-ATLaS"
  adapter_version TEXT                 -- e.g. "health-lora-v1", null if base model
  source          TEXT NOT NULL        -- 'text' | 'voice' | 'batch'
  status          TEXT NOT NULL        -- 'success' | 'parse_error' | 'failed'
  created_at      DATETIME NOT NULL
  updated_at      DATETIME NOT NULL

stock_observations
  id          INTEGER PRIMARY KEY
  report_id   INTEGER NOT NULL REFERENCES reports(id)
  item        TEXT NOT NULL
  quantity    INTEGER
  unit        TEXT
  status      TEXT                     -- 'in_stock' | 'low_stock' | 'out_of_stock'
  created_at  DATETIME NOT NULL
```

No `report_date` field distinct from `created_at`: the synthetic reports don't reliably contain a parseable explicit date in the free text (confirmed by re-checking `training/prepare_dataset.py`'s templates), so `created_at` (server-assigned, real) is the actual timestamp used for trends — this is stated explicitly rather than inventing a "report date" field that would mostly be null or fabricated.

**Known limitation to flag now, not discover later:** if the backend keeps running inside an ephemeral Colab session (as it does today), the SQLite file lives in `/content` and is **wiped on every Colab disconnect** — exactly the same problem that bit the LoRA adapter twice already this session. For the demo this is acceptable (a fresh demo doesn't need historical data), but "trend analysis" as a feature is only meaningful if the database persists across sessions. Proposed mitigation for now: write the SQLite file to a path that can be pointed at a mounted Google Drive folder in Colab (`DATABASE_PATH` env var, defaults to local `./data/app.db`), documented as a known limitation rather than solved outright — a real fix (persistent Postgres, e.g. on Hugging Face Spaces later) is out of scope for this phase.

### 2.2 FastAPI additions

```
api/
├── main.py                 # extend: init DB on startup, mount new routers
├── config.py                # NEW — centralizes all env var reads (currently scattered
│                             #   across natlas_service.py / run_benchmark.py / train.py)
├── auth.py                  # NEW — simple Bearer API-key dependency, env-var based
├── db/
│   ├── session.py           # NEW — SQLAlchemy engine/session
│   └── models.py            # NEW — Report, StockObservation ORM models
├── routes/
│   ├── health.py             # unchanged
│   ├── inference.py          # extend: persist each successful /analyze call to DB
│   ├── reports.py            # NEW — list/get reports, dashboard stats
│   ├── batch.py              # NEW — CSV upload + batch processing
│   ├── analytics.py          # NEW — trend queries
│   └── exports.py            # NEW — CSV/Excel export (report-level + stock-item-level)
└── services/
    ├── natlas_service.py      # UNCHANGED — reused as-is by batch processing too
    ├── report_service.py      # NEW — DB CRUD
    ├── batch_service.py       # NEW — orchestrates CSV row → natlas_service.analyze() → DB
    ├── analytics_service.py   # NEW — trend/aggregation queries
    ├── export_service.py      # NEW — pandas-based CSV/XLSX generation
    └── transcription_service.py # NEW — speech-to-text (see 2.4)
```

**Auth approach (deliberately simple, per the spec's own "do not over-engineer" instruction):** a single `API_KEY` env var (optionally comma-separated for multiple keys), checked via a FastAPI `Depends()` that reads the `Authorization: Bearer <key>` header. No user table, no OAuth, no key-management UI. Applied to `/analyze`, `/batch`, and report-mutating endpoints; `/health` stays open.

### 2.3 Streamlit — convert to native multipage app

Streamlit's built-in `pages/` directory mechanism (not a hand-rolled nav system) auto-generates the sidebar:

```
streamlit_app/
├── app.py              # becomes a thin landing page (or redirects to Dashboard)
├── pages/
│   ├── 1_Dashboard.py
│   ├── 2_New_Report.py   # today's app.py content moves here, unchanged behavior
│   ├── 3_Voice_Report.py
│   ├── 4_History.py
│   ├── 5_Trends.py
│   ├── 6_Batch_Processing.py
│   └── 7_Exports.py
├── components/
│   ├── charts.py
│   ├── tables.py
│   └── metrics.py
└── api_client.py        # NEW — single place for all `requests.*` calls + the API key
                          #   header, instead of duplicating raw requests code per page
```

The existing `2_New_Report.py` is a near-verbatim move of the current `app.py` — same behavior, same example dropdown, same result rendering. **Nothing about today's working analysis flow changes.**

### 2.4 Voice input

No existing ASR/voice code or dependency anywhere in the repo, and the NCAIR1/N-ATLaS Hugging Face repo (inspected earlier this session) contains only the LLM — no bundled ASR component to "reuse." Per the spec's own instruction not to claim N-ATLaS ASR unless it's genuinely used, the plan is: **`openai-whisper` (the `base` or `small` model)** for transcription — well-supported, reproducible, modest dependency size, runs acceptably on either CPU or the same GPU session as the LLM. The UI will label this plainly as "speech-to-text," never as "N-ATLaS voice." Flow: record/upload audio → transcribe → **editable text box shown to the user** → only on explicit submission does it go through the same `/api/v1/analyze` path as typed text. No auto-submission of unreviewed transcripts.

### 2.5 Batch processing & exports

Batch reuses `NAtlasService.analyze()` per row — no second inference path. Per-row failures are caught and recorded individually (row number + error), never aborting the whole batch. Export is two-tier exactly as the spec suggests: a report-level CSV/XLSX (one row per report, `structured_data` kept as a JSON string column so nothing is lossily flattened) and a stock-item-level CSV/XLSX (one row per report×item, suited to spreadsheet pivoting). Needs `openpyxl` added to `requirements.txt` for `.xlsx` writing.

---

## 3. Risks

1. **Timing conflict with the demo video.** You said you're recording the demo now. Adding API-key auth changes how `/analyze` is called (needs an `Authorization` header) — if recording is already in progress using today's unauthenticated flow, we should sequence auth last (Phase 7, as the spec itself orders it) so it doesn't invalidate footage you've already captured.
2. **Ephemeral Colab storage** (see 2.1) — history/trends will reset on every Colab disconnect unless the DB file is pointed somewhere persistent (Drive mount). Documented as a known limitation for now rather than solved.
3. **New dependencies**: `sqlalchemy`, `openpyxl`, `python-multipart` (file uploads), `openai-whisper` (+ `ffmpeg` system dependency for audio decoding — needs verifying it's available in the Colab image). None of these are heavy/exotic, but they do need installing and verifying in a fresh Colab session, same as every dependency so far.
4. **Local testing stays partial.** Exactly as with the existing model code, anything that needs real N-ATLaS inference (batch processing end-to-end, the full `/analyze`-with-persistence flow) can only be fully verified on a GPU (Colab) — local testing will cover DB CRUD, auth, CSV validation, and export logic with `NAtlasService` mocked, consistent with how this project has handled the hardware constraint from the start.

---

## 4. Implementation order (as specified)

1. ~~Inspection + this plan~~ ✅
2. Database/report persistence (SQLAlchemy models, wire into `/analyze`)
3. Dashboard + report history (API endpoints + Streamlit pages)
4. Trend analysis/comparison
5. CSV/Excel export
6. Batch processing
7. API-key authentication (sequenced after the above so it doesn't block iterating on them, and after your demo recording per risk #1)
8. Voice input
9. Testing (pytest + httpx, DB/auth/CSV unit tests locally; model-dependent tests documented as GPU-only)
10. Documentation (`DATABASE.md`, `FEATURES.md`, `VOICE.md`, `DEPLOYMENT.md` + updates to `ARCHITECTURE.md`, `API.md`, `README.md`) and cleanup (remove stale `trl` dependency, add `ADAPTER_PATH`/`API_KEY`/`DATABASE_PATH` to `.env.example`)

Will work in small, logical commits per phase as the spec requests, on a `feature/health-intelligence-platform` branch rather than directly on `main`.

---

**Stopping here per Phase 1 instructions.** Confirm this plan (or redirect anything in it) before Phase 2 implementation starts.
