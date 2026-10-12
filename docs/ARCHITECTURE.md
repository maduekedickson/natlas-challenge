# Architecture

## System overview

```
                         User (browser)
                               │
                               ▼
                      Streamlit UI (multipage)
             Dashboard / New Report / Voice / History /
                  Trends / Batch / Exports
                               │ HTTP (localhost)
                               ▼
                          FastAPI (api/main.py)
             ┌─────────────────┼──────────────────┐
             │                 │                  │
             ▼                 ▼                  ▼
      NAtlasService      report_service      analytics_service
     (singleton model    batch_service        export_service
      + LoRA adapter,    transcription_service
      unchanged)              │                  │
             │                 ▼                  ▼
             ▼            SQLite (reports,    SQLite (same DB,
      N-ATLaS (base or    stock_observations)  read-only queries)
      LoRA-adapted)
```

Streamlit never calls the model or database directly — it only talks to FastAPI, per the Streamlit → FastAPI → N-ATLaS separation this project is built around. Every new feature (dashboard, history, trends, batch, exports, voice) is a new FastAPI route + service module; `NAtlasService` itself — model loading, prompting, parsing — is completely unchanged from the original extraction demo, and is the single inference path reused by `/analyze`, batch processing, and (after transcription) voice reports alike.

## Request flow (one `/analyze` call)

1. User enters free-text report in Streamlit, clicks "Analyze."
2. Streamlit POSTs `{"report_text": "..."}` to `FastAPI /api/v1/analyze`.
3. `NAtlasService.analyze()` renders the instruction + report into N-ATLaS's native Llama-3 chat template (`tokenizer.apply_chat_template`), not a raw string — this materially affects output quality (see `docs/FEASIBILITY.md` §4's documented limitation when this was skipped).
4. Model generates via greedy decoding (`do_sample=False`, deterministic — important for reproducibility).
5. Output is parsed by a tolerant brace-extraction JSON parser (handles a model wrapping output in markdown fences, etc., without crashing — see the base-model example in `docs/EVALUATION.md`).
6. Structured fields (or `parsed_ok: false` plus the raw text, never a silent failure) are persisted via `report_service.create_report()` (one `reports` row + one `stock_observations` row per stock item) and returned to Streamlit, now including a `report_id`.

Batch processing and voice reports funnel into this same flow after their own pre-step (CSV row iteration; transcription + user review respectively) — there is no second extraction implementation anywhere in the codebase.

## Components and where they live

| Component | Path | Responsibility |
|---|---|---|
| Demo UI | `streamlit_app/app.py` + `pages/` | Input/output rendering only, no model or DB logic directly |
| API client | `streamlit_app/api_client.py` | Single place for every `requests` call Streamlit makes |
| API service | `api/main.py`, `api/routes/`, `api/schemas/` | HTTP contract, validation, error handling |
| Model wrapper | `api/services/natlas_service.py` | Singleton load-once model access — **unchanged** by this expansion |
| Config | `api/config.py` | Centralizes env vars for the new features (DB path, adapter version, Whisper model) |
| DB layer | `api/db/models.py`, `api/db/session.py` | SQLAlchemy models + session/engine setup |
| Report persistence | `api/services/report_service.py` | CRUD, dashboard stats |
| Analytics | `api/services/analytics_service.py` | Trend/aggregation queries |
| Batch processing | `api/services/batch_service.py` | CSV → `NAtlasService.analyze()` per row → DB, with per-row failure handling |
| Exports | `api/services/export_service.py` | Report-level and stock-item-level CSV/XLSX generation |
| Voice | `api/services/transcription_service.py`, `api/routes/voice.py` | Whisper transcription (see `docs/VOICE.md`) |
| Dataset generation | `training/prepare_dataset.py` | Synthetic data (documented in `docs/DATASET.md`) |
| Fine-tuning | `training/train.py` | LoRA via `transformers.Trainer` + `peft`, manual completion-only loss masking |
| Evaluation | `evaluation/metrics.py`, `run_benchmark.py`, `benchmark.py` | Field-level P/R/F1 + exact match, base-vs-adapted comparison — **untouched** |
| Feasibility tooling | `scripts/hardware_check.py`, `scripts/test_natlas.py` | Phase 0 — hardware detection, genuine-inference smoke test |
| Tests | `tests/test_platform.py` | DB/analytics/batch/export logic, model mocked (real inference verified separately via Colab) |

## Why this stack

- **PyTorch + Hugging Face Transformers, no framework conversion** — `AutoModelForCausalLM`/`AutoTokenizer` directly against N-ATLaS's native checkpoint, per `CLAUDE.md` §17.
- **LoRA via `peft`, not full fine-tuning** — 41.9M trainable params (0.52% of 8.07B), fast enough to iterate inside a single Colab session; trained in ~2 minutes (`docs/EVALUATION.md`).
- **Plain `transformers.Trainer` instead of TRL's `SFTTrainer`** — a deliberate deviation from the originally planned stack. TRL's current completion-masking API (`assistant_only_loss`) requires conversational-format datasets and chat-template generation markers that aren't guaranteed to work cleanly with N-ATLaS's custom template; a manually-masked `Trainer` loop was verified correct against the real tokenizer (prompt is a token-level prefix of the full sequence) and avoided that risk under a tight deadline.
- **FastAPI as the service boundary** — the one thing another developer needs to integrate against for PS3; documented independently in `docs/API.md`.
- **Streamlit as the demo layer, converted to native multipage** — Streamlit's own `pages/` directory mechanism (not a hand-rolled nav system) auto-generates the sidebar; each feature is its own page file.
- **SQLite via SQLAlchemy for persistence** — per `CLAUDE.md`'s own stated default, zero infrastructure, works identically on a laptop, in Colab, or later on Hugging Face Spaces. See `docs/DATABASE.md` for schema and its Colab-ephemeral-storage limitation.
- **Stock items normalized into their own table**, not left inside the JSON blob — so trend queries are plain SQL, not per-request JSON parsing.
- **`openai-whisper` for voice, explicitly not claimed as N-ATLaS ASR** — the NCAIR1/N-ATLaS repo has no bundled ASR component (confirmed by inspection); see `docs/VOICE.md`.

## Deployment topology (as actually run)

Both services currently run as background processes inside a single interactive **Google Colab A100** session, each exposed to a browser via Colab's `kernel.proxyPort`. This is a working prototype deployment, proving the full pipeline genuinely functions — it is **not** a persistent production deployment:

- The session is ephemeral; it ends when the Colab runtime disconnects.
- No authentication on either service.
- Single in-process model instance — concurrent requests serialize behind the GPU (no request queue).

A real deployment would move this to a persistent GPU-backed host (dedicated cloud GPU VM, or AWS per `CLAUDE.md` §27) behind a process manager, with auth added to the FastAPI layer — out of scope for this submission's timeline but recorded here as explicit future work, not hidden.

## Related documentation

- `docs/FEASIBILITY.md` — hardware investigation, local vs. cloud GPU findings
- `docs/PROJECT-DEFINITION.md` — why PS3 was selected under the deadline
- `docs/DATASET.md` — synthetic dataset provenance and schema
- `docs/EVALUATION.md` — baseline vs. adapted benchmark results
- `docs/API.md` — full API contract and example
- `docs/DATABASE.md` — schema and persistence limitations
- `docs/FEATURES.md` — what was added in the platform expansion, and what was deliberately descoped
- `docs/VOICE.md` — speech-to-text implementation and boundaries
- `docs/DEPLOYMENT.md` — how to run the full system
- `docs/IMPLEMENTATION_PLAN.md` — the plan this expansion was built from
