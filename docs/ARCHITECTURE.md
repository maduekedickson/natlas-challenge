# Architecture

## System overview

```
                User (browser)
                     │
                     ▼
        ┌─────────────────────────┐
        │   Streamlit UI           │  streamlit_app/app.py
        │   (port 8501)             │  - text input, example picker
        └────────────┬─────────────┘  - renders structured result
                     │ HTTP (localhost)
                     ▼
        ┌─────────────────────────┐
        │   FastAPI service         │  api/main.py
        │   (port 8000)             │  - /api/v1/health
        └────────────┬─────────────┘  - /api/v1/analyze
                     │
                     ▼
        ┌─────────────────────────┐
        │   NAtlasService            │  api/services/natlas_service.py
        │   (singleton, loaded once  │  - tokenizer + model + optional
        │    at API startup)         │    LoRA adapter, loaded once
        └────────────┬─────────────┘
                     │
                     ▼
        ┌─────────────────────────┐
        │   N-ATLaS (base or         │  NCAIR1/N-ATLaS (Llama-3 8B)
        │   LoRA-adapted)             │  + training/output/lora-adapter
        └─────────────────────────┘
```

Streamlit never calls the model directly — it only talks to FastAPI, per the Streamlit → FastAPI → N-ATLaS separation this project is built around. This keeps the model/API reusable by other clients (CLI, another frontend, another developer integrating per PS3's requirement), not locked behind the demo UI.

## Request flow (one `/analyze` call)

1. User enters free-text report in Streamlit, clicks "Analyze."
2. Streamlit POSTs `{"report_text": "..."}` to `FastAPI /api/v1/analyze`.
3. `NAtlasService.analyze()` renders the instruction + report into N-ATLaS's native Llama-3 chat template (`tokenizer.apply_chat_template`), not a raw string — this materially affects output quality (see `docs/FEASIBILITY.md` §4's documented limitation when this was skipped).
4. Model generates via greedy decoding (`do_sample=False`, deterministic — important for reproducibility).
5. Output is parsed by a tolerant brace-extraction JSON parser (handles a model wrapping output in markdown fences, etc., without crashing — see the base-model example in `docs/EVALUATION.md`).
6. Structured fields (or `parsed_ok: false` plus the raw text, never a silent failure) are returned to Streamlit and rendered as a table.

## Components and where they live

| Component | Path | Responsibility |
|---|---|---|
| Demo UI | `streamlit_app/app.py` | Input/output rendering only, no model logic |
| API service | `api/main.py`, `api/routes/`, `api/schemas/` | HTTP contract, validation, error handling |
| Model wrapper | `api/services/natlas_service.py` | Singleton load-once model access |
| Dataset generation | `training/prepare_dataset.py` | Synthetic data (documented in `docs/DATASET.md`) |
| Fine-tuning | `training/train.py` | LoRA via `transformers.Trainer` + `peft`, manual completion-only loss masking |
| Evaluation | `evaluation/metrics.py`, `run_benchmark.py`, `benchmark.py` | Field-level P/R/F1 + exact match, base-vs-adapted comparison |
| Feasibility tooling | `scripts/hardware_check.py`, `scripts/test_natlas.py` | Phase 0 — hardware detection, genuine-inference smoke test |

## Why this stack

- **PyTorch + Hugging Face Transformers, no framework conversion** — `AutoModelForCausalLM`/`AutoTokenizer` directly against N-ATLaS's native checkpoint, per `CLAUDE.md` §17.
- **LoRA via `peft`, not full fine-tuning** — 41.9M trainable params (0.52% of 8.07B), fast enough to iterate inside a single Colab session; trained in ~2 minutes (`docs/EVALUATION.md`).
- **Plain `transformers.Trainer` instead of TRL's `SFTTrainer`** — a deliberate deviation from the originally planned stack. TRL's current completion-masking API (`assistant_only_loss`) requires conversational-format datasets and chat-template generation markers that aren't guaranteed to work cleanly with N-ATLaS's custom template; a manually-masked `Trainer` loop was verified correct against the real tokenizer (prompt is a token-level prefix of the full sequence) and avoided that risk under a tight deadline.
- **FastAPI as the service boundary** — the one thing another developer needs to integrate against for PS3; documented independently in `docs/API.md`.
- **Streamlit as the demo layer** — fastest path to a usable UI without frontend engineering, per `CLAUDE.md` §22/§28.

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
