# Deployment

Kept deliberately simple — no Kubernetes, microservices, Redis, Celery, or Kafka. Per `CLAUDE.md`: "the competition version should remain simple and reproducible."

```
Streamlit  →  FastAPI  →  N-ATLaS (+ LoRA)  →  SQLite
```

## Running everything (any GPU-backed environment)

```bash
pip install -r requirements.txt

export HF_TOKEN=...
export ADAPTER_PATH=training/output/lora-adapter/final   # omit to serve base model
export ADAPTER_VERSION=health-lora-v1
export DATABASE_PATH=data/app.db

python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 &
streamlit run streamlit_app/app.py --server.port 8501
```

## Running it in Google Colab (this project's actual demo environment)

A GPU is required — see `docs/FEASIBILITY.md` for why this doesn't run on a typical laptop CPU. The pattern used throughout this project: `!git clone` the repo, install deps, train/load the adapter, then launch both services as background subprocesses with `proxyPort` for browser access. Full copy-pasteable cells are kept in the project's own Colab notebook workflow (see conversation history / `notebooks/`), not duplicated here to avoid drift between two copies of the same instructions.

Key points specific to this expanded platform:
- `ffmpeg` should already be present on Colab's standard image (required by Whisper) — if not: `!apt-get install -y ffmpeg`
- `DATABASE_PATH` defaults to a local file inside the ephemeral Colab filesystem — see `docs/DATABASE.md`'s limitation note on this resetting every session, and the Google Drive mount workaround if persistence across sessions matters for a given demo.

## Known limitations of this deployment pattern

- **Ephemeral**: an interactive Colab session isn't a persistent service. Fine for recording a demo video or live-driving a demonstration; not a standing public URL.
- **No authentication**: deliberately descoped for this pass (see `docs/IMPLEMENTATION_PLAN.md`).
- **Single process, no request queue**: concurrent requests serialize behind the one GPU. Also relevant given `outline.md` §15's 1,000-active-user license ceiling on N-ATLaS itself — this prototype isn't trying to scale past that anyway.

## A persistent deployment, if needed later

Options considered (not implemented): a dedicated cloud GPU VM (AWS/GCP) running both services continuously, or Hugging Face Spaces with a GPU tier (purpose-built for exactly this: model + API + UI together, discussed but intentionally deferred — see conversation record). Either would also want the SQLite database replaced with a managed Postgres instance, since a persistent service without persistent storage would still lose history on every restart/redeploy.
