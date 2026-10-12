# N-ATLaS Health Intelligence Platform

National AI Innovation Challenge 2026 — Problem Statement 3 (Sectoral Fine-Tuning), Health domain.

N-ATLaS is an initiative of the Federal Ministry of Communications, Innovation and Digital Economy, powered by Awarri Technologies.

**This is an information-extraction and operational reporting tool. It is not a diagnostic system and does not provide medical advice, treatment recommendations, or clinical decision-making.**

## What this is

A LoRA-fine-tuned [NCAIR1/N-ATLaS](https://huggingface.co/NCAIR1/N-ATLaS) that turns free-text health facility reports (stock levels, services, facility status — written in English, including Nigerian-Pidgin-flavoured phrasing) into structured data, with a full operational platform built around it: persistent report history, a dashboard, trend analysis, CSV/Excel export, batch CSV processing, and voice input — all served through a FastAPI backend with a multipage Streamlit UI.

```
Streamlit UI (multipage)  →  FastAPI  →  N-ATLaS (base + LoRA adapter)  →  SQLite
```

Features: Dashboard · New Report · Voice Report · History · Trends & Comparison · Batch Processing · Exports. See [`docs/FEATURES.md`](docs/FEATURES.md) for what each does and what was deliberately left out of this pass (API-key auth).

## Results

Base vs. LoRA-adapted N-ATLaS on a 520-example held-out benchmark (full details: [`docs/EVALUATION.md`](docs/EVALUATION.md)):

| Metric | Base | Adapted |
|---|---|---|
| Exact match | 1.35% | 99.81% |
| Field-level F1 | 0.677 | 0.9998 |

Verified on a Google Colab A100 GPU — see [`docs/FEASIBILITY.md`](docs/FEASIBILITY.md) for hardware/latency measurements.

## Project docs

- [`docs/PROJECT-DEFINITION.md`](docs/PROJECT-DEFINITION.md) — scope and why this problem statement
- [`docs/FEASIBILITY.md`](docs/FEASIBILITY.md) — hardware investigation, local vs. cloud GPU
- [`docs/DATASET.md`](docs/DATASET.md) — synthetic dataset provenance and schema
- [`docs/EVALUATION.md`](docs/EVALUATION.md) — baseline vs. adapted benchmark results
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — system design
- [`docs/API.md`](docs/API.md) — API contract and example
- [`docs/DATABASE.md`](docs/DATABASE.md) — schema and persistence notes
- [`docs/FEATURES.md`](docs/FEATURES.md) — the platform features and their scope
- [`docs/VOICE.md`](docs/VOICE.md) — speech-to-text implementation and boundaries
- [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) — how to run the full system

## Running it

Requires a GPU with enough VRAM for an 8B model (verified on an A100; see `docs/FEASIBILITY.md` for why this does not run on a typical laptop CPU).

```bash
pip install -r requirements.txt

export HF_TOKEN=...                 # accepted-license token for NCAIR1/N-ATLaS
python training/prepare_dataset.py  # generates data/{train,validation,benchmark}
python training/train.py --epochs 3 # LoRA fine-tune, ~2 min on an A100

export ADAPTER_PATH=training/output/lora-adapter/final
export ADAPTER_VERSION=health-lora-v1
export DATABASE_PATH=data/app.db

python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 &
streamlit run streamlit_app/app.py --server.port 8501
```

Run the test suite (DB/analytics/batch/export logic — doesn't need a GPU):
```bash
pytest tests/test_platform.py -v
```

Reproduce the benchmark comparison:
```bash
python evaluation/run_benchmark.py --model base --output evaluation/results/base_results.json
python evaluation/run_benchmark.py --model adapted --adapter-path training/output/lora-adapter/final --output evaluation/results/adapted_results.json
python evaluation/benchmark.py --base evaluation/results/base_results.json --adapted evaluation/results/adapted_results.json
```

## License

N-ATLaS is distributed under its own Open-Source Research and Innovation License (see the [model card](https://huggingface.co/NCAIR1/N-ATLaS)); this project's own code has no separate license declared.
