# N-ATLaS Health Report Extraction

National AI Innovation Challenge 2026 — Problem Statement 3 (Sectoral Fine-Tuning), Health domain.

N-ATLaS is an initiative of the Federal Ministry of Communications, Innovation and Digital Economy, powered by Awarri Technologies.

## What this is

A LoRA-fine-tuned [NCAIR1/N-ATLaS](https://huggingface.co/NCAIR1/N-ATLaS) that turns free-text health facility reports (stock levels, services, facility status — written in English, including Nigerian-Pidgin-flavoured phrasing) into structured JSON, served through a FastAPI backend with a Streamlit demo UI.

```
Streamlit UI  →  FastAPI  →  N-ATLaS (base + LoRA adapter)
```

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

## Running it

Requires a GPU with enough VRAM for an 8B model (verified on an A100; see `docs/FEASIBILITY.md` for why this does not run on a typical laptop CPU).

```bash
pip install -r requirements.txt

export HF_TOKEN=...                 # accepted-license token for NCAIR1/N-ATLaS
python training/prepare_dataset.py  # generates data/{train,validation,benchmark}
python training/train.py --epochs 3 # LoRA fine-tune, ~2 min on an A100

export ADAPTER_PATH=training/output/lora-adapter/final
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 &
streamlit run streamlit_app/app.py --server.port 8501
```

Reproduce the benchmark comparison:
```bash
python evaluation/run_benchmark.py --model base --output evaluation/results/base_results.json
python evaluation/run_benchmark.py --model adapted --adapter-path training/output/lora-adapter/final --output evaluation/results/adapted_results.json
python evaluation/benchmark.py --base evaluation/results/base_results.json --adapted evaluation/results/adapted_results.json
```

## License

N-ATLaS is distributed under its own Open-Source Research and Innovation License (see the [model card](https://huggingface.co/NCAIR1/N-ATLaS)); this project's own code has no separate license declared.
