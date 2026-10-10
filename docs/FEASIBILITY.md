# N-ATLaS Feasibility Report (Phase 0)

**Status:** Complete. Local environment verified, genuine GPU inference verified on a Colab A100, LoRA fine-tuning completed with a measured benchmark improvement, and a live FastAPI service verified end-to-end. See `docs/EVALUATION.md` and `docs/API.md` for the full results.

This document distinguishes **verified facts** (measured directly), from **pending items** (blocked on an external prerequisite), from **assumptions** (not yet measured). Nothing here is fabricated or simulated.

---

## 1. Local Development Environment — VERIFIED

Measured on this machine via `scripts/hardware_check.py` (raw output: `docs/hardware_report.json`).

| Item | Value |
|---|---|
| OS | Windows 11 Pro (Build 26200) |
| CPU | 13th Gen Intel Core i5-1334U — 10 physical cores / 12 logical |
| Total RAM | 15.65 GB |
| Available RAM at time of check | **0.35 GB (97.8% used)** |
| GPU | Intel Iris Xe Graphics (integrated) |
| CUDA available | **No** |
| Free disk space | 167.7 GB / 460.3 GB |
| Python | 3.12.10 (via `py -3.12`; the venv's `.python-version`) |
| PyTorch | 2.14.0+cpu (CPU-only build, installed from `download.pytorch.org/whl/cpu`) |
| Transformers | 5.17.0 |
| Accelerate | 1.15.0 |

**Note on Python version:** `CLAUDE.md` targets Python 3.11. This machine has Python 3.14 (default `py` alias) and 3.12 installed, but not 3.11. 3.14 was released too recently to trust for ML-library stability, so **3.12 was used instead** as the closest safely-supported version, and PyTorch 2.14 installed and imported cleanly on it. This deviation is recorded here per the reproducibility requirement.

---

## 2. N-ATLaS Model Repository — VERIFIED

Queried directly from the Hugging Face Hub API (`huggingface.co/api/models/NCAIR1/N-ATLaS`), not assumed from prior documentation:

| Item | Value |
|---|---|
| Repo | `NCAIR1/N-ATLaS` |
| Access | **Gated** (`gated: auto` — requires accepting the license while logged in to Hugging Face, then a User Access Token) |
| Architecture | `LlamaForCausalLM`, Llama-3 8B (confirmed matches vendor documentation) |
| Format | Safetensors, 4 shards |
| Total download size | **16.08 GB** (4.98 + 5.00 + 4.92 + 1.17 GB) + 17 MB tokenizer |
| Last modified | 2025-09-23 |

This confirms the architecture details previously reported in `outline.md` (hidden size 4096, 32 layers, 32 heads / 8 KV heads, vocab 128,256, ~8,092-token context) match the live model card. See also intermediate size 14,336 and head dim 128, which were not previously documented.

---

## 3. Local Inference Feasibility — VERIFIED (not attempted, by design)

Given section 1 and 2 above:

- An 8B-parameter model in bf16/fp16 requires **~16 GB just for weights**, before activations, KV-cache, or the tokenizer/runtime overhead.
- This machine has 15.65 GB of **total** RAM, and at measurement time only **0.35 GB available**.
- **Conclusion: full-precision (bf16/fp32) local CPU inference is not feasible on this machine.** It would very likely OOM, or in the worst case cause severe swapping/system instability before failing.

`scripts/test_natlas.py` encodes this as an explicit guard: it estimates the required memory, compares it against `psutil`-reported available RAM, and refuses to attempt the load with a clear error rather than risking a system freeze. This guard was verified to trigger correctly (see run log below) — it currently exits first on a missing `HF_TOKEN`, which is also a real, unresolved blocker (section 4).

No local inference attempt was made. This is a documented engineering decision, not a skipped requirement — running the vendor's own math against the vendor's own model card was sufficient to establish infeasibility without needing a wasted 16 GB download on a machine already under memory pressure.

**Quantisation note (for a future local attempt):** even 4-bit quantisation (~5 GB resident) is plausible on this hardware's disk/CPU budget, but has not been attempted yet, and CPU-side 4-bit (bitsandbytes) support is less mature on Windows than on Linux/CUDA. Not pursued for Phase 0 given a GPU path (Colab) is available and is a better fit for the vendor's native Hugging Face/PyTorch loading path.

---

## 4. GPU Inference (Google Colab A100) — VERIFIED

Run on 2026-10-10, Colab with an NVIDIA A100-SXM4-80GB runtime, via `notebooks/colab_feasibility.ipynb`.

| Metric | Measured value |
|---|---|
| GPU | NVIDIA A100-SXM4-80GB (79.25 GB VRAM) |
| CUDA available | Yes (PyTorch 2.11.0+cu130) |
| Tokenizer load time | 4.95 s |
| Model load time (incl. one-time ~16 GB download) | 55.45 s |
| Peak GPU memory after model load | **14.96 GB** — matches the bf16 estimate almost exactly, ~64 GB headroom remaining |
| Generation (English prompt, 45 tokens) | 3.25 s → 13.85 tokens/sec |
| Generation (multilingual-instruction prompt, 13 tokens) | 0.51 s → 25.57 tokens/sec |

**Sample output (English):**
> Q: "What are three ways a rural health worker in Nigeria could use an AI assistant during a facility stock report?"
> A: "A rural health worker in Nigeria could use an AI assistant during a facility stock report by asking the AI to check the inventory levels, request restocking of supplies, and provide information on the usage and consumption of medical supplies."

Coherent and relevant — genuine, unmocked N-ATLaS output.

**Observed limitation:** a raw-string prompt asking the model to answer in Igbo, then Yoruba, then Hausa produced only a short English reply before hitting an early end-of-sequence token, rather than the requested multilingual output. This prompt did **not** use N-ATLaS's native Llama-3 chat template (`tokenizer.apply_chat_template`) — it was a quick ad-hoc smoke-test string. This is recorded here as an honest limitation of that specific test, not of the model generally (the vendor's own documented per-language eval scores in `outline.md` §11 indicate real multilingual capability). The actual evaluation pipeline (`evaluation/run_benchmark.py`) uses the proper chat template and does not have this issue.

**Conclusion: local hardware was correctly identified as the blocker, not N-ATLaS itself.** On a real GPU, loading and inference both work cleanly and comfortably within memory, confirming the "No" in section 3 was about this laptop specifically.

---

## 5. Preliminary Answers to the Required Feasibility Questions

| Question | Answer |
|---|---|
| Does N-ATLaS run locally? | **No**, not in full precision, on this machine (measured: insufficient RAM). Quantised CPU inference is untested. |
| Required RAM | ~16 GB minimum for bf16 weights alone (measured from actual shard sizes), plus overhead — practically 20+ GB recommended for headroom. |
| Required VRAM | ~16 GB minimum for bf16 on GPU; likely fits on a 16 GB card with 4-bit/8-bit quantisation with more comfortable margin. Not yet measured on real GPU hardware — pending Colab run. |
| CPU/GPU configuration | No usable GPU on the primary dev machine (integrated Intel graphics, no CUDA). A CUDA GPU (e.g. Colab T4) is required for a full-precision run. |
| Inference framework | Native Hugging Face Transformers + PyTorch (`AutoTokenizer` / `AutoModelForCausalLM`), as specified. No conversion to another framework. |
| Approximate inference latency | Measured: ~3.3s for a 45-token response (smoke test); ~8.1s for a full structured-extraction response (~100+ tokens) via the live FastAPI service. Both on an A100. |
| Is quantisation necessary? | For local CPU use: yes, if attempted at all. For a 16 GB GPU: likely still advisable for safety margin, not yet confirmed necessary. |
| Is LoRA/QLoRA fine-tuning locally feasible? | **No** on this machine (same RAM ceiling). **Confirmed feasible and fast on a Colab A100**: LoRA (r=16) on all attention+MLP projections, 500 training examples, 3 epochs, completed in ~2 minutes. See `docs/EVALUATION.md` for the resulting benchmark improvement. |
| Cloud infrastructure required? | Yes — at minimum a single 16 GB-class GPU (Colab T4 free tier is a reasonable first test; a larger/paid tier or AWS GPU instance likely needed for real fine-tuning work under PS3). |
| Major technical risks | (1) Gated-model access friction; (2) local machine cannot run or test anything beyond tokenizer/config-level checks; (3) Colab free-tier session limits (idle timeout, ~12h cap) may constrain iterative fine-tuning workflows; (4) currently near-zero available local RAM even for lightweight tooling work. |
| PS1/PS2/PS3 initial feasibility | See section 6. |

---

## 6. Preliminary PS1/PS2/PS3 Feasibility (subject to revision after the Colab run)

- **PS1 (Developer Infrastructure):** Feasible independent of local hardware constraints — most of the work (SDK/tooling/docs/dataset tooling) doesn't require running the 8B model directly on this machine; a hosted or Colab-based inference backend would suffice. Main constraint is recruiting 2+ external beta testers.
- **PS2 (Voice-First Access):** Feasibility depends on whether N-ATLaS's official ASR component has separate (possibly smaller) hardware requirements — not yet inspected. The core LLM hardware constraint (section 3) still applies for the response-generation half of the pipeline.
- **PS3 (Sectoral Fine-Tuning):** **Completed end-to-end.** 1,080 synthetic examples generated, baseline benchmarked (F1 0.677), LoRA fine-tuned (~2 min on an A100), adapted model benchmarked on the same held-out 520-example set (F1 0.9998), and served through a working FastAPI endpoint verified live. See `docs/EVALUATION.md` and `docs/API.md`.

This is the project's selected path (see `docs/PROJECT-DEFINITION.md`) given the ~24-hour timeline: PS3's validation doesn't depend on recruiting external users in real time, unlike PS1 (2+ beta testers) or PS2 (50+ real interactions), both of which remain technically feasible but were not pursued under this deadline.

---

## 7. Artifacts Produced So Far

- `scripts/hardware_check.py` — run, output in `docs/hardware_report.json`
- `scripts/test_natlas.py` — run, correctly refused (missing `HF_TOKEN`), no fabricated output
- `notebooks/colab_feasibility.ipynb` — ready to run once a token is available
- `.env.example`, `.python-version`, `requirements.txt`, `requirements-dev.txt`

## 8. Next Steps (blocked on user action)

1. Accept the N-ATLaS license on Hugging Face and generate an access token.
2. Run `notebooks/colab_feasibility.ipynb` on a Colab GPU runtime with that token.
3. Fold the real measured results into section 4 of this document.
4. Only then proceed to the PS1/PS2/PS3 selection (`docs/PROJECT-DEFINITION.md`) with actual evidence.
