"""Attempt genuine N-ATLaS inference and measure loading time, latency, and memory use.

This script performs NO mocking: it downloads the real NCAIR1/N-ATLaS weights
(gated repository) and runs real generation. It requires an HF_TOKEN with
accepted access to the model, set via environment variable or a .env file.

Usage:
    python scripts/test_natlas.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil
import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

load_dotenv()

MODEL_NAME = os.environ.get("MODEL_NAME", "NCAIR1/N-ATLaS")
MODEL_REVISION = os.environ.get("MODEL_REVISION") or None
HF_TOKEN = os.environ.get("HF_TOKEN")

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "docs"
OUTPUT_PATH = OUTPUT_DIR / "test_natlas_output.json"

# Rough fp16/bf16 memory footprint estimate for an 8B-parameter model:
# 8e9 params * 2 bytes ~= 16 GB, plus activation/KV-cache overhead.
ESTIMATED_BF16_WEIGHTS_GB = 16.0
SAFETY_MARGIN_GB = 4.0

PROMPT = (
    "You are N-ATLaS, a multilingual Nigerian AI assistant. "
    "Respond briefly in English.\n\n"
    "Question: What are three ways a rural health worker in Nigeria could use "
    "an AI assistant during a facility stock report?\nAnswer:"
)


def check_feasible_or_exit() -> str:
    """Pick a device and abort early rather than risk an OOM/system freeze."""
    if not HF_TOKEN:
        logger.error(
            "HF_TOKEN is not set. NCAIR1/N-ATLaS is a gated repository: accept the "
            "license at https://huggingface.co/NCAIR1/N-ATLaS while logged in, then "
            "create a read token at https://huggingface.co/settings/tokens and set "
            "HF_TOKEN in your .env file."
        )
        sys.exit(1)

    if torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info("CUDA GPU detected with %.2f GB VRAM.", vram_gb)
        if vram_gb < ESTIMATED_BF16_WEIGHTS_GB - SAFETY_MARGIN_GB:
            logger.warning(
                "GPU VRAM (%.2f GB) is below the ~%.0f GB an 8B model needs in "
                "bf16. Consider 4-bit/8-bit quantisation instead of a full run.",
                vram_gb,
                ESTIMATED_BF16_WEIGHTS_GB,
            )
        return "cuda"

    available_gb = psutil.virtual_memory().available / 1024**3
    logger.info("No CUDA GPU. Available system RAM: %.2f GB.", available_gb)
    if available_gb < ESTIMATED_BF16_WEIGHTS_GB + SAFETY_MARGIN_GB:
        logger.error(
            "Refusing to attempt a full bf16 load: available RAM (%.2f GB) is well "
            "below the ~%.0f GB an 8B-parameter model requires, plus overhead. "
            "This would very likely OOM or freeze the machine. Run this on a GPU "
            "runtime (e.g. Google Colab) or use a quantised load path instead.",
            available_gb,
            ESTIMATED_BF16_WEIGHTS_GB,
        )
        sys.exit(1)
    return "cpu"


def main() -> None:
    device = check_feasible_or_exit()
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    result: dict = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model_name": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "device": device,
        "dtype": str(dtype),
    }

    logger.info("Loading tokenizer for %s ...", MODEL_NAME)
    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, revision=MODEL_REVISION, token=HF_TOKEN
    )
    tokenizer_load_s = time.perf_counter() - t0
    result["tokenizer_load_seconds"] = round(tokenizer_load_s, 2)
    logger.info("Tokenizer loaded in %.2fs", tokenizer_load_s)

    logger.info("Loading model for %s (this downloads ~16 GB on first run) ...", MODEL_NAME)
    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        revision=MODEL_REVISION,
        token=HF_TOKEN,
        torch_dtype=dtype,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model_load_s = time.perf_counter() - t0
    result["model_load_seconds"] = round(model_load_s, 2)
    logger.info("Model loaded in %.2fs", model_load_s)

    inputs = tokenizer(PROMPT, return_tensors="pt").to(model.device)

    logger.info("Running generation ...")
    t0 = time.perf_counter()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            temperature=None,
            top_p=None,
        )
    generation_s = time.perf_counter() - t0

    generated_text = tokenizer.decode(
        output_ids[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )
    new_tokens = output_ids.shape[1] - inputs["input_ids"].shape[1]

    result.update(
        {
            "prompt": PROMPT,
            "generated_text": generated_text,
            "new_tokens": int(new_tokens),
            "generation_seconds": round(generation_s, 2),
            "tokens_per_second": round(new_tokens / generation_s, 2) if generation_s > 0 else None,
        }
    )

    if device == "cuda":
        result["peak_gpu_memory_gb"] = round(torch.cuda.max_memory_allocated() / 1024**3, 2)
    else:
        result["process_ram_gb"] = round(
            psutil.Process().memory_info().rss / 1024**3, 2
        )

    logger.info("Generated text: %s", generated_text)
    logger.info("Result: %s", json.dumps(result, indent=2))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(result, indent=2))
    logger.info("Saved output artifact to %s", OUTPUT_PATH)


if __name__ == "__main__":
    main()
