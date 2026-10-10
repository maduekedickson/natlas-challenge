"""Run the held-out benchmark against either base N-ATLaS or an adapted
(LoRA fine-tuned) version, and score it with evaluation/metrics.py.

Designed to run on a GPU (Colab A100). No mocked generation: this always
calls the real model.

Usage:
    python evaluation/run_benchmark.py --model base \
        --benchmark data/benchmark/benchmark.jsonl \
        --output evaluation/results/base_results.json

    python evaluation/run_benchmark.py --model adapted \
        --adapter-path training/output/lora-adapter \
        --benchmark data/benchmark/benchmark.jsonl \
        --output evaluation/results/adapted_results.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

from metrics import score_benchmark, score_example

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

load_dotenv()

MODEL_NAME = os.environ.get("MODEL_NAME", "NCAIR1/N-ATLaS")
HF_TOKEN = os.environ.get("HF_TOKEN")


def load_model(adapter_path: str | None):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        token=HF_TOKEN,
        torch_dtype=dtype,
        device_map="auto",
        low_cpu_mem_usage=True,
    )

    if adapter_path:
        from peft import PeftModel

        logger.info("Loading LoRA adapter from %s", adapter_path)
        model = PeftModel.from_pretrained(model, adapter_path)
        logger.info("Merging LoRA weights into base model for fast inference")
        model = model.merge_and_unload()

    model.eval()
    return tokenizer, model


def build_prompt(tokenizer, instruction: str, input_text: str) -> str:
    messages = [{"role": "user", "content": f"{instruction}\n\nReport:\n{input_text}"}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def run(args: argparse.Namespace) -> None:
    tokenizer, model = load_model(args.adapter_path)

    examples = []
    with open(args.benchmark, encoding="utf-8") as f:
        for line in f:
            examples.append(json.loads(line))
    if args.limit:
        examples = examples[: args.limit]
    logger.info("Running %d benchmark examples (model=%s)", len(examples), args.model)

    per_example_scores = []
    raw_outputs = []
    t_start = time.perf_counter()

    for i, ex in enumerate(examples):
        prompt = build_prompt(tokenizer, ex["instruction"], ex["input"])
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
            )
        generated_text = tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )

        score = score_example(ex["id"], ex["output"], generated_text)
        per_example_scores.append(score)
        raw_outputs.append({"id": ex["id"], "generated_text": generated_text})

        if (i + 1) % 25 == 0:
            logger.info("Progress: %d/%d", i + 1, len(examples))

    total_s = time.perf_counter() - t_start
    aggregate = score_benchmark(per_example_scores)

    result = {
        "model": args.model,
        "model_name": MODEL_NAME,
        "adapter_path": args.adapter_path,
        "n_examples": len(examples),
        "total_seconds": round(total_s, 2),
        "seconds_per_example": round(total_s / len(examples), 3) if examples else None,
        "metrics": aggregate.to_dict(),
        "raw_outputs": raw_outputs,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2))
    logger.info("Metrics: %s", aggregate.to_dict())
    logger.info("Saved results to %s", args.output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["base", "adapted"], required=True)
    parser.add_argument("--adapter-path", default=None, help="Required when --model adapted")
    parser.add_argument("--benchmark", default="data/benchmark/benchmark.jsonl")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--limit", type=int, default=None, help="Limit examples for a quick smoke test")
    args = parser.parse_args()

    if args.model == "adapted" and not args.adapter_path:
        parser.error("--adapter-path is required when --model adapted")

    run(args)


if __name__ == "__main__":
    main()
