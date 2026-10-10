"""Singleton service wrapping N-ATLaS: load once at startup, serve many requests.

Does not mock or substitute the model. If HF_TOKEN is missing or the model
can't be loaded (e.g. insufficient hardware), this raises rather than
falling back to fake output.
"""

from __future__ import annotations

import json
import logging
import os
import time

import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

load_dotenv()
logger = logging.getLogger(__name__)

MODEL_NAME = os.environ.get("MODEL_NAME", "NCAIR1/N-ATLaS")
HF_TOKEN = os.environ.get("HF_TOKEN")
ADAPTER_PATH = os.environ.get("ADAPTER_PATH")  # optional path to a LoRA adapter

EXTRACTION_INSTRUCTION = (
    "Extract structured information from this health facility report. "
    "Return a single JSON object with exactly these fields: "
    "facility_name (string), reporter_name (string or null), "
    "stock_items (list of {item, quantity, unit, status}), "
    "services_available (list of strings), "
    "facility_status (\"open\" or \"closed\")."
)


def _safe_parse_json(text: str) -> dict | None:
    text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    try:
        parsed = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


class NAtlasService:
    """Lazily-instantiated singleton. First call loads the model; later calls reuse it."""

    _instance: "NAtlasService | None" = None

    def __new__(cls) -> "NAtlasService":
        if cls._instance is None:
            instance = super().__new__(cls)
            instance._load()
            cls._instance = instance
        return cls._instance

    def _load(self) -> None:
        if not HF_TOKEN:
            raise RuntimeError("HF_TOKEN not set. See .env.example.")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32

        logger.info("Loading tokenizer for %s", MODEL_NAME)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)

        logger.info("Loading model for %s on %s", MODEL_NAME, self.device)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            token=HF_TOKEN,
            torch_dtype=dtype,
            device_map="auto",
            low_cpu_mem_usage=True,
        )

        self.adapter_loaded = False
        if ADAPTER_PATH:
            from peft import PeftModel

            logger.info("Loading LoRA adapter from %s", ADAPTER_PATH)
            self.model = PeftModel.from_pretrained(self.model, ADAPTER_PATH)
            self.adapter_loaded = True

        self.model.eval()
        logger.info(
            "N-ATLaS service ready (device=%s, adapter_loaded=%s)",
            self.device,
            self.adapter_loaded,
        )

    def analyze(self, report_text: str, max_new_tokens: int = 256) -> dict:
        messages = [
            {"role": "user", "content": f"{EXTRACTION_INSTRUCTION}\n\nReport:\n{report_text}"}
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        t0 = time.perf_counter()
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        generation_s = time.perf_counter() - t0

        generated_text = self.tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )

        parsed = _safe_parse_json(generated_text)
        result = {
            "raw_model_output": generated_text,
            "parsed_ok": parsed is not None,
            "generation_seconds": round(generation_s, 2),
            "facility_name": None,
            "reporter_name": None,
            "stock_items": [],
            "services_available": [],
            "facility_status": None,
        }
        if parsed:
            result.update(
                {
                    "facility_name": parsed.get("facility_name"),
                    "reporter_name": parsed.get("reporter_name"),
                    "stock_items": parsed.get("stock_items", []),
                    "services_available": parsed.get("services_available", []),
                    "facility_status": parsed.get("facility_status"),
                }
            )
        return result

    def health(self) -> dict:
        return {
            "status": "ok",
            "model_name": MODEL_NAME,
            "model_loaded": hasattr(self, "model"),
            "device": getattr(self, "device", "unknown"),
            "adapter_loaded": getattr(self, "adapter_loaded", False),
        }
