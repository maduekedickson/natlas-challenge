"""LoRA fine-tune N-ATLaS on the synthetic health-report extraction dataset.

Uses plain transformers.Trainer + peft (not TRL's SFTTrainer) with manual
prompt-masking: loss is computed only on the assistant's completion tokens,
not the instruction/input. This was verified locally against the real
N-ATLaS tokenizer: the prompt is a token-level prefix of the full sequence,
and the unmasked labels decode to exactly the target JSON output.

Intended to run on a GPU (Colab A100). Requires: transformers, accelerate, peft.

Usage:
    python training/train.py --train data/train/train.jsonl --epochs 3
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import torch
from dotenv import load_dotenv
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

load_dotenv()

MODEL_NAME = os.environ.get("MODEL_NAME", "NCAIR1/N-ATLaS")
HF_TOKEN = os.environ.get("HF_TOKEN")
SEED = 42

LORA_CONFIG = {
    "r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    "task_type": "CAUSAL_LM",
    "bias": "none",
}


def load_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def build_example(tokenizer, example: dict, max_length: int) -> dict:
    user_msg = [
        {"role": "user", "content": f"{example['instruction']}\n\nReport:\n{example['input']}"}
    ]
    prompt_text = tokenizer.apply_chat_template(
        user_msg, tokenize=False, add_generation_prompt=True
    )
    full_messages = user_msg + [{"role": "assistant", "content": example["output"]}]
    full_text = tokenizer.apply_chat_template(full_messages, tokenize=False)

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"][:max_length]

    prompt_len = min(len(prompt_ids), len(full_ids))
    labels = list(full_ids)
    for i in range(prompt_len):
        labels[i] = -100

    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
    }


class TokenizedDataset(Dataset):
    def __init__(self, examples: list[dict]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        return self.examples[idx]


def make_collate_fn(pad_token_id: int):
    def collate(batch: list[dict]) -> dict:
        max_len = max(len(ex["input_ids"]) for ex in batch)
        input_ids, attention_mask, labels = [], [], []
        for ex in batch:
            pad_len = max_len - len(ex["input_ids"])
            input_ids.append(ex["input_ids"] + [pad_token_id] * pad_len)
            attention_mask.append(ex["attention_mask"] + [0] * pad_len)
            labels.append(ex["labels"] + [-100] * pad_len)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    return collate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/train/train.jsonl")
    parser.add_argument("--output-dir", default="training/output/lora-adapter")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-length", type=int, default=1024)
    args = parser.parse_args()

    if not HF_TOKEN:
        raise SystemExit("HF_TOKEN not set. See .env.example.")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        token=HF_TOKEN,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model = get_peft_model(model, LoraConfig(**LORA_CONFIG))
    model.print_trainable_parameters()

    raw_examples = load_jsonl(args.train)
    tokenized = [build_example(tokenizer, ex, args.max_length) for ex in raw_examples]
    train_dataset = TokenizedDataset(tokenized)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
        seed=SEED,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=make_collate_fn(tokenizer.pad_token_id),
    )

    logger.info("Starting LoRA fine-tuning: %d examples, %d epochs", len(raw_examples), args.epochs)
    trainer.train()

    final_dir = Path(args.output_dir) / "final"
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    logger.info("Saved LoRA adapter to %s", final_dir)

    run_config = {
        "base_model": MODEL_NAME,
        "train_file": args.train,
        "n_train_examples": len(raw_examples),
        "lora_config": LORA_CONFIG,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "learning_rate": args.lr,
        "max_length": args.max_length,
        "seed": SEED,
        "framework": "transformers.Trainer + peft (not TRL)",
    }
    (Path(args.output_dir) / "run_config.json").write_text(json.dumps(run_config, indent=2))
    logger.info("Saved run config alongside adapter.")


if __name__ == "__main__":
    main()
