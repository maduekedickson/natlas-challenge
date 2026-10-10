"""Field-level precision/recall/F1 and exact-match metrics for the structured
health-facility-report extraction task.

No model dependency: pure comparison of two JSON-shaped objects, so this can
be (and is) unit-tested without ever loading N-ATLaS.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field


def _norm_str(value) -> str | None:
    if value is None:
        return None
    return str(value).strip().lower()


def _flatten_stock_items(items) -> set[tuple]:
    flat = set()
    if not isinstance(items, list):
        return flat
    for it in items:
        if not isinstance(it, dict):
            continue
        flat.add(
            (
                "stock_item",
                _norm_str(it.get("item")),
                it.get("quantity"),
                _norm_str(it.get("unit")),
                _norm_str(it.get("status")),
            )
        )
    return flat


def _flatten_services(services) -> set[tuple]:
    if not isinstance(services, list):
        return set()
    return {("service", _norm_str(s)) for s in services}


def flatten_fields(obj: dict) -> set[tuple]:
    """Turn a parsed extraction object into a flat set of (field, value) facts."""
    facts: set[tuple] = set()
    facts.add(("facility_name", _norm_str(obj.get("facility_name"))))
    facts.add(("reporter_name", _norm_str(obj.get("reporter_name"))))
    facts.add(("facility_status", _norm_str(obj.get("facility_status"))))
    facts |= _flatten_stock_items(obj.get("stock_items", []))
    facts |= _flatten_services(obj.get("services_available", []))
    return facts


def safe_parse_json(text: str) -> dict | None:
    """Best-effort parse of a model's raw text output as a JSON object."""
    text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    candidate = text[start : end + 1]
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


@dataclass
class ExampleScore:
    example_id: str
    parsed_ok: bool
    exact_match: bool
    precision: float
    recall: float
    f1: float


@dataclass
class BenchmarkScore:
    n_examples: int
    n_parsed_ok: int
    exact_match_rate: float
    mean_precision: float
    mean_recall: float
    mean_f1: float
    per_example: list[ExampleScore] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "n_examples": self.n_examples,
            "n_parsed_ok": self.n_parsed_ok,
            "parse_success_rate": round(self.n_parsed_ok / self.n_examples, 4) if self.n_examples else 0.0,
            "exact_match_rate": round(self.exact_match_rate, 4),
            "mean_precision": round(self.mean_precision, 4),
            "mean_recall": round(self.mean_recall, 4),
            "mean_f1": round(self.mean_f1, 4),
        }


def score_example(example_id: str, expected_output: str, predicted_text: str) -> ExampleScore:
    expected_obj = json.loads(expected_output)
    predicted_obj = safe_parse_json(predicted_text)

    if predicted_obj is None:
        return ExampleScore(example_id, parsed_ok=False, exact_match=False, precision=0.0, recall=0.0, f1=0.0)

    expected_facts = flatten_fields(expected_obj)
    predicted_facts = flatten_fields(predicted_obj)

    tp = len(expected_facts & predicted_facts)
    precision = tp / len(predicted_facts) if predicted_facts else 0.0
    recall = tp / len(expected_facts) if expected_facts else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    exact_match = expected_facts == predicted_facts

    return ExampleScore(example_id, parsed_ok=True, exact_match=exact_match, precision=precision, recall=recall, f1=f1)


def score_benchmark(scores: list[ExampleScore]) -> BenchmarkScore:
    n = len(scores)
    if n == 0:
        return BenchmarkScore(0, 0, 0.0, 0.0, 0.0, 0.0, [])
    n_parsed = sum(1 for s in scores if s.parsed_ok)
    exact_rate = sum(1 for s in scores if s.exact_match) / n
    mean_p = sum(s.precision for s in scores) / n
    mean_r = sum(s.recall for s in scores) / n
    mean_f1 = sum(s.f1 for s in scores) / n
    return BenchmarkScore(n, n_parsed, exact_rate, mean_p, mean_r, mean_f1, scores)
