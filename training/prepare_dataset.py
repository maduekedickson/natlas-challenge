"""Generate the synthetic health-facility-report extraction dataset.

Produces instruction-tuning JSONL examples for the task:
    free-text facility report -> structured JSON (facility, stock, services, status)

The dataset is template-generated, NOT real field data. This is documented
explicitly in docs/DATASET.md. See docs/PROJECT-DEFINITION.md for scope.

Usage:
    python training/prepare_dataset.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path

SEED = 42
random.seed(SEED)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

TASK_INSTRUCTION = (
    "Extract structured information from this health facility report. "
    "Return a single JSON object with exactly these fields: "
    "facility_name (string), reporter_name (string or null), "
    "stock_items (list of {item, quantity, unit, status}), "
    "services_available (list of strings), "
    "facility_status (\"open\" or \"closed\")."
)

TOWNS = [
    "Owerri", "Enugu", "Kano", "Ibadan", "Lagos", "Abuja", "Kaduna", "Jos",
    "Uyo", "Asaba", "Onitsha", "Benin City", "Sokoto", "Maiduguri", "Calabar",
    "Ilorin", "Abeokuta", "Warri", "Minna", "Zaria", "Nsukka", "Awka",
]

FACILITY_TYPES = [
    "Primary Health Centre", "General Hospital", "Health Post",
    "Maternity Clinic", "Comprehensive Health Centre", "Cottage Hospital",
]

NAMED_FACILITY_TEMPLATES = [
    "{town} {ftype}",
    "{ftype}, {town}",
    "St. Mary's {ftype}, {town}",
    "{town} Central {ftype}",
]

REPORTER_NAMES = [
    "Ngozi", "Chidinma", "Amaka", "Ifeoma", "Aisha", "Fatima", "Ibrahim",
    "Musa", "Folake", "Tunde", "Bisi", "Adewale", "Chinedu", "Emeka",
    "Hauwa", "Grace", "Blessing", "Yusuf", "Kemi", "Uche", None, None,
]

STOCK_ITEMS = [
    ("measles vaccine", "doses"),
    ("BCG vaccine", "doses"),
    ("OPV (polio) vaccine", "doses"),
    ("pentavalent vaccine", "doses"),
    ("yellow fever vaccine", "doses"),
    ("tetanus toxoid", "doses"),
    ("ORS", "sachets"),
    ("paracetamol", "tablets"),
    ("amoxicillin", "capsules"),
    ("artemether-lumefantrine (ACT)", "packs"),
    ("ferrous sulfate", "tablets"),
    ("folic acid", "tablets"),
    ("gloves", "pairs"),
    ("syringes", "pieces"),
    ("malaria RDT kits", "kits"),
    ("HIV test kits", "kits"),
    ("cotton wool", "packs"),
]

SERVICES = [
    "antenatal care", "immunization", "family planning", "malaria testing",
    "HIV testing and counselling", "general outpatient services",
    "postnatal care", "nutrition counselling",
]

CLOSED_REASONS = [
    "due to staff shortage", "for fumigation", "because of a power outage",
    "for routine maintenance",
]

PIDGIN_OPENERS = [
    "Good morning o, ", "Abeg make I report, ", "Na so e be, ", "",
    "Wetin dey happen for our side: ", "",
]

PLAIN_OPENERS = [
    "Good morning, ", "Hello, ", "Daily report: ", "Submitting today's report. ",
    "", "",
]


def stock_status(qty: int) -> str:
    if qty == 0:
        return "out_of_stock"
    if qty < 10:
        return "low_stock"
    return "in_stock"


def make_facility_name(rng: random.Random) -> str:
    town = rng.choice(TOWNS)
    ftype = rng.choice(FACILITY_TYPES)
    template = rng.choice(NAMED_FACILITY_TEMPLATES)
    return template.format(town=town, ftype=ftype)


def make_example(rng: random.Random, idx: int) -> dict:
    facility = make_facility_name(rng)
    reporter = rng.choice(REPORTER_NAMES)
    n_items = rng.randint(1, 4)
    chosen_items = rng.sample(STOCK_ITEMS, n_items)

    stock_items = []
    stock_phrases = []
    for item, unit in chosen_items:
        qty = rng.choice([0, 0, rng.randint(1, 9), rng.randint(10, 200)])
        status = stock_status(qty)
        stock_items.append({
            "item": item,
            "quantity": qty,
            "unit": unit if qty > 0 else None,
            "status": status,
        })
        if qty == 0:
            stock_phrases.append(f"we are out of {item}")
        else:
            stock_phrases.append(f"we have {qty} {unit} of {item}")

    n_services = rng.randint(1, 3)
    services = rng.sample(SERVICES, n_services)

    is_open = rng.random() > 0.15
    facility_status = "open" if is_open else "closed"

    use_pidgin = rng.random() < 0.3
    opener = rng.choice(PIDGIN_OPENERS if use_pidgin else PLAIN_OPENERS)

    reporter_phrase = f"This is {reporter} from {facility}. " if reporter else f"Report from {facility}. "
    stock_sentence = "; ".join(stock_phrases) + ". "
    services_sentence = f"We offer {', '.join(services)} here. "

    if is_open:
        status_sentence = "The facility is open today."
    else:
        reason = rng.choice(CLOSED_REASONS)
        status_sentence = f"The facility is closed today {reason}."

    report_text = opener + reporter_phrase + stock_sentence + services_sentence + status_sentence

    output_obj = {
        "facility_name": facility,
        "reporter_name": reporter,
        "stock_items": stock_items,
        "services_available": services,
        "facility_status": facility_status,
    }

    return {
        "id": f"health-extract-{idx:05d}",
        "instruction": TASK_INSTRUCTION,
        "input": report_text,
        "output": json.dumps(output_obj, ensure_ascii=False),
        "language": "english_pidgin" if use_pidgin else "english",
        "domain": "health",
    }


def write_jsonl(path: Path, examples: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")


def main() -> None:
    rng = random.Random(SEED)

    n_train = 500
    n_validation = 60
    n_benchmark = 520  # >=500 per outline.md section 28

    total = n_train + n_validation + n_benchmark
    all_examples = [make_example(rng, i) for i in range(total)]
    rng.shuffle(all_examples)

    train = all_examples[:n_train]
    validation = all_examples[n_train:n_train + n_validation]
    benchmark = all_examples[n_train + n_validation:]

    write_jsonl(DATA_DIR / "raw" / "health_reports_all.jsonl", all_examples)
    write_jsonl(DATA_DIR / "train" / "train.jsonl", train)
    write_jsonl(DATA_DIR / "validation" / "validation.jsonl", validation)
    write_jsonl(DATA_DIR / "benchmark" / "benchmark.jsonl", benchmark)

    print(f"train:      {len(train)}")
    print(f"validation: {len(validation)}")
    print(f"benchmark:  {len(benchmark)} (must be >= 500: {len(benchmark) >= 500})")

    train_ids = {e["id"] for e in train}
    bench_ids = {e["id"] for e in benchmark}
    overlap = train_ids & bench_ids
    print(f"train/benchmark id overlap: {len(overlap)} (must be 0)")


if __name__ == "__main__":
    main()
