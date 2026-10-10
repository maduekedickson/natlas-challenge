# Dataset: Health Facility Report Extraction

## Task

Free-text health-facility report → structured JSON with fields:
`facility_name`, `reporter_name`, `stock_items[]` (`item`, `quantity`, `unit`, `status`), `services_available[]`, `facility_status`.

## Provenance — IMPORTANT

**This dataset is 100% synthetic / template-generated**, produced by `training/prepare_dataset.py` (seed=42, fully reproducible). It is **not** real field data from any health facility, and must never be presented as such. This is an explicit, documented engineering decision made under a ~24-hour competition deadline (see `docs/PROJECT-DEFINITION.md`) — `outline.md` §17/§23 explicitly permit "carefully generated synthetic examples" as a valid dataset source.

Generation method: randomized combination of Nigerian town names, facility-type templates, a fixed vocabulary of vaccines/medicines/supplies, service types, and sentence templates (including a ~30% Nigerian-Pidgin-flavoured phrasing variant, to preserve realistic code-switching per `CLAUDE.md` §15 rather than normalizing it away).

## Schema (JSONL)

```json
{
  "id": "health-extract-00001",
  "instruction": "Extract structured information from this health facility report. ...",
  "input": "<free-text report>",
  "output": "<JSON string: the structured answer>",
  "language": "english | english_pidgin",
  "domain": "health"
}
```

## Splits

| Split | File | Count |
|---|---|---|
| Train | `data/train/train.jsonl` | 500 |
| Validation | `data/validation/validation.jsonl` | 60 |
| Benchmark (held out) | `data/benchmark/benchmark.jsonl` | 520 |

Benchmark size satisfies the ≥500 requirement in `outline.md` §28. Train/benchmark are generated from disjoint example IDs (verified: 0 overlap at generation time) — the benchmark is never seen during fine-tuning.

## Known limitations (recorded, not hidden)

- **English + Pidgin-flavoured English only.** No native Igbo/Hausa/Yoruba examples are included in this version — hand-generating those without native-speaker review risked introducing incorrect or ungrammatical text into the training/eval data, which is a worse outcome than a narrower honest scope. Flagged as future work.
- Vocabulary (towns, vaccines, medicines, services) is a fixed, modest list — real-world report phrasing is more varied. This is a controlled first dataset for proving the fine-tuning pipeline works and produces a measurable improvement, not a production-scale dataset.
- No human review pass was possible given the timeline. Spot-checked samples (see below) read as coherent and internally consistent with their JSON labels.

## Quality spot-check

Three random training examples were manually inspected for input/output consistency before use; all three were correct and well-formed (see conversation record / `training/prepare_dataset.py` output for reproducing this check).
