# Evaluation: Base N-ATLaS vs. LoRA-Adapted (Health Report Extraction)

All numbers below are generated programmatically by `evaluation/run_benchmark.py` and `evaluation/benchmark.py` — none are hand-typed. Raw per-example outputs are preserved in the result JSON files for audit.

## Baseline (Base N-ATLaS, zero-shot, proper chat template)

Run: 2026-10-10, Colab A100, full held-out benchmark (`data/benchmark/benchmark.jsonl`, 520 examples), greedy decoding, `max_new_tokens=256`.

| Metric | Value |
|---|---|
| Examples | 520 |
| Parse success rate | 99.81% (519/520 valid JSON) |
| Exact match rate | 1.35% |
| Mean field-level precision | 0.6765 |
| Mean field-level recall | 0.6774 |
| Mean field-level F1 | **0.6765** |
| Total wall time | ~51.5 minutes (≈5.9s/example, generation-bound) |

**Interpretation:** base N-ATLaS already follows the extraction instruction reasonably well zero-shot when prompted with the correct chat template (near-perfect JSON formatting), but gets the actual field values fully correct less than 2% of the time. This is the before-fine-tuning reference point — the adapted model is evaluated on the exact same 520 examples for a fair comparison.

## Adapted (LoRA fine-tuned)

Run: 2026-10-10, same Colab A100 session, same held-out benchmark (`data/benchmark/benchmark.jsonl`, 520 examples), greedy decoding, same `max_new_tokens=256`.

**Training:** `training/train.py`, 3 epochs over the 500-example train split, LoRA (r=16, alpha=32, dropout=0.05) on all attention + MLP projections, 41.9M trainable params (0.52% of the 8.07B total). Loss dropped from ~0.048 to ~0.0002 over training — consistent with the dataset's templated structure, which is exactly why the held-out benchmark (disjoint example IDs, same templates) is the meaningful test, not the training loss.

| Metric | Value |
|---|---|
| Examples | 520 (identical benchmark file as baseline) |
| Parse success rate | 100% (520/520) |
| Exact match rate | 99.81% |
| Mean field-level precision | 1.0 |
| Mean field-level recall | 0.9997 |
| Mean field-level F1 | **0.9998** |

## Comparison

Generated programmatically via `python evaluation/benchmark.py --base evaluation/results/base_results.json --adapted evaluation/results/adapted_results.json` → `evaluation/results/comparison.csv` and `comparison.png`.

| Metric | Base | Adapted | Improvement |
|---|---|---|---|
| Parse success rate | 0.9981 | 1.0000 | +0.0019 |
| Exact match rate | 0.0135 | 0.9981 | **+0.9846** |
| Mean precision | 0.6765 | 1.0000 | +0.3235 |
| Mean recall | 0.6774 | 0.9997 | +0.3223 |
| Mean F1 | 0.6765 | 0.9998 | **+0.3233** |

**Conclusion:** LoRA fine-tuning produced a large, measurable improvement on the held-out benchmark — exact match went from 1.35% to 99.81%, and field-level F1 from 0.68 to 1.00. This demonstrates genuine domain adaptation (not just prompt engineering — the baseline already used the correct chat-template prompting and still only reached 0.68 F1).

**Qualitative example** (benchmark ID `health-extract-00050`, real outputs, both truncated):

> Input: "Daily report: This is Yusuf from General Hospital, Abuja. we have 75 packs of cotton wool; we are out of amoxicillin; we are out of syringes. We offer postnatal care..."
>
> Expected: `{"facility_name": "General Hospital, Abuja", "reporter_name": "Yusuf", "stock_items": [{"item": "cotton wool", "quantity": 75, "unit": "packs", "status": "in_stock"}, {"item": "amoxicillin", "quantity": 0, "unit": null, "status": "out_of_stock"}, ...]}`
>
> **Base**: wraps the JSON in a ` ```json ` code fence, and uses near-but-not-exact vocabulary: `"status": "available"` (not `"in_stock"`), `"unit": "units"` for an out-of-stock item (expected `null`).
>
> **Adapted**: clean JSON, no code fence, exact schema vocabulary throughout.

This is the concrete mechanism behind the F1 gap: the base model clearly understands the *task* (it extracts largely correct facts) but doesn't reliably follow the *exact* output contract a downstream system would parse against. The fine-tune fixes format/vocabulary compliance, not just "understanding."

**Honest caveat:** this dataset is synthetic and template-generated (see `docs/DATASET.md`). The near-perfect adapted score is partly a reflection of the dataset's structural regularity — a held-out benchmark drawn from the same template family is a genuine test of generalization *within that distribution*, but is not evidence the model would perform this well on real, unstructured field reports with vocabulary or phrasing outside the generator's fixed lists. This limitation is recorded here rather than glossed over, consistent with `CLAUDE.md`'s rule against overstating unmeasured performance.
