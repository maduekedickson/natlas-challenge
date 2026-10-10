# Project Definition — PS3: Sectoral Fine-Tuning (Health)

**Status:** Locked for execution given a ~24-hour deadline. Revisit only if blocked.

## Why PS3, and why now

With Phase 0 complete (`docs/FEASIBILITY.md`), the picture is:

- A Colab **A100** is available (confirmed by the user), which removes the hardware blocker entirely — 16 GB bf16 weights fit with large headroom (40/80 GB VRAM), and LoRA fine-tuning of an 8B model is fast on this GPU.
- With **~1 day left**, the deciding factor is no longer compute, it's **what depends on other people**:
  - PS1 requires 2+ external beta testers actually using the artefact and giving feedback — depends on other people responding in time.
  - PS2 requires 50+ documented real user interactions — depends on reaching real users at scale in <24h; high risk of falling short or rushing low-quality "evidence."
  - **PS3's validation is self-contained**: a held-out benchmark comparison (base vs. adapted N-ATLaS) that one person can run end-to-end without depending on anyone else's availability.
- PS3 also matches the domain hypothesis already recorded in `outline.md` §6 (health), so no time is lost re-deriving a direction.

## Scoped task: Structured Health Facility Report Extraction

**Why this exact task:** it's explicitly listed in `outline.md` §6 ("facility reporting," "structured health data extraction," "supply/stock reporting," "service availability"), it is *not* autonomous clinical decision-making (stays inside the explicit safety boundary in §6), and it has an objective, programmatic metric (field-level precision/recall/F1, exact match — §18) that needs no human judges. That combination is what makes it finishable solo in 24 hours.

**Input:** a free-text message a health worker might send (text or transcribed voice) reporting on their facility.

**Output (structured JSON):**
```json
{
  "facility_name": "Eziobodo Primary Health Centre",
  "reporter_name": "Ngozi",
  "stock_items": [
    {"item": "measles vaccine", "quantity": 20, "unit": "doses", "status": "in_stock"},
    {"item": "ORS", "quantity": 0, "unit": null, "status": "out_of_stock"}
  ],
  "services_available": ["antenatal care"],
  "facility_status": "open"
}
```

## Honest scope limitations (recorded now, not discovered later)

- **Dataset is synthetic/template-generated**, not real field reports — documented as such in `docs/DATASET.md`, never presented as real-world data. This is explicitly permitted by `outline.md` §17/§23 ("carefully generated synthetic examples") given no time to source/clean a real dataset.
- **Language scope for v1: English + light Nigerian-Pidgin-flavoured English only.** Native Igbo/Hausa/Yoruba training examples are *not* hand-authored for this dataset — risk of introducing ungrammatical or incorrect text at scale without native-speaker review (a real quality requirement per `outline.md` §16) is too high to accept blind under this deadline. Multilingual extension is recorded here as **future work**, not silently dropped.
- Benchmark size target: 500+ held-out examples per `outline.md` §28, generated alongside train/validation from the same template engine, split so **no template-instance overlaps between train and benchmark** (leakage prevention, §24).

## Pipeline (self-contained, no external dependencies)

```
Synthetic dataset generator (training/prepare_dataset.py)
        ↓
train.jsonl / validation.jsonl / benchmark.jsonl (held out, ≥500)
        ↓
Baseline: base N-ATLaS on benchmark.jsonl → baseline_results.json
        ↓
LoRA fine-tune on train.jsonl (A100, Colab)
        ↓
Adapted N-ATLaS on the SAME benchmark.jsonl → adapted_results.json
        ↓
comparison.csv / comparison.png (generated programmatically, not hand-typed)
```

## Fallback

If LoRA training or the benchmark comparison cannot be completed in time, the fallback submission is the baseline-only evaluation plus a clearly-labelled "fine-tuning in progress / future work" note — never a fabricated "adapted" result.
