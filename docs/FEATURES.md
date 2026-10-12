# Features

The original extraction demo (free text → structured JSON) is unchanged and remains central — everything below is additive, persisted on top of it.

## Dashboard
`GET /api/v1/dashboard`, Streamlit page `Dashboard`. Total reports, distinct facilities represented, reports submitted today, count of open/closed-status reports, stock-out observation count, and the 10 most recent reports.

## Report History
`GET /api/v1/reports` (list, filterable by facility), `GET /api/v1/reports/{id}` (detail: original text, full structured JSON, normalized stock items, model/adapter version). Streamlit page `History`.

## Trends & Comparison
`GET /api/v1/analytics/*` — stock quantity over time for a facility+commodity, facility open/closed status over time, reports-by-facility, commodity frequency, stock-out occurrences. All computed from real stored data; **empty results are returned (and shown as "no data yet") rather than fabricated** when there isn't enough history. Streamlit page `Trends`.

## CSV / Excel Export
`GET /api/v1/exports/{reports|stock-items}.{csv|xlsx}`, optionally filtered by facility. Two tiers, per the feature spec: **report-level** (one row per report, full structured JSON kept as a column — nothing lossily flattened) and **stock-item-level** (one row per report×item, suited to spreadsheet pivoting). Streamlit page `Exports`.

## Batch Processing
`POST /api/v1/batch` (multipart CSV upload). Expects a `report` (or `report_text`/`text`) column, optionally `facility`/`reporter`. **Reuses `NAtlasService.analyze()` per row — no second inference implementation.** Per-row failures (empty text, inference error) are recorded individually and never abort the batch; response includes total/successful/failed counts, processing time, and per-row results. Streamlit page `Batch Processing`.

## Voice Input
See `docs/VOICE.md`.

## What stayed exactly the same
- `POST /api/v1/analyze` still returns the same shape it always did (plus a new optional `report_id`)
- `NAtlasService` — the model loading, prompting, and parsing logic — is completely untouched
- The 520-example benchmark and its results are untouched; none of this touches `data/benchmark/` or `evaluation/results/`

## What was deliberately *not* added
- **API-key authentication** — explicitly descoped for this pass to keep the demo video flow unauthenticated and simple; can be added later (see `docs/IMPLEMENTATION_PLAN.md` Phase 7, not implemented).
- Components split (`streamlit_app/components/charts.py` etc.) — the suggested structure was simplified to keep chart/table rendering inline in each page. Reason: Streamlit's multipage import resolution couldn't be fully tested locally (same class of import-path issue hit earlier with `evaluation/run_benchmark.py`), so fewer cross-file import paths means fewer things that can only be debugged via a slow Colab round-trip.
