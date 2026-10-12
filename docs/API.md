# API: N-ATLaS Health Intelligence Platform

FastAPI service exposing LoRA-adapted N-ATLaS for structured extraction from free-text health facility reports, plus report persistence, trend analytics, batch processing, exports, and voice transcription. Source: `api/`.

**No authentication in this version** — deliberately descoped for the current demo pass (see `docs/IMPLEMENTATION_PLAN.md`).

## Running it

Requires a GPU with enough VRAM to hold N-ATLaS (confirmed working on an A100-80GB — see `docs/FEASIBILITY.md`). Not runnable on a CPU-only machine.

```bash
export HF_TOKEN=...                                   # accepted-license HF token
export ADAPTER_PATH=training/output/lora-adapter/final # omit to serve the base model instead
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Interactive docs are auto-generated at `/docs` (Swagger UI) and `/redoc`.

## Endpoints

### `GET /api/v1/health`

Returns service status — does not re-run inference, just reports what's currently loaded.

**Response:**
```json
{
  "status": "ok",
  "model_name": "NCAIR1/N-ATLaS",
  "model_loaded": true,
  "device": "cuda",
  "adapter_loaded": true
}
```

### `POST /api/v1/analyze`

Extracts structured facility/stock/service data from a free-text report.

**Request:**
```json
{
  "report_text": "Good morning, this is Ngozi from Owerri Primary Health Centre. We have 20 doses of measles vaccine, we are out of ORS, and we offer antenatal care. The facility is open today."
}
```

**Response** (real example, captured 2026-10-10 from the live service on a Colab A100):
```json
{
  "facility_name": "Owerri Primary Health Centre",
  "reporter_name": "Ngozi",
  "stock_items": [
    {"item": "measles vaccine", "quantity": 20, "unit": "doses", "status": "in_stock"},
    {"item": "ORS", "quantity": 0, "unit": null, "status": "out_of_stock"}
  ],
  "services_available": ["antenatal care"],
  "facility_status": "open",
  "raw_model_output": "{...the raw JSON string the model generated...}",
  "parsed_ok": true,
  "generation_seconds": 8.14,
  "report_id": 1
}
```

Field notes:
- `parsed_ok`: `false` if the model's raw output couldn't be parsed as JSON — in that case the structured fields are `null`/empty but `raw_model_output` is still returned, so callers always have something to fall back to (never a silent failure).
- `generation_seconds`: real measured latency for that single request (not batched) — included so integrators can plan for actual response times rather than assume.
- `report_id`: the persisted row's ID (new) — use it with `GET /api/v1/reports/{id}` to retrieve it later.
- Error responses: malformed/empty `report_text` → `422` (FastAPI/Pydantic validation). Inference failure → `500` with a generic message (`"Inference failed. See server logs."`) — internals are never leaked to the client.

### `GET /api/v1/reports` / `GET /api/v1/reports/{id}` / `GET /api/v1/dashboard`

List (filterable by `facility`), retrieve one report's full detail (original text, structured data, normalized stock items), or get aggregate dashboard stats. See `docs/DATABASE.md` for the underlying schema.

### `GET /api/v1/analytics/*`

`facilities`, `stock-items` (for populating filter dropdowns), `stock-trend?facility=...&item=...`, `status-trend?facility=...`, `reports-by-facility`, `commodity-frequency`, `stockouts`. Return `[]` rather than fabricated data when there isn't enough history yet.

### `POST /api/v1/batch`

Multipart CSV upload (`file`). Expects a `report`/`report_text`/`text` column. Reuses the same `/analyze` inference path per row; per-row failures are recorded, never abort the batch. Response: `{total_rows, successful, failed, processing_seconds, successes: [...], failures: [...]}`.

### `GET /api/v1/exports/{reports|stock-items}.{csv|xlsx}`

Optional `?facility=` filter. Report-level export keeps the full structured JSON as a column; stock-item-level export is one row per report×item.

### `POST /api/v1/transcribe`

Multipart audio upload (`file`). Returns `{"transcript": "..."}` via Whisper — **not N-ATLaS's own ASR**, see `docs/VOICE.md`. Callers are expected to let a human review/edit the transcript before passing it to `/analyze`; the API itself doesn't enforce that (the Streamlit voice page does, via its UI flow).

## For external integrators

This is the one integration point PS3 requires another developer be able to use: a single `POST` with free text in, structured JSON out, over plain HTTP, documented via standard OpenAPI (`/openapi.json`) — compatible with any HTTP client or auto-generated client library. No N-ATLaS-specific knowledge, Hugging Face account, or GPU access needed by the integrator; those are the service operator's concerns, not the caller's.

## Known limitations

- Single in-process model instance — no request queuing/batching. Concurrent requests will serialize behind the GPU. Fine for a demo/prototype; would need a request queue or multiple workers for real concurrent load (see `outline.md` §15's 1,000-user license cap, which is the relevant ceiling anyway).
- No authentication on the API itself in this prototype — add an API key/auth layer before any real deployment.
