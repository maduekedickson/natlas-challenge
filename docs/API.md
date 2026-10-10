# API: N-ATLaS Health Report Extraction

FastAPI service exposing LoRA-adapted N-ATLaS for structured extraction from free-text health facility reports. Source: `api/`.

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
  "generation_seconds": 8.14
}
```

Field notes:
- `parsed_ok`: `false` if the model's raw output couldn't be parsed as JSON — in that case the structured fields are `null`/empty but `raw_model_output` is still returned, so callers always have something to fall back to (never a silent failure).
- `generation_seconds`: real measured latency for that single request (not batched) — included so integrators can plan for actual response times rather than assume.
- Error responses: malformed/empty `report_text` → `422` (FastAPI/Pydantic validation). Inference failure → `500` with a generic message (`"Inference failed. See server logs."`) — internals are never leaked to the client.

## For external integrators

This is the one integration point PS3 requires another developer be able to use: a single `POST` with free text in, structured JSON out, over plain HTTP, documented via standard OpenAPI (`/openapi.json`) — compatible with any HTTP client or auto-generated client library. No N-ATLaS-specific knowledge, Hugging Face account, or GPU access needed by the integrator; those are the service operator's concerns, not the caller's.

## Known limitations

- Single in-process model instance — no request queuing/batching. Concurrent requests will serialize behind the GPU. Fine for a demo/prototype; would need a request queue or multiple workers for real concurrent load (see `outline.md` §15's 1,000-user license cap, which is the relevant ceiling anyway).
- No authentication on the API itself in this prototype — add an API key/auth layer before any real deployment.
