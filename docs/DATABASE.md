# Database

SQLite via SQLAlchemy — chosen per `CLAUDE.md`'s own stated default ("SQLite initially / PostgreSQL if required") and because it needs zero infrastructure: identical setup on a laptop, in Colab, or later on Hugging Face Spaces.

File location: `DATABASE_PATH` env var (default `data/app.db`). See the Colab limitation note below before relying on history across sessions.

## Schema

```
reports
  id                   INTEGER PRIMARY KEY
  facility_name        TEXT, indexed
  reporter_name        TEXT
  report_text          TEXT NOT NULL      -- original free text, always kept
  structured_data       TEXT NOT NULL      -- full raw JSON string (audit trail)
  services_available    TEXT                -- JSON-encoded list
  facility_status       TEXT                -- 'open' | 'closed' | null
  model_name            TEXT NOT NULL
  adapter_version       TEXT                -- e.g. 'health-lora-v1', null if base model
  source                TEXT NOT NULL      -- 'text' | 'voice' | 'batch'
  status                TEXT NOT NULL      -- 'success' | 'parse_error'
  created_at            DATETIME NOT NULL
  updated_at            DATETIME NOT NULL

stock_observations
  id          INTEGER PRIMARY KEY
  report_id   INTEGER NOT NULL  -> reports.id
  item        TEXT NOT NULL, indexed
  quantity    INTEGER
  unit        TEXT
  status      TEXT                -- 'in_stock' | 'low_stock' | 'out_of_stock'
  created_at  DATETIME NOT NULL
```

## Why stock items are normalized, not left in the JSON blob

`structured_data` keeps the full raw extraction result (nothing is lost), but stock items are *also* written into their own `stock_observations` rows. This is what makes trend queries ("ORS quantity at Asaba PHC over time") plain SQL instead of parsing JSON in Python on every request.

## Why there's no separate `report_date` field

The synthetic report text doesn't reliably contain a parseable explicit date (checked against `training/prepare_dataset.py`'s templates). Rather than invent a field that would mostly be null, `created_at` (server-assigned, real) is the timestamp actually used for all trend/history features.

## Known limitation: ephemeral Colab storage

If the backend runs inside an interactive Colab session (as it has throughout this project), the SQLite file lives in `/content` and is **wiped on every Colab disconnect** — the same issue that affected the LoRA adapter earlier. For a live demo this is fine (a fresh demo doesn't need pre-existing history — submit a few reports live and the dashboard/trends populate immediately). For persistence across sessions, point `DATABASE_PATH` at a mounted Google Drive folder:

```python
from google.colab import drive
drive.mount('/content/drive')
# then set DATABASE_PATH=/content/drive/MyDrive/natlas-app.db before starting the API
```

A real fix (managed Postgres, e.g. on Hugging Face Spaces) is out of scope for this phase and noted as future work.

## Migrations

None — `init_db()` (called at FastAPI startup) just does `Base.metadata.create_all()`, which creates missing tables but never alters existing ones. Fine for a prototype with a stable schema; a real deployment would want Alembic.
