"""Batch CSV processing. Reuses NAtlasService.analyze() per row -- no second
inference implementation. Per-row failures are recorded, never crash the batch."""

from __future__ import annotations

import io
import time

import pandas as pd
from sqlalchemy.orm import Session

from api.services import report_service
from api.services.natlas_service import NAtlasService

REPORT_COLUMN_CANDIDATES = ["report", "report_text", "text"]

# Excel on Windows commonly exports CSV as cp1252 (or adds a UTF-8 BOM), not
# plain UTF-8 -- e.g. "smart quotes" decode as 0x93/0x94 in cp1252, not valid
# UTF-8. Try the common ones in order rather than failing on the first user
# who opens a template in Excel and saves it.
CSV_ENCODINGS_TO_TRY = ["utf-8-sig", "utf-8", "cp1252", "latin-1"]


def find_report_column(columns: list[str]) -> str | None:
    lowered = {str(c).lower(): c for c in columns}
    for candidate in REPORT_COLUMN_CANDIDATES:
        if candidate in lowered:
            return lowered[candidate]
    return None


def _format_stock_items(items: list[dict] | None) -> str:
    if not items:
        return ""
    parts = []
    for it in items:
        qty, unit, status = it.get("quantity"), it.get("unit"), it.get("status")
        qty_str = f"{qty} {unit}".strip() if qty is not None else "unknown qty"
        parts.append(f"{it.get('item')}: {qty_str} ({status})" if status else f"{it.get('item')}: {qty_str}")
    return "; ".join(parts)


def _format_services(services: list[str] | None) -> str:
    return ", ".join(services) if services else ""


def _read_csv_robust(file_bytes: bytes) -> pd.DataFrame:
    last_error: Exception | None = None
    for encoding in CSV_ENCODINGS_TO_TRY:
        try:
            return pd.read_csv(io.BytesIO(file_bytes), encoding=encoding)
        except UnicodeError as e:
            last_error = e
            continue
    raise last_error  # latin-1 can decode any byte, so this should be unreachable


def _read_uploaded_file(file_bytes: bytes, filename: str | None) -> pd.DataFrame:
    """Dispatch on file extension. An .xlsx is a binary (zip-based) format --
    no amount of text-encoding fallback can read it as CSV text, which is
    what previously made Excel uploads fail with a confusing decode error."""
    ext = (filename or "").rsplit(".", 1)[-1].lower() if filename and "." in filename else ""
    if ext in ("xlsx", "xls"):
        return pd.read_excel(io.BytesIO(file_bytes))
    return _read_csv_robust(file_bytes)


def process_csv(
    db: Session, file_bytes: bytes, *, adapter_version: str | None, filename: str | None = None
) -> dict:
    t0 = time.perf_counter()
    try:
        df = _read_uploaded_file(file_bytes, filename)
    except Exception as e:
        return {"error": f"Could not parse file: {e}"}

    if df.empty:
        return {"error": "CSV file is empty."}

    report_col = find_report_column(list(df.columns))
    if report_col is None:
        return {
            "error": (
                "No report-text column found. Expected one of "
                f"{REPORT_COLUMN_CANDIDATES}. Found columns: {list(df.columns)}"
            )
        }

    service = NAtlasService()
    health = service.health()
    model_name = health.get("model_name")
    effective_adapter_version = adapter_version if health.get("adapter_loaded") else None

    successes: list[dict] = []
    failures: list[dict] = []

    for idx, row in df.iterrows():
        report_text = row.get(report_col)
        if pd.isna(report_text) or not str(report_text).strip():
            failures.append({"row": int(idx), "error": "Empty report text"})
            continue
        try:
            result = service.analyze(str(report_text))
            report = report_service.create_report(
                db,
                report_text=str(report_text),
                analysis_result=result,
                model_name=model_name,
                adapter_version=effective_adapter_version,
                source="batch",
            )
            successes.append(
                {
                    "row": int(idx),
                    "report_id": report.id,
                    "facility": report.facility_name,
                    "facility_status": result.get("facility_status"),
                    "stock_items": _format_stock_items(result.get("stock_items")),
                    "services_available": _format_services(result.get("services_available")),
                }
            )
        except Exception as e:
            failures.append({"row": int(idx), "error": str(e)})

    total_s = time.perf_counter() - t0

    return {
        "total_rows": int(len(df)),
        "successful": len(successes),
        "failed": len(failures),
        "processing_seconds": round(total_s, 2),
        "successes": successes,
        "failures": failures,
    }
