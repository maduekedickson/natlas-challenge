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


def find_report_column(columns: list[str]) -> str | None:
    lowered = {str(c).lower(): c for c in columns}
    for candidate in REPORT_COLUMN_CANDIDATES:
        if candidate in lowered:
            return lowered[candidate]
    return None


def process_csv(db: Session, file_bytes: bytes, *, adapter_version: str | None) -> dict:
    t0 = time.perf_counter()
    try:
        df = pd.read_csv(io.BytesIO(file_bytes))
    except Exception as e:
        return {"error": f"Could not parse CSV: {e}"}

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
                {"row": int(idx), "report_id": report.id, "facility": report.facility_name}
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
