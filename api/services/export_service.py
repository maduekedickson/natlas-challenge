"""CSV/Excel export, report-level and stock-item-level. The report-level
export keeps the full structured JSON as a column so nothing is lossily
flattened; the stock-item-level export is normalized for spreadsheet pivoting."""

from __future__ import annotations

import io

import pandas as pd
from sqlalchemy.orm import Session

from api.db.models import Report


def _reports_query(db: Session, facility: str | None = None) -> list[Report]:
    query = db.query(Report).order_by(Report.created_at.asc())
    if facility:
        query = query.filter(Report.facility_name == facility)
    return query.all()


def report_level_dataframe(db: Session, facility: str | None = None) -> pd.DataFrame:
    reports = _reports_query(db, facility)
    rows = [
        {
            "report_id": r.id,
            "facility": r.facility_name,
            "reporter": r.reporter_name,
            "facility_status": r.facility_status,
            "services_available": r.services_available,
            "structured_data_json": r.structured_data,
            "model_name": r.model_name,
            "adapter_version": r.adapter_version,
            "source": r.source,
            "status": r.status,
            "created_at": r.created_at,
        }
        for r in reports
    ]
    return pd.DataFrame(rows)


def stock_item_level_dataframe(db: Session, facility: str | None = None) -> pd.DataFrame:
    reports = _reports_query(db, facility)
    rows = [
        {
            "report_id": r.id,
            "facility": r.facility_name,
            "created_at": r.created_at,
            "item": s.item,
            "quantity": s.quantity,
            "unit": s.unit,
            "status": s.status,
        }
        for r in reports
        for s in r.stock_observations
    ]
    return pd.DataFrame(rows)


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


def to_xlsx_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="data")
    return buf.getvalue()
