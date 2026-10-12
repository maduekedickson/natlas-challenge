"""Trend and aggregation queries over stored reports. Returns [] / empty
results when there isn't enough data yet -- never fabricates history."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import func
from sqlalchemy.orm import Session

from api.db.models import Report, StockObservation


def list_facilities(db: Session) -> list[str]:
    rows = db.query(Report.facility_name).filter(Report.facility_name.isnot(None)).distinct().all()
    return sorted({r[0] for r in rows})


def list_stock_items(db: Session) -> list[str]:
    rows = db.query(StockObservation.item).distinct().all()
    return sorted({r[0] for r in rows})


def stock_trend(
    db: Session,
    *,
    facility: str,
    item: str,
    start: datetime | None = None,
    end: datetime | None = None,
) -> list[dict]:
    query = (
        db.query(Report.created_at, StockObservation.quantity, StockObservation.status)
        .join(StockObservation, StockObservation.report_id == Report.id)
        .filter(Report.facility_name == facility, StockObservation.item == item)
        .order_by(Report.created_at.asc())
    )
    if start:
        query = query.filter(Report.created_at >= start)
    if end:
        query = query.filter(Report.created_at <= end)
    return [{"date": r[0], "quantity": r[1], "status": r[2]} for r in query.all()]


def facility_status_trend(
    db: Session, *, facility: str, start: datetime | None = None, end: datetime | None = None
) -> list[dict]:
    query = (
        db.query(Report.created_at, Report.facility_status)
        .filter(Report.facility_name == facility)
        .order_by(Report.created_at.asc())
    )
    if start:
        query = query.filter(Report.created_at >= start)
    if end:
        query = query.filter(Report.created_at <= end)
    return [{"date": r[0], "status": r[1]} for r in query.all()]


def reports_by_facility(db: Session) -> list[dict]:
    rows = (
        db.query(Report.facility_name, func.count(Report.id))
        .filter(Report.facility_name.isnot(None))
        .group_by(Report.facility_name)
        .order_by(func.count(Report.id).desc())
        .all()
    )
    return [{"facility": r[0], "count": r[1]} for r in rows]


def commodity_frequency(db: Session) -> list[dict]:
    rows = (
        db.query(StockObservation.item, func.count(StockObservation.id))
        .group_by(StockObservation.item)
        .order_by(func.count(StockObservation.id).desc())
        .all()
    )
    return [{"item": r[0], "count": r[1]} for r in rows]


def stockout_occurrences(db: Session) -> list[dict]:
    rows = (
        db.query(StockObservation.item, func.count(StockObservation.id))
        .filter(StockObservation.status == "out_of_stock")
        .group_by(StockObservation.item)
        .order_by(func.count(StockObservation.id).desc())
        .all()
    )
    return [{"item": r[0], "count": r[1]} for r in rows]
