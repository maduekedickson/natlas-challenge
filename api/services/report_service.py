"""DB CRUD for reports. Reuses NAtlasService's output dict as-is — no second
inference path, no reshaping of what the model actually produced."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from sqlalchemy import func
from sqlalchemy.orm import Session

from api.db.models import Report, StockObservation


def create_report(
    db: Session,
    *,
    report_text: str,
    analysis_result: dict,
    model_name: str,
    adapter_version: str | None,
    source: str = "text",
) -> Report:
    parsed_ok = analysis_result.get("parsed_ok", False)
    status = "success" if parsed_ok else "parse_error"

    report = Report(
        facility_name=analysis_result.get("facility_name"),
        reporter_name=analysis_result.get("reporter_name"),
        report_text=report_text,
        structured_data=json.dumps(analysis_result, ensure_ascii=False),
        services_available=json.dumps(analysis_result.get("services_available") or []),
        facility_status=analysis_result.get("facility_status"),
        model_name=model_name,
        adapter_version=adapter_version,
        source=source,
        status=status,
    )
    db.add(report)
    db.flush()

    for item in analysis_result.get("stock_items") or []:
        db.add(
            StockObservation(
                report_id=report.id,
                item=item.get("item"),
                quantity=item.get("quantity"),
                unit=item.get("unit"),
                status=item.get("status"),
            )
        )

    db.commit()
    db.refresh(report)
    return report


def list_reports(
    db: Session, *, limit: int = 50, offset: int = 0, facility: str | None = None
) -> list[Report]:
    query = db.query(Report).order_by(Report.created_at.desc())
    if facility:
        query = query.filter(Report.facility_name == facility)
    return query.offset(offset).limit(limit).all()


def get_report(db: Session, report_id: int) -> Report | None:
    return db.query(Report).filter(Report.id == report_id).first()


def report_to_summary_dict(report: Report) -> dict:
    return {
        "id": report.id,
        "facility_name": report.facility_name,
        "reporter_name": report.reporter_name,
        "facility_status": report.facility_status,
        "status": report.status,
        "source": report.source,
        "created_at": report.created_at,
    }


def report_to_detail_dict(report: Report) -> dict:
    base = report_to_summary_dict(report)
    base.update(
        {
            "report_text": report.report_text,
            "structured_data": json.loads(report.structured_data),
            "services_available": (
                json.loads(report.services_available) if report.services_available else []
            ),
            "model_name": report.model_name,
            "adapter_version": report.adapter_version,
            "stock_items": [
                {
                    "item": s.item,
                    "quantity": s.quantity,
                    "unit": s.unit,
                    "status": s.status,
                }
                for s in report.stock_observations
            ],
        }
    )
    return base


def dashboard_stats(db: Session) -> dict:
    total_reports = db.query(func.count(Report.id)).scalar() or 0
    facilities = (
        db.query(func.count(func.distinct(Report.facility_name)))
        .filter(Report.facility_name.isnot(None))
        .scalar()
        or 0
    )

    today_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    reports_today = (
        db.query(func.count(Report.id)).filter(Report.created_at >= today_start).scalar() or 0
    )

    open_facilities = (
        db.query(func.count(Report.id)).filter(Report.facility_status == "open").scalar() or 0
    )
    closed_facilities = (
        db.query(func.count(Report.id)).filter(Report.facility_status == "closed").scalar() or 0
    )

    stockouts = (
        db.query(func.count(StockObservation.id))
        .filter(StockObservation.status == "out_of_stock")
        .scalar()
        or 0
    )

    recent = db.query(Report).order_by(Report.created_at.desc()).limit(10).all()

    return {
        "total_reports": total_reports,
        "facilities_represented": facilities,
        "reports_today": reports_today,
        "open_facilities": open_facilities,
        "closed_facilities": closed_facilities,
        "stock_out_observations": stockouts,
        "recent_reports": [report_to_summary_dict(r) for r in recent],
    }
