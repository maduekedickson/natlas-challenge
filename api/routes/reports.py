from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from api.db.session import get_session
from api.schemas.responses import DashboardStatsResponse, ReportDetailResponse, ReportSummaryResponse
from api.services import report_service

router = APIRouter()


@router.get("/reports", response_model=list[ReportSummaryResponse])
def list_reports(
    limit: int = 50,
    offset: int = 0,
    facility: str | None = None,
    db: Session = Depends(get_session),
):
    reports = report_service.list_reports(db, limit=limit, offset=offset, facility=facility)
    return [report_service.report_to_summary_dict(r) for r in reports]


@router.get("/reports/{report_id}", response_model=ReportDetailResponse)
def get_report(report_id: int, db: Session = Depends(get_session)):
    report = report_service.get_report(db, report_id)
    if report is None:
        raise HTTPException(status_code=404, detail="Report not found")
    return report_service.report_to_detail_dict(report)


@router.get("/dashboard", response_model=DashboardStatsResponse)
def dashboard(db: Session = Depends(get_session)):
    return report_service.dashboard_stats(db)
