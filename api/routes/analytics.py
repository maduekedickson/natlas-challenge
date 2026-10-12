from datetime import datetime

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from api.db.session import get_session
from api.services import analytics_service

router = APIRouter()


@router.get("/analytics/facilities")
def facilities(db: Session = Depends(get_session)):
    return analytics_service.list_facilities(db)


@router.get("/analytics/stock-items")
def stock_items(db: Session = Depends(get_session)):
    return analytics_service.list_stock_items(db)


@router.get("/analytics/stock-trend")
def stock_trend(
    facility: str,
    item: str,
    start: datetime | None = None,
    end: datetime | None = None,
    db: Session = Depends(get_session),
):
    return analytics_service.stock_trend(db, facility=facility, item=item, start=start, end=end)


@router.get("/analytics/status-trend")
def status_trend(
    facility: str,
    start: datetime | None = None,
    end: datetime | None = None,
    db: Session = Depends(get_session),
):
    return analytics_service.facility_status_trend(db, facility=facility, start=start, end=end)


@router.get("/analytics/reports-by-facility")
def reports_by_facility(db: Session = Depends(get_session)):
    return analytics_service.reports_by_facility(db)


@router.get("/analytics/commodity-frequency")
def commodity_frequency(db: Session = Depends(get_session)):
    return analytics_service.commodity_frequency(db)


@router.get("/analytics/stockouts")
def stockouts(db: Session = Depends(get_session)):
    return analytics_service.stockout_occurrences(db)
