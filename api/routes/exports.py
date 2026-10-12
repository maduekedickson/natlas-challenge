from fastapi import APIRouter, Depends
from fastapi.responses import Response
from sqlalchemy.orm import Session

from api.db.session import get_session
from api.services import export_service

router = APIRouter()

XLSX_MEDIA_TYPE = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


@router.get("/exports/reports.csv")
def export_reports_csv(facility: str | None = None, db: Session = Depends(get_session)):
    df = export_service.report_level_dataframe(db, facility)
    content = export_service.to_csv_bytes(df)
    return Response(
        content=content,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=reports.csv"},
    )


@router.get("/exports/reports.xlsx")
def export_reports_xlsx(facility: str | None = None, db: Session = Depends(get_session)):
    df = export_service.report_level_dataframe(db, facility)
    content = export_service.to_xlsx_bytes(df)
    return Response(
        content=content,
        media_type=XLSX_MEDIA_TYPE,
        headers={"Content-Disposition": "attachment; filename=reports.xlsx"},
    )


@router.get("/exports/stock-items.csv")
def export_stock_items_csv(facility: str | None = None, db: Session = Depends(get_session)):
    df = export_service.stock_item_level_dataframe(db, facility)
    content = export_service.to_csv_bytes(df)
    return Response(
        content=content,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=stock_items.csv"},
    )


@router.get("/exports/stock-items.xlsx")
def export_stock_items_xlsx(facility: str | None = None, db: Session = Depends(get_session)):
    df = export_service.stock_item_level_dataframe(db, facility)
    content = export_service.to_xlsx_bytes(df)
    return Response(
        content=content,
        media_type=XLSX_MEDIA_TYPE,
        headers={"Content-Disposition": "attachment; filename=stock_items.xlsx"},
    )
