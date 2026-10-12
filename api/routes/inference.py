import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from api.config import ADAPTER_VERSION
from api.db.session import get_session
from api.schemas.requests import AnalyzeRequest
from api.schemas.responses import AnalyzeResponse
from api.services import report_service
from api.services.natlas_service import NAtlasService

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/analyze", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest, db: Session = Depends(get_session)) -> AnalyzeResponse:
    service = NAtlasService()
    try:
        result = service.analyze(request.report_text)
    except Exception:
        logger.exception("Inference failed")
        raise HTTPException(status_code=500, detail="Inference failed. See server logs.")

    health = service.health()
    adapter_version = ADAPTER_VERSION if health.get("adapter_loaded") else None

    report_id = None
    try:
        report = report_service.create_report(
            db,
            report_text=request.report_text,
            analysis_result=result,
            model_name=health.get("model_name"),
            adapter_version=adapter_version,
            source="text",
        )
        report_id = report.id
    except Exception:
        logger.exception("Failed to persist report -- returning the analysis result anyway")
        db.rollback()

    result["report_id"] = report_id
    return AnalyzeResponse(**result)
