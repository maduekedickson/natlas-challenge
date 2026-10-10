import logging

from fastapi import APIRouter, HTTPException

from api.schemas.requests import AnalyzeRequest
from api.schemas.responses import AnalyzeResponse
from api.services.natlas_service import NAtlasService

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/analyze", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    service = NAtlasService()
    try:
        result = service.analyze(request.report_text)
    except Exception:
        logger.exception("Inference failed")
        raise HTTPException(status_code=500, detail="Inference failed. See server logs.")
    return AnalyzeResponse(**result)
