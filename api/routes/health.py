from fastapi import APIRouter

from api.schemas.responses import HealthResponse
from api.services.natlas_service import NAtlasService

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    service = NAtlasService()
    return HealthResponse(**service.health())
