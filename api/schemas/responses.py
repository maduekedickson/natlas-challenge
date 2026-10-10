from pydantic import BaseModel


class StockItem(BaseModel):
    item: str
    quantity: int | None = None
    unit: str | None = None
    status: str | None = None


class AnalyzeResponse(BaseModel):
    facility_name: str | None = None
    reporter_name: str | None = None
    stock_items: list[StockItem] = []
    services_available: list[str] = []
    facility_status: str | None = None
    raw_model_output: str
    parsed_ok: bool
    generation_seconds: float


class HealthResponse(BaseModel):
    status: str
    model_name: str
    model_loaded: bool
    device: str
    adapter_loaded: bool
