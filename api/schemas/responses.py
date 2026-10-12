from datetime import datetime

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
    report_id: int | None = None


class HealthResponse(BaseModel):
    status: str
    model_name: str
    model_loaded: bool
    device: str
    adapter_loaded: bool


class ReportSummaryResponse(BaseModel):
    id: int
    facility_name: str | None = None
    reporter_name: str | None = None
    facility_status: str | None = None
    status: str
    source: str
    created_at: datetime


class ReportDetailResponse(ReportSummaryResponse):
    report_text: str
    structured_data: dict
    services_available: list[str] = []
    stock_items: list[StockItem] = []
    model_name: str
    adapter_version: str | None = None


class DashboardStatsResponse(BaseModel):
    total_reports: int
    facilities_represented: int
    reports_today: int
    open_facilities: int
    closed_facilities: int
    stock_out_observations: int
    recent_reports: list[ReportSummaryResponse] = []
