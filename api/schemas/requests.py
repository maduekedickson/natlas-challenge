from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    report_text: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        description="Free-text health facility report to extract structured data from.",
    )
