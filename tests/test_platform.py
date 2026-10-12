"""Tests for the new platform features (DB persistence, dashboard, analytics,
batch processing, exports).

These mock NAtlasService rather than loading the real model -- deliberately.
Per docs/IMPLEMENTATION_PLAN.md: real N-ATLaS inference requires a GPU and has
been verified manually via Colab throughout this project (see
docs/EVALUATION.md, docs/API.md). These are unit/integration tests for
everything that doesn't need the model: persistence, batch CSV handling,
export generation, and analytics queries -- consistent with "use mocks only
for unit tests where appropriate" from the feature-expansion spec.
"""

from __future__ import annotations

import io
import os
import tempfile

# Must happen before importing anything from `api` -- api.db.session binds
# its SQLite engine to DATABASE_PATH at import time.
_TMP_DB = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
os.environ["DATABASE_PATH"] = _TMP_DB.name
os.environ.setdefault("HF_TOKEN", "test-token-not-used")  # natlas_service is mocked below

import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.services.natlas_service import NAtlasService


class FakeNAtlasService:
    """Stands in for the real model in tests that aren't about inference itself."""

    def analyze(self, report_text: str, max_new_tokens: int = 256) -> dict:
        return {
            "raw_model_output": (
                '{"facility_name": "Test PHC", "reporter_name": "Tester", '
                '"stock_items": [{"item": "ORS", "quantity": 10, "unit": "sachets", '
                '"status": "in_stock"}], "services_available": ["antenatal care"], '
                '"facility_status": "open"}'
            ),
            "parsed_ok": True,
            "generation_seconds": 0.01,
            "facility_name": "Test PHC",
            "reporter_name": "Tester",
            "stock_items": [{"item": "ORS", "quantity": 10, "unit": "sachets", "status": "in_stock"}],
            "services_available": ["antenatal care"],
            "facility_status": "open",
        }

    def health(self) -> dict:
        return {
            "status": "ok",
            "model_name": "NCAIR1/N-ATLaS",
            "model_loaded": True,
            "device": "cpu",
            "adapter_loaded": True,
        }


@pytest.fixture
def fake_model():
    NAtlasService._instance = FakeNAtlasService()
    yield
    NAtlasService._instance = None


@pytest.fixture
def client(fake_model):
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    assert r.json()["adapter_loaded"] is True


def test_analyze_persists_report(client):
    r = client.post("/api/v1/analyze", json={"report_text": "we have 10 sachets of ORS"})
    assert r.status_code == 200
    body = r.json()
    assert body["facility_name"] == "Test PHC"
    assert body["report_id"] is not None

    r2 = client.get(f"/api/v1/reports/{body['report_id']}")
    assert r2.status_code == 200
    detail = r2.json()
    assert detail["facility_name"] == "Test PHC"
    assert detail["stock_items"][0]["item"] == "ORS"


def test_invalid_analyze_request(client):
    r = client.post("/api/v1/analyze", json={"report_text": ""})
    assert r.status_code == 422


def test_analyze_handles_non_numeric_quantity_without_crashing(client):
    """Regression test: a real report ("...one malaria patient...") made the
    heavily-fine-tuned model emit a word quantity instead of a digit, which
    crashed response validation with a raw 500. NAtlasService now sanitizes
    stock_items before returning them -- this confirms the fix end-to-end."""

    class FakeWithWordQuantity(FakeNAtlasService):
        def analyze(self, report_text: str, max_new_tokens: int = 256) -> dict:
            return {
                "raw_model_output": '{"stock_items": [{"item": "malaria patient", "quantity": "one"}]}',
                "parsed_ok": True,
                "generation_seconds": 0.01,
                "facility_name": "FMCA Sabah",
                "reporter_name": "Dixie",
                "stock_items": [{"item": "malaria patient", "quantity": None, "unit": None, "status": None}],
                "services_available": [],
                "facility_status": None,
            }

    NAtlasService._instance = FakeWithWordQuantity()
    r = client.post("/api/v1/analyze", json={"report_text": "we have one malaria patient here"})
    assert r.status_code == 200
    assert r.json()["stock_items"][0]["quantity"] is None


def test_report_not_found(client):
    r = client.get("/api/v1/reports/999999")
    assert r.status_code == 404


def test_dashboard_reflects_submitted_reports(client):
    client.post("/api/v1/analyze", json={"report_text": "report one"})
    client.post("/api/v1/analyze", json={"report_text": "report two"})
    r = client.get("/api/v1/dashboard")
    assert r.status_code == 200
    stats = r.json()
    assert stats["total_reports"] >= 2
    assert stats["open_facilities"] >= 2


def test_analytics_endpoints(client):
    client.post("/api/v1/analyze", json={"report_text": "some report"})

    facilities = client.get("/api/v1/analytics/facilities").json()
    assert "Test PHC" in facilities

    items = client.get("/api/v1/analytics/stock-items").json()
    assert "ORS" in items

    trend = client.get(
        "/api/v1/analytics/stock-trend", params={"facility": "Test PHC", "item": "ORS"}
    ).json()
    assert len(trend) >= 1
    assert trend[0]["quantity"] == 10


def test_batch_valid_csv(client):
    csv_content = 'facility,reporter,report\nA,John,"we have stock"\nB,Mary,"we are out"\n'
    files = {"file": ("batch.csv", io.BytesIO(csv_content.encode()), "text/csv")}
    r = client.post("/api/v1/batch", files=files)
    assert r.status_code == 200
    body = r.json()
    assert body["total_rows"] == 2
    assert body["successful"] == 2
    assert body["failed"] == 0


def test_batch_handles_cp1252_encoded_csv(client):
    """Regression test: a CSV saved by Excel on Windows (cp1252, with 'smart
    quotes' that aren't valid UTF-8) failed with a raw UnicodeDecodeError
    instead of being read. Real example byte: 0x93 (left smart quote in
    cp1252) is not a valid UTF-8 start byte."""
    csv_text = 'report\n"we have “some” stock"\n'
    csv_bytes = csv_text.encode("cp1252")
    files = {"file": ("batch.csv", io.BytesIO(csv_bytes), "text/csv")}
    r = client.post("/api/v1/batch", files=files)
    assert r.status_code == 200
    body = r.json()
    assert "error" not in body
    assert body["total_rows"] == 1
    assert body["successful"] == 1


def test_batch_accepts_xlsx_upload(client):
    """Regression test: uploading an actual Excel file previously hit the
    CSV text parser (an .xlsx is a binary zip-based format, not text), which
    failed with a confusing decode error. Filename extension now dispatches
    to pd.read_excel() instead."""
    import pandas as pd

    df = pd.DataFrame({"facility": ["A"], "reporter": ["John"], "report": ["we have stock"]})
    buf = io.BytesIO()
    df.to_excel(buf, index=False, engine="openpyxl")
    buf.seek(0)

    files = {
        "file": (
            "batch.xlsx",
            buf,
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    }
    r = client.post("/api/v1/batch", files=files)
    assert r.status_code == 200
    body = r.json()
    assert "error" not in body
    assert body["total_rows"] == 1
    assert body["successful"] == 1


def test_batch_missing_report_column(client):
    csv_content = "facility,reporter\nA,John\n"
    files = {"file": ("batch.csv", io.BytesIO(csv_content.encode()), "text/csv")}
    r = client.post("/api/v1/batch", files=files)
    assert r.status_code == 200
    assert "error" in r.json()


def test_batch_empty_file(client):
    files = {"file": ("empty.csv", io.BytesIO(b""), "text/csv")}
    r = client.post("/api/v1/batch", files=files)
    assert r.status_code == 200
    assert "error" in r.json()


def test_batch_handles_malformed_and_empty_rows_without_crashing(client):
    csv_content = 'report\n"a real report"\n""\n'
    files = {"file": ("batch.csv", io.BytesIO(csv_content.encode()), "text/csv")}
    r = client.post("/api/v1/batch", files=files)
    assert r.status_code == 200
    body = r.json()
    assert body["successful"] >= 1
    assert body["successful"] + body["failed"] == body["total_rows"]


def test_exports_csv_and_xlsx(client):
    client.post("/api/v1/analyze", json={"report_text": "exportable report"})

    r_csv = client.get("/api/v1/exports/reports.csv")
    assert r_csv.status_code == 200
    assert b"facility" in r_csv.content

    r_xlsx = client.get("/api/v1/exports/reports.xlsx")
    assert r_xlsx.status_code == 200
    assert len(r_xlsx.content) > 0

    r_items_csv = client.get("/api/v1/exports/stock-items.csv")
    assert r_items_csv.status_code == 200
    assert b"item" in r_items_csv.content
