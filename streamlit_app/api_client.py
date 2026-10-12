"""Centralized HTTP client for the Streamlit app -> FastAPI backend.

Every page goes through this module rather than calling `requests` directly,
so the base URL and endpoint paths live in one place.

Note: imported as a flat module (`import api_client`), not
`from streamlit_app import api_client` -- when Streamlit runs app.py (or a
page) directly, it adds that script's own directory to sys.path, not the
parent, so a package-qualified import would fail the same way
`evaluation/run_benchmark.py`'s did earlier in this project.
"""

from __future__ import annotations

import os

import requests

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")
ANALYZE_TIMEOUT = 120


def health() -> dict:
    r = requests.get(f"{API_BASE_URL}/api/v1/health", timeout=10)
    r.raise_for_status()
    return r.json()


def analyze(report_text: str) -> dict:
    r = requests.post(
        f"{API_BASE_URL}/api/v1/analyze", json={"report_text": report_text}, timeout=ANALYZE_TIMEOUT
    )
    r.raise_for_status()
    return r.json()


def transcribe(audio_bytes: bytes, filename: str = "audio.wav") -> str:
    files = {"file": (filename, audio_bytes)}
    r = requests.post(f"{API_BASE_URL}/api/v1/transcribe", files=files, timeout=ANALYZE_TIMEOUT)
    r.raise_for_status()
    return r.json()["transcript"]


def dashboard_stats() -> dict:
    r = requests.get(f"{API_BASE_URL}/api/v1/dashboard", timeout=30)
    r.raise_for_status()
    return r.json()


def list_reports(limit: int = 50, offset: int = 0, facility: str | None = None) -> list[dict]:
    params = {"limit": limit, "offset": offset}
    if facility:
        params["facility"] = facility
    r = requests.get(f"{API_BASE_URL}/api/v1/reports", params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def get_report(report_id: int) -> dict:
    r = requests.get(f"{API_BASE_URL}/api/v1/reports/{report_id}", timeout=30)
    r.raise_for_status()
    return r.json()


def list_facilities() -> list[str]:
    r = requests.get(f"{API_BASE_URL}/api/v1/analytics/facilities", timeout=30)
    r.raise_for_status()
    return r.json()


def list_stock_items() -> list[str]:
    r = requests.get(f"{API_BASE_URL}/api/v1/analytics/stock-items", timeout=30)
    r.raise_for_status()
    return r.json()


def stock_trend(facility: str, item: str) -> list[dict]:
    r = requests.get(
        f"{API_BASE_URL}/api/v1/analytics/stock-trend",
        params={"facility": facility, "item": item},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


def status_trend(facility: str) -> list[dict]:
    r = requests.get(
        f"{API_BASE_URL}/api/v1/analytics/status-trend", params={"facility": facility}, timeout=30
    )
    r.raise_for_status()
    return r.json()


def reports_by_facility() -> list[dict]:
    r = requests.get(f"{API_BASE_URL}/api/v1/analytics/reports-by-facility", timeout=30)
    r.raise_for_status()
    return r.json()


def commodity_frequency() -> list[dict]:
    r = requests.get(f"{API_BASE_URL}/api/v1/analytics/commodity-frequency", timeout=30)
    r.raise_for_status()
    return r.json()


def stockouts() -> list[dict]:
    r = requests.get(f"{API_BASE_URL}/api/v1/analytics/stockouts", timeout=30)
    r.raise_for_status()
    return r.json()


def batch_process(file_bytes: bytes, filename: str = "batch.csv") -> dict:
    """filename's extension (.csv vs .xlsx/.xls) determines how the server
    parses it -- content-type is left for requests/FastAPI to infer."""
    files = {"file": (filename, file_bytes)}
    r = requests.post(f"{API_BASE_URL}/api/v1/batch", files=files, timeout=600)
    r.raise_for_status()
    return r.json()


def export_url(kind: str, fmt: str, facility: str | None = None) -> str:
    """kind: 'reports' | 'stock-items'; fmt: 'csv' | 'xlsx'"""
    url = f"{API_BASE_URL}/api/v1/exports/{kind}.{fmt}"
    if facility:
        url += f"?facility={facility}"
    return url
