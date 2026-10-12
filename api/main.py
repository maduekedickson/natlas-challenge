"""FastAPI service exposing N-ATLaS-based health-report structured extraction,
plus report persistence, analytics, batch processing, exports, and voice
transcription.

Run:
    python -m uvicorn api.main:app --host 0.0.0.0 --port 8000

Requires a GPU-backed environment with enough VRAM to hold N-ATLaS (see
docs/FEASIBILITY.md) and HF_TOKEN set (see .env.example). Not intended to
run on a CPU-only / low-RAM machine.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.db.session import init_db
from api.routes import analytics, batch, exports, health, inference, reports, voice
from api.services.natlas_service import NAtlasService

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing database ...")
    init_db()
    logger.info("Loading N-ATLaS at startup (this can take ~1 minute on first run) ...")
    NAtlasService()  # forces load now, not on the first request
    logger.info("N-ATLaS loaded. API ready.")
    yield


app = FastAPI(
    title="N-ATLaS Health Intelligence Platform API",
    description=(
        "Extracts structured facility/stock/service information from free-text "
        "health facility reports using a LoRA-adapted N-ATLaS, with report "
        "persistence, trend analytics, batch processing, and exports. "
        "N-ATLaS is an initiative of the Federal Ministry of Communications, "
        "Innovation and Digital Economy, powered by Awarri Technologies. "
        "This is an information-extraction and operational reporting tool, "
        "not a diagnostic system."
    ),
    version="0.2.0",
    lifespan=lifespan,
)

app.include_router(health.router, prefix="/api/v1", tags=["health"])
app.include_router(inference.router, prefix="/api/v1", tags=["inference"])
app.include_router(reports.router, prefix="/api/v1", tags=["reports"])
app.include_router(analytics.router, prefix="/api/v1", tags=["analytics"])
app.include_router(batch.router, prefix="/api/v1", tags=["batch"])
app.include_router(exports.router, prefix="/api/v1", tags=["exports"])
app.include_router(voice.router, prefix="/api/v1", tags=["voice"])
