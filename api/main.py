"""FastAPI service exposing N-ATLaS-based health-report structured extraction.

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

from api.routes import health, inference
from api.services.natlas_service import NAtlasService

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading N-ATLaS at startup (this can take ~1 minute on first run) ...")
    NAtlasService()  # forces load now, not on the first request
    logger.info("N-ATLaS loaded. API ready.")
    yield


app = FastAPI(
    title="N-ATLaS Health Report Extraction API",
    description=(
        "Extracts structured facility/stock/service information from free-text "
        "health facility reports using a LoRA-adapted N-ATLaS. "
        "N-ATLaS is an initiative of the Federal Ministry of Communications, "
        "Innovation and Digital Economy, powered by Awarri Technologies."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(health.router, prefix="/api/v1", tags=["health"])
app.include_router(inference.router, prefix="/api/v1", tags=["inference"])
