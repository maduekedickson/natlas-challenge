"""Centralized environment configuration for the new platform features.

Existing modules (natlas_service.py, train.py, run_benchmark.py) keep reading
their own env vars directly, unchanged, to avoid touching working code. This
module is for the new database/export/voice features only.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

DATABASE_PATH = os.environ.get("DATABASE_PATH", "data/app.db")
ADAPTER_VERSION = os.environ.get("ADAPTER_VERSION", "health-lora-v1")
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "base")
