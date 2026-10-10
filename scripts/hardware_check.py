"""Detect and report the local hardware/software environment for N-ATLaS feasibility.

Usage:
    python scripts/hardware_check.py
"""

from __future__ import annotations

import json
import logging
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import psutil
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "docs" / "hardware_report.json"


def get_cpu_info() -> dict:
    return {
        "processor": platform.processor(),
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cores": psutil.cpu_count(logical=True),
        "max_frequency_mhz": getattr(psutil.cpu_freq(), "max", None),
    }


def get_memory_info() -> dict:
    vm = psutil.virtual_memory()
    return {
        "total_gb": round(vm.total / 1024**3, 2),
        "available_gb": round(vm.available / 1024**3, 2),
        "used_percent": vm.percent,
    }


def get_disk_info() -> dict:
    du = psutil.disk_usage(str(Path(__file__).resolve().parent.parent))
    return {
        "total_gb": round(du.total / 1024**3, 2),
        "free_gb": round(du.free / 1024**3, 2),
        "used_percent": du.percent,
    }


def get_gpu_info() -> dict:
    if not torch.cuda.is_available():
        return {"cuda_available": False, "devices": []}

    devices = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        devices.append(
            {
                "index": i,
                "name": props.name,
                "total_memory_gb": round(props.total_memory / 1024**3, 2),
            }
        )
    return {
        "cuda_available": True,
        "cuda_version": torch.version.cuda,
        "devices": devices,
    }


def get_software_info() -> dict:
    versions = {"torch": torch.__version__}
    try:
        import transformers

        versions["transformers"] = transformers.__version__
    except ImportError:
        versions["transformers"] = None
    try:
        import accelerate

        versions["accelerate"] = accelerate.__version__
    except ImportError:
        versions["accelerate"] = None

    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        **versions,
    }


def build_report() -> dict:
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "software": get_software_info(),
        "cpu": get_cpu_info(),
        "memory": get_memory_info(),
        "disk": get_disk_info(),
        "gpu": get_gpu_info(),
    }


def main() -> None:
    report = build_report()

    logger.info("Software: %s", report["software"])
    logger.info("CPU: %s", report["cpu"])
    logger.info("Memory: %s", report["memory"])
    logger.info("Disk: %s", report["disk"])
    logger.info("GPU: %s", report["gpu"])

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(report, indent=2))
    logger.info("Report written to %s", OUTPUT_PATH)

    if not report["gpu"]["cuda_available"]:
        logger.warning(
            "No CUDA-capable GPU detected. Large-model inference/fine-tuning will be "
            "CPU-bound and constrained by system RAM (%.2f GB total).",
            report["memory"]["total_gb"],
        )


if __name__ == "__main__":
    sys.exit(main())
