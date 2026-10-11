"""Lazy optional-runtime discovery; runnable as ``python -m hearing.device``."""

import importlib
import json
import os
import platform
from typing import Literal

Device = Literal["cuda", "mps", "cpu"]


def _optional_import(name: str):
    try:
        return importlib.import_module(name)
    except ImportError:
        return None


def pick() -> Device:
    """Use an explicit override, otherwise CUDA, MPS, then CPU."""
    override = os.environ.get("HEARING_DEVICE")
    if override is not None:
        if override not in {"cuda", "mps", "cpu"}:
            raise ValueError("HEARING_DEVICE must be cuda, mps, or cpu")
        return override
    torch = _optional_import("torch")
    if torch is not None:
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    return "cpu"


def describe() -> dict:
    """Return JSON-safe runtime and machine metadata, without loading models."""
    device = pick()
    torch = _optional_import("torch")
    ort = _optional_import("onnxruntime")
    return {
        "device": device,
        "torch": torch.__version__ if torch is not None else None,
        "onnxruntime_providers": ort.get_available_providers()
        if ort is not None
        else None,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "hostname": platform.node(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
    }


if __name__ == "__main__":
    print(json.dumps(describe(), indent=2))
