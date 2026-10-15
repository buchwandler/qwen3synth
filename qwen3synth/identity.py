from __future__ import annotations

from collections.abc import Mapping

import onnxvoice

from . import __version__
from .api_contract import REQUEST_API_VERSION


def runtime_identity() -> Mapping[str, object]:
    return {
        "engine": "qwen3synth",
        "engine_version": __version__,
        "request_api_version": REQUEST_API_VERSION,
        "onnxvoice_version": onnxvoice.__version__,
        "layouts": ("qwen3-wavekat-v1",),
        "sample_rate": 24_000,
    }
