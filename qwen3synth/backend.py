"""Compatibility shim for the retired Qwen3Synth-owned ONNX backend."""

from __future__ import annotations

from typing import Any

from .runtime import LAYOUT, normalize_language

__all__ = ["LAYOUT", "normalize_language", "WaveKatBackend"]


class WaveKatBackend:
    """Removed runtime; create :class:`Qwen3Runtime` to use OnnxVoice instead."""

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("WaveKatBackend was removed; use Qwen3Runtime backed by OnnxVoice")
