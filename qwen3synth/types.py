from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .audio import write_wav
from .errors import InvalidReferenceTranscriptError


@dataclass(frozen=True)
class GenerationConfig:
    temperature: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.05
    max_new_tokens: int = 2048
    seed: int | None = None

    def __post_init__(self) -> None:
        if not 0.0 < self.temperature <= 5.0:
            raise ValueError("temperature must be in (0, 5]")
        if self.top_k < 0:
            raise ValueError("top_k must be >= 0")
        if self.repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be > 0")
        if not 1 <= self.max_new_tokens <= 8192:
            raise ValueError("max_new_tokens must be between 1 and 8192")


@dataclass(frozen=True)
class SynthesisRequest:
    id: str
    text: str
    language: str = "en"

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("request id must not be empty")
        if not self.text.strip():
            raise ValueError("text must not be empty")


@dataclass(frozen=True)
class VoiceDesignRequest:
    id: str
    text: str
    description: str
    language: str = "en"

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("request id must not be empty")
        if not self.text.strip():
            raise ValueError("text must not be empty")
        if not self.description.strip():
            raise ValueError("voice description must not be empty")


@dataclass(frozen=True)
class ReferenceVoice:
    audio: str | Path
    transcript: str
    language: str = "en"
    source_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.transcript, str) or not self.transcript.strip():
            raise InvalidReferenceTranscriptError("reference transcript must not be empty")


@dataclass(frozen=True)
class RequestMeasure:
    amount: int
    maximum: int
    unit: str = "tokens"
    fits: bool = True


@dataclass(frozen=True)
class SynthesisResult:
    request_id: str
    samples: np.ndarray
    sample_rate: int
    language: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def duration_seconds(self) -> float:
        return float(len(self.samples) / self.sample_rate)

    def save_wav(self, path: str | Path) -> None:
        write_wav(path, self.samples, self.sample_rate)
