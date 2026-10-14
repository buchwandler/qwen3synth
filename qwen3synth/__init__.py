from __future__ import annotations

try:
    from ._version import __version__
except ImportError:  # source tree before setuptools_scm generated _version.py
    __version__ = "0.0.0"

from .errors import (
    CatalogError,
    IncompatibleVoiceProfileError,
    InvalidModelLayoutError,
    InvalidReferenceAudioError,
    InvalidReferenceTranscriptError,
    ModelGenerationError,
    ModelNotInstalledError,
    Qwen3SynthError,
    SynthesisInputTooLongError,
    UnsupportedModelCapabilityError,
)
from .profile import PreparedVoice
from .runtime import Qwen3Runtime
from .types import (
    GenerationConfig,
    ReferenceVoice,
    RequestMeasure,
    SynthesisRequest,
    SynthesisResult,
    VoiceDesignRequest,
)

__all__ = [
    "__version__",
    "Qwen3Runtime",
    "GenerationConfig",
    "ReferenceVoice",
    "PreparedVoice",
    "RequestMeasure",
    "SynthesisRequest",
    "SynthesisResult",
    "VoiceDesignRequest",
    "Qwen3SynthError",
    "CatalogError",
    "ModelNotInstalledError",
    "UnsupportedModelCapabilityError",
    "SynthesisInputTooLongError",
    "InvalidReferenceAudioError",
    "InvalidReferenceTranscriptError",
    "IncompatibleVoiceProfileError",
    "InvalidModelLayoutError",
    "ModelGenerationError",
]
