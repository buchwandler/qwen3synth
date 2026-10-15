from __future__ import annotations

try:
    from ._version import __version__
except ImportError:  # source tree before setuptools_scm generated _version.py
    __version__ = "0.0.0"

from .api_contract import REQUEST_API_VERSION, RequestApiContract, request_api_contract
from .discovery import DiscoveredBundle, discover_bundles
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
from .identity import runtime_identity
from .profile import PreparedVoice, VoiceProfileInfo, inspect_voice_profile
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
    "REQUEST_API_VERSION",
    "RequestApiContract",
    "request_api_contract",
    "runtime_identity",
    "DiscoveredBundle",
    "discover_bundles",
    "Qwen3Runtime",
    "GenerationConfig",
    "ReferenceVoice",
    "PreparedVoice",
    "VoiceProfileInfo",
    "inspect_voice_profile",
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
