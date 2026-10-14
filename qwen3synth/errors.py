class Qwen3SynthError(Exception):
    """Base exception for public qwen3synth failures."""


class CatalogError(Qwen3SynthError):
    pass


class ModelNotInstalledError(Qwen3SynthError):
    pass


class UnsupportedModelCapabilityError(Qwen3SynthError):
    pass


class SynthesisInputTooLongError(Qwen3SynthError):
    pass


class InvalidReferenceAudioError(Qwen3SynthError):
    pass


class InvalidReferenceTranscriptError(Qwen3SynthError):
    pass


class IncompatibleVoiceProfileError(Qwen3SynthError):
    pass


class InvalidModelLayoutError(Qwen3SynthError):
    pass


class ModelGenerationError(Qwen3SynthError):
    pass
