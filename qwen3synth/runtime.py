from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .audio import TARGET_SAMPLE_RATE, read_wav, resample_linear
from .backend import LAYOUT, WaveKatBackend
from .catalog import find_installed
from .errors import (
    IncompatibleVoiceProfileError,
    InvalidReferenceAudioError,
    UnsupportedModelCapabilityError,
)
from .profile import PreparedVoice
from .types import (
    GenerationConfig,
    ReferenceVoice,
    RequestMeasure,
    SynthesisRequest,
    SynthesisResult,
    VoiceDesignRequest,
)


class Qwen3Runtime:
    def __init__(self, model_dir: str | Path, *, quality: str = "int4", provider: str = "cpu"):
        self.model_dir = Path(model_dir)
        self.quality = quality
        self.provider = provider
        self.backend = WaveKatBackend(self.model_dir, variant=quality, provider=provider)
        self._install = self._read_install_manifest()

    def _read_install_manifest(self) -> dict:
        p = self.model_dir / ".qwen3synth-install.json"
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
        return {}

    @classmethod
    def load(
        cls, directory: str | Path, *, quality: str = "int4", provider: str = "cpu"
    ) -> Qwen3Runtime:
        return cls(directory, quality=quality, provider=provider)

    @classmethod
    def from_pretrained(
        cls,
        bundle: str,
        *,
        quality: str = "int4",
        provider: str = "cpu",
        catalog: str | Path | None = None,
        cache_dir: str | Path | None = None,
    ) -> Qwen3Runtime:
        directory = find_installed(
            bundle,
            quality,
            catalog_source=catalog,
            cache_dir=Path(cache_dir) if cache_dir else None,
        )
        return cls(directory, quality=quality, provider=provider)

    def __enter__(self) -> Qwen3Runtime:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        self.backend._sessions.clear()

    @property
    def model_type(self) -> str:
        return self.backend.model_type

    def measure_text(self, text: str, *, description: str | None = None) -> RequestMeasure:
        amount = self.backend.token_count(text, description=description)
        maximum = int(self.backend.config.get("max_position_embeddings", 32768))
        return RequestMeasure(amount=amount, maximum=maximum, fits=amount <= maximum)

    def measure_request(self, request: SynthesisRequest) -> RequestMeasure:
        return self.measure_text(request.text)

    def design_voice(
        self, request: VoiceDesignRequest, *, config: GenerationConfig | None = None
    ) -> SynthesisResult:
        if self.model_type != "voice_design":
            raise UnsupportedModelCapabilityError("design_voice requires a voice_design bundle")
        measure = self.measure_text(request.text, description=request.description)
        if not measure.fits:
            from .errors import SynthesisInputTooLongError

            raise SynthesisInputTooLongError(
                f"request uses {measure.amount} tokens; limit is {measure.maximum}"
            )
        gen = config or GenerationConfig()
        wav = self.backend.generate_voice_design(
            request.text, request.description, request.language, gen
        )
        return SynthesisResult(
            request.id,
            wav,
            TARGET_SAMPLE_RATE,
            request.language,
            {
                "model_id": self.backend.model_id,
                "quality": self.quality,
                "layout": LAYOUT,
                "kind": "voice_design",
            },
        )

    def prepare_voice(self, reference: ReferenceVoice) -> PreparedVoice:
        if self.model_type != "base":
            raise UnsupportedModelCapabilityError("prepare_voice requires a base bundle")
        samples, sr, raw_hash = read_wav(reference.audio)
        samples = resample_linear(samples, sr, TARGET_SAMPLE_RATE)
        duration = len(samples) / TARGET_SAMPLE_RATE
        if duration < 0.5:
            raise InvalidReferenceAudioError(
                "reference audio is too short; use a clean 3-10 second excerpt"
            )
        if duration > 10.0:
            raise InvalidReferenceAudioError(
                "WaveKat Base ONNX accepts at most 10 seconds in this MVP; select a matching excerpt and transcript"
            )
        prepared_hash = hashlib.sha256(np.asarray(samples, dtype="<f4").tobytes()).hexdigest()
        speaker, codes = self.backend.encode_reference(samples)
        return PreparedVoice(
            speaker,
            codes,
            reference.transcript,
            reference.language,
            raw_hash,
            prepared_hash,
            self.backend.model_id,
            self._install.get("revision"),
            source_id=reference.source_id,
        )

    def synthesize(
        self,
        request: SynthesisRequest,
        *,
        voice: PreparedVoice,
        config: GenerationConfig | None = None,
    ) -> SynthesisResult:
        if self.model_type != "base":
            raise UnsupportedModelCapabilityError(
                "synthesize with PreparedVoice requires a base bundle"
            )
        if voice.layout != LAYOUT:
            raise IncompatibleVoiceProfileError(
                f"profile layout {voice.layout!r} is incompatible with {LAYOUT}"
            )
        if voice.model_id != self.backend.model_id:
            raise IncompatibleVoiceProfileError(
                f"profile was prepared for {voice.model_id}, loaded model is {self.backend.model_id}"
            )
        measure = self.measure_request(request)
        if not measure.fits:
            from .errors import SynthesisInputTooLongError

            raise SynthesisInputTooLongError(
                f"request uses {measure.amount} tokens; limit is {measure.maximum}"
            )
        gen = config or GenerationConfig()
        wav = self.backend.generate_clone(
            request.text,
            voice.transcript,
            request.language,
            voice.speaker_embedding,
            voice.reference_codes,
            gen,
        )
        return SynthesisResult(
            request.id,
            wav,
            TARGET_SAMPLE_RATE,
            request.language,
            {
                "model_id": self.backend.model_id,
                "quality": self.quality,
                "layout": LAYOUT,
                "kind": "voice_clone",
                "reference_audio_sha256": voice.audio_sha256,
            },
        )
