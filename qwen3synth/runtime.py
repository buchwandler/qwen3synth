from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import numpy as np
from onnxvoice import NotInstalledError, OnnxVoice, Qwen3VoiceState
from onnxvoice.errors import CapabilityError, RuntimeContractError

from .audio import TARGET_SAMPLE_RATE, read_wav, resample_linear
from .errors import (
    IncompatibleVoiceProfileError,
    InvalidModelLayoutError,
    InvalidReferenceAudioError,
    InvalidReferenceTranscriptError,
    ModelGenerationError,
    ModelNotInstalledError,
    UnsupportedModelCapabilityError,
)
from .frontend import encode_instruction, encode_reference, encode_target, normalize_language
from .profile import PreparedVoice
from .tokenizer import ByteLevelBPETokenizer
from .types import (
    GenerationConfig,
    ReferenceVoice,
    RequestMeasure,
    SynthesisRequest,
    SynthesisResult,
    VoiceDesignRequest,
)

LAYOUT = "qwen3-wavekat-v1"
_MAX_REFERENCE_SECONDS = 10.0
_MIN_REFERENCE_SECONDS = 0.5
_CODEC_SAMPLES_PER_FRAME = 1_920
_TOKENIZER_FILES = {
    "tokenizer_json": "tokenizer.json",
    "vocab": "vocab.json",
    "merges": "merges.txt",
    "tokenizer_config": "tokenizer_config.json",
    "special_tokens": "special_tokens_map.json",
    "added_tokens": "added_tokens.json",
}
_EMBEDDINGS = (
    "text_embedding",
    "text_projection_fc1_weight",
    "text_projection_fc1_bias",
    "text_projection_fc2_weight",
    "text_projection_fc2_bias",
    "talker_codec_embedding",
    *(f"cp_codec_embedding_{index}" for index in range(15)),
)
_GENERATION_GRAPHS = ("talker_prefill", "talker_decode", "code_predictor", "vocoder")
_ENCODER_GRAPHS = ("speaker_encoder", "tokenizer_encoder")


class Qwen3Runtime:
    """Qwen3Synth frontend backed by OnnxVoice's WaveKat runtime."""

    def __init__(
        self,
        model_dir: str | Path | None = None,
        *,
        quality: str = "int4",
        provider: str = "cpu",
        adapter: Any | None = None,
        manager: OnnxVoice | None = None,
    ) -> None:
        if quality not in {"fp32", "int4"}:
            raise ValueError("quality must be 'fp32' or 'int4'")
        if adapter is None:
            if model_dir is None:
                raise ValueError("model_dir is required when no OnnxVoice adapter is supplied")
            adapter = self._open_local(Path(model_dir), quality=quality, provider=provider)
        self.model_dir = Path(model_dir) if model_dir is not None else None
        self.quality = quality
        self.provider = provider
        self.adapter = adapter
        self._onnxvoice_manager = manager
        self.model_type = str(adapter.model_type)
        self.config = adapter.config
        self.tokenizer = self._load_tokenizer(adapter)

    @staticmethod
    def _load_tokenizer(adapter: Any) -> ByteLevelBPETokenizer:
        vocab_path = adapter.installation.artifact("tokenizer", component="vocab").path
        return ByteLevelBPETokenizer(vocab_path.parent)

    @staticmethod
    def _open_local(directory: Path, *, quality: str, provider: str) -> Any:
        files: dict[str, Path] = {"config": directory / "config.json"}
        for component, filename in _TOKENIZER_FILES.items():
            files[f"tokenizer:{component}"] = directory / "tokenizer" / filename
        for component in _EMBEDDINGS:
            files[f"embedding:{component}"] = directory / "embeddings" / f"{component}.npy"
        for component in _GENERATION_GRAPHS:
            graph = directory / quality / f"{component}.onnx"
            files[f"model:{component}"] = graph
            files[f"model_data:{component}"] = Path(str(graph) + ".data")
        has_encoder_files = any(
            (directory / f"{component}.onnx").exists()
            or Path(str(directory / f"{component}.onnx") + ".data").exists()
            for component in _ENCODER_GRAPHS
        )
        if has_encoder_files:
            for component in _ENCODER_GRAPHS:
                graph = directory / f"{component}.onnx"
                files[f"model:{component}"] = graph
                files[f"model_data:{component}"] = Path(str(graph) + ".data")
        model_type = "base" if has_encoder_files else "voice_design"
        try:
            return OnnxVoice.open_local(
                system="qwen3",
                files=files,
                runtime={"layout": LAYOUT, "model_type": model_type, "quality": quality},
                sample_rate=TARGET_SAMPLE_RATE,
                providers=provider,
            )
        except CapabilityError as exc:
            raise UnsupportedModelCapabilityError(str(exc)) from exc
        except RuntimeContractError as exc:
            raise InvalidModelLayoutError(str(exc)) from exc

    @staticmethod
    def _manager(
        *,
        cache_dir: str | Path | None = None,
        catalog: str | Path | None = None,
        offline: bool = False,
    ) -> OnnxVoice:
        if cache_dir is None:
            cache_dir = os.environ.get("QWEN3SYNTH_CACHE")
        sources = {"qwen3": str(catalog)} if catalog is not None else None
        return OnnxVoice(cache_dir=cache_dir, catalog_sources=sources, offline=offline)

    @staticmethod
    def _open_managed(manager: OnnxVoice, installation: Any, provider: str) -> Any:
        try:
            return manager.open(installation, providers=provider)
        except NotInstalledError as exc:
            raise ModelNotInstalledError(str(exc)) from exc
        except CapabilityError as exc:
            raise UnsupportedModelCapabilityError(str(exc)) from exc
        except RuntimeContractError as exc:
            raise InvalidModelLayoutError(str(exc)) from exc

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
        manager = cls._manager(cache_dir=cache_dir, catalog=catalog)
        ref = bundle if bundle.startswith("qwen3:") else f"qwen3:{bundle}"
        installation = manager.install(ref, quality=quality)
        adapter = cls._open_managed(manager, installation, provider)
        return cls(quality=quality, provider=provider, adapter=adapter, manager=manager)

    @classmethod
    def from_installed(
        cls,
        bundle: str,
        *,
        quality: str = "int4",
        provider: str = "cpu",
        catalog: str | Path | None = None,
        cache_dir: str | Path | None = None,
    ) -> Qwen3Runtime:
        """Open an installed bundle without network or asset acquisition."""
        manager = cls._manager(cache_dir=cache_dir, catalog=catalog, offline=True)
        ref = bundle if bundle.startswith("qwen3:") else f"qwen3:{bundle}"
        try:
            installation = manager.resolve(ref, quality=quality)
        except NotInstalledError as exc:
            raise ModelNotInstalledError(str(exc)) from exc
        adapter = cls._open_managed(manager, installation, provider)
        return cls(quality=quality, provider=provider, adapter=adapter, manager=manager)

    def __enter__(self) -> Qwen3Runtime:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        self.adapter.close()

    @property
    def backend(self) -> Any:
        """Deprecated compatibility alias for the OnnxVoice adapter."""
        return self.adapter

    @property
    def model_id(self) -> str:
        value = getattr(self.config, "model_id", "")
        return str(value or self.adapter.installation.id)

    @property
    def model_revision(self) -> str | None:
        value = self.adapter.installation.metadata.get("source_revision")
        return value if isinstance(value, str) else None

    def measure_text(self, text: str, *, description: str | None = None) -> RequestMeasure:
        amount = len(encode_target(text, self.tokenizer))
        if description:
            amount += len(encode_instruction(description, self.tokenizer))
        maximum = int(getattr(self.config, "max_position_embeddings", None) or 32768)
        return RequestMeasure(amount=amount, maximum=maximum, fits=amount <= maximum)

    def measure_request(self, request: SynthesisRequest) -> RequestMeasure:
        return self.measure_text(request.text)

    def design_voice(
        self, request: VoiceDesignRequest, *, config: GenerationConfig | None = None
    ) -> SynthesisResult:
        if self.model_type != "voice_design":
            raise UnsupportedModelCapabilityError("design_voice requires a voice_design bundle")
        measure = self.measure_text(request.text, description=request.description)
        self._check_measure(measure)
        generation = config or GenerationConfig()
        inference = self._infer(
            encode_target(request.text, self.tokenizer),
            instruction_token_ids=encode_instruction(request.description, self.tokenizer),
            language=normalize_language(request.language),
            temperature=generation.temperature,
            top_k=generation.top_k,
            repetition_penalty=generation.repetition_penalty,
            max_new_tokens=generation.max_new_tokens,
            seed=generation.seed,
        )
        return SynthesisResult(
            request.id,
            inference.audio,
            inference.sample_rate,
            request.language,
            {
                **dict(inference.metadata),
                "model_id": self.model_id,
                "quality": self.quality,
                "layout": LAYOUT,
                "kind": "voice_design",
            },
        )

    def _infer(self, target_token_ids: tuple[int, ...], **kwargs: Any) -> Any:
        try:
            return self.adapter.infer(target_token_ids, **kwargs)
        except CapabilityError as exc:
            raise UnsupportedModelCapabilityError(str(exc)) from exc
        except RuntimeContractError as exc:
            raise ModelGenerationError(str(exc)) from exc

    def prepare_voice(self, reference: ReferenceVoice) -> PreparedVoice:
        if self.model_type != "base":
            raise UnsupportedModelCapabilityError("prepare_voice requires a base bundle")
        samples, source_rate, raw_hash = read_wav(reference.audio)
        samples = resample_linear(samples, source_rate, TARGET_SAMPLE_RATE)
        duration = len(samples) / TARGET_SAMPLE_RATE
        if duration < _MIN_REFERENCE_SECONDS:
            raise InvalidReferenceAudioError(
                "reference audio is too short; use a clean 3-10 second excerpt"
            )
        if duration > _MAX_REFERENCE_SECONDS:
            raise InvalidReferenceAudioError(
                "WaveKat Base ONNX accepts at most 10 seconds; select a matching excerpt and transcript"
            )
        prepared_hash = hashlib.sha256(np.asarray(samples, dtype="<f4").tobytes()).hexdigest()
        try:
            reference_ids = encode_reference(reference.transcript, self.tokenizer)
        except ValueError as exc:
            raise InvalidReferenceTranscriptError(
                "reference transcript could not be encoded by the model tokenizer"
            ) from exc
        try:
            state = self.adapter.prepare_voice(
                samples,
                sample_rate=TARGET_SAMPLE_RATE,
                reference_token_ids=reference_ids,
            )
        except CapabilityError as exc:
            raise UnsupportedModelCapabilityError(str(exc)) from exc
        except RuntimeContractError as exc:
            raise ModelGenerationError(str(exc)) from exc
        return PreparedVoice(
            speaker_embedding=state.speaker_embedding,
            reference_codes=state.reference_codes,
            transcript=reference.transcript,
            language=reference.language,
            audio_sha256=raw_hash,
            prepared_audio_sha256=prepared_hash,
            model_id=state.model_id,
            model_revision=self.model_revision,
            source_id=reference.source_id,
            reference_samples=state.reference_samples,
            reference_token_ids=state.reference_token_ids,
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
        if voice.model_id != self.model_id:
            raise IncompatibleVoiceProfileError(
                f"profile was prepared for {voice.model_id}, loaded model is {self.model_id}"
            )
        if (
            voice.model_revision is not None
            and self.model_revision is not None
            and voice.model_revision != self.model_revision
        ):
            raise IncompatibleVoiceProfileError(
                f"profile model revision {voice.model_revision!r} is incompatible with "
                f"loaded model revision {self.model_revision!r}"
            )
        measure = self.measure_request(request)
        self._check_measure(measure)
        generation = config or GenerationConfig()
        reference_ids = voice.reference_token_ids or encode_reference(
            voice.transcript, self.tokenizer
        )
        reference_samples = voice.reference_samples or min(
            voice.reference_codes.shape[2] * _CODEC_SAMPLES_PER_FRAME,
            int(_MAX_REFERENCE_SECONDS * TARGET_SAMPLE_RATE),
        )
        state = Qwen3VoiceState(
            speaker_embedding=voice.speaker_embedding,
            reference_codes=voice.reference_codes,
            reference_token_ids=reference_ids,
            sample_rate=TARGET_SAMPLE_RATE,
            reference_samples=reference_samples,
            model_id=voice.model_id,
            layout=voice.layout,
        )
        inference = self._infer(
            encode_target(request.text, self.tokenizer),
            voice_state=state,
            language=normalize_language(request.language),
            temperature=generation.temperature,
            top_k=generation.top_k,
            repetition_penalty=generation.repetition_penalty,
            max_new_tokens=generation.max_new_tokens,
            seed=generation.seed,
        )
        return SynthesisResult(
            request.id,
            inference.audio,
            inference.sample_rate,
            request.language,
            {
                **dict(inference.metadata),
                "model_id": self.model_id,
                "quality": self.quality,
                "layout": LAYOUT,
                "kind": "voice_clone",
                "reference_audio_sha256": voice.audio_sha256,
            },
        )

    @staticmethod
    def _check_measure(measure: RequestMeasure) -> None:
        if not measure.fits:
            from .errors import SynthesisInputTooLongError

            raise SynthesisInputTooLongError(
                f"request uses {measure.amount} tokens; limit is {measure.maximum}"
            )
