from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import numpy as np
from onnxvoice import OnnxVoice, Qwen3VoiceState

from .audio import TARGET_SAMPLE_RATE, read_wav, resample_linear
from .errors import (
    IncompatibleVoiceProfileError,
    InvalidReferenceAudioError,
    UnsupportedModelCapabilityError,
)
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


def normalize_language(language: str) -> str:
    aliases = {
        "en": "english",
        "zh": "chinese",
        "ja": "japanese",
        "ko": "korean",
        "de": "german",
        "fr": "french",
        "es": "spanish",
        "it": "italian",
        "pt": "portuguese",
        "ru": "russian",
    }
    key = language.strip().lower()
    return aliases.get(key, key)


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
        return OnnxVoice.open_local(
            system="qwen3",
            files=files,
            runtime={"layout": LAYOUT, "model_type": model_type, "quality": quality},
            sample_rate=TARGET_SAMPLE_RATE,
            providers=provider,
        )

    @staticmethod
    @staticmethod
    def _manager(
        *, cache_dir: str | Path | None = None, catalog: str | Path | None = None
    ) -> OnnxVoice:
        if cache_dir is None:
            cache_dir = os.environ.get("QWEN3SYNTH_CACHE")
        sources = {"qwen3": str(catalog)} if catalog is not None else None
        return OnnxVoice(cache_dir=cache_dir, catalog_sources=sources)

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
        adapter = manager.open(installation, providers=provider)
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

    def _encode_assistant(self, text: str, *, open_assistant: bool) -> tuple[int, ...]:
        suffix = "<|im_start|>assistant\n" if open_assistant else ""
        wrapped = f"<|im_start|>assistant\n{text}<|im_end|>\n{suffix}"
        return tuple(self.tokenizer.encode(wrapped))

    def _encode_instruction(self, description: str) -> tuple[int, ...]:
        return tuple(self.tokenizer.encode(f"<|im_start|>user\n{description}<|im_end|>\n"))

    def measure_text(self, text: str, *, description: str | None = None) -> RequestMeasure:
        amount = len(self._encode_assistant(text, open_assistant=True))
        if description:
            amount += len(self._encode_instruction(description))
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
        inference = self.adapter.infer(
            self._encode_assistant(request.text, open_assistant=True),
            instruction_token_ids=self._encode_instruction(request.description),
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
        reference_ids = tuple(
            self.tokenizer.encode(f"<|im_start|>assistant\n{reference.transcript}<|im_end|>\n")
        )
        state = self.adapter.prepare_voice(
            samples,
            sample_rate=TARGET_SAMPLE_RATE,
            reference_token_ids=reference_ids,
        )
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
        measure = self.measure_request(request)
        self._check_measure(measure)
        generation = config or GenerationConfig()
        reference_ids = voice.reference_token_ids or tuple(
            self.tokenizer.encode(f"<|im_start|>assistant\n{voice.transcript}<|im_end|>\n")
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
        inference = self.adapter.infer(
            self._encode_assistant(request.text, open_assistant=True),
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
