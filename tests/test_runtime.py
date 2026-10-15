from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from onnxvoice import Qwen3VoiceState
from onnxvoice.types import InferenceResult

import qwen3synth.runtime as runtime_module
from qwen3synth.audio import write_wav
from qwen3synth.profile import PreparedVoice
from qwen3synth.runtime import Qwen3Runtime
from qwen3synth.types import (
    GenerationConfig,
    ReferenceVoice,
    SynthesisRequest,
    VoiceDesignRequest,
)


class _Tokenizer:
    instances: list[_Tokenizer] = []

    def __init__(self, _directory: str | Path) -> None:
        self.inputs: list[str] = []
        self.instances.append(self)

    def encode(self, text: str) -> list[int]:
        self.inputs.append(text)
        return [4, 5, 6]


class _Adapter:
    def __init__(self, root: Path, model_type: str = "base") -> None:
        self.model_type = model_type
        self.quality = "int4"
        self.config = SimpleNamespace(model_id="fixture/qwen3", max_position_embeddings=4096)
        self.installation = SimpleNamespace(
            id="base-0.6b",
            metadata={"source_revision": "a" * 40},
            artifact=lambda _role, *, component: SimpleNamespace(
                path=root / "tokenizer" / ("vocab.json" if component == "vocab" else component)
            ),
        )
        self.prepare_calls: list[dict[str, Any]] = []
        self.infer_calls: list[tuple[tuple[int, ...], dict[str, Any]]] = []
        self.closed = False

    def prepare_voice(
        self, audio: np.ndarray, *, sample_rate: int, reference_token_ids: tuple[int, ...]
    ) -> Qwen3VoiceState:
        self.prepare_calls.append(
            {
                "audio": audio.copy(),
                "sample_rate": sample_rate,
                "reference_token_ids": tuple(reference_token_ids),
            }
        )
        frames = (len(audio) + 1_919) // 1_920
        return Qwen3VoiceState(
            speaker_embedding=np.zeros((1, 4), dtype=np.float32),
            reference_codes=np.zeros((1, 16, frames), dtype=np.int64),
            reference_token_ids=reference_token_ids,
            sample_rate=sample_rate,
            reference_samples=len(audio),
            model_id=self.config.model_id,
        )

    def infer(self, token_ids: Any, **kwargs: Any) -> InferenceResult:
        self.infer_calls.append((tuple(token_ids), kwargs))
        return InferenceResult(
            audio=np.asarray([0.25, -0.25], dtype=np.float32),
            sample_rate=24_000,
            metadata={"providers_active": ("CPUExecutionProvider",)},
        )

    def close(self) -> None:
        self.closed = True


class _Manager:
    instance: _Manager | None = None

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.installs: list[tuple[str, dict[str, Any]]] = []
        self.opens: list[tuple[Any, dict[str, Any]]] = []
        self.adapter = _Adapter(Path("/fixture"))
        _Manager.instance = self

    def install(self, ref: str, **kwargs: Any) -> Any:
        self.installs.append((ref, kwargs))
        return SimpleNamespace(ref=ref)

    def open(self, installation: Any, **kwargs: Any) -> _Adapter:
        self.opens.append((installation, kwargs))
        return self.adapter


@pytest.fixture(autouse=True)
def _stub_tokenizer(monkeypatch: pytest.MonkeyPatch) -> None:
    _Tokenizer.instances.clear()
    monkeypatch.setattr(runtime_module, "ByteLevelBPETokenizer", _Tokenizer)


def test_from_pretrained_delegates_install_and_open_to_onnxvoice(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(runtime_module, "OnnxVoice", _Manager)

    runtime = Qwen3Runtime.from_pretrained(
        "base-0.6b",
        quality="fp32",
        provider="cuda",
        catalog=tmp_path / "catalog.json",
        cache_dir=tmp_path / "cache",
    )

    manager = _Manager.instance
    assert manager is not None
    assert manager.kwargs == {
        "cache_dir": tmp_path / "cache",
        "catalog_sources": {"qwen3": str(tmp_path / "catalog.json")},
    }
    assert manager.installs == [("qwen3:base-0.6b", {"quality": "fp32"})]
    assert manager.opens == [(SimpleNamespace(ref="qwen3:base-0.6b"), {"providers": "cuda"})]
    assert runtime.adapter is manager.adapter
    runtime.close()
    assert manager.adapter.closed


def test_voice_design_uses_frontend_chat_tokens_and_generation_policy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(runtime_module, "OnnxVoice", _Manager)
    manager = _Manager()
    manager.adapter = _Adapter(tmp_path, model_type="voice_design")
    runtime = Qwen3Runtime(adapter=manager.adapter)
    config = GenerationConfig(
        temperature=0.4, top_k=7, repetition_penalty=1.2, max_new_tokens=9, seed=17
    )

    result = runtime.design_voice(
        VoiceDesignRequest("design-1", "Hello there.", "Warm alto voice", "en"),
        config=config,
    )

    tokenizer = _Tokenizer.instances[-1]
    assert (
        tokenizer.inputs
        == [
            "<|im_start|>assistant\nHello there.<|im_end|>\n<|im_start|>assistant\n",
            "<|im_start|>user\nWarm alto voice<|im_end|>\n",
        ]
        * 2
    )
    token_ids, kwargs = manager.adapter.infer_calls[0]
    assert token_ids == (4, 5, 6)
    assert kwargs == {
        "instruction_token_ids": (4, 5, 6),
        "language": "english",
        "temperature": 0.4,
        "top_k": 7,
        "repetition_penalty": 1.2,
        "max_new_tokens": 9,
        "seed": 17,
    }
    assert result.samples.tolist() == [0.25, -0.25]
    assert result.metadata["kind"] == "voice_design"
    assert result.metadata["providers_active"] == ("CPUExecutionProvider",)


def test_prepare_profile_and_clone_route_reference_state_through_onnxvoice(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(runtime_module, "OnnxVoice", _Manager)
    manager = _Manager()
    manager.adapter = _Adapter(tmp_path)
    runtime = Qwen3Runtime(adapter=manager.adapter)
    reference_path = tmp_path / "reference.wav"
    write_wav(reference_path, np.zeros(16_000, dtype=np.float32), 16_000)

    prepared = runtime.prepare_voice(ReferenceVoice(reference_path, "spoken words", "en"))
    prepare_call = manager.adapter.prepare_calls[0]
    assert prepare_call["sample_rate"] == 24_000
    assert prepare_call["audio"].shape == (24_000,)
    assert prepare_call["reference_token_ids"] == (4, 5, 6)
    assert prepared.reference_samples == 24_000
    assert prepared.model_revision == "a" * 40

    profile_path = tmp_path / "voice.qvoice"
    prepared.save(profile_path)
    loaded = PreparedVoice.load(profile_path)
    assert loaded.reference_samples == 24_000
    assert loaded.reference_token_ids == (4, 5, 6)

    generation = GenerationConfig(temperature=0.6, top_k=11, seed=5)
    result = runtime.synthesize(
        SynthesisRequest("clone-1", "A new sentence.", "en"),
        voice=loaded,
        config=generation,
    )
    token_ids, kwargs = manager.adapter.infer_calls[0]
    assert token_ids == (4, 5, 6)
    state = kwargs.pop("voice_state")
    assert isinstance(state, Qwen3VoiceState)
    assert state.reference_token_ids == (4, 5, 6)
    assert state.reference_samples == 24_000
    assert kwargs == {
        "language": "english",
        "temperature": 0.6,
        "top_k": 11,
        "repetition_penalty": 1.05,
        "max_new_tokens": 2048,
        "seed": 5,
    }
    assert result.metadata["kind"] == "voice_clone"


def test_local_load_maps_bundle_files_to_onnxvoice_open_local(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[dict[str, Any]] = []
    adapter = _Adapter(tmp_path, model_type="voice_design")

    def open_local(**kwargs: Any) -> _Adapter:
        calls.append(kwargs)
        return adapter

    monkeypatch.setattr(runtime_module.OnnxVoice, "open_local", staticmethod(open_local))
    runtime = Qwen3Runtime.load(tmp_path, quality="fp32", provider="cpu")

    assert calls[0]["system"] == "qwen3"
    assert calls[0]["runtime"] == {
        "layout": "qwen3-wavekat-v1",
        "model_type": "voice_design",
        "quality": "fp32",
    }
    assert calls[0]["files"]["model:talker_prefill"] == tmp_path / "fp32" / "talker_prefill.onnx"
    assert calls[0]["providers"] == "cpu"
    assert runtime.adapter is adapter
