from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from onnxvoice import Qwen3VoiceState
from onnxvoice.types import InferenceResult

import qwen3synth.__main__ as cli
import qwen3synth.runtime as runtime_module
from qwen3synth.audio import write_wav
from qwen3synth.profile import PreparedVoice


class _Tokenizer:
    def __init__(self, _directory: str | Path) -> None:
        pass

    def encode(self, _text: str) -> list[int]:
        return [1, 2, 3]


class _Adapter:
    def __init__(self, root: Path, model_type: str) -> None:
        self.model_type = model_type
        self.config = SimpleNamespace(model_id="fixture/qwen3", max_position_embeddings=4096)
        self.installation = SimpleNamespace(
            id="fixture-bundle",
            metadata={"source_revision": "a" * 40},
            artifact=lambda _role, *, component: SimpleNamespace(
                path=root / "tokenizer" / f"{component}.json"
            ),
        )

    def prepare_voice(
        self, audio: np.ndarray, *, sample_rate: int, reference_token_ids: tuple[int, ...]
    ) -> Qwen3VoiceState:
        frames = (len(audio) + 1_919) // 1_920
        return Qwen3VoiceState(
            speaker_embedding=np.zeros((1, 4), dtype=np.float32),
            reference_codes=np.zeros((1, 16, frames), dtype=np.int64),
            reference_token_ids=reference_token_ids,
            sample_rate=sample_rate,
            reference_samples=len(audio),
            model_id=self.config.model_id,
        )

    def infer(self, _token_ids: Any, **_kwargs: Any) -> InferenceResult:
        return InferenceResult(
            audio=np.asarray([0.25, -0.25], dtype=np.float32),
            sample_rate=24_000,
            metadata={},
        )

    def close(self) -> None:
        pass


class _Manager:
    instances: list[_Manager] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.installs: list[tuple[str, dict[str, Any]]] = []
        self.resolves: list[tuple[str, dict[str, Any]]] = []
        self.opens: list[tuple[Any, dict[str, Any]]] = []
        self.instances.append(self)

    def install(self, ref: str, **kwargs: Any) -> Any:
        self.installs.append((ref, kwargs))
        return SimpleNamespace(ref=ref)

    def resolve(self, ref: str, **kwargs: Any) -> Any:
        self.resolves.append((ref, kwargs))
        return SimpleNamespace(ref=ref, quality=kwargs["quality"])

    def open(self, installation: Any, **kwargs: Any) -> _Adapter:
        self.opens.append((installation, kwargs))
        model_type = "voice_design" if "voice-design" in installation.ref else "base"
        return _Adapter(Path("/fixture"), model_type)


@pytest.fixture
def managers(monkeypatch: pytest.MonkeyPatch) -> list[_Manager]:
    _Manager.instances.clear()
    monkeypatch.setattr(runtime_module, "OnnxVoice", _Manager)
    monkeypatch.setattr(runtime_module, "ByteLevelBPETokenizer", _Tokenizer)
    return _Manager.instances


def test_cli_voice_design_does_not_install(managers: list[_Manager], tmp_path: Path) -> None:
    output = tmp_path / "design.wav"
    assert (
        cli.main(
            [
                "voice",
                "design",
                "--text",
                "Hello.",
                "--prompt",
                "Warm narrator",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    manager = managers[-1]
    assert manager.kwargs["offline"] is True
    assert manager.resolves == [("qwen3:voice-design-1.7b", {"quality": "int4"})]
    assert manager.installs == []
    assert output.is_file()


def test_cli_voice_prepare_does_not_install(managers: list[_Manager], tmp_path: Path) -> None:
    reference = tmp_path / "reference.wav"
    output = tmp_path / "voice.qvoice"
    write_wav(reference, np.zeros(16_000, dtype=np.float32), 16_000)

    assert (
        cli.main(
            [
                "voice",
                "prepare",
                "--reference",
                str(reference),
                "--transcript",
                "Reference words.",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    manager = managers[-1]
    assert manager.kwargs["offline"] is True
    assert manager.resolves == [("qwen3:base-0.6b", {"quality": "int4"})]
    assert manager.installs == []
    assert output.is_file()


def test_cli_synthesize_does_not_install(managers: list[_Manager], tmp_path: Path) -> None:
    profile = tmp_path / "voice.qvoice"
    PreparedVoice(
        speaker_embedding=np.zeros((1, 4), dtype=np.float32),
        reference_codes=np.zeros((1, 16, 1), dtype=np.int64),
        transcript="Reference words.",
        language="en",
        audio_sha256="0" * 64,
        prepared_audio_sha256="1" * 64,
        model_id="fixture/qwen3",
        model_revision="a" * 40,
        reference_samples=1_920,
        reference_token_ids=(1, 2, 3),
    ).save(profile)
    output = tmp_path / "clone.wav"

    assert (
        cli.main(
            [
                "synthesize",
                "--voice",
                str(profile),
                "--text",
                "New words.",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    manager = managers[-1]
    assert manager.kwargs["offline"] is True
    assert manager.resolves == [("qwen3:base-0.6b", {"quality": "int4"})]
    assert manager.installs == []
    assert output.is_file()
