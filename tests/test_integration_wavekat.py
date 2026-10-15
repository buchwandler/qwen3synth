from __future__ import annotations

import os
import urllib.request
from importlib.util import find_spec
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import qwen3synth.runtime as runtime_module
from qwen3synth.errors import ModelNotInstalledError
from qwen3synth.profile import PreparedVoice
from qwen3synth.runtime import Qwen3Runtime
from qwen3synth.types import GenerationConfig, ReferenceVoice, SynthesisRequest, VoiceDesignRequest

pytestmark = [pytest.mark.integration, pytest.mark.resource_heavy]
_REFERENCE_TEXT = "Hello from Qwen3Synth."


def _required() -> bool:
    return os.environ.get("QWEN3SYNTH_REQUIRE_REAL") == "1"


def _runtime_or_skip(bundle: str, quality: str, cache_dir: Path | None) -> Qwen3Runtime:
    try:
        return Qwen3Runtime.from_installed(
            bundle,
            quality=quality,
            provider="cpu",
            cache_dir=cache_dir,
        )
    except ModelNotInstalledError as exc:
        message = f"Required Qwen3 bundle {bundle}:{quality} is not installed: {exc}"
        if _required():
            pytest.fail(message)
        pytest.skip(message)


@pytest.fixture(scope="session", params=("int4", "fp32"), ids=("int4", "fp32"))
def real_reference(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
) -> tuple[str, Path, Any]:
    quality = str(request.param)
    if find_spec("onnxruntime") is None:
        message = "ONNX Runtime is required for real Qwen3Synth integration"
        if _required():
            pytest.fail(message)
        pytest.skip(message)

    cache_value = os.environ.get("QWEN3SYNTH_TEST_CACHE")
    if _required() and not cache_value:
        pytest.fail("QWEN3SYNTH_TEST_CACHE must select the installed model cache")
    cache_dir = Path(cache_value).expanduser() if cache_value else None

    patches = pytest.MonkeyPatch()

    def fail_install(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("integration opened an acquiring path")

    def fail_network(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("integration attempted network access")

    patches.setattr(runtime_module.OnnxVoice, "install", fail_install)
    patches.setattr(urllib.request, "urlopen", fail_network)
    try:
        with _runtime_or_skip("voice-design-1.7b", quality, cache_dir) as designer:
            result = designer.design_voice(
                VoiceDesignRequest(
                    id=f"integration-{quality}",
                    text=_REFERENCE_TEXT,
                    description="Warm, clear narrator, calm pace, neutral English accent.",
                    language="en",
                ),
                config=GenerationConfig(seed=42, max_new_tokens=64),
            )
        wav_path = tmp_path_factory.mktemp(f"qwen3-reference-{quality}") / "reference.wav"
        result.save_wav(wav_path)
        yield quality, wav_path, result
    finally:
        patches.undo()


def _assert_audio(result: Any) -> None:
    assert result.sample_rate == 24_000
    assert isinstance(result.samples, np.ndarray)
    assert result.samples.dtype == np.float32
    assert result.samples.ndim == 1
    assert result.samples.size > 0
    assert np.isfinite(result.samples).all()
    assert np.max(np.abs(result.samples)) > 1e-5


def test_real_voice_design_frontend(real_reference: tuple[str, Path, Any]) -> None:
    _, _, result = real_reference
    _assert_audio(result)
    assert result.metadata["kind"] == "voice_design"


def test_real_base_prepare_profile_clone_and_offline_repeat(
    real_reference: tuple[str, Path, Any], tmp_path: Path
) -> None:
    quality, wav_path, _ = real_reference
    cache_value = os.environ.get("QWEN3SYNTH_TEST_CACHE")
    cache_dir = Path(cache_value).expanduser() if cache_value else None
    profile_path = tmp_path / f"voice-{quality}.qvoice"

    with _runtime_or_skip("base-0.6b", quality, cache_dir) as base:
        voice = base.prepare_voice(ReferenceVoice(wav_path, _REFERENCE_TEXT, "en"))
        voice.save(profile_path)
        loaded = PreparedVoice.load(profile_path)
        result = base.synthesize(
            SynthesisRequest(f"clone-{quality}", "A new sentence in the same voice.", "en"),
            voice=loaded,
            config=GenerationConfig(seed=43, max_new_tokens=64),
        )
    _assert_audio(result)
    assert result.metadata["kind"] == "voice_clone"

    with _runtime_or_skip("base-0.6b", quality, cache_dir) as offline_base:
        repeated = offline_base.synthesize(
            SynthesisRequest(f"offline-{quality}", "A second offline sentence.", "en"),
            voice=PreparedVoice.load(profile_path),
            config=GenerationConfig(seed=44, max_new_tokens=64),
        )
    _assert_audio(repeated)
    assert repeated.metadata["kind"] == "voice_clone"
