from __future__ import annotations

from pathlib import Path
from typing import Any

import onnxvoice
import pytest
from onnxvoice.types import CatalogItem

import qwen3synth
import qwen3synth.discovery as discovery
from qwen3synth import (
    REQUEST_API_VERSION,
    DiscoveredBundle,
    RequestApiContract,
    discover_bundles,
    request_api_contract,
    runtime_identity,
)


def _item(bundle_id: str, model_type: str, revision: str) -> CatalogItem:
    return CatalogItem(
        system="qwen3",
        id=bundle_id,
        kind="bundle",
        artifacts=(),
        sample_rate=24_000,
        metadata={
            "layout": "qwen3-wavekat-v1",
            "model_type": model_type,
            "source_revision": revision,
            "profiles": {"int4": {}, "fp32": {}},
            "languages": ["en", "de"],
        },
    )


def test_request_api_contract_is_stable_and_public() -> None:
    expected = RequestApiContract(
        version=1,
        supports_reference_voice=True,
        supports_request_measurement=True,
        supports_named_voices=False,
        supports_voice_design=True,
    )
    assert REQUEST_API_VERSION == 1
    assert request_api_contract() == expected
    assert request_api_contract() == expected
    assert qwen3synth.RequestApiContract is RequestApiContract


def test_runtime_identity_is_deterministic_and_metadata_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("metadata APIs must not use the network or open models")

    monkeypatch.setattr("urllib.request.urlopen", fail)
    monkeypatch.setattr(onnxvoice.OnnxVoice, "open", fail)

    identity = runtime_identity()
    assert identity == runtime_identity()
    assert identity == {
        "engine": "qwen3synth",
        "engine_version": qwen3synth.__version__,
        "request_api_version": 1,
        "onnxvoice_version": onnxvoice.__version__,
        "layouts": ("qwen3-wavekat-v1",),
        "sample_rate": 24_000,
    }


def test_discovery_maps_onnxvoice_metadata_deterministically(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[dict[str, Any]] = []
    items = [
        _item("voice-design-1.7b", "voice_design", "b" * 40),
        _item("base-0.6b", "base", "a" * 40),
    ]

    class FakeOnnxVoice:
        def __init__(self, **kwargs: Any) -> None:
            calls.append({"init": kwargs})

        def list(self, system: str, *, refresh: bool = False) -> list[CatalogItem]:
            calls.append({"list": (system, refresh)})
            return items

        def open(self, *args: Any, **kwargs: Any) -> None:
            raise AssertionError("discovery must not open models")

    monkeypatch.setattr(discovery, "OnnxVoice", FakeOnnxVoice)
    monkeypatch.setattr(discovery, "find_spec", lambda name: object())
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("network access")),
    )
    catalog_path = tmp_path / "catalog.json"

    bundles = discover_bundles(
        offline=True,
        refresh=False,
        cache_dir=tmp_path / "cache",
        catalog=catalog_path,
    )

    assert isinstance(bundles[0], DiscoveredBundle)
    assert [bundle.id for bundle in bundles] == ["base-0.6b", "voice-design-1.7b"]
    base, design = bundles
    assert base.ref == "qwen3:base-0.6b"
    assert base.display_name == base.id
    assert base.model_type == "base"
    assert base.layout == "qwen3-wavekat-v1"
    assert base.sample_rate == 24_000
    assert base.languages == ("en", "de")
    assert base.precisions == ("fp32", "int4")
    assert base.source_revision == "a" * 40
    assert base.runtime_available
    assert design.model_type == "voice_design"
    assert design.source_revision == "b" * 40
    assert calls == [
        {
            "init": {
                "cache_dir": tmp_path / "cache",
                "catalog_sources": {"qwen3": str(catalog_path)},
                "offline": True,
            }
        },
        {"list": ("qwen3", True)},
    ]


def test_discovery_does_not_refresh_remote_catalog_while_offline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, bool]] = []

    class FakeOnnxVoice:
        def __init__(self, **kwargs: Any) -> None:
            pass

        def list(self, system: str, *, refresh: bool = False) -> list[CatalogItem]:
            calls.append((system, refresh))
            return []

    monkeypatch.setattr(discovery, "OnnxVoice", FakeOnnxVoice)
    monkeypatch.setattr(discovery, "find_spec", lambda name: None)
    bundles = discover_bundles(
        offline=True,
        refresh=True,
        catalog="https://example.invalid/qwen3.json",
    )
    assert bundles == ()
    assert calls == [("qwen3", False)]
