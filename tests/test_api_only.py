from __future__ import annotations

import importlib.util
from pathlib import Path

import qwen3synth
import qwen3synth.catalog as catalog

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_API_NAMES = {
    "GenerationConfig",
    "ModelGenerationError",
    "ModelNotInstalledError",
    "PreparedVoice",
    "ReferenceVoice",
    "SynthesisRequest",
    "SynthesisResult",
    "VoiceDesignRequest",
}


def test_package_has_no_console_script_or_main_module() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert "[project.scripts]" not in pyproject
    assert not (ROOT / "qwen3synth" / "__main__.py").exists()
    assert importlib.util.find_spec("qwen3synth.__main__") is None


def test_supported_python_api_exports_remain_available() -> None:
    assert PUBLIC_API_NAMES <= set(qwen3synth.__all__)
    assert all(getattr(qwen3synth, name) is not None for name in PUBLIC_API_NAMES)


def test_catalog_retains_metadata_operations_without_asset_facades() -> None:
    assert callable(catalog.load_catalog)
    assert callable(catalog.bundle_record)
    assert hasattr(catalog, "BundleRecord")
    assert not hasattr(catalog, "install_bundle")
    assert not hasattr(catalog, "find_installed")


def test_six_examples_are_directly_executable_scripts() -> None:
    names = (
        "00_discover_bundles.py",
        "01_voice_design_prompt_to_wav.py",
        "02_prepare_voice_profile.py",
        "03_synthesize_cloned_voice.py",
        "04_local_bundle.py",
        "05_inspect_voice_profile.py",
    )
    for name in names:
        path = ROOT / "examples" / name
        source = path.read_text(encoding="utf-8")
        compile(source, str(path), "exec")
        assert "import argparse" in source
        assert 'if __name__ == "__main__":' in source
