# Qwen3Synth API and runtime contract

Qwen3Synth owns the Qwen3 frontend, request policy, WAV handling, and reusable voice profiles. ONNXVoice owns the normalized bundle catalog, installation store, model validation, ONNX sessions, generation, and vocoding.

## API-only package and migration

The package is API-only. It installs no `qwen3synth` console command and provides no `qwen3synth.__main__` module, so `python -m qwen3synth` is not supported. The former `install_bundle()` and `find_installed()` catalog helpers are removed in v0.1.0. Use ONNXVoice for asset management and keep structured catalog inspection in `load_catalog()` and `bundle_record()`.

## Installation and model opening

Acquire model assets explicitly through ONNXVoice:

```bash
onnxvoice install qwen3:voice-design-1.7b --quality int4
```

Voice design, profile preparation, and synthesis are Python methods. In Python:

- `Qwen3Runtime.from_pretrained(...)` is an explicit convenience factory that may install the selected bundle.
- `Qwen3Runtime.from_installed(...)` resolves and opens an existing managed installation with ONNXVoice offline mode. It does not acquire assets.
- `Qwen3Runtime.load(path)` opens an explicit local bundle directory without registering or copying its files.

The example at [`examples/01_voice_design_prompt_to_wav.py`](../examples/01_voice_design_prompt_to_wav.py) uses `from_installed()` and expects the model to have been installed first.

## Cache and local bundles

`Qwen3Runtime.from_pretrained()` is the explicit acquisition convenience and delegates installation to `OnnxVoice.install()`. `from_installed()` configures ONNXVoice offline mode and only resolves an existing installation. `load(path)` opens an unmanaged local bundle without catalog acquisition or a separate cache. A complete local bundle includes `config.json`, tokenizer files, required embeddings, all required ONNX graphs, and graph-adjacent `.onnx.data` files; an isolated graph is insufficient.

Cache precedence is explicit `cache_dir=`, then `QWEN3SYNTH_CACHE`, then `ONNXVOICE_CACHE_DIR`, then the ONNXVoice default. Use `ONNXVOICE_CACHE_DIR` to share installed assets between `onnxvoice` and Qwen3Synth.

Base reference WAVs must be 0.5 to 10 seconds and their transcript must match the speech exactly. `.qvoice` profiles contain sensitive voice information and should be stored privately and used only for authorized voices.

## Metadata APIs

The package root exports these deterministic, metadata-only interfaces:

```python
from qwen3synth import (
    REQUEST_API_VERSION,
    RequestApiContract,
    request_api_contract,
    runtime_identity,
    DiscoveredBundle,
    discover_bundles,
    VoiceProfileInfo,
    inspect_voice_profile,
)
```

`REQUEST_API_VERSION` is `1`. `request_api_contract()` reports support for reference voices, request measurement, and VoiceDesign. Named voices are not supported by this API version.

`runtime_identity()` returns the frontend and ONNXVoice versions, the request API version, supported layout identifiers, and sample rate. It does not open a model or ONNX session.

`discover_bundles(offline=False, refresh=False, cache_dir=None, catalog=None)` delegates catalog parsing to ONNXVoice and returns bundles in ID order. Each result exposes the canonical reference, model type, layout, sample rate, available precision profiles, source revision, runtime availability, and normalized ONNXVoice metadata. The current catalog declares `base-0.6b` and `voice-design-1.7b`, both at 24 kHz with `fp32` and `int4` profiles. The normalized catalog does not declare per-bundle language lists, so `languages` is empty unless that metadata is supplied by the catalog.

Offline discovery never contacts the network. A remote `refresh` request is not sent in offline mode; provide a cached catalog or a local `catalog` path. A local catalog can be read without network access.

`inspect_voice_profile(path)` validates the bounded profile archive and returns a `VoiceProfileInfo`, including `profile_sha256`, the SHA-256 digest of the complete `.qvoice` file. It does not open ONNX models, ONNX sessions, or the network.

## `.qvoice` schema 2

Schema 2 stores the layout, model ID and revision, transcript and language, source and audio hashes, reference sample count, exact reference token IDs, and SHA-256 checksums for the speaker embedding and reference-code arrays. The ZIP must contain exactly the manifest and two numeric NumPy arrays. The loader rejects unsafe or duplicate member names, unsupported schemas, malformed fields, checksum mismatches, invalid array dtypes and shapes, and oversized input. Current limits are 32 MiB per archive, 1 MiB for the manifest, and 15 MiB per array member. NumPy object/pickle arrays are not accepted.

Synthesis rejects a profile whose layout or model ID differs from the active Base runtime. If both the profile and runtime have model revisions, they must match. Profiles without a recorded revision can be used only when their other compatibility checks pass.

## Stable errors

Public failures derive from `Qwen3SynthError`. The package exports `ModelNotInstalledError`, `UnsupportedModelCapabilityError`, `SynthesisInputTooLongError`, `InvalidReferenceAudioError`, `InvalidReferenceTranscriptError`, `IncompatibleVoiceProfileError`, `InvalidModelLayoutError`, and `ModelGenerationError`. Catalog failures use `CatalogError`.

Relevant ONNXVoice errors are translated at frontend entry points. Missing installations become `ModelNotInstalledError`; unsupported capabilities become `UnsupportedModelCapabilityError`; runtime-contract failures while opening a bundle become `InvalidModelLayoutError`; runtime-contract failures during prepare or generation become `ModelGenerationError`. The original ONNXVoice exception remains available as `__cause__`.

## Verification status

Offline unit tests cover tokenizer parity fixtures, shared prompt encoders, runtime request behavior, profile validation, public metadata APIs, and error mapping. The fixture-complete non-integration run completed with 81 passed, 4 integration tests deselected, and no skips, using pinned tokenizer assets and exact integer IDs.

The current real-model integration run used the locally installed INT4 bundles at pinned revisions `177cd39e2ddab0d4d7ce5152a7abc2d382621158` (Base) and `62c7863a68800d72bcee4f2931148c441571ace7` (VoiceDesign), after both passed ONNXVoice checksum verification. With `QWEN3SYNTH_REQUIRE_REAL=1`, `pytest tests/test_integration_wavekat.py -k int4 -ra` completed with 2 passed and 2 FP32 cases deselected in 113.35s. It exercised VoiceDesign output structure and Base profile preparation, save/load, cloning, and offline repeat. The end-to-end examples also generated and inspected mono 24-kHz PCM WAVs; the VoiceDesign reference and cloned output were each 2.96 seconds. These runtime checks used the local ONNXVoice 0.2.9 checkout and preinstalled assets. They do not prove released ONNXVoice 0.2.9 resolves from the configured package index. No manual listening was performed; perceived audio quality and FP32 integration remain unverified.
