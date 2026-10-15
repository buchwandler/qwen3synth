# Qwen3Synth API and runtime contract

Qwen3Synth owns the Qwen3 frontend, request policy, WAV handling, and reusable voice profiles. ONNXVoice owns the normalized bundle catalog, installation store, model validation, ONNX sessions, generation, and vocoding.

## Installation and model opening

Model acquisition is explicit:

```bash
qwen3synth models install voice-design-1.7b --quality int4
```

CLI `voice design`, `voice prepare`, and `synthesize` commands only open installed assets. In Python:

- `Qwen3Runtime.from_pretrained(...)` is an explicit convenience factory that may install the selected bundle.
- `Qwen3Runtime.from_installed(...)` resolves and opens an existing managed installation with ONNXVoice offline mode. It does not acquire assets.
- `Qwen3Runtime.load(path)` opens an explicit local bundle directory without registering or copying its files.

The example at [`examples/01_voice_design_prompt_to_wav.py`](../examples/01_voice_design_prompt_to_wav.py) uses `from_installed()` and expects the model to have been installed first.

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

Offline unit tests cover tokenizer parity fixtures, shared prompt encoders, runtime request behavior, profile validation, public metadata APIs, and error mapping. The fixture tests use pinned tokenizer assets and exact integer IDs. Strict real-model integration completed with ONNX Runtime 1.31.0 (`CPUExecutionProvider`), using the pinned Base revision `177cd39e2ddab0d4d7ce5152a7abc2d382621158` and VoiceDesign revision `62c7863a68800d72bcee4f2931148c441571ace7` at `int4` and `fp32`. With `QWEN3SYNTH_REQUIRE_REAL=1` and `QWEN3SYNTH_TEST_CACHE` set to the installed cache, `pytest tests/test_integration_wavekat.py` completed with 4 passed in 1218.36s. This validates generated audio structure, Base profile prepare/save/load/clone, and offline repeat; it does not verify perceived audio quality. Manual listening remains unverified. A skipped integration test is not a pass.
