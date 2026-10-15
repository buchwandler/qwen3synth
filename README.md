# qwen3synth

`qwen3synth` is a Qwen3-TTS frontend for the WaveKat ONNX bundles. It keeps the user-facing text and voice workflow while delegating catalogs, model installation, integrity verification, ONNX sessions, graph validation, providers, generation, and vocoding to [OnnxVoice](https://github.com/buchwandler/onnxvoice).

The supported bundles are `voice-design-1.7b` and `base-0.6b`, with `fp32` and `int4` profiles and the `qwen3-wavekat-v1` ABI. VoiceDesign creates a voice from a description; Base clones from a reference recording and its exact transcript. Both return mono 24-kHz WAV when saved by this frontend.

## Ownership and dependencies

Qwen3Synth owns Qwen byte-level BPE tokenization, chat templates, language normalization, WAV read/write and resampling, request/generation policy, reference-profile persistence, and CLI behavior. OnnxVoice owns the catalog and model-runtime lifecycle. Qwen3Synth does not maintain a separate model downloader/cache or ONNX Runtime wrapper.

Install a provider extra through this package:

```bash
python -m pip install -e '.[cpu]'
# or, on a compatible ONNX Runtime platform:
python -m pip install -e '.[gpu]'
```

The package does not depend on `qwen-tts`, PyTorch, Transformers, `huggingface_hub`, librosa, or soundfile. OnnxVoice manages pinned asset acquisition through its own catalog/store. The assets retain their upstream licenses; install and use them only under those terms, and clone only voices you have permission to use.

## Explicit model installation

Only `qwen3synth models install` and Python `Qwen3Runtime.from_pretrained()` acquire model assets. CLI design, prepare, and synthesis commands only open installed models. For offline or production use, prefer `Qwen3Runtime.from_installed()`.

```bash
qwen3synth models list
qwen3synth models install voice-design-1.7b --quality int4
qwen3synth models install base-0.6b --quality int4
```

The shared OnnxVoice catalog supports the canonical refs `qwen3:voice-design-1.7b` and `qwen3:base-0.6b`; the short names above are aliases. `--cache-dir` selects the OnnxVoice cache for supported CLI commands. `QWEN3SYNTH_CACHE` sets the frontend's cache directory. See the [OnnxVoice Qwen3 runtime contract](https://github.com/buchwandler/onnxvoice/blob/main/docs/systems/qwen3.md) for provider, local-bundle, ABI, and parity details.

## Design a voice

```bash
qwen3synth voice design \
  --bundle voice-design-1.7b --quality int4 \
  --prompt 'Warm, measured, clear British baritone.' \
  --text 'This sample demonstrates the voice.' \
  --language en --seed 42 --output reference.wav
```

VoiceDesign duration is model-controlled, not an exact requested length. Save a suitable generated WAV and retain its exact spoken transcript if you plan to use Base cloning.

## Prepare a Base voice profile

The frontend reads WAV, converts it to mono 24-kHz samples, and enforces its reference duration policy. Choose a clean 0.5-to-10-second excerpt and provide the matching transcript; Qwen3Synth will not silently truncate a longer recording while keeping the wrong transcript.

```bash
qwen3synth voice prepare \
  --bundle base-0.6b --quality int4 \
  --reference reference.wav \
  --transcript 'This sample demonstrates the voice.' \
  --language en --output narrator.qvoice
```

The `.qvoice` profile uses strict schema 2: JSON metadata, numeric conditioning arrays, reference sample count, exact transcript token IDs, and checksums. It includes a profile content digest for identity. Loading enforces archive/member size limits, array shape/type checks, checksums, and `qwen3-wavekat-v1` layout. Synthesis rejects model ID or recorded model revision mismatches.

## Synthesize with the prepared profile

```bash
qwen3synth synthesize \
  --bundle base-0.6b --quality int4 \
  --voice narrator.qvoice \
  --text 'Now the same voice can read new text.' \
  --language en --seed 42 --output cloned.wav
```

For a previously installed local directory, use `--bundle-dir PATH` instead of `--bundle`. The local directory must contain the complete selected WaveKat inventory, including graph-adjacent `.onnx.data` files. Local opening is unmanaged and does not copy or register those paths.

## Python API

`Qwen3Runtime.from_pretrained()` is an explicit convenience factory that may install the requested bundle into the OnnxVoice cache. `Qwen3Runtime.from_installed()` resolves and opens an existing installation without acquisition. `Qwen3Runtime.load()` opens an explicit local bundle directory without catalog access.

```python
from qwen3synth import (
    GenerationConfig,
    Qwen3Runtime,
    ReferenceVoice,
    SynthesisRequest,
    VoiceDesignRequest,
)

with Qwen3Runtime.from_pretrained(
    "voice-design-1.7b", quality="int4", provider="cpu"
) as designer:
    sample = designer.design_voice(
        VoiceDesignRequest(
            id="design-1",
            text="A short sample to select a voice.",
            description="Warm, measured, clear British baritone.",
            language="en",
        ),
        config=GenerationConfig(seed=42),
    )
    sample.save_wav("reference.wav")

with Qwen3Runtime.from_pretrained(
    "base-0.6b", quality="int4", provider="cpu"
) as clone:
    voice = clone.prepare_voice(
        ReferenceVoice(
            audio="reference.wav",
            transcript="A short sample to select a voice.",
            language="en",
        )
    )
    voice.save("narrator.qvoice")
    result = clone.synthesize(
        SynthesisRequest(id="line-1", text="Read this in the selected voice.", language="en"),
        voice=voice,
        config=GenerationConfig(seed=42),
    )
    result.save_wav("cloned.wav")
```

`GenerationConfig` exposes temperature, top-k, repetition penalty, maximum new codec frames, and seed. `Qwen3Runtime.measure_request()` and `measure_text()` report token capacity using the same frontend tokenizer as synthesis.

## Metadata and stable API

The package root exports `REQUEST_API_VERSION == 1`, `RequestApiContract`, and `request_api_contract()` for request capability negotiation. `runtime_identity()` returns deterministic frontend/ONNXVoice version and layout metadata without opening models.

`discover_bundles()` delegates normalized catalog parsing to ONNXVoice and reports both Qwen3 bundles, their `base` or `voice_design` model type, `fp32`/`int4` profiles, pinned source revision, and 24-kHz sample rate. Use `offline=True` to prohibit network access. `inspect_voice_profile(path)` returns schema and model metadata plus the SHA-256 digest of the complete `.qvoice` file.

The public error boundary includes `ModelNotInstalledError`, `UnsupportedModelCapabilityError`, `InvalidReferenceAudioError`, `InvalidReferenceTranscriptError`, `IncompatibleVoiceProfileError`, `InvalidModelLayoutError`, and `ModelGenerationError`. See [the API contract](docs/api-contract.md) for behavior and error mapping details.

A short installed-model example is available in [`examples/README.md`](examples/README.md).

## Runtime and parity status

Qwen3Synth's regular unit tests are designed to run offline and cover exact tokenizer parity, prompt templates, WAV handling, runtime behavior, profile validation, metadata APIs, and stable error mapping. Strict real-model integration completed on CPU with ONNX Runtime 1.31.0 (`CPUExecutionProvider`) and the pinned Base (`177cd39e2ddab0d4d7ce5152a7abc2d382621158`) and VoiceDesign (`62c7863a68800d72bcee4f2931148c441571ace7`) bundles at both `int4` and `fp32`: `QWEN3SYNTH_REQUIRE_REAL=1 pytest tests/test_integration_wavekat.py` completed with 4 passed in 1218.36s using an installed cache selected by `QWEN3SYNTH_TEST_CACHE`. This verified generated audio structure, Base profile prepare/save/load/clone, and offline repeat. It does not establish perceived audio quality. Manual listening remains unverified, and a skipped model test is not a pass.

ONNX Runtime provider support is platform-dependent. The presence of a `cpu` or `gpu` extra does not establish that a provider wheel is available or supported on a given mobile/Termux environment. Check the [OnnxVoice provider guide](https://github.com/buchwandler/onnxvoice/blob/main/docs/providers.md) and model asset requirements before installing multi-gigabyte profiles.
