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

Synthesis does not install weights implicitly. Install a model first with Qwen3Synth's compatibility CLI (delegated to OnnxVoice), or use the OnnxVoice API directly:

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

The `.qvoice` profile is a versioned ZIP containing JSON metadata, numeric conditioning arrays, reference sample count, transcript token IDs, and checksums. It does not use pickle. A profile is checked against the selected model identity and `qwen3-wavekat-v1` layout before inference.

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

`Qwen3Runtime.from_pretrained()` explicitly resolves/installs the requested bundle into the OnnxVoice cache. `Qwen3Runtime.load()` opens an existing local bundle directory without catalog acquisition.

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

## Runtime and parity status

Ordinary tests are offline and use synthetic graph/session fixtures plus committed tokenizer/audio-reference data. The OnnxVoice repository contains CPU integration cases marked `integration` and `resource_heavy` for VoiceDesign/Base at both qualities. They require matching model installations already present in an offline cache and may consume substantial disk, RAM, and CPU time; absent bundles cause explicit skips. Pytest's default discovery does not exclude integration markers, so use `pytest -m 'not integration'` for an offline-only suite. A skip is not a pass. This checkout has no installed WaveKat bundle, so real-model inference, waveform parity, and listening quality have not been verified here.

ONNX Runtime provider support is platform-dependent. The presence of a `cpu` or `gpu` extra does not establish that a provider wheel is available or supported on a given mobile/Termux environment. Check the [OnnxVoice provider guide](https://github.com/buchwandler/onnxvoice/blob/main/docs/providers.md) and model asset requirements before installing multi-gigabyte profiles.
