# qwen3synth

`qwen3synth` is an API-only Python frontend for Qwen3-TTS voice design and cloning on WaveKat ONNX bundles. It owns tokenization, prompt templates, language normalization, WAV handling and resampling, reference conditioning, `.qvoice` profiles, and synthesis requests. [ONNXVoice](https://github.com/buchwandler/onnxvoice) owns catalog and model installation, integrity checks, ONNX sessions, generation, and vocoding.

The supported bundles are `voice-design-1.7b` and `base-0.6b`, with `fp32` and `int4` profiles and the `qwen3-wavekat-v1` layout. Saved synthesis output is mono 24-kHz WAV. Use only model assets under their licenses and clone voices you are authorized to use.

## Install provider support and model assets

From a source checkout, install the provider extra supported by your platform:

```bash
python -m pip install -e '.[cpu]'
# Or use the GPU extra on a supported ONNX Runtime platform:
python -m pip install -e '.[gpu]'
```

Catalog listing reads model metadata and does not initialize ONNX sessions. A catalog refresh may access the network if metadata is not cached. Installing a bundle downloads and verifies its model assets, which can require several gigabytes:

```bash
onnxvoice list --system qwen3
onnxvoice install qwen3:voice-design-1.7b --quality int4
onnxvoice install qwen3:base-0.6b --quality int4
```

`Qwen3Runtime.from_installed()` and the examples below open already installed assets without downloading models. `Qwen3Runtime.from_pretrained()` is the explicit Python convenience that delegates installation to `OnnxVoice.install()`. For applications that need a fixed location, pass `cache_dir=`.

ONNXVoice and Qwen3Synth share the ONNXVoice cache. Prefer `ONNXVOICE_CACHE_DIR` when setting one cache for installation and examples. Cache selection precedence is explicit `cache_dir=`, then the frontend override `QWEN3SYNTH_CACHE`, then `ONNXVOICE_CACHE_DIR`, then the ONNXVoice default.

## Design, prepare, and synthesize

The examples are Python scripts, not installed commands. This complete path designs a short reference, prepares its reusable profile using the exact spoken transcript, and synthesizes different words:

```bash
python examples/01_voice_design_prompt_to_wav.py \
  --prompt 'Warm, measured, clear British baritone.' \
  --text 'This sample demonstrates the voice.' \
  --language en --quality int4 --provider cpu --seed 42 \
  --max-new-tokens 128 --output reference.wav

python examples/02_prepare_voice_profile.py \
  --reference reference.wav \
  --transcript 'This sample demonstrates the voice.' \
  --language en --quality int4 --provider cpu --output narrator.qvoice

python examples/03_synthesize_cloned_voice.py \
  --voice narrator.qvoice \
  --text 'Now the voice speaks different words.' \
  --language en --quality int4 --provider cpu --seed 42 \
  --max-new-tokens 128 --output cloned.wav

python examples/05_inspect_voice_profile.py narrator.qvoice
```

Base reference audio must be a WAV from 0.5 to 10 seconds, and the transcript must match the recording exactly. The profile contains sensitive voice information. Keep it private. VoiceDesign duration is model-controlled; `max-new-tokens` is a generation cap, not an exact audio duration.

See [`examples/README.md`](examples/README.md) for discovery, local-bundle use, and troubleshooting. A local bundle opened with `Qwen3Runtime.load()` must be complete, including configuration, tokenizer files, embeddings, every required graph, and adjacent `.onnx.data` files. A lone ONNX graph is not a complete bundle.

## Python API and package boundary

There is no `qwen3synth` console command, `qwen3synth.__main__`, or supported `python -m qwen3synth` invocation. Use `onnxvoice` for catalog and asset operations, the Python API for synthesis, and the examples for runnable workflows. The former `install_bundle()` and `find_installed()` catalog helpers are removed in v0.1.0. Use `onnxvoice install` for acquisition and `load_catalog()` / `bundle_record()` for structured catalog metadata.

```python
from qwen3synth import GenerationConfig, Qwen3Runtime, VoiceDesignRequest

with Qwen3Runtime.from_installed(
    "voice-design-1.7b", quality="int4", provider="cpu"
) as runtime:
    result = runtime.design_voice(
        VoiceDesignRequest(
            id="example",
            text="A short spoken sentence.",
            description="Warm, clear baritone.",
            language="en",
        ),
        config=GenerationConfig(seed=42, max_new_tokens=128),
    )
    result.save_wav("reference.wav")
```

Qwen3Synth does not add PyTorch, Transformers, `qwen-tts`, `huggingface_hub`, librosa, or soundfile as runtime dependencies. See [`docs/api-contract.md`](docs/api-contract.md) for stable APIs, metadata behavior, and error handling.
