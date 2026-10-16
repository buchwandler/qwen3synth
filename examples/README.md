# Qwen3Synth examples

These directly executable Python scripts demonstrate the supported library workflows. They are not installed console commands. Run them from the repository root after installing a provider extra and the required bundles.

## Install and discover

```bash
python -m pip install -e '.[cpu]'
python examples/00_discover_bundles.py
onnxvoice list --system qwen3
onnxvoice install qwen3:voice-design-1.7b --quality int4
onnxvoice install qwen3:base-0.6b --quality int4
```

`00_discover_bundles.py` shows catalog metadata without opening model sessions. Fetching uncached catalog metadata may access the network. `onnxvoice install` acquires the larger model assets and verifies them. The synthesis examples use `Qwen3Runtime.from_installed()` and do not download models.

## Design, prepare, and clone

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

The transcript supplied to profile preparation must be the exact speech in the reference WAV. Base accepts reference audio from 0.5 to 10 seconds. `.qvoice` files contain sensitive voice data, so store them privately and use only authorized voices. VoiceDesign sample duration is model-controlled, not fixed by the token cap.

## Open a complete local bundle

`04_local_bundle.py` uses `Qwen3Runtime.load()` to open an unmanaged local VoiceDesign bundle. It does not install or download assets:

```bash
python examples/04_local_bundle.py \
  --bundle-dir /models/voice-design-1.7b \
  --prompt 'Warm, measured, clear British baritone.' \
  --text 'This sample demonstrates the voice.' \
  --language en --quality int4 --provider cpu --seed 42 \
  --max-new-tokens 128 --output local-reference.wav
```

The directory must contain `config.json`, the tokenizer inventory, required embeddings, all selected graph files, and each graph's adjacent `.onnx.data` file. A single `.onnx` file is insufficient. For managed installation instead, use `onnxvoice install qwen3:voice-design-1.7b --quality int4` and run example 01.

## Cache and troubleshooting

Use `ONNXVOICE_CACHE_DIR` to share an explicit cache between `onnxvoice install` and examples. Runtime cache precedence is `cache_dir=` first, then `QWEN3SYNTH_CACHE`, then `ONNXVOICE_CACHE_DIR`, then ONNXVoice's default. The examples use the shared environment configuration and do not define a second download store.

If a managed bundle is missing, install the matching asset explicitly, for example:

```bash
onnxvoice install qwen3:base-0.6b --quality int4
```

Use `onnxvoice info qwen3:base-0.6b --quality int4` to inspect installed details. See `onnxvoice --help` for catalog and asset-management commands. The `05_inspect_voice_profile.py` script validates profile metadata without opening an ONNX model or using the network.
