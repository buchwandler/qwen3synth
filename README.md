# qwen3synth

MVP Qwen3-TTS synthesis engine for published ONNX bundles. It reimplements the host-side inference pipeline and deliberately does **not** depend on `qwen-tts`, PyTorch, Transformers, `huggingface_hub`, librosa, or soundfile.

The package has a flat repository layout: `qwen3synth/` is directly under the repository root; there is no `src/` layer. Package versions are generated from Git tags by `setuptools_scm`.

## Supported MVP bundles

- `voice-design-1.7b`: WaveKat Qwen3-TTS 1.7B VoiceDesign ONNX.
- `base-0.6b`: WaveKat Qwen3-TTS 0.6B Base ONNX voice cloning.
- Profiles: `int4` and `fp32`.
- Output: mono 24 kHz WAV.

The built-in seed catalog pins the WaveKat repositories to immutable revisions. For production, point `--catalog` at the companion `qwen3-onnx-bundles` repository.

## Install

```bash
pip install -e '.[cpu]'
```

## Model installation

No model is installed implicitly during synthesis.

```bash
qwen3synth models list
qwen3synth models install voice-design-1.7b --quality int4
qwen3synth models install base-0.6b --quality int4
```

The installer uses standard-library HTTP against pinned Hugging Face revisions and records SHA-256 hashes for every downloaded file in the local installation manifest. It does not use the Hugging Face Python client.

## Design a voice

```bash
qwen3synth voice design \
  --bundle voice-design-1.7b --quality int4 \
  --prompt 'Warm mature British male baritone, measured and clear.' \
  --text 'This is the reference voice that I will reuse later.' \
  --language en --seed 42 --output reference.wav
```

VoiceDesign duration is generated, not an exact requested duration. If you want a ~20-second master, choose reference text of the appropriate length.

## Prepare a cloning profile

WaveKat's Base ONNX reference encoder uses a canonical ten-second input. This MVP therefore requires a **matching 0.5-10 second excerpt and exact transcript**; it will not silently truncate a 20-second WAV while retaining an incorrect transcript.

```bash
qwen3synth voice prepare \
  --bundle base-0.6b --quality int4 \
  --reference reference_excerpt.wav \
  --transcript 'Exact words spoken in the selected excerpt.' \
  --language en --output narrator.qvoice
```

The `.qvoice` file is a versioned ZIP containing JSON, a speaker embedding, reference codec codes, and checksums. It never uses pickle.

## Synthesize

```bash
qwen3synth synthesize \
  --bundle base-0.6b --quality int4 \
  --voice narrator.qvoice \
  --text 'This is a new sentence in the cloned voice.' \
  --language en --seed 42 --output cloned.wav
```

## Important MVP limitations

This code follows the published WaveKat ONNX reference algorithm and graph names, but the ZIP itself does not contain multi-gigabyte models and real-model execution is not exercised by ordinary unit tests. The tokenizer is an independent pure-Python byte-level BPE implementation and should be parity-tested against pinned tokenizer fixtures before a production release. The NumPy mel frontend is designed to mirror the documented librosa parameters but should likewise be numerically compared against audited reference fixtures.

Only use voice cloning for voices you have permission to clone.
