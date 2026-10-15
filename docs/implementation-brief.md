# qwen3synth — implementation brief

**Status:** implementation specification / feasibility reviewed 2026-10-14
**Target:** a standalone Qwen3-TTS producer package backed by `onnxvoice`; entirely independent of Qwen's Python inference stack at development, install, and runtime.
**Repository / distribution / import / CLI:** `qwen3synth` / `qwen3synth` / `qwen3synth` / `qwen3synth`
**Companion repository:** [`qwen3-onnx-bundles`](./qwen3-onnx-bundles-implementation-brief.md)

> **Implementation status:** This brief records the original proposal and feasibility decisions; its proposed API and component ownership are not the current contract. The implementation delegates catalog, assets, sessions, generation, and vocoding to OnnxVoice while Qwen3Synth retains frontend and profile behavior. See the current [Qwen3Synth README](../README.md) and [OnnxVoice Qwen3 runtime documentation](https://github.com/buchwandler/onnxvoice/blob/main/docs/systems/qwen3.md) for implemented APIs and verification limits.

## 0. Decision and feasibility

**Proceed, in incremental, independently testable phases.** This is technically possible with downloadable, pre-exported ONNX assets and independently written Python code. A model download is **not** sufficient by itself: autoregressive generation, model-specific embedding assembly, logit sampling, text tokenization, reference-audio processing, KV-cache transfer, codebook prediction, vocoder calls, and waveform trimming require implementation.

Public reference exports checked on 2026-10-14:

| Export                                                                                                                    | Core capability                                                     | Candidate role                          | Caveat                                                              |
| ------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------- | --------------------------------------- | ------------------------------------------------------------------- |
| [`wavekat/Qwen3-TTS-1.7B-VoiceDesign-ONNX`](https://huggingface.co/wavekat/Qwen3-TTS-1.7B-VoiceDesign-ONNX)               | Description + text → speech; FP32 and INT4                          | **MVP VoiceDesign**                     | 4 graph pipeline; substantial memory/disk                           |
| [`wavekat/Qwen3-TTS-0.6B-Base-ONNX`](https://huggingface.co/wavekat/Qwen3-TTS-0.6B-Base-ONNX)                             | Reference audio + _exact transcript_ + text → speech; FP32 and INT4 | **MVP cloning**                         | 6 graph pipeline; reference preprocessing must match                |
| [`xkos/Qwen3-TTS-12Hz-1.7B-ONNX`](https://huggingface.co/xkos/Qwen3-TTS-12Hz-1.7B-ONNX)                                   | VoiceDesign + Base in different export structure                    | Alternative after MVP                   | Different graph layout: requires separate adapter and test fixtures |
| [`onnx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign`](https://huggingface.co/onnx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign) | VoiceDesign export variants                                         | Later investigation                     | Export status is implementation-specific, not an ABI guarantee      |
| [`QwenLM/Qwen3-TTS`](https://github.com/QwenLM/Qwen3-TTS)                                                                 | Official specifications and model taxonomy                          | **Documentation/parity reference only** | Must **not** become a package dependency or execution requirement   |

WaveKat's 1.7B VoiceDesign variant lists `talker_prefill`, `talker_decode`, `code_predictor`, and `vocoder`. Its 0.6B Base variant additionally needs `speaker_encoder` and `tokenizer_encoder`, with reference conditioning. Reported output rate is 24 kHz; 12 Hz is the semantic codec-frame rate, **not** the WAV sample rate. The WaveKat model cards describe 10 supported languages. These facts apply to this export family; **never silently assume another exporter has the same graph ABI**.

### Hard constraints

1. Implement _all host-side application logic ourselves_; **do not import or invoke** `qwen_tts`, `torch`, `transformers`, `huggingface_hub`, `librosa`, `soundfile`, upstream `generate_onnx.py`, upstream `onnx_inference.py`, `tiktoken`, or Hugging Face model-loader helpers, either at runtime or in normal packaging/tests. No `subprocess` fallback to a bundled Python reference script.
2. Allowed infrastructure: `numpy` for numeric arrays, `onnxvoice` for owned bundle/cache/runtime functionality, and ONNX Runtime through `onnxvoice[cpu]` / `[gpu]`. Python standard library may implement WAV, STFT, resampling, JSON, HTTP (owned by `onnxvoice`), and text BPE. Do not embed the model in the wheel.
3. No ONNX **exporter** in this project. Consume published ONNX and data files only. Model export/quantization requires a separate, explicitly opt-in effort.
4. No `/src` layout. Typed Python package rooted at `./qwen3synth/`; tests in `./tests/`; `setuptools_scm`; supported Python 3.10+ consistent with `pocketsynth` and `onnxvoice` snapshots.
5. Use the existing `onnxvoice` system/asset store instead of creating a second HTTP downloader, cache, model registry, or ONNX Runtime wrapper.
6. Keep upstream provenance, third-party licenses, and a clear independent-project/non-affiliation notice. Only clone voices with permission/appropriate rights.

## 1. Ownership and system boundary

Mirror the **real** `pocketsynth` architecture, not its internals verbatim:

```text
Caller / document orchestration / SSMD
    └── prepared speakable text + requested language + voice choice
         └── qwen3synth (producer)
              ├── Public requests / typed validation / capability checks
              ├── Qwen BPE tokenizer and prompt templates
              ├── Qwen text normalization + token-capacity measurement
              ├── Reference WAV validation and voice-profile lifecycle
              ├── Design/clone orchestration and generation controls
              ├── Explicit convenience chunking and WAV output
              └── Stable synthesis result + runtime identity
                   └── onnxvoice (existing asset and ONNX runtime layer)
                        ├── qwen3 catalog install/verify/store/provenance
                        ├── provider selection / graph inspection / sessions
                        ├── ONNX tensors / embeddings / graph layout adapter
                        ├── Qwen autoregressive decode and KV-cache updates
                        ├── speech ref encoder and codebook prediction
                        ├── vocoder + reference audio decoding/trimming
                        └── native float32 audio + diagnostics
                             └── pinned ONNX bundle files from catalog
```

**Important boundary:** The _model-generation algorithm and loop_ belong to `onnxvoice`, not `qwen3synth`. The latter chooses validated generation parameters and submits model-ready token IDs, instruction IDs, prepared reference data, and metadata. That keeps the shared low-level asset/session layer consistent with Pocket's `Flow-LM` ownership. A dedicated `onnxvoice.systems.qwen3` adapter may expose typed model-specific entry points beyond its generic `infer()` API, without leaking private ONNX session objects to the producer.

`qwen3-onnx-bundles` is **catalog-only**, with no importable Python module, wheel, model mirror, downloader or runtime.

Out of scope: document parsing, SSMD interpretation, language-policy routing across engines, word alignments, external audio mixing, timeline composition, audio mastering, voice training/fine-tuning, arbitrary inference against original PyTorch checkpoints.

## 2. Package structure

```text
qwen3synth/
├── pyproject.toml
├── README.md
├── LICENSE
├── NOTICE
├── .pre-commit-config.yaml
├── .github/workflows/{tests.yml,pre-commit.yml,python-publish.yml}
├── docs/{architecture.md,api.md,voices.md,cli.md,onnx-abi.md,limitations.md}
├── qwen3synth/
│   ├── __init__.py
│   ├── __main__.py                 # CLI entry
│   ├── _version.py                 # generated by setuptools_scm
│   ├── py.typed
│   ├── errors.py
│   ├── types.py                    # requests/results/measurement metadata
│   ├── api_contract.py
│   ├── config.py                   # generation policy/validation
│   ├── tokenizer.py                # our own Qwen-compatible BPE implementation
│   ├── text.py                     # prompt templates, languages, normalization
│   ├── frontend.py                 # encode strict request once
│   ├── audio.py                    # WAV parse/write and bounded preprocessing
│   ├── voice.py                    # prepared references, voice identities
│   ├── profile.py                  # reusable clone profile persistence
│   ├── bundle.py                   # onnxvoice bundle façade
│   ├── discovery.py                # catalog inspection without model open
│   ├── _onnxvoice.py               # narrow adapter/protocol
│   ├── runtime.py                  # Qwen3Runtime strict operations
│   ├── diagnostics.py
│   ├── convenience.py              # explicitly separate from strict runtime
│   └── text_split.py
└── tests/
    ├── fixtures/{tokenization,graph_abi,profiles,wav,expected_audio}/
    ├── test_tokenizer.py
    ├── test_prompt_templates.py
    ├── test_request_contract.py
    ├── test_runtime_strict.py
    ├── test_voice_profiles.py
    ├── test_wav.py
    ├── test_discovery.py
    ├── test_onnxvoice_boundary.py
    └── test_integration_wavekat.py
```

This is an organizational proposal, not a copy of file names demanded by upstream Qwen.

## 3. Model and capability taxonomy

Represent these as **different model capabilities**, not runtime flags whose behavior is guessed from filenames:

| Model type                   | Supported inputs                                                   | Public entry point                | MVP                               |
| ---------------------------- | ------------------------------------------------------------------ | --------------------------------- | --------------------------------- |
| `voice_design`               | `text`, `language`, freeform `description`                         | `design_voice(...)`               | Yes, WaveKat 1.7B                 |
| `base`                       | `text`, `language`, reference audio + transcript (ICL)             | `prepare_voice()`, `synthesize()` | Yes, WaveKat 0.6B                 |
| `custom_voice`               | `text`, `language`, predefined speaker, optional style instruction | `synthesize_custom_voice(...)`    | Later, after verified ONNX export |
| `base`, x-vector-only        | `text` + speaker representation without ICL transcript             | Explicit capability-gated mode    | Later, only after parity          |
| Alternative exporter layouts | Same semantic capabilities, different graph contracts              | Adapter selected by `layout`      | Later                             |

Do not pretend a `voice_design` model can clone references or a `base` model can generate from a description directly. The workflow **design → save WAV → create clone profile with Base → synthesize** requires two _different_ model installations.

## 4. Public API (proposed)

Retain the spirit of `pocketsynth`'s `REQUEST_API_VERSION=1`, strict requests, `measure_request`, prepared voices, typed errors, and offline metadata inspection. Do not promise an ABI-equivalent drop-in module without deliberate contract comparison.

```python
from qwen3synth import (
    Qwen3Runtime, SynthesisRequest, VoiceDesignRequest,
    GenerationConfig, ReferenceVoice,
)

# One-time: create a synthetic reference from a description.
with Qwen3Runtime.from_pretrained("voice-design-1.7b", quality="int4") as designer:
    sample = designer.design_voice(
        VoiceDesignRequest(
            id="design-001",
            text="Welcome to this demonstration of a clear and natural speaking voice.",
            language="en",
            description="Warm mature British male baritone, measured, natural, clear.",
        ),
        config=GenerationConfig(seed=42),
    )
    sample.save_wav("reference.wav")

# Reuse the chosen waveform and exact spoken transcript with Base.
with Qwen3Runtime.from_pretrained("base-0.6b", quality="int4") as clone:
    voice = clone.prepare_voice(ReferenceVoice(
        audio="reference.wav",
        transcript="Welcome to this demonstration of a clear and natural speaking voice.",
        language="en",
    ))
    voice.save("voices/british_narrator.qvoice")
    result = clone.synthesize(
        SynthesisRequest(id="line-001", text="Now I can read new text.", language="en"),
        voice=voice,
        config=GenerationConfig(seed=42),
    )
    result.save_wav("cloned.wav")
```

**Examples are target APIs, not evidence they already run.** The `.qvoice` profile is our proposed versioned container, not an upstream standard.

### Required types and behavior

- `SynthesisRequest(id, text, language)`: strictly **one** complete speakable text request. No silent splitting, no document parsing, no auto language detection, no unsupported SSMD directives.
- `VoiceDesignRequest(id, text, language, description)`: explicit natural-language voice prompt. Must reject empty description and wrong model type; keep design and cloning workflows distinct.
- `ReferenceVoice(audio, transcript, language, source_id=None)`: supports local PCM WAV initially, optionally in-memory normalized samples later. Reject unreadable/compressed audio, nonfinite samples, insufficient duration, unsupported sample rate, channels/layout without defined policy, and missing ICL transcript.
- `PreparedVoice`: typed, reusable, immutable identity anchored to normalized audio hash, exact reference transcript hash, bundle ID + graph ABI + catalog revision + profile schema. Only reusable against compatible Base bundle. Profile reuse must not depend on private Python object serialization.
- `GenerationConfig`: validated temperature, top-k/top-p where supported, repetition penalty (only if supported), max codec frames, stop/EOS controls, seed. Enforce _actual exported model capabilities_, not fictional options.
- `SynthesisResult`: finite mono `numpy.float32`, actual sample rate (24,000 for WaveKat), request ID, language, generation configuration, metadata, backend timings and diagnostics. Empty word-timing array unless measured by a real aligner.
- `RequestMeasure(amount, maximum, unit="tokens", fits)` uses the **same encoder** as strict synthesis. Over-capacity measurement reports `fits=False`; synthesis raises before inference.
- `request_api_contract()` and `runtime_identity()` are metadata-only and deterministic; no model loading or downloads.
- Typed errors: `UnsupportedModelCapabilityError`, `SynthesisInputTooLongError`, `InvalidReferenceAudioError`, `InvalidReferenceTranscriptError`, `IncompatibleVoiceProfileError`, `InvalidModelLayoutError`, `ModelGenerationError`, plus re-exported stable `onnxvoice` asset/session errors.

### Strict vs convenience

`runtime.synthesize()` encodes one request once; capacity-checks once; invokes one complete model generation operation or raises. _One generation_ internally has many ONNX decode steps; never confuse “one synthesis request” with “one ONNX `session.run()`”. `design_voice()` follows the same strict semantics. Convenience `synthesize_to_wav()` may explicitly split at model limits and join, with opt-in sentence segmentation, keeping all joining/mixing outside the strict API. Do not reuse Phrasplit as a hard dependency unless intentionally permitted; a small in-project deterministic splitter is sufficient for MVP.

## 5. Required from `onnxvoice` (separate changes, same engineering milestone)

Add Qwen to `onnxvoice` with **an explicit versioned layout** such as `qwen3-wavekat-v1`; do not hardcode it to `qwen3` globally. All graph signatures must be discovered and asserted via ONNX Runtime metadata.

### Low-level pipeline: VoiceDesign

1. Load pinned config, tokenizer assets, embedding arrays, and 4 graph roles.
2. Receive prepared text token IDs and instruction token IDs; assemble the exact exporter-specific prompt embedding layout.
3. Run talker prefill; initialize and validate KV tensors.
4. Autoregressively invoke talker decode; select first codec-group tokens with validated sampling/EOS handling.
5. Invoke code predictor to generate remaining codec groups with exporter-specific ordering.
6. Decode codec groups through vocoder to 24 kHz waveform; validate finite samples; handle frame/audio alignment precisely.
7. Return backend metadata: frame count, stop reason, seed and sampling policy, timings, graph variant and provider(s).

### Low-level pipeline: Base ICL voice clone

1. Decode a permitted mono WAV; normalize/resample according to the **exact graph contract**. Preserve raw source hash and prepared audio hash separately.
2. Compute mel/STFT frontend if the speaker encoder expects mel rather than raw audio. Inspect actual graph inputs instead of guessing: WaveKat's model card describes log-mel speaker conditioning and an audio tokenizer encoder.
3. Execute FP32 `speaker_encoder` and `tokenizer_encoder` once per prepared voice; capture the speaker vector and 16-group reference codes (12 Hz).
4. Assemble reference transcript + target text, speaker embedding, reference audio codes, special tokens, and attention masks exactly as indicated by validated exporter behavior.
5. Prefill, decode, code-predict, prepend/condition reference codes as appropriate, vocode, and trim the reference portion to yield **target-only** audio.
6. Cache reference conditioning only when bound to exact model revision/ABI, audio-transcript hash, sample rate, and preprocessing version.

### Graph and numerical correctness gates

- Record the **actual input/output names, ranks, dtypes, optional inputs, dynamic axes, and external initializer paths** of all six WaveKat graphs. Never invent a KV-cache tensor naming convention.
- Assert cache growth, position indexing, causal attention, prefill-vs-decode behavior, multimodal codebook group ordering, EOS, and stop-token handling.
- Avoid CPU copying of large tensors where supported, but optimize only after a correct baseline.
- Ensure ONNX external `.onnx.data` files remain adjacent to their owning graph when building content-addressed installations; ONNX Runtime cannot automatically resolve an arbitrarily relocated external tensor file.
- INT4 is **weight-only** quantization and can still require FP32 operations and large peak memory. `cpu` and `cuda` are capability-checked separately; do not promise all execution providers work.
- Open local bundle must never invoke HTTP. `onnxvoice` validates roles, hashes, graph layout and provider before synthesis.

## 6. Own text tokenizer; no Python tokenizer toolkits

Implement a deterministic, tested **Qwen-compatible BPE** frontend in `qwen3synth/tokenizer.py` using shipped `tokenizer/` JSON assets. This is more than `text.split()`:

- Interpret the exact tokenizer configuration of the **selected export** (`tokenizer.json`, vocabulary, merges, normalizer, pre-tokenizer, special tokens, added tokens, Unicode handling).
- Implement BPE merge-rank selection and byte-level decoding/encoding, respecting duplicate/unknown/special token rules and offsets. Do not assume every exporter has the same tokenizer assets.
- Produce exact token ID fixtures for ASCII, punctuation, multiple spaces, emoji, Chinese, Japanese, German umlauts, mixed-script text, newlines, unknown strings, and explicit special tokens.
- `encode(text)` and `encode_instruction(description)` must apply distinct Qwen prompt templates and language/control tokens as required by the model type. Token IDs and embeddings may require nontrivial multimodal alignment.
- Add a tokenizer conformance suite with **committed expected token IDs** generated independently from an accepted pinned upstream corpus. Normal tests cannot import, download, or execute Qwen/Transformers/tokenizers.
- If faithful tokenizer parity is not achieved with standard-library Unicode handling, **block release**; do not substitute a simplified tokenizer and claim correctness.

Normalizing text is a limited, deterministic engine policy; document-level number/date expansion and pronunciation guidance remain caller-owned.

## 7. Reference audio and reusable voice profiles

Implement owned WAV parsing/writing (PCM16 and possibly PCM float32) with `wave`, `struct`, and NumPy; use a precisely tested resampler and mel frontend (matching each export's requirements), not a silent dependency on `librosa`. A high-quality polyphase resampler can be implemented with NumPy, with fixture-based phase/length checks.

**20-second design → clone requirement:**

1. VoiceDesign accepts arbitrary descriptive prompt plus a sufficiently long natural reference transcript. Generate once, select a preferred sample and save `reference.wav` + exact transcript + prompt. Do **not** promise exact 20.000 s from a text length; duration is model-controlled.
2. Report measured duration, peak/dBFS, clipping, RMS, sample rate, frame count. Reject silence, NaN and implausible values.
3. For the WaveKat Base export, upstream reports **3–10 seconds works well**. Support extraction of a voiced 3–10-second segment from a 20-second master along with **a matching transcript for that selected segment**. Never auto-truncate the WAV while keeping the full transcript.
4. Prepared profiles persist versioned metadata and deterministic numeric arrays; never use pickle. Prefer a ZIP container with JSON plus `.npy` arrays and checksums, archive traversal protection, size limits and `allow_pickle=False`.
5. Store `voice_design_prompt`, `reference_text`, raw/prepared audio fingerprint, profile schema, selected bundle revision and export layout. The clone profile is portable only to a demonstrably compatible layout/revision; incompatible profiles raise typed errors, not warnings.

A voice description is not a stable voice identity. The selected reference audio (and its exact transcript) is the reusable artifact.

## 8. CLI proposal

```bash
qwen3synth models list
qwen3synth models inspect voice-design-1.7b
qwen3synth models install voice-design-1.7b --quality int4
qwen3synth models install base-0.6b --quality int4
qwen3synth check --bundle base-0.6b --offline

qwen3synth voice design \
  --bundle voice-design-1.7b --quality int4 \
  --prompt-file voice.txt --text-file reference.txt \
  --language en --seed 42 --output reference.wav

qwen3synth voice inspect reference.wav --transcript-file reference.txt
qwen3synth voice prepare \
  --bundle base-0.6b --reference reference.wav \
  --transcript-file reference.txt --output voices/narrator.qvoice

qwen3synth synthesize \
  --bundle base-0.6b --voice voices/narrator.qvoice \
  --text 'A different paragraph in the same synthetic voice.' \
  --language en --output output.wav --offline
```

Additional: `--bundle-dir`, `--provider`, `--cache-dir`, `--force-download` (install only), `--refresh-catalog` (discovery/install only), `--seed`, `--max-frames`, `--temperature`, `--json`, `--verbose`; require options to be actually supported before advertising them. Output errors must have stable exit codes and actionable descriptions. Local/offline modes never reach the network. Default `voice design` and `synthesize` must **not** implicitly install models: require explicit install or deliberate opt-in.

## 9. PyPI, packaging, CI, constraints

Use `setuptools_scm` with `write_to = "qwen3synth/_version.py"`; `./qwen3synth` package root (no `src`), `py.typed`, `python -m qwen3synth` and console entry point. Initial base deps should be limited to `numpy`, `onnxvoice` (range resolved after adapter release). `[cpu]` and `[gpu]` extras select `onnxvoice` providers. No Torch/Transformers/HF Python client/ONNX export dependency even in optional extras.

CI matrix: Python 3.10–3.14 where supported by dependency wheels and OS; Linux/Windows/macOS appropriately gated, and a **Termux/Android feasibility probe** for ONNX Runtime wheel availability rather than claiming support. Default tests entirely offline, small mock ONNX graphs/fixtures only; large real models in manually triggered scheduled/integration CI or a dedicated host. Lint `ruff`; typecheck `mypy`; `pytest` with coverage; build both wheel/sdist; verify dependency graph for forbidden inference dependencies. Do not put 2–10 GB model data in Git, PyPI, source distributions, or ordinary CI artifacts.

License: choose an explicit independent source-code license (Apache-2.0 matches `onnxvoice`/`pocketsynth` snapshots). Preserve original Qwen model and conversion metadata licenses; do not imply official Qwen/Alibaba endorsement. Name/PyPI checks are availability signals, **not trademark clearance**.

## 10. Test strategy, release gates, and operational realism

**Layer A — unit (offline, no ONNX assets):** text/token fixture parity; special tokens; languages; input validation; metadataless imports; WAV headers, resampling and mel behavior; graph role validation with fake backend; capacity limits; generation options; typed error mapping; deterministic profile serialization/deserialization and incompatibility.

**Layer B — small graph-contract tests:** synthetic ONNX models validate the exact names/ranks/types from recorded graph manifests; simulate prefill, decode, KV append, EOS, codebook and trim behavior. Ensure a fake `onnxvoice` raises when graph inputs shift unexpectedly.

**Layer C — real model tests (explicit, tagged `integration`):** pinned WaveKat INT4 VoiceDesign and Base CPU runs. Validate non-silent finite WAV, expected 24 kHz, voice-design instruction changes output, Base voice-clone produces target speech, deterministic run with same seed where supported, offline redownload avoidance, and compatibility with selected GPU EPs where available. Measure RTF, memory peak and cold/warm load; **do not set arbitrary pass/fail speed claims**.

**Layer D — perceptual/parity tests:** pinned reference waveforms and deterministic greedy token traces where available. Compare BPE token IDs exactly; reference speaker embeddings/mel frontends within justified numeric tolerances; group-0 and residual codebook traces; waveform lengths/trimming; listening evaluation across at least English and German. Golden fixtures may be prepared once from audited exporter behavior, but normal tests never execute the exporter helper code.

Security and reproducibility: pinned 40-character revisions; expected file sizes + SHA-256 or materialized verified hash; immutable source and selected variant metadata; no following arbitrary untrusted JSON URLs; timeout/retry/cancellation; bounded memory/file extraction; forbid `.npy`/`.npz` object or pickle; revalidate on reopen.

## 11. Implementation phases and acceptance criteria

| Phase                             | Work                                                                                                                                                                        | Hard exit condition                                                                                        |
| --------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| P0 — proof of ABI                 | Pin and inspect one WaveKat INT4 VoiceDesign and one INT4 Base export; list every component, external data file, inputs/outputs, tokenizer, special tokens, required memory | Signed-off `docs/onnx-abi.md`; demonstrated real ONNX session loading                                      |
| P1 — `onnxvoice` Qwen design      | Add layout-specific adapter; talker prefill/decode, KV state, sampling, code predictor, vocoder                                                                             | Generate valid audible text from prompt tokens and instruction tokens without upstream Python scripts      |
| P2 — Qwen tokenizer + strict API  | Implement faithful BPE, template assembly, capability API, request validation, strict `design_voice()`                                                                      | Exact test fixture token matches, stable typed results, no hidden splitting                                |
| P3 — `onnxvoice` Base voice clone | Add reference speaker/tokenizer encoders, mel/resampler parity, ICL assembly and trimming                                                                                   | Two distinct references result in meaningfully distinct output voices; transcript/ABI constraints enforced |
| P4 — profile + CLI                | Reusable versioned voice profile, local/offline paths, 20s-master-to-short-reference workflow                                                                               | Design sample → cloned new utterance with zero network during cached synthesis                             |
| P5 — catalog + docs + release     | Pair with verified `qwen3-onnx-bundles`, full docs/CI, publish                                                                                                              | Real-model integration and artifact integrity gates pass; forbidden dependencies absent                    |
| P6 — later                        | CustomVoice, 1.7B Base, other exporters, streaming, other providers                                                                                                         | Separate explicit model- and layout-specific parity gates                                                  |

**MVP definition of done:** Two locally installed ONNX bundles can (1) synthesize an original voice from a written description and text; (2) save the reference; (3) prepare a compatible Base clone profile using its matching transcript; and (4) render new utterances. The entire pipeline works offline after model installation, with no `qwen-tts`, PyTorch, Transformers, Hugging Face Python client, librosa, soundfile, or upstream inference-script dependency.

## 12. Open risks and decisions that must be proven, not guessed

1. **BPE parity** (including Unicode pretokenization and control tokens) is a potential critical path.
2. **ONNX KV cache** layout and reference-code interleaving are exporter-specific; lock the layout and test it before generalizing.
3. **Audio conditioning parity** (speaker mel + codec encoder + reference trim) requires precise numerical tests; audible output alone does not prove correct cloning.
4. **Quantization and RAM**: WaveKat lists ~4.27 GB for its VoiceDesign INT4 directory and ~2.2 GB for its Base INT4 directory, excluding other costs. CPU inference may be slow, and peaks can be higher than on-disk size. Do not assert mobile/Termux support without benchmarking.
5. **Long prompt/duration control**: 20s is a _target_, not an exact-duration capability. Seed support does not guarantee stable speaker identity across separate VoiceDesign runs.
6. **Multilingual content**: advertised supported languages are not proof of equal pronunciation/voice similarity quality.
7. **Publisher changes**: revisions, tokenizer and graph files can change. Catalog pin, hash and layout-adapter version must prevent silent mutation.

## 13. Research sources and local reference design

**Local repository snapshots supplied for this brief:** `context_onnxvoice.md` and `context_pocketsynth.md`, reconstructed from their accompanying Codecrate unpackers (strict header/hash check). Relevant design precedent: `onnxvoice/docs/architecture.md`, `onnxvoice/onnxvoice/systems/base.py`, `pocketsynth/docs/architecture.md`, `pocketsynth/pocketsynth/runtime.py`, `pocketsynth/pocketsynth/types.py`, and `pocketsynth/pyproject.toml`.

**Public sources verified 2026-10-14:**

- [Official Qwen3-TTS repository](https://github.com/QwenLM/Qwen3-TTS)
- [WaveKat VoiceDesign 1.7B ONNX model card](https://huggingface.co/wavekat/Qwen3-TTS-1.7B-VoiceDesign-ONNX)
- [WaveKat Base 0.6B ONNX model card](https://huggingface.co/wavekat/Qwen3-TTS-0.6B-Base-ONNX)
- [xkos alternative ONNX export](https://huggingface.co/xkos/Qwen3-TTS-12Hz-1.7B-ONNX)
- [ONNX Community exporter status](https://huggingface.co/onnx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign)

**Scope caveat:** Feasibility is supported by third-party model cards and published graph sets; it does **not** mean that a complete run of the proposed reimplementation was executed in this review. The implementation must earn its own graph-ABI and quality acceptance tests.
