# Examples

The prompt-to-WAV example uses an already installed VoiceDesign model. Install it explicitly, then run the example:

```bash
qwen3synth models install voice-design-1.7b --quality int4
python examples/01_voice_design_prompt_to_wav.py
```

It writes `qwen3_prompt_demo.wav` in the current directory. The script opens the installed model with `Qwen3Runtime.from_installed()`, which does not acquire model assets.

Equivalent CLI command:

```bash
qwen3synth voice design \
  --bundle voice-design-1.7b \
  --quality int4 \
  --provider cpu \
  --prompt 'Warm, clear adult narrator, calm pace, natural delivery.' \
  --text 'Hello from Qwen3Synth.' \
  --language en \
  --seed 42 \
  --max-tokens 64 \
  --output qwen3_prompt_demo.wav
```

`max_new_tokens` (CLI `--max-tokens`) is a generation cap, not an exact duration request. VoiceDesign controls the resulting duration.
