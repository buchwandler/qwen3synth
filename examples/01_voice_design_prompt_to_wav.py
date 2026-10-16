from __future__ import annotations

import argparse
import sys
from pathlib import Path

from qwen3synth import GenerationConfig, ModelNotInstalledError, Qwen3Runtime, VoiceDesignRequest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Design a voice with an installed VoiceDesign bundle and write a WAV."
    )
    parser.add_argument("--prompt", required=True, help="description of the desired voice")
    parser.add_argument("--text", required=True, help="exact words to synthesize")
    parser.add_argument("--language", default="en")
    parser.add_argument("--quality", choices=("int4", "fp32"), default="int4")
    parser.add_argument("--provider", default="cpu")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--output", type=Path, default=Path("voice-design.wav"))
    args = parser.parse_args()
    if not args.prompt.strip():
        parser.error("--prompt must not be empty")
    if not args.text.strip():
        parser.error("--text must not be empty")

    try:
        with Qwen3Runtime.from_installed(
            "voice-design-1.7b", quality=args.quality, provider=args.provider
        ) as runtime:
            result = runtime.design_voice(
                VoiceDesignRequest(
                    id="voice-design-example",
                    text=args.text,
                    description=args.prompt,
                    language=args.language,
                ),
                config=GenerationConfig(seed=args.seed, max_new_tokens=args.max_new_tokens),
            )
            args.output.parent.mkdir(parents=True, exist_ok=True)
            result.save_wav(args.output)
    except ModelNotInstalledError as exc:
        print(
            f"VoiceDesign bundle is not installed. Run `onnxvoice install "
            f"qwen3:voice-design-1.7b --quality {args.quality}`.\n{exc}",
            file=sys.stderr,
        )
        return 2

    print(f"Wrote {args.output} ({result.duration_seconds:.2f}s, {result.sample_rate} Hz)")
    print(f"Spoken transcript: {args.text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
