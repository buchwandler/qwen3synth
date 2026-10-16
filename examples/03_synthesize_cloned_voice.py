from __future__ import annotations

import argparse
import sys
from pathlib import Path

from qwen3synth import (
    GenerationConfig,
    IncompatibleVoiceProfileError,
    ModelNotInstalledError,
    PreparedVoice,
    Qwen3Runtime,
    SynthesisRequest,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Synthesize new words with an installed Base bundle and a .qvoice profile."
    )
    parser.add_argument("--voice", required=True, type=Path, help="prepared .qvoice profile")
    parser.add_argument("--text", required=True, help="new text to synthesize")
    parser.add_argument("--language", default="en")
    parser.add_argument("--quality", choices=("int4", "fp32"), default="int4")
    parser.add_argument("--provider", default="cpu")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--output", type=Path, default=Path("cloned-voice.wav"))
    args = parser.parse_args()
    if not args.voice.is_file():
        parser.error(f"voice profile does not exist: {args.voice}")
    if not args.text.strip():
        parser.error("--text must not be empty")
    try:
        voice = PreparedVoice.load(args.voice)
    except IncompatibleVoiceProfileError as exc:
        parser.error(f"invalid .qvoice profile: {exc}")

    try:
        with Qwen3Runtime.from_installed(
            "base-0.6b", quality=args.quality, provider=args.provider
        ) as runtime:
            result = runtime.synthesize(
                SynthesisRequest(
                    id="cloned-synthesis-example", text=args.text, language=args.language
                ),
                voice=voice,
                config=GenerationConfig(seed=args.seed, max_new_tokens=args.max_new_tokens),
            )
            args.output.parent.mkdir(parents=True, exist_ok=True)
            result.save_wav(args.output)
    except ModelNotInstalledError as exc:
        print(
            f"Base bundle is not installed. Run `onnxvoice install "
            f"qwen3:base-0.6b --quality {args.quality}`.\n{exc}",
            file=sys.stderr,
        )
        return 2

    print(f"Wrote {args.output} ({result.duration_seconds:.2f}s, {result.sample_rate} Hz)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
