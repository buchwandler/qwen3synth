from __future__ import annotations

import argparse
import sys
from pathlib import Path

from qwen3synth import (
    GenerationConfig,
    InvalidModelLayoutError,
    Qwen3Runtime,
    VoiceDesignRequest,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Open a complete unmanaged local VoiceDesign bundle and write a WAV."
    )
    parser.add_argument(
        "--bundle-dir", required=True, type=Path, help="complete local bundle directory"
    )
    parser.add_argument("--prompt", required=True, help="description of the desired voice")
    parser.add_argument("--text", required=True, help="exact words to synthesize")
    parser.add_argument("--language", default="en")
    parser.add_argument("--quality", choices=("int4", "fp32"), default="int4")
    parser.add_argument("--provider", default="cpu")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--output", type=Path, default=Path("local-voice-design.wav"))
    args = parser.parse_args()
    if not args.bundle_dir.is_dir():
        parser.error(f"local bundle directory does not exist: {args.bundle_dir}")
    if not args.prompt.strip() or not args.text.strip():
        parser.error("--prompt and --text must not be empty")

    try:
        with Qwen3Runtime.load(
            args.bundle_dir, quality=args.quality, provider=args.provider
        ) as runtime:
            result = runtime.design_voice(
                VoiceDesignRequest(
                    id="local-bundle-example",
                    text=args.text,
                    description=args.prompt,
                    language=args.language,
                ),
                config=GenerationConfig(seed=args.seed, max_new_tokens=args.max_new_tokens),
            )
            args.output.parent.mkdir(parents=True, exist_ok=True)
            result.save_wav(args.output)
    except InvalidModelLayoutError as exc:
        print(
            f"The local directory is not a complete VoiceDesign bundle: {exc}\n"
            "Provide its config, tokenizer, embeddings, graph files, and adjacent .onnx.data files. "
            f"For a managed bundle, run `onnxvoice install "
            f"qwen3:voice-design-1.7b --quality {args.quality}` and use example 01.",
            file=sys.stderr,
        )
        return 2

    print(f"Wrote {args.output} ({result.duration_seconds:.2f}s, {result.sample_rate} Hz)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
