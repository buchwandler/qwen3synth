from __future__ import annotations

import argparse
import sys
from pathlib import Path

from qwen3synth import ModelNotInstalledError, Qwen3Runtime, ReferenceVoice


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare a reusable .qvoice profile from a reference WAV and its exact transcript."
    )
    parser.add_argument(
        "--reference", required=True, type=Path, help="reference WAV, 0.5 to 10 seconds"
    )
    parser.add_argument(
        "--transcript", required=True, help="exact words spoken in the reference WAV"
    )
    parser.add_argument("--language", default="en")
    parser.add_argument("--quality", choices=("int4", "fp32"), default="int4")
    parser.add_argument("--provider", default="cpu")
    parser.add_argument("--output", type=Path, default=Path("narrator.qvoice"))
    args = parser.parse_args()
    if not args.reference.is_file():
        parser.error(f"reference WAV does not exist: {args.reference}")
    if not args.transcript.strip():
        parser.error("--transcript must not be empty")

    try:
        with Qwen3Runtime.from_installed(
            "base-0.6b", quality=args.quality, provider=args.provider
        ) as runtime:
            profile = runtime.prepare_voice(
                ReferenceVoice(
                    audio=args.reference,
                    transcript=args.transcript,
                    language=args.language,
                )
            )
            profile.save(args.output)
    except ModelNotInstalledError as exc:
        print(
            f"Base bundle is not installed. Run `onnxvoice install "
            f"qwen3:base-0.6b --quality {args.quality}`.\n{exc}",
            file=sys.stderr,
        )
        return 2

    if not args.output.is_file():
        raise RuntimeError(f"profile was not written: {args.output}")
    print(f"Saved {args.output}")
    print(
        "The .qvoice profile contains sensitive voice data. Keep it private and use authorized voices."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
