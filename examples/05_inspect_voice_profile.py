from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from qwen3synth import IncompatibleVoiceProfileError, inspect_voice_profile


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate and display .qvoice metadata without opening a model or using the network."
    )
    parser.add_argument("voice", type=Path, help="profile to inspect")
    args = parser.parse_args()
    if not args.voice.is_file():
        parser.error(f"voice profile does not exist: {args.voice}")
    try:
        profile = inspect_voice_profile(args.voice)
    except IncompatibleVoiceProfileError as exc:
        parser.error(f"invalid .qvoice profile: {exc}")
    print(json.dumps(asdict(profile), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
