from __future__ import annotations

import argparse

from qwen3synth import discover_bundles


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Show Qwen3 bundle catalog metadata without opening ONNX sessions."
    )
    parser.add_argument("--offline", action="store_true", help="use cached catalog metadata only")
    parser.add_argument("--cache-dir", help="ONNXVoice shared cache directory")
    args = parser.parse_args()

    if not args.offline:
        print("Uncached catalog metadata may be fetched from the network.")
    for bundle in discover_bundles(offline=args.offline, cache_dir=args.cache_dir):
        profiles = ", ".join(bundle.precisions)
        print(
            f"{bundle.ref}: type={bundle.model_type}, layout={bundle.layout}, "
            f"sample_rate={bundle.sample_rate}, profiles={profiles}, "
            f"revision={bundle.source_revision}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
