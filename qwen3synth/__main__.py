from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import __version__
from .catalog import bundle_record, install_bundle, load_catalog
from .profile import PreparedVoice
from .runtime import Qwen3Runtime
from .types import GenerationConfig, ReferenceVoice, SynthesisRequest, VoiceDesignRequest


def _gen(args: argparse.Namespace) -> GenerationConfig:
    return GenerationConfig(
        temperature=args.temperature,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        max_new_tokens=args.max_tokens,
        seed=args.seed,
    )


def _add_gen(p: argparse.ArgumentParser) -> None:
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--repetition-penalty", type=float, default=1.05)
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--seed", type=int)


def _runtime(args: argparse.Namespace) -> Qwen3Runtime:
    if getattr(args, "bundle_dir", None):
        return Qwen3Runtime.load(args.bundle_dir, quality=args.quality, provider=args.provider)
    return Qwen3Runtime.from_installed(
        args.bundle,
        quality=args.quality,
        provider=args.provider,
        catalog=getattr(args, "catalog", None),
        cache_dir=getattr(args, "cache_dir", None),
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="qwen3synth")
    ap.add_argument("--version", action="version", version=__version__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    models = sub.add_parser("models")
    ms = models.add_subparsers(dest="models_cmd", required=True)
    ml = ms.add_parser("list")
    ml.add_argument("--catalog")
    mi = ms.add_parser("inspect")
    mi.add_argument("bundle")
    mi.add_argument("--catalog")
    minst = ms.add_parser("install")
    minst.add_argument("bundle")
    minst.add_argument("--quality", default="int4", choices=["int4", "fp32"])
    minst.add_argument("--catalog")
    minst.add_argument("--cache-dir")
    minst.add_argument("--force", action="store_true")

    voice = sub.add_parser("voice")
    vs = voice.add_subparsers(dest="voice_cmd", required=True)
    vd = vs.add_parser("design")
    vd.add_argument("--bundle", default="voice-design-1.7b")
    vd.add_argument("--bundle-dir")
    vd.add_argument("--quality", default="int4")
    vd.add_argument("--provider", default="cpu")
    vd.add_argument("--catalog")
    vd.add_argument("--cache-dir")
    vd.add_argument("--text", required=True)
    vd.add_argument("--prompt", required=True)
    vd.add_argument("--language", default="en")
    vd.add_argument("--output", required=True)
    _add_gen(vd)
    vp = vs.add_parser("prepare")
    vp.add_argument("--bundle", default="base-0.6b")
    vp.add_argument("--bundle-dir")
    vp.add_argument("--quality", default="int4")
    vp.add_argument("--provider", default="cpu")
    vp.add_argument("--catalog")
    vp.add_argument("--cache-dir")
    vp.add_argument("--reference", required=True)
    vp.add_argument("--transcript", required=True)
    vp.add_argument("--language", default="en")
    vp.add_argument("--output", required=True)

    sy = sub.add_parser("synthesize")
    sy.add_argument("--bundle", default="base-0.6b")
    sy.add_argument("--bundle-dir")
    sy.add_argument("--quality", default="int4")
    sy.add_argument("--provider", default="cpu")
    sy.add_argument("--catalog")
    sy.add_argument("--cache-dir")
    sy.add_argument("--voice", required=True)
    sy.add_argument("--text", required=True)
    sy.add_argument("--language", default="en")
    sy.add_argument("--output", required=True)
    _add_gen(sy)

    args = ap.parse_args(argv)
    if args.cmd == "models":
        cat = load_catalog(args.catalog)
        if args.models_cmd == "list":
            for key, b in cat["bundles"].items():
                print(f"{key}\t{b['model_type']}\t{','.join(b['profiles'])}")
            return 0
        if args.models_cmd == "inspect":
            rec = bundle_record(args.bundle, cat)
            print(json.dumps(rec.__dict__, indent=2, sort_keys=True))
            return 0
        if args.models_cmd == "install":
            print(
                install_bundle(
                    args.bundle,
                    args.quality,
                    catalog_source=args.catalog,
                    cache_dir=Path(args.cache_dir) if args.cache_dir else None,
                    force=args.force,
                )
            )
            return 0
    if args.cmd == "voice" and args.voice_cmd == "design":
        with _runtime(args) as rt:
            result = rt.design_voice(
                VoiceDesignRequest("cli", args.text, args.prompt, args.language), config=_gen(args)
            )
            result.save_wav(args.output)
        return 0
    if args.cmd == "voice" and args.voice_cmd == "prepare":
        with _runtime(args) as rt:
            v = rt.prepare_voice(ReferenceVoice(args.reference, args.transcript, args.language))
            v.save(args.output)
        return 0
    if args.cmd == "synthesize":
        voice_obj = PreparedVoice.load(args.voice)
        with _runtime(args) as rt:
            result = rt.synthesize(
                SynthesisRequest("cli", args.text, args.language),
                voice=voice_obj,
                config=_gen(args),
            )
            result.save_wav(args.output)
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
