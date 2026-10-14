from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import shutil
import tempfile
import urllib.parse
import urllib.request
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any
from collections.abc import Iterable

from .errors import CatalogError, ModelNotInstalledError


@dataclass(frozen=True)
class BundleRecord:
    id: str
    repository: str
    revision: str
    model_type: str
    layout: str
    profiles: dict[str, dict[str, Any]]
    license: str


def default_cache_dir() -> Path:
    env = os.environ.get("QWEN3SYNTH_CACHE")
    if env:
        return Path(env).expanduser()
    return Path.home() / ".cache" / "qwen3synth"


def load_catalog(source: str | Path | None = None) -> dict[str, Any]:
    if source is None:
        return json.loads(
            files("qwen3synth").joinpath("catalog_seed.json").read_text(encoding="utf-8")
        )
    s = str(source)
    if s.startswith(("http://", "https://")):
        with urllib.request.urlopen(s, timeout=30) as r:
            return json.load(r)
    return json.loads(Path(s).read_text(encoding="utf-8"))


def bundle_record(bundle_id: str, catalog: dict[str, Any]) -> BundleRecord:
    bundles = catalog.get("bundles", {})
    match = bundle_id
    if match not in bundles:
        for key, value in bundles.items():
            if bundle_id in value.get("aliases", []):
                match = key
                break
    if match not in bundles:
        raise CatalogError(f"unknown bundle: {bundle_id}")
    b = bundles[match]
    src = b["source"]
    return BundleRecord(
        match,
        src["repository"],
        src["revision"],
        b["model_type"],
        b["layout"],
        b["profiles"],
        src.get("license", "unknown"),
    )


def installation_dir(bundle: BundleRecord, quality: str, cache_dir: Path | None = None) -> Path:
    root = cache_dir or default_cache_dir()
    return root / "models" / bundle.id / bundle.revision / quality


def find_installed(
    bundle_id: str,
    quality: str = "int4",
    *,
    catalog_source: str | Path | None = None,
    cache_dir: Path | None = None,
) -> Path:
    catalog = load_catalog(catalog_source)
    record = bundle_record(bundle_id, catalog)
    p = installation_dir(record, quality, cache_dir)
    if not (p / ".qwen3synth-install.json").exists():
        raise ModelNotInstalledError(f"bundle {record.id}:{quality} is not installed at {p}")
    return p


def _hf_tree(repository: str, revision: str) -> list[dict[str, Any]]:
    repo = urllib.parse.quote(repository, safe="/")
    rev = urllib.parse.quote(revision, safe="")
    url = (
        f"https://huggingface.co/api/models/{repo}/tree/{rev}?recursive=true&expand=true&limit=1000"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "qwen3synth/0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        data = json.load(r)
    if not isinstance(data, list):
        raise CatalogError("unexpected Hugging Face tree response")
    return [x for x in data if isinstance(x, dict) and x.get("type") == "file"]


def _matches(path: str, includes: Iterable[str]) -> bool:
    return any(fnmatch.fnmatch(path, pattern) for pattern in includes)


def _download(repository: str, revision: str, rel: str, destination: Path) -> tuple[int, str]:
    url = f"https://huggingface.co/{repository}/resolve/{revision}/{urllib.parse.quote(rel)}?download=true"
    destination.parent.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256()
    size = 0
    fd, tmp_name = tempfile.mkstemp(prefix=destination.name + ".", dir=str(destination.parent))
    try:
        with (
            os.fdopen(fd, "wb") as out,
            urllib.request.urlopen(
                urllib.request.Request(url, headers={"User-Agent": "qwen3synth/0"}), timeout=120
            ) as r,
        ):
            while True:
                chunk = r.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
                h.update(chunk)
                size += len(chunk)
        os.replace(tmp_name, destination)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    return size, h.hexdigest()


def install_bundle(
    bundle_id: str,
    quality: str = "int4",
    *,
    catalog_source: str | Path | None = None,
    cache_dir: Path | None = None,
    force: bool = False,
) -> Path:
    catalog = load_catalog(catalog_source)
    record = bundle_record(bundle_id, catalog)
    try:
        profile = record.profiles[quality]
    except KeyError as exc:
        raise CatalogError(f"bundle {record.id} has no quality {quality!r}") from exc
    dest = installation_dir(record, quality, cache_dir)
    marker = dest / ".qwen3synth-install.json"
    if marker.exists() and not force:
        return dest
    if force and dest.exists():
        shutil.rmtree(dest)
    files_meta = _hf_tree(record.repository, record.revision)
    includes = list(profile["include"])
    selected = sorted({str(x["path"]) for x in files_meta if _matches(str(x["path"]), includes)})
    if not selected:
        raise CatalogError("catalog include patterns selected no upstream files")
    manifest_files = []
    for rel in selected:
        size, digest = _download(record.repository, record.revision, rel, dest / rel)
        manifest_files.append({"path": rel, "size": size, "sha256": digest})
    marker.write_text(
        json.dumps(
            {
                "schema": 1,
                "bundle": record.id,
                "quality": quality,
                "repository": record.repository,
                "revision": record.revision,
                "layout": record.layout,
                "files": manifest_files,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return dest
