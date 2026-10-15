from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from onnxvoice import NotInstalledError, OnnxVoice
from onnxvoice.errors import CatalogError as OnnxCatalogError
from onnxvoice.errors import OfflineError

from .errors import CatalogError, ModelNotInstalledError


@dataclass(frozen=True)
class BundleRecord:
    """Qwen3Synth-compatible view of an OnnxVoice Qwen3 catalog item."""

    id: str
    repository: str
    revision: str
    model_type: str
    layout: str
    profiles: dict[str, dict[str, Any]]
    license: str


def _manager(source: str | Path | None = None, cache_dir: str | Path | None = None) -> OnnxVoice:
    sources = {"qwen3": str(source)} if source is not None else None
    return OnnxVoice(cache_dir=cache_dir, catalog_sources=sources)


def load_catalog(
    source: str | Path | None = None, *, cache_dir: str | Path | None = None
) -> dict[str, Any]:
    """Load the normalized Qwen3 catalog through OnnxVoice."""
    manager = _manager(source, cache_dir)
    try:
        return manager.catalog.load_raw("qwen3", refresh=source is not None)
    except (OnnxCatalogError, OfflineError) as exc:
        raise CatalogError(str(exc)) from exc


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
    bundle = bundles[match]
    source = bundle["source"]
    return BundleRecord(
        id=match,
        repository=source["repository"],
        revision=source["revision"],
        model_type=bundle["model_type"],
        layout=bundle["layout"],
        profiles=bundle["profiles"],
        license=source.get("license", "unknown"),
    )


def find_installed(
    bundle_id: str,
    quality: str = "int4",
    *,
    catalog_source: str | Path | None = None,
    cache_dir: Path | None = None,
) -> Path:
    """Resolve an existing OnnxVoice installation without acquiring assets."""
    catalog = load_catalog(catalog_source, cache_dir=cache_dir)
    record = bundle_record(bundle_id, catalog)
    manager = _manager(catalog_source, cache_dir)
    try:
        return manager.resolve(f"qwen3:{record.id}", quality=quality).path
    except NotInstalledError as exc:
        raise ModelNotInstalledError(f"bundle {record.id}:{quality} is not installed") from exc


def install_bundle(
    bundle_id: str,
    quality: str = "int4",
    *,
    catalog_source: str | Path | None = None,
    cache_dir: Path | None = None,
    force: bool = False,
) -> Path:
    """Install a Qwen3 bundle using OnnxVoice's shared catalog and asset store."""
    catalog = load_catalog(catalog_source, cache_dir=cache_dir)
    record = bundle_record(bundle_id, catalog)
    installation = _manager(catalog_source, cache_dir).install(
        f"qwen3:{record.id}", quality=quality, force=force
    )
    return installation.path
