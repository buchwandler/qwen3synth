from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path
from typing import Any

from onnxvoice import OnnxVoice
from onnxvoice.errors import CatalogError as OnnxCatalogError
from onnxvoice.errors import OfflineError

from .errors import CatalogError


@dataclass(frozen=True)
class DiscoveredBundle:
    id: str
    ref: str
    display_name: str
    model_type: str
    layout: str
    sample_rate: int
    languages: tuple[str, ...]
    precisions: tuple[str, ...]
    source_revision: str
    runtime_available: bool
    metadata: dict[str, Any]


def discover_bundles(
    *,
    offline: bool = False,
    refresh: bool = False,
    cache_dir: str | Path | None = None,
    catalog: str | Path | None = None,
) -> tuple[DiscoveredBundle, ...]:
    """Discover Qwen3 bundle metadata through ONNXVoice without opening models."""
    catalog_source = str(catalog) if catalog is not None else None
    local_catalog = catalog_source is not None and not catalog_source.startswith(
        ("http://", "https://")
    )
    manager = OnnxVoice(
        cache_dir=cache_dir,
        catalog_sources={"qwen3": catalog_source} if catalog_source is not None else None,
        offline=offline,
    )
    refresh_catalog = (refresh or local_catalog) and (not offline or local_catalog)
    try:
        items = manager.list("qwen3", refresh=refresh_catalog)
    except (OnnxCatalogError, OfflineError) as exc:
        raise CatalogError(str(exc)) from exc
    runtime_available = find_spec("onnxruntime") is not None

    bundles: list[DiscoveredBundle] = []
    for item in items:
        metadata = item.metadata
        assert item.sample_rate is not None
        bundles.append(
            DiscoveredBundle(
                id=item.id,
                ref=item.ref,
                display_name=item.id,
                model_type=metadata["model_type"],
                layout=metadata["layout"],
                sample_rate=item.sample_rate,
                languages=tuple(metadata.get("languages", ())),
                precisions=tuple(sorted(metadata["profiles"])),
                source_revision=metadata["source_revision"],
                runtime_available=runtime_available,
                metadata=dict(metadata),
            )
        )
    return tuple(sorted(bundles, key=lambda bundle: bundle.id))
