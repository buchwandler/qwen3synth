import json

from qwen3synth.catalog import bundle_record, load_catalog


def test_catalog_facade_uses_onnxvoice_catalog_source(tmp_path):
    source = {
        "bundles": {
            "base-0.6b": {
                "aliases": ["base", "clone"],
                "layout": "qwen3-wavekat-v1",
                "model_type": "base",
                "profiles": {"fp32": {}, "int4": {}},
                "source": {
                    "repository": "fixture/qwen3-base",
                    "revision": "a" * 40,
                    "license": "apache-2.0",
                },
            }
        }
    }
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(json.dumps(source), encoding="utf-8")

    catalog = load_catalog(catalog_path, cache_dir=tmp_path / "cache")
    record = bundle_record("clone", catalog)

    assert record.id == "base-0.6b"
    assert record.revision == "a" * 40
    assert record.layout == "qwen3-wavekat-v1"
    assert set(record.profiles) == {"fp32", "int4"}
