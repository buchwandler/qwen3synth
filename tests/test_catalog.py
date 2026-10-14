from qwen3synth.catalog import bundle_record, load_catalog


def test_seed_catalog_has_two_mvp_models():
    c = load_catalog()
    assert set(c["bundles"]) == {"voice-design-1.7b", "base-0.6b"}
    assert len(bundle_record("base", c).revision) == 40
