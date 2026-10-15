from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pytest

from qwen3synth.tokenizer import ByteLevelBPETokenizer

_FIXTURE = Path(__file__).parent / "fixtures" / "qwen3_tokenizer_parity.json"
_ASSET_FILES = {
    "tokenizer_json": "tokenizer.json",
    "vocab": "vocab.json",
    "merges": "merges.txt",
    "tokenizer_config": "tokenizer_config.json",
    "special_tokens": "special_tokens_map.json",
    "added_tokens": "added_tokens.json",
}


def _tokenizer_directory() -> Path:
    configured = os.environ.get("QWEN3SYNTH_TEST_TOKENIZER_DIR")
    if configured:
        return Path(configured)
    if os.environ.get("QWEN3SYNTH_REQUIRE_TOKENIZER_PARITY") == "1":
        pytest.fail("QWEN3SYNTH_TEST_TOKENIZER_DIR must point to the pinned tokenizer assets")
    pytest.skip("set QWEN3SYNTH_TEST_TOKENIZER_DIR to the pinned WaveKat tokenizer assets")


@pytest.fixture(scope="session")
def qwen_parity_data() -> tuple[ByteLevelBPETokenizer, list[dict[str, Any]]]:
    fixture = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    tokenizer_dir = _tokenizer_directory()
    expected_hashes = fixture["provenance"]["tokenizer_asset_sha256"]
    for component, filename in _ASSET_FILES.items():
        digest = hashlib.sha256((tokenizer_dir / filename).read_bytes()).hexdigest()
        assert digest == expected_hashes[component]
    return ByteLevelBPETokenizer(tokenizer_dir), fixture["cases"]
