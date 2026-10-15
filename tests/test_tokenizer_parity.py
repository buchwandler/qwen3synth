from __future__ import annotations

import pytest

from qwen3synth.tokenizer import ByteLevelBPETokenizer


@pytest.mark.parametrize("case_index", range(19))
def test_qwen_tokenizer_exact_parity(
    case_index: int,
    qwen_parity_data: tuple[ByteLevelBPETokenizer, list[dict[str, object]]],
) -> None:
    tokenizer, cases = qwen_parity_data
    case = cases[case_index]
    assert tokenizer.encode(case["text"]) == case["ids"]
