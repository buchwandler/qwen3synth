from __future__ import annotations

from typing import Any

from qwen3synth.frontend import (
    encode_instruction,
    encode_reference,
    encode_target,
    normalize_language,
)
from qwen3synth.tokenizer import ByteLevelBPETokenizer


def test_frontend_encoders_match_pinned_qwen_parity_cases(
    qwen_parity_data: tuple[ByteLevelBPETokenizer, list[dict[str, Any]]],
) -> None:
    tokenizer, cases = qwen_parity_data
    expected = {case["name"]: tuple(case["ids"]) for case in cases}

    assert (
        encode_target("Hello from Qwen3Synth.", tokenizer) == expected["assistant_wrapped_target"]
    )
    assert (
        encode_instruction("Warm, clear adult narrator, calm pace, natural delivery.", tokenizer)
        == expected["user_wrapped_voice_design"]
    )
    assert (
        encode_reference("A bright day ahead.", tokenizer)
        == expected["base_reference_transcript_wrapped"]
    )


def test_normalize_language_aliases() -> None:
    assert normalize_language("en") == "english"
    assert normalize_language(" DE ") == "german"
    assert normalize_language("Klingon") == "klingon"
