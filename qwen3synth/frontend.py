from __future__ import annotations

from .tokenizer import ByteLevelBPETokenizer


def encode_target(text: str, tokenizer: ByteLevelBPETokenizer) -> tuple[int, ...]:
    wrapped = f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
    return tuple(tokenizer.encode(wrapped))


def encode_reference(transcript: str, tokenizer: ByteLevelBPETokenizer) -> tuple[int, ...]:
    wrapped = f"<|im_start|>assistant\n{transcript}<|im_end|>\n"
    return tuple(tokenizer.encode(wrapped))


def encode_instruction(description: str, tokenizer: ByteLevelBPETokenizer) -> tuple[int, ...]:
    wrapped = f"<|im_start|>user\n{description}<|im_end|>\n"
    return tuple(tokenizer.encode(wrapped))


def normalize_language(language: str) -> str:
    aliases = {
        "en": "english",
        "zh": "chinese",
        "ja": "japanese",
        "ko": "korean",
        "de": "german",
        "fr": "french",
        "es": "spanish",
        "it": "italian",
        "pt": "portuguese",
        "ru": "russian",
    }
    key = language.strip().lower()
    return aliases.get(key, key)
