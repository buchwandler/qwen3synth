from __future__ import annotations

import json
import unicodedata
from pathlib import Path


def _bytes_to_unicode() -> dict[int, str]:
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs), strict=True))


def _is_letter(c: str) -> bool:
    return unicodedata.category(c).startswith("L")


def _is_number(c: str) -> bool:
    return unicodedata.category(c).startswith("N")


def _pretokenize(text: str) -> list[str]:
    # Pure-Python approximation of the Qwen/GPT byte-level pre-tokenizer.
    # It intentionally preserves leading spaces with the following token.
    out: list[str] = []
    i = 0
    contractions = ("'s", "'t", "'re", "'ve", "'m", "'ll", "'d")
    while i < len(text):
        lower = text[i:].lower()
        matched = next((c for c in contractions if lower.startswith(c)), None)
        if matched:
            out.append(text[i : i + len(matched)])
            i += len(matched)
            continue
        c = text[i]
        if c in "\r\n":
            j = i + 1
            while j < len(text) and text[j] in "\r\n":
                j += 1
            out.append(text[i:j])
            i = j
            continue
        if c.isspace():
            j = i
            while j < len(text) and text[j].isspace() and text[j] not in "\r\n":
                j += 1
            if j < len(text) and (_is_letter(text[j]) or not text[j].isspace()):
                prefix = text[i:j]
                if _is_letter(text[j]):
                    k = j + 1
                    while k < len(text) and _is_letter(text[k]):
                        k += 1
                    out.append(prefix + text[j:k])
                    i = k
                    continue
                if _is_number(text[j]):
                    k = j
                    while k < len(text) and _is_number(text[k]) and k - j < 3:
                        k += 1
                    out.append(prefix + text[j:k])
                    i = k
                    continue
                k = j + 1
                while (
                    k < len(text)
                    and not text[k].isspace()
                    and not _is_letter(text[k])
                    and not _is_number(text[k])
                ):
                    k += 1
                out.append(prefix + text[j:k])
                i = k
                continue
            out.append(text[i:j])
            i = j
            continue
        if _is_letter(c):
            j = i + 1
            while j < len(text) and _is_letter(text[j]):
                j += 1
            out.append(text[i:j])
            i = j
            continue
        if _is_number(c):
            j = i
            while j < len(text) and _is_number(text[j]) and j - i < 3:
                j += 1
            out.append(text[i:j])
            i = j
            continue
        j = i + 1
        while (
            j < len(text)
            and not text[j].isspace()
            and not _is_letter(text[j])
            and not _is_number(text[j])
        ):
            j += 1
        out.append(text[i:j])
        i = j
    return out


class ByteLevelBPETokenizer:
    """Small, dependency-free byte-level BPE reader for WaveKat's Qwen tokenizer assets."""

    def __init__(self, directory: str | Path):
        root = Path(directory)
        vocab_path = root / "vocab.json"
        merges_path = root / "merges.txt"
        if not vocab_path.exists() or not merges_path.exists():
            raise FileNotFoundError("tokenizer requires vocab.json and merges.txt")
        self.encoder: dict[str, int] = json.loads(vocab_path.read_text(encoding="utf-8"))
        self.byte_encoder = _bytes_to_unicode()
        self.bpe_ranks: dict[tuple[str, str], int] = {}
        rank = 0
        for line in merges_path.read_text(encoding="utf-8").splitlines():
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) == 2:
                self.bpe_ranks[(parts[0], parts[1])] = rank
                rank += 1
        self.special: dict[str, int] = {}
        added = root / "added_tokens.json"
        if added.exists():
            obj = json.loads(added.read_text(encoding="utf-8"))
            if isinstance(obj, dict):
                self.special.update({str(k): int(v) for k, v in obj.items()})
        cfg = root / "tokenizer_config.json"
        if cfg.exists():
            obj = json.loads(cfg.read_text(encoding="utf-8"))
            dec = obj.get("added_tokens_decoder", {})
            if isinstance(dec, dict):
                for key, value in dec.items():
                    if isinstance(value, dict) and "content" in value:
                        self.special[str(value["content"])] = int(key)
        self._special_sorted = sorted(self.special, key=len, reverse=True)
        self._bpe_cache: dict[str, tuple[str, ...]] = {}

    @staticmethod
    def _pairs(word: tuple[str, ...]) -> set[tuple[str, str]]:
        return set(zip(word, word[1:], strict=False))

    def _bpe(self, token: str) -> tuple[str, ...]:
        if token in self._bpe_cache:
            return self._bpe_cache[token]
        if not token:
            return ()
        word = tuple(token)
        pairs = self._pairs(word)
        if not pairs:
            if len(self._bpe_cache) < 100_000:
                self._bpe_cache[token] = word
            return word
        while True:
            bigram = min(pairs, key=lambda p: self.bpe_ranks.get(p, 10**18))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new: list[str] = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                except ValueError:
                    new.extend(word[i:])
                    break
                new.extend(word[i:j])
                i = j
                if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
                    new.append(first + second)
                    i += 2
                else:
                    new.append(word[i])
                    i += 1
            word = tuple(new)
            if len(word) == 1:
                break
            pairs = self._pairs(word)
        if len(self._bpe_cache) < 100_000:
            self._bpe_cache[token] = word
        return word

    def _ordinary(self, text: str) -> list[int]:
        ids: list[int] = []
        for piece in _pretokenize(text):
            encoded = "".join(self.byte_encoder[b] for b in piece.encode("utf-8"))
            for tok in self._bpe(encoded):
                try:
                    ids.append(self.encoder[tok])
                except KeyError as exc:
                    raise ValueError(
                        f"tokenizer vocabulary does not contain BPE token {tok!r}"
                    ) from exc
        return ids

    def encode(self, text: str) -> list[int]:
        if not self.special:
            return self._ordinary(text)
        out: list[int] = []
        pos = 0
        while pos < len(text):
            best_at = len(text)
            best_token: str | None = None
            for tok in self._special_sorted:
                idx = text.find(tok, pos)
                if idx != -1 and idx < best_at:
                    best_at = idx
                    best_token = tok
            if best_token is None:
                out.extend(self._ordinary(text[pos:]))
                break
            if best_at > pos:
                out.extend(self._ordinary(text[pos:best_at]))
            out.append(self.special[best_token])
            pos = best_at + len(best_token)
        return out
