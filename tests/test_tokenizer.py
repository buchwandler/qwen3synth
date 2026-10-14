import json
from qwen3synth.tokenizer import ByteLevelBPETokenizer, _bytes_to_unicode


def test_tiny_byte_bpe(tmp_path):
    enc = _bytes_to_unicode()
    h, i = enc[ord("h")], enc[ord("i")]
    vocab = {h: 0, i: 1, h + i: 2, "<x>": 3}
    (tmp_path / "vocab.json").write_text(json.dumps(vocab))
    (tmp_path / "merges.txt").write_text("#version: 0.2\n" + h + " " + i + "\n")
    (tmp_path / "added_tokens.json").write_text(json.dumps({"<x>": 3}))
    t = ByteLevelBPETokenizer(tmp_path)
    assert t.encode("hi<x>") == [2, 3]
