import numpy as np
from qwen3synth.profile import PreparedVoice


def test_profile_roundtrip(tmp_path):
    p = PreparedVoice(
        np.zeros((1, 4), np.float32),
        np.zeros((1, 16, 3), np.int64),
        "hello",
        "en",
        "a" * 64,
        "b" * 64,
        "model",
        "rev",
    )
    f = tmp_path / "v.qvoice"
    p.save(f)
    q = PreparedVoice.load(f)
    assert q.transcript == "hello"
    assert q.reference_codes.shape == (1, 16, 3)
