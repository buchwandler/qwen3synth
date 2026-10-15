import numpy as np

from qwen3synth.audio import read_wav, resample_linear, write_wav


def test_wav_roundtrip(tmp_path):
    x = np.linspace(-0.5, 0.5, 2400, dtype=np.float32)
    p = tmp_path / "x.wav"
    write_wav(p, x, 24000)
    y, sr, digest = read_wav(p)
    assert sr == 24000
    assert len(y) == len(x)
    assert len(digest) == 64
    assert np.max(np.abs(x - y)) < 1e-4


def test_resample_shape():
    x = np.zeros(16000, dtype=np.float32)
    assert len(resample_linear(x, 16000, 24000)) == 24000
