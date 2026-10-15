from __future__ import annotations

import hashlib
import wave
from pathlib import Path

import numpy as np

from .errors import InvalidReferenceAudioError

TARGET_SAMPLE_RATE = 24_000


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _pcm24_to_float(raw: bytes) -> np.ndarray:
    b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
    x = (
        b[:, 0].astype(np.int32)
        | (b[:, 1].astype(np.int32) << 8)
        | (b[:, 2].astype(np.int32) << 16)
    )
    x = np.where(x & 0x800000, x | ~0xFFFFFF, x)
    return x.astype(np.float32) / 8388608.0


def read_wav(path: str | Path, *, mono: bool = True) -> tuple[np.ndarray, int, str]:
    p = Path(path)
    try:
        raw_file = p.read_bytes()
        with wave.open(str(p), "rb") as wf:
            channels = wf.getnchannels()
            sr = wf.getframerate()
            width = wf.getsampwidth()
            n = wf.getnframes()
            comptype = wf.getcomptype()
            raw = wf.readframes(n)
    except (wave.Error, OSError) as exc:
        raise InvalidReferenceAudioError(f"not a readable PCM WAV: {p}") from exc
    digest = sha256_bytes(raw_file)
    if comptype != "NONE":
        raise InvalidReferenceAudioError("compressed WAV is not supported")
    if channels < 1 or channels > 8:
        raise InvalidReferenceAudioError(f"unsupported channel count: {channels}")
    if sr <= 0:
        raise InvalidReferenceAudioError("invalid sample rate")
    if width == 1:
        x = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif width == 2:
        x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 3:
        x = _pcm24_to_float(raw)
    elif width == 4:
        x = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise InvalidReferenceAudioError(f"unsupported PCM sample width: {width}")
    if len(x) % channels:
        raise InvalidReferenceAudioError("malformed interleaved PCM data")
    x = x.reshape(-1, channels)
    if mono:
        x = x.mean(axis=1)
    else:
        x = x[:, 0]
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0 or not np.all(np.isfinite(x)):
        raise InvalidReferenceAudioError("reference WAV is empty or non-finite")
    return x, sr, digest


def resample_linear(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    x = np.asarray(samples, dtype=np.float32).reshape(-1)
    if source_rate == target_rate:
        return x.copy()
    if source_rate <= 0 or target_rate <= 0:
        raise ValueError("sample rates must be positive")
    out_len = max(1, int(round(len(x) * target_rate / source_rate)))
    old = np.arange(len(x), dtype=np.float64)
    new = np.linspace(0.0, max(len(x) - 1, 0), out_len, dtype=np.float64)
    return np.interp(new, old, x).astype(np.float32)


def write_wav(path: str | Path, samples: np.ndarray, sample_rate: int = TARGET_SAMPLE_RATE) -> None:
    x = np.asarray(samples, dtype=np.float32).reshape(-1)
    if not np.all(np.isfinite(x)):
        raise ValueError("audio contains non-finite samples")
    pcm = np.round(np.clip(x, -1.0, 1.0) * 32767.0).astype("<i2")
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(p), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
