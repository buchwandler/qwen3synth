from __future__ import annotations

import hashlib
import math
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
    raw_file = p.read_bytes()
    digest = sha256_bytes(raw_file)
    try:
        with wave.open(str(p), "rb") as wf:
            channels = wf.getnchannels()
            sr = wf.getframerate()
            width = wf.getsampwidth()
            n = wf.getnframes()
            comptype = wf.getcomptype()
            raw = wf.readframes(n)
    except (wave.Error, OSError) as exc:
        raise InvalidReferenceAudioError(f"not a readable PCM WAV: {p}") from exc
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


def _hz_to_mel(freq: np.ndarray) -> np.ndarray:
    # Slaney-style mel scale used by librosa when htk=False.
    f_min = 0.0
    f_sp = 200.0 / 3
    mels = (freq - f_min) / f_sp
    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_min) / f_sp
    logstep = math.log(6.4) / 27.0
    mask = freq >= min_log_hz
    mels = np.asarray(mels)
    mels[mask] = min_log_mel + np.log(freq[mask] / min_log_hz) / logstep
    return mels


def _mel_to_hz(mels: np.ndarray) -> np.ndarray:
    f_min = 0.0
    f_sp = 200.0 / 3
    freqs = f_min + f_sp * mels
    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_min) / f_sp
    logstep = math.log(6.4) / 27.0
    mask = mels >= min_log_mel
    freqs = np.asarray(freqs)
    freqs[mask] = min_log_hz * np.exp(logstep * (mels[mask] - min_log_mel))
    return freqs


def mel_filterbank(sr: int, n_fft: int, n_mels: int, fmin: float, fmax: float) -> np.ndarray:
    fftfreqs = np.linspace(0.0, sr / 2, 1 + n_fft // 2)
    min_mel, max_mel = _hz_to_mel(np.array([fmin, fmax], dtype=np.float64))
    mel_f = _mel_to_hz(np.linspace(min_mel, max_mel, n_mels + 2))
    fdiff = np.diff(mel_f)
    ramps = mel_f[:, np.newaxis] - fftfreqs[np.newaxis, :]
    weights = np.zeros((n_mels, len(fftfreqs)), dtype=np.float64)
    for i in range(n_mels):
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i + 2] / fdiff[i + 1]
        weights[i] = np.maximum(0.0, np.minimum(lower, upper))
    enorm = 2.0 / (mel_f[2 : n_mels + 2] - mel_f[:n_mels])
    weights *= enorm[:, np.newaxis]
    return weights.astype(np.float32)


def compute_log_mel(
    samples: np.ndarray,
    *,
    sr: int = TARGET_SAMPLE_RATE,
    n_fft: int = 1024,
    hop_length: int = 256,
    win_length: int = 1024,
    n_mels: int = 128,
    fmin: float = 0.0,
    fmax: float = 12000.0,
) -> np.ndarray:
    x = np.asarray(samples, dtype=np.float32).reshape(-1)
    if len(x) < win_length:
        x = np.pad(x, (0, win_length - len(x)))
    frame_count = 1 + (len(x) - win_length) // hop_length
    shape = (frame_count, win_length)
    strides = (x.strides[0] * hop_length, x.strides[0])
    frames = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides).copy()
    window = np.hanning(win_length + 1)[:-1].astype(np.float32)
    frames *= window
    spectrum = np.fft.rfft(frames, n=n_fft, axis=1)
    power = (np.abs(spectrum) ** 2).astype(np.float32)
    filt = mel_filterbank(sr, n_fft, n_mels, fmin, fmax)
    mel = power @ filt.T
    mel = np.log(np.clip(mel, 1e-5, None)).astype(np.float32)
    return mel[np.newaxis, :, :]
