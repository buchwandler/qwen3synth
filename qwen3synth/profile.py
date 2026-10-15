from __future__ import annotations

import hashlib
import io
import json
import math
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import numpy as np

from .errors import IncompatibleVoiceProfileError

PROFILE_SCHEMA = 2
_PROFILE_LAYOUT = "qwen3-wavekat-v1"
_SAMPLE_RATE = 24_000
_MAX_REFERENCE_SAMPLES = 10 * _SAMPLE_RATE
_MAX_ARCHIVE_BYTES = 32 * 1024 * 1024
_MAX_MANIFEST_BYTES = 1024 * 1024
_MAX_ARRAY_BYTES = 15 * 1024 * 1024
_MEMBERS = {"manifest.json", "speaker_embedding.npy", "reference_codes.npy"}
_MEMBER_LIMITS = {
    "manifest.json": _MAX_MANIFEST_BYTES,
    "speaker_embedding.npy": _MAX_ARRAY_BYTES,
    "reference_codes.npy": _MAX_ARRAY_BYTES,
}
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def _npy_bytes(array: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, array, allow_pickle=False)
    return buf.getvalue()


def _invalid(reason: str) -> IncompatibleVoiceProfileError:
    return IncompatibleVoiceProfileError(reason)


def _validate_digest(value: Any, field: str) -> None:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise _invalid(f"invalid {field} SHA-256")


def _validate_manifest(manifest: Any) -> dict[str, Any]:
    if not isinstance(manifest, dict):
        raise _invalid("qvoice manifest must be a JSON object")
    required = {
        "schema",
        "layout",
        "model_id",
        "model_revision",
        "transcript",
        "language",
        "source_id",
        "audio_sha256",
        "prepared_audio_sha256",
        "reference_samples",
        "reference_token_ids",
        "files",
    }
    missing = required - manifest.keys()
    if missing:
        raise _invalid(f"qvoice manifest is missing required keys: {', '.join(sorted(missing))}")
    if type(manifest["schema"]) is not int or manifest["schema"] != PROFILE_SCHEMA:
        raise _invalid("unsupported qvoice schema")
    if manifest["layout"] != _PROFILE_LAYOUT:
        raise _invalid("incompatible qvoice layout")
    if not isinstance(manifest["model_id"], str) or not manifest["model_id"].strip():
        raise _invalid("qvoice model_id must not be empty")
    for field in ("transcript", "language"):
        if not isinstance(manifest[field], str):
            raise _invalid(f"qvoice {field} must be a string")
    for field in ("model_revision", "source_id"):
        value = manifest[field]
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise _invalid(f"qvoice {field} must be a non-empty string or null")
    for field in ("audio_sha256", "prepared_audio_sha256"):
        _validate_digest(manifest[field], field)

    samples = manifest["reference_samples"]
    if type(samples) is not int or not 0 < samples <= _MAX_REFERENCE_SAMPLES:
        raise _invalid("invalid qvoice reference_samples")
    token_ids = manifest["reference_token_ids"]
    if (
        not isinstance(token_ids, list)
        or not token_ids
        or any(type(token_id) is not int or token_id < 0 for token_id in token_ids)
    ):
        raise _invalid("qvoice reference_token_ids must contain non-negative integers")

    files = manifest["files"]
    if not isinstance(files, dict) or set(files) != {
        "speaker_embedding.npy",
        "reference_codes.npy",
    }:
        raise _invalid("qvoice manifest must checksum both profile arrays")
    for name, digest in files.items():
        _validate_digest(digest, f"{name} checksum")
    return manifest


def _validate_archive_members(archive: zipfile.ZipFile) -> dict[str, zipfile.ZipInfo]:
    infos = archive.infolist()
    names = [info.filename for info in infos]
    if len(set(names)) != len(names):
        raise _invalid("duplicate qvoice archive member names")
    if len(infos) != len(_MEMBERS):
        if "manifest.json" not in names:
            raise _invalid("qvoice archive is missing manifest.json")
        raise _invalid("unexpected qvoice archive members")

    for name in names:
        posix = PurePosixPath(name)
        windows = PureWindowsPath(name)
        if (
            posix.is_absolute()
            or windows.is_absolute()
            or windows.drive
            or "\\" in name
            or ".." in posix.parts
        ):
            raise _invalid("unsafe qvoice archive path")
    if "manifest.json" not in names:
        raise _invalid("qvoice archive is missing manifest.json")
    if set(names) != _MEMBERS:
        raise _invalid("unexpected qvoice archive members")

    members = {info.filename: info for info in infos}
    for name, info in members.items():
        if info.flag_bits & 0x1:
            raise _invalid("encrypted qvoice archive members are not supported")
        if info.file_size > _MEMBER_LIMITS[name]:
            if name == "manifest.json":
                raise _invalid("qvoice manifest is too large")
            raise _invalid(f"qvoice archive member is too large: {name}")
    return members


def _load_array(data: bytes, name: str) -> np.ndarray:
    try:
        stream = io.BytesIO(data)
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            shape, _, dtype = np.lib.format.read_array_header_1_0(stream, max_header_size=16 * 1024)
        elif version == (2, 0):
            shape, _, dtype = np.lib.format.read_array_header_2_0(stream, max_header_size=16 * 1024)
        else:
            raise _invalid(f"unsupported qvoice NumPy format: {name}")
    except IncompatibleVoiceProfileError:
        raise
    except (ValueError, OSError, EOFError, UnicodeError) as exc:
        raise _invalid(f"invalid qvoice NumPy array: {name}") from exc

    if dtype.hasobject:
        raise _invalid(f"qvoice object arrays are not supported: {name}")
    if name == "speaker_embedding.npy":
        if dtype.kind != "f":
            raise _invalid("qvoice speaker embedding must have a floating dtype")
        if len(shape) != 2 or shape[0] != 1 or shape[1] < 1:
            raise _invalid("qvoice speaker embedding must have shape (1, embedding_size)")
    else:
        if dtype.kind not in "iu":
            raise _invalid("qvoice reference codes must have an integer dtype")
        if len(shape) != 3 or shape[0] != 1 or shape[1] != 16 or shape[2] < 1:
            raise _invalid("qvoice reference codes must have shape (1, 16, frames)")

    expected_size = math.prod(shape) * dtype.itemsize
    payload_size = len(data) - stream.tell()
    if expected_size > _MAX_ARRAY_BYTES or expected_size != payload_size:
        raise _invalid(f"invalid or oversized qvoice NumPy array: {name}")
    try:
        array = np.load(io.BytesIO(data), allow_pickle=False)
    except (ValueError, OSError, EOFError) as exc:
        raise _invalid(f"invalid qvoice NumPy array: {name}") from exc
    if not isinstance(array, np.ndarray):
        raise _invalid(f"invalid qvoice NumPy array: {name}")
    return array


def _validate_arrays(speaker: np.ndarray, codes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if speaker.dtype.kind != "f":
        raise _invalid("qvoice speaker embedding must have a floating dtype")
    if speaker.ndim != 2 or speaker.shape[0] != 1 or speaker.shape[1] < 1:
        raise _invalid("qvoice speaker embedding must have shape (1, embedding_size)")
    if not np.isfinite(speaker).all():
        raise _invalid("qvoice speaker embedding contains non-finite values")

    if codes.dtype.kind not in "iu":
        raise _invalid("qvoice reference codes must have an integer dtype")
    if codes.ndim != 3 or codes.shape[0] != 1 or codes.shape[1] != 16 or codes.shape[2] < 1:
        raise _invalid("qvoice reference codes must have shape (1, 16, frames)")
    if codes.dtype.kind == "u" and np.any(codes > np.iinfo(np.int64).max):
        raise _invalid("qvoice reference code ID exceeds int64 range")
    if np.any(codes < 0):
        raise _invalid("qvoice reference code IDs must be non-negative")
    speaker = np.asarray(speaker, dtype=np.float32)
    if not np.isfinite(speaker).all():
        raise _invalid("qvoice speaker embedding contains non-finite float32 values")
    return speaker, np.asarray(codes, dtype=np.int64)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_profile(path: str | Path) -> tuple[dict[str, Any], np.ndarray, np.ndarray, str]:
    profile_path = Path(path)
    archive_size = profile_path.stat().st_size
    if archive_size > _MAX_ARCHIVE_BYTES:
        raise _invalid("qvoice archive is too large")

    try:
        with zipfile.ZipFile(profile_path, "r") as archive:
            members = _validate_archive_members(archive)
            if members["manifest.json"].file_size > _MAX_MANIFEST_BYTES:
                raise _invalid("qvoice manifest is too large")
            manifest_bytes = archive.read("manifest.json")
            try:
                manifest = _validate_manifest(json.loads(manifest_bytes))
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                raise _invalid("malformed qvoice manifest JSON") from exc

            blobs = {
                name: archive.read(name)
                for name in ("speaker_embedding.npy", "reference_codes.npy")
            }
    except IncompatibleVoiceProfileError:
        raise
    except (zipfile.BadZipFile, RuntimeError, OSError, EOFError, NotImplementedError) as exc:
        raise _invalid("malformed qvoice archive") from exc

    for name, data in blobs.items():
        if hashlib.sha256(data).hexdigest() != manifest["files"][name]:
            raise _invalid(f"qvoice checksum mismatch: {name}")
    speaker = _load_array(blobs["speaker_embedding.npy"], "speaker_embedding.npy")
    codes = _load_array(blobs["reference_codes.npy"], "reference_codes.npy")
    speaker, codes = _validate_arrays(speaker, codes)
    return manifest, speaker, codes, _sha256_file(profile_path)


@dataclass(frozen=True)
class VoiceProfileInfo:
    schema: int
    layout: str
    model_id: str
    model_revision: str | None
    language: str
    reference_samples: int
    audio_sha256: str
    prepared_audio_sha256: str
    profile_sha256: str


@dataclass(frozen=True)
class PreparedVoice:
    speaker_embedding: np.ndarray
    reference_codes: np.ndarray
    transcript: str
    language: str
    audio_sha256: str
    prepared_audio_sha256: str
    model_id: str
    model_revision: str | None
    layout: str = _PROFILE_LAYOUT
    source_id: str | None = None
    reference_samples: int = 0
    reference_token_ids: tuple[int, ...] = ()

    def save(self, path: str | Path) -> None:
        speaker = _npy_bytes(np.asarray(self.speaker_embedding, dtype=np.float32))
        codes = _npy_bytes(np.asarray(self.reference_codes, dtype=np.int64))
        manifest: dict[str, Any] = {
            "schema": PROFILE_SCHEMA,
            "layout": self.layout,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "transcript": self.transcript,
            "language": self.language,
            "source_id": self.source_id,
            "audio_sha256": self.audio_sha256,
            "prepared_audio_sha256": self.prepared_audio_sha256,
            "reference_samples": self.reference_samples,
            "reference_token_ids": list(self.reference_token_ids),
            "files": {
                "speaker_embedding.npy": hashlib.sha256(speaker).hexdigest(),
                "reference_codes.npy": hashlib.sha256(codes).hexdigest(),
            },
        }
        _validate_manifest(manifest)
        _validate_arrays(
            _load_array(speaker, "speaker_embedding.npy"),
            _load_array(codes, "reference_codes.npy"),
        )
        if len(speaker) > _MAX_ARRAY_BYTES or len(codes) > _MAX_ARRAY_BYTES:
            raise _invalid("qvoice profile arrays exceed the maximum size")
        manifest_bytes = json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8") + b"\n"
        if len(manifest_bytes) > _MAX_MANIFEST_BYTES:
            raise _invalid("qvoice manifest is too large")

        profile_path = Path(path)
        profile_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(profile_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("manifest.json", manifest_bytes)
            archive.writestr("speaker_embedding.npy", speaker)
            archive.writestr("reference_codes.npy", codes)

    @classmethod
    def load(cls, path: str | Path) -> PreparedVoice:
        manifest, speaker, codes, _ = _read_profile(path)
        return cls(
            speaker_embedding=speaker,
            reference_codes=codes,
            transcript=manifest["transcript"],
            language=manifest["language"],
            audio_sha256=manifest["audio_sha256"],
            prepared_audio_sha256=manifest["prepared_audio_sha256"],
            model_id=manifest["model_id"],
            model_revision=manifest["model_revision"],
            layout=manifest["layout"],
            source_id=manifest["source_id"],
            reference_samples=manifest["reference_samples"],
            reference_token_ids=tuple(manifest["reference_token_ids"]),
        )


def inspect_voice_profile(path: str | Path) -> VoiceProfileInfo:
    manifest, _, _, profile_sha256 = _read_profile(path)
    return VoiceProfileInfo(
        schema=manifest["schema"],
        layout=manifest["layout"],
        model_id=manifest["model_id"],
        model_revision=manifest["model_revision"],
        language=manifest["language"],
        reference_samples=manifest["reference_samples"],
        audio_sha256=manifest["audio_sha256"],
        prepared_audio_sha256=manifest["prepared_audio_sha256"],
        profile_sha256=profile_sha256,
    )
