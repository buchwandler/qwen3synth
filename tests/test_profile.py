from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from qwen3synth.errors import IncompatibleVoiceProfileError
from qwen3synth.profile import (
    _MAX_ARCHIVE_BYTES,
    _MAX_ARRAY_BYTES,
    PROFILE_SCHEMA,
    PreparedVoice,
    VoiceProfileInfo,
    inspect_voice_profile,
)


def _voice() -> PreparedVoice:
    return PreparedVoice(
        speaker_embedding=np.zeros((1, 4), dtype=np.float32),
        reference_codes=np.zeros((1, 16, 3), dtype=np.int64),
        transcript="hello",
        language="en",
        audio_sha256="a" * 64,
        prepared_audio_sha256="b" * 64,
        model_id="wavekat/base-0.6b",
        model_revision="c" * 40,
        reference_samples=5_760,
        reference_token_ids=(1, 2, 3),
    )


def _npy_bytes(array: np.ndarray, *, allow_pickle: bool = False) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=allow_pickle)
    return buffer.getvalue()


def _read_archive(path: Path) -> tuple[dict[str, Any], bytes, bytes]:
    with zipfile.ZipFile(path) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        speaker = archive.read("speaker_embedding.npy")
        codes = archive.read("reference_codes.npy")
    return manifest, speaker, codes


def _rewrite_profile(
    path: Path,
    *,
    manifest: dict[str, Any] | None = None,
    raw_manifest: bytes | None = None,
    speaker: np.ndarray | bytes | None = None,
    codes: np.ndarray | bytes | None = None,
    refresh_checksums: bool = True,
    entry_names: list[str] | None = None,
    renames: dict[str, str] | None = None,
    extras: tuple[tuple[str, bytes], ...] = (),
) -> None:
    original_manifest, speaker_data, codes_data = _read_archive(path)
    manifest = original_manifest if manifest is None else manifest
    if isinstance(speaker, np.ndarray):
        speaker_data = _npy_bytes(speaker, allow_pickle=speaker.dtype.hasobject)
    elif isinstance(speaker, bytes):
        speaker_data = speaker
    if isinstance(codes, np.ndarray):
        codes_data = _npy_bytes(codes, allow_pickle=codes.dtype.hasobject)
    elif isinstance(codes, bytes):
        codes_data = codes
    if refresh_checksums:
        manifest["files"] = {
            "speaker_embedding.npy": hashlib.sha256(speaker_data).hexdigest(),
            "reference_codes.npy": hashlib.sha256(codes_data).hexdigest(),
        }
    manifest_data = raw_manifest or (json.dumps(manifest).encode("utf-8"))
    contents = {
        "manifest.json": manifest_data,
        "speaker_embedding.npy": speaker_data,
        "reference_codes.npy": codes_data,
    }
    names = entry_names or ["manifest.json", "speaker_embedding.npy", "reference_codes.npy"]
    renames = renames or {}
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in names:
            archive.writestr(renames.get(name, name), contents[name])
        for name, data in extras:
            archive.writestr(name, data)


@pytest.fixture
def profile_path(tmp_path: Path) -> Path:
    path = tmp_path / "voice.qvoice"
    _voice().save(path)
    return path


def test_profile_schema2_roundtrip_and_content_identity(profile_path: Path) -> None:
    voice = PreparedVoice.load(profile_path)
    assert voice.reference_samples == 5_760
    assert voice.reference_token_ids == (1, 2, 3)
    assert voice.reference_codes.shape == (1, 16, 3)

    info = inspect_voice_profile(profile_path)
    assert isinstance(info, VoiceProfileInfo)
    assert info.schema == PROFILE_SCHEMA == 2
    assert info.layout == "qwen3-wavekat-v1"
    assert info.model_id == "wavekat/base-0.6b"
    assert info.model_revision == "c" * 40
    assert info.reference_samples == 5_760
    assert info.profile_sha256 == hashlib.sha256(profile_path.read_bytes()).hexdigest()
    assert inspect_voice_profile(profile_path).profile_sha256 == info.profile_sha256


def test_profile_rejects_missing_manifest(profile_path: Path) -> None:
    _rewrite_profile(profile_path, entry_names=["speaker_embedding.npy", "reference_codes.npy"])
    with pytest.raises(IncompatibleVoiceProfileError, match="missing manifest"):
        PreparedVoice.load(profile_path)


def test_profile_rejects_duplicate_member_names(profile_path: Path) -> None:
    with pytest.warns(UserWarning, match="Duplicate name"):
        _rewrite_profile(
            profile_path,
            entry_names=["manifest.json", "speaker_embedding.npy", "speaker_embedding.npy"],
        )
    with pytest.raises(IncompatibleVoiceProfileError, match="duplicate"):
        PreparedVoice.load(profile_path)


@pytest.mark.parametrize(
    "member_name",
    ["../manifest.json", "/manifest.json", "C:\\manifest.json", "folder\\manifest.json"],
)
def test_profile_rejects_unsafe_member_paths(profile_path: Path, member_name: str) -> None:
    _rewrite_profile(profile_path, renames={"manifest.json": member_name})
    with pytest.raises(IncompatibleVoiceProfileError, match="unsafe"):
        PreparedVoice.load(profile_path)


def test_profile_rejects_extra_members(profile_path: Path) -> None:
    _rewrite_profile(profile_path, extras=(("extra.bin", b"extra"),))
    with pytest.raises(IncompatibleVoiceProfileError, match="unexpected"):
        PreparedVoice.load(profile_path)


def test_profile_rejects_oversized_archive(tmp_path: Path) -> None:
    path = tmp_path / "oversized.qvoice"
    with path.open("wb") as stream:
        stream.truncate(_MAX_ARCHIVE_BYTES + 1)
    with pytest.raises(IncompatibleVoiceProfileError, match="archive is too large"):
        inspect_voice_profile(path)


def test_profile_rejects_non_zip_archive(tmp_path: Path) -> None:
    path = tmp_path / "invalid.qvoice"
    path.write_bytes(b"not a zip archive")
    with pytest.raises(IncompatibleVoiceProfileError, match="malformed qvoice archive"):
        inspect_voice_profile(path)


def test_profile_rejects_oversized_manifest(profile_path: Path) -> None:
    _rewrite_profile(profile_path, raw_manifest=b"0" * (1024 * 1024 + 1))
    with pytest.raises(IncompatibleVoiceProfileError, match="manifest is too large"):
        PreparedVoice.load(profile_path)


def test_profile_rejects_oversized_member(profile_path: Path) -> None:
    _rewrite_profile(profile_path, speaker=b"0" * (_MAX_ARRAY_BYTES + 1))
    with pytest.raises(IncompatibleVoiceProfileError, match="member is too large"):
        PreparedVoice.load(profile_path)


def test_profile_rejects_oversized_npy_shape_before_allocation(profile_path: Path) -> None:
    header = io.BytesIO()
    np.lib.format.write_array_header_1_0(
        header,
        {
            "descr": np.lib.format.dtype_to_descr(np.dtype(np.float32)),
            "fortran_order": False,
            "shape": (1, _MAX_ARRAY_BYTES),
        },
    )
    _rewrite_profile(profile_path, speaker=header.getvalue())
    with pytest.raises(IncompatibleVoiceProfileError, match="oversized qvoice NumPy array"):
        PreparedVoice.load(profile_path)


def test_profile_rejects_malformed_json(profile_path: Path) -> None:
    _rewrite_profile(profile_path, raw_manifest=b"{")
    with pytest.raises(IncompatibleVoiceProfileError, match="malformed qvoice manifest"):
        PreparedVoice.load(profile_path)


def test_profile_rejects_missing_manifest_keys(profile_path: Path) -> None:
    manifest, _, _ = _read_archive(profile_path)
    del manifest["model_revision"]
    _rewrite_profile(profile_path, manifest=manifest)
    with pytest.raises(IncompatibleVoiceProfileError, match="missing required keys"):
        PreparedVoice.load(profile_path)


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"schema": 1}, "unsupported qvoice schema"),
        ({"audio_sha256": "z" * 64}, "invalid audio_sha256 SHA-256"),
        ({"model_id": ""}, "model_id must not be empty"),
        ({"layout": "other-layout"}, "incompatible qvoice layout"),
        ({"reference_samples": 0}, "invalid qvoice reference_samples"),
        ({"reference_samples": 240_001}, "invalid qvoice reference_samples"),
        ({"reference_samples": True}, "invalid qvoice reference_samples"),
        ({"reference_token_ids": []}, "reference_token_ids"),
        ({"reference_token_ids": [-1]}, "reference_token_ids"),
        ({"reference_token_ids": [True]}, "reference_token_ids"),
    ],
)
def test_profile_rejects_invalid_manifest_values(
    profile_path: Path, updates: dict[str, Any], message: str
) -> None:
    manifest, _, _ = _read_archive(profile_path)
    manifest.update(updates)
    _rewrite_profile(profile_path, manifest=manifest)
    with pytest.raises(IncompatibleVoiceProfileError, match=message):
        PreparedVoice.load(profile_path)


@pytest.mark.parametrize(
    ("speaker", "message"),
    [
        (np.ones((1, 4), dtype=np.int64), "floating dtype"),
        (np.asarray([[np.nan]], dtype=np.float32), "non-finite"),
        (np.ones(4, dtype=np.float32), "shape"),
        (np.ones((2, 4), dtype=np.float32), "shape"),
    ],
)
def test_profile_rejects_invalid_speaker_embeddings(
    profile_path: Path, speaker: np.ndarray, message: str
) -> None:
    _rewrite_profile(profile_path, speaker=speaker)
    with pytest.raises(IncompatibleVoiceProfileError, match=message):
        PreparedVoice.load(profile_path)


def test_profile_rejects_object_arrays(profile_path: Path) -> None:
    object_data = _npy_bytes(np.asarray([[object()]], dtype=object), allow_pickle=True)
    _rewrite_profile(profile_path, speaker=object_data)
    with pytest.raises(
        IncompatibleVoiceProfileError, match="qvoice object arrays are not supported"
    ):
        PreparedVoice.load(profile_path)


@pytest.mark.parametrize(
    ("codes", "message"),
    [
        (np.zeros((1, 16, 3), dtype=np.float32), "integer dtype"),
        (np.zeros((16, 3), dtype=np.int64), "shape"),
        (np.zeros((1, 15, 3), dtype=np.int64), "shape"),
        (np.full((1, 16, 3), -1, dtype=np.int64), "non-negative"),
    ],
)
def test_profile_rejects_invalid_reference_codes(
    profile_path: Path, codes: np.ndarray, message: str
) -> None:
    _rewrite_profile(profile_path, codes=codes)
    with pytest.raises(IncompatibleVoiceProfileError, match=message):
        PreparedVoice.load(profile_path)


def test_profile_rejects_checksum_mismatch(profile_path: Path) -> None:
    _rewrite_profile(profile_path, speaker=b"tampered", refresh_checksums=False)
    with pytest.raises(IncompatibleVoiceProfileError, match="checksum mismatch"):
        PreparedVoice.load(profile_path)
