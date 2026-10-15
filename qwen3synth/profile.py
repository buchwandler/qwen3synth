from __future__ import annotations

import hashlib
import io
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .errors import IncompatibleVoiceProfileError

PROFILE_SCHEMA = 1


def _npy_bytes(array: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, array, allow_pickle=False)
    return buf.getvalue()


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
    layout: str = "qwen3-wavekat-v1"
    source_id: str | None = None

    reference_samples: int = 0
    reference_token_ids: tuple[int, ...] = ()

    def save(self, path: str | Path) -> None:
        speaker = _npy_bytes(np.asarray(self.speaker_embedding, dtype=np.float32))
        codes = _npy_bytes(np.asarray(self.reference_codes, dtype=np.int64))
        manifest: dict[str, Any] = {
            "schema": PROFILE_SCHEMA,
            "layout": self.layout,
            "transcript": self.transcript,
            "language": self.language,
            "audio_sha256": self.audio_sha256,
            "prepared_audio_sha256": self.prepared_audio_sha256,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "source_id": self.source_id,
            "reference_samples": self.reference_samples,
            "reference_token_ids": list(self.reference_token_ids),
            "files": {
                "speaker_embedding.npy": hashlib.sha256(speaker).hexdigest(),
                "reference_codes.npy": hashlib.sha256(codes).hexdigest(),
            },
        }
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(p, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
            zf.writestr("speaker_embedding.npy", speaker)
            zf.writestr("reference_codes.npy", codes)

    @classmethod
    def load(cls, path: str | Path) -> PreparedVoice:
        with zipfile.ZipFile(path, "r") as zf:
            names = set(zf.namelist())
            allowed = {"manifest.json", "speaker_embedding.npy", "reference_codes.npy"}
            if names != allowed:
                raise IncompatibleVoiceProfileError("unexpected qvoice archive members")
            if any(name.startswith("/") or ".." in Path(name).parts for name in names):
                raise IncompatibleVoiceProfileError("unsafe qvoice archive path")
            manifest = json.loads(zf.read("manifest.json"))
            if manifest.get("schema") != PROFILE_SCHEMA:
                raise IncompatibleVoiceProfileError("unsupported qvoice schema")
            blobs = {
                name: zf.read(name) for name in ("speaker_embedding.npy", "reference_codes.npy")
            }
            for name, data in blobs.items():
                expected = manifest["files"].get(name)
                if hashlib.sha256(data).hexdigest() != expected:
                    raise IncompatibleVoiceProfileError(f"profile checksum mismatch: {name}")
            speaker = np.load(io.BytesIO(blobs["speaker_embedding.npy"]), allow_pickle=False)
            codes = np.load(io.BytesIO(blobs["reference_codes.npy"]), allow_pickle=False)
        return cls(
            speaker_embedding=np.asarray(speaker, dtype=np.float32),
            reference_codes=np.asarray(codes, dtype=np.int64),
            transcript=str(manifest["transcript"]),
            language=str(manifest["language"]),
            audio_sha256=str(manifest["audio_sha256"]),
            prepared_audio_sha256=str(manifest["prepared_audio_sha256"]),
            model_id=str(manifest["model_id"]),
            model_revision=manifest.get("model_revision"),
            layout=str(manifest.get("layout", "qwen3-wavekat-v1")),
            source_id=manifest.get("source_id"),
            reference_samples=int(manifest.get("reference_samples", 0)),
            reference_token_ids=tuple(
                int(token_id) for token_id in manifest.get("reference_token_ids", ())
            ),
        )
