from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .errors import InvalidModelLayoutError, ModelGenerationError
from .tokenizer import ByteLevelBPETokenizer
from .types import GenerationConfig

LAYOUT = "qwen3-wavekat-v1"


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


def _import_ort() -> Any:
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError(
            "ONNX Runtime is required; install qwen3synth[cpu] or qwen3synth[gpu]"
        ) from exc
    return ort


def _text_project(token_ids: list[int] | np.ndarray, emb: dict[str, Any]) -> np.ndarray:
    ids = np.asarray(token_ids, dtype=np.int64)
    x = emb["text_embedding"][ids]
    hidden = x @ emb["text_projection_fc1_weight"].T + emb["text_projection_fc1_bias"]
    hidden = hidden * (1.0 / (1.0 + np.exp(-hidden)))
    return hidden @ emb["text_projection_fc2_weight"].T + emb["text_projection_fc2_bias"]


def _sample(logits: np.ndarray, config: GenerationConfig, rng: np.random.Generator) -> int:
    x = np.asarray(logits, dtype=np.float64).copy()
    if config.temperature != 1.0:
        x /= config.temperature
    if 0 < config.top_k < x.size:
        idx = np.argpartition(x, -config.top_k)[-config.top_k :]
        mask = np.full_like(x, -np.inf)
        mask[idx] = x[idx]
        x = mask
    m = np.max(x)
    probs = np.exp(x - m)
    total = probs.sum()
    if not np.isfinite(total) or total <= 0:
        return int(np.argmax(x))
    probs /= total
    return int(rng.choice(len(probs), p=probs))


@dataclass
class WaveKatBackend:
    model_dir: Path
    variant: str = "int4"
    provider: str = "cpu"

    def __post_init__(self) -> None:
        self.model_dir = Path(self.model_dir)
        cfg = self.model_dir / "config.json"
        if not cfg.exists():
            raise InvalidModelLayoutError(f"missing {cfg}")
        self.config: dict[str, Any] = json.loads(cfg.read_text(encoding="utf-8"))
        self.model_type = str(self.config.get("tts_model_type", ""))
        if self.model_type not in {"voice_design", "base"}:
            raise InvalidModelLayoutError(f"unsupported tts_model_type: {self.model_type!r}")
        self.tokenizer = ByteLevelBPETokenizer(self.model_dir / "tokenizer")
        self.emb = self._load_embeddings()
        self._sessions: dict[str, Any] = {}

    @property
    def model_id(self) -> str:
        return str(self.config.get("model_id", self.model_dir.name))

    def _load_embeddings(self) -> dict[str, Any]:
        root = self.model_dir / "embeddings"
        names = [
            "text_embedding",
            "text_projection_fc1_weight",
            "text_projection_fc1_bias",
            "text_projection_fc2_weight",
            "text_projection_fc2_bias",
            "talker_codec_embedding",
        ]
        result: dict[str, Any] = {}
        for name in names:
            p = root / f"{name}.npy"
            if not p.exists():
                raise InvalidModelLayoutError(f"missing embedding {p}")
            result[name] = np.load(p, mmap_mode="r", allow_pickle=False)
        cp = []
        for i in range(64):
            p = root / f"cp_codec_embedding_{i}.npy"
            if not p.exists():
                break
            cp.append(np.load(p, mmap_mode="r", allow_pickle=False))
        if not cp:
            raise InvalidModelLayoutError("no cp_codec_embedding_*.npy files found")
        result["cp_codec_embeddings"] = cp
        return result

    def _providers(self) -> list[str] | None:
        if self.provider in {"", "auto"}:
            return None
        mapping = {
            "cpu": "CPUExecutionProvider",
            "cuda": "CUDAExecutionProvider",
            "directml": "DmlExecutionProvider",
        }
        name = mapping.get(self.provider.lower(), self.provider)
        return [name]

    def _session(self, role: str, *, root: bool = False) -> Any:
        key = ("root:" if root else "variant:") + role
        if key in self._sessions:
            return self._sessions[key]
        ort = _import_ort()
        p = self.model_dir / (f"{role}.onnx" if root else f"{self.variant}/{role}.onnx")
        if not p.exists():
            raise InvalidModelLayoutError(f"missing ONNX graph {p}")
        kwargs: dict[str, Any] = {}
        providers = self._providers()
        if providers:
            kwargs["providers"] = providers
        try:
            sess = ort.InferenceSession(str(p), **kwargs)
        except Exception as exc:
            raise InvalidModelLayoutError(f"failed to load {p}: {exc}") from exc
        self._sessions[key] = sess
        return sess

    def token_count(self, text: str, *, description: str | None = None) -> int:
        chat = f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
        total = len(self.tokenizer.encode(chat))
        if description:
            prompt = f"<|im_start|>user\n{description}<|im_end|>\n"
            total += len(self.tokenizer.encode(prompt))
        return total

    def _codec_prefix(self, language: str) -> list[int]:
        lang = normalize_language(language)
        language_id = self.config.get("codec_language_id", {}).get(lang)
        if language_id is not None:
            return [
                self.config["codec_think_id"],
                self.config["codec_think_bos_id"],
                language_id,
                self.config["codec_think_eos_id"],
            ]
        return [
            self.config["codec_nothink_id"],
            self.config["codec_think_bos_id"],
            self.config["codec_think_eos_id"],
        ]

    def _decode_codes(
        self, prefill_embeds: np.ndarray, trailing_hidden: np.ndarray, gen: GenerationConfig
    ) -> np.ndarray:
        prefill = self._session("talker_prefill")
        decode = self._session("talker_decode")
        cp = self._session("code_predictor")
        vocoder = self._session("vocoder")
        cfg = self.config
        num_layers = int(cfg["talker_num_layers"])
        num_groups = int(cfg["talker_num_code_groups"])
        cp_layers = int(cfg["cp_num_layers"])
        cp_heads = int(cfg["cp_num_kv_heads"])
        cp_dim = int(cfg["cp_head_dim"])
        vocab_size = int(cfg["talker_vocab_size"])
        eos = int(cfg["codec_eos_token_id"])
        codec_emb = self.emb["talker_codec_embedding"]
        cp_embs = self.emb["cp_codec_embeddings"]
        if len(cp_embs) < num_groups - 1:
            raise InvalidModelLayoutError("not enough code predictor embeddings")
        T = int(prefill_embeds.shape[1])
        attention_mask = np.ones((1, T), dtype=np.int64)
        position_ids = np.arange(T, dtype=np.int64).reshape(1, 1, T).repeat(3, axis=0)
        try:
            out = prefill.run(
                None,
                {
                    "inputs_embeds": prefill_embeds,
                    "attention_mask": attention_mask,
                    "position_ids": position_ids,
                },
            )
        except Exception as exc:
            raise ModelGenerationError(f"talker prefill failed: {exc}") from exc
        logits, hidden = out[0], out[1]
        kv = out[2:]
        if len(kv) != num_layers * 2:
            raise InvalidModelLayoutError(
                f"prefill returned {len(kv)} KV outputs, expected {num_layers * 2}"
            )
        past_keys = np.stack([kv[i * 2] for i in range(num_layers)])
        past_values = np.stack([kv[i * 2 + 1] for i in range(num_layers)])
        suppress = np.zeros(vocab_size, dtype=bool)
        suppress[max(0, vocab_size - 1024) : vocab_size] = True
        suppress[eos] = False
        rng = np.random.default_rng(gen.seed)
        generated: list[int] = []
        frames: list[list[int]] = []
        current_pos = T
        for step in range(gen.max_new_tokens):
            scores = logits[0, -1, :].astype(np.float64, copy=True)
            scores[suppress] = -np.inf
            if step < 2:
                scores[eos] = -np.inf
            if gen.repetition_penalty != 1.0 and generated:
                seen = np.asarray(generated, dtype=np.int64)
                s = scores[seen]
                scores[seen] = np.where(
                    s > 0, s / gen.repetition_penalty, s * gen.repetition_penalty
                )
            group0 = _sample(scores, gen, rng)
            if group0 == eos:
                break
            generated.append(group0)
            frame = [group0]
            talker_hidden = hidden[0, -1:, :]
            group0_embed = codec_emb[group0].reshape(1, -1)
            cp_input = np.concatenate([talker_hidden, group0_embed], axis=0)[
                np.newaxis, :, :
            ].astype(np.float32)
            cp_keys = np.zeros((cp_layers, 1, cp_heads, 0, cp_dim), dtype=np.float32)
            cp_values = np.zeros((cp_layers, 1, cp_heads, 0, cp_dim), dtype=np.float32)
            for g in range(num_groups - 1):
                cp_out = cp.run(
                    None,
                    {
                        "inputs_embeds": cp_input,
                        "generation_steps": np.array([g], dtype=np.int64),
                        "past_keys": cp_keys,
                        "past_values": cp_values,
                    },
                )
                cp_keys, cp_values = cp_out[1], cp_out[2]
                token = _sample(cp_out[0][0, -1, :], gen, rng)
                frame.append(token)
                cp_input = cp_embs[g][token].reshape(1, 1, -1).astype(np.float32)
            frames.append(frame)
            next_embed = codec_emb[group0].astype(np.float32, copy=True)
            for g in range(num_groups - 1):
                next_embed += cp_embs[g][frame[g + 1]]
            next_embed += trailing_hidden[0]
            next_embed = next_embed.reshape(1, 1, -1).astype(np.float32)
            mask = np.ones((1, current_pos + 1), dtype=np.int64)
            pos = np.array([[[current_pos]]], dtype=np.int64).repeat(3, axis=0)
            dec = decode.run(
                None,
                {
                    "inputs_embeds": next_embed,
                    "attention_mask": mask,
                    "position_ids": pos,
                    "past_keys": past_keys,
                    "past_values": past_values,
                },
            )
            logits, hidden, past_keys, past_values = dec[0], dec[1], dec[2], dec[3]
            current_pos += 1
        if not frames:
            raise ModelGenerationError("model produced no codec frames")
        codes = np.asarray(frames, dtype=np.int64).T[np.newaxis, :, :]
        wav = vocoder.run(None, {"codes": codes})[0].reshape(-1).astype(np.float32)
        if not np.all(np.isfinite(wav)):
            raise ModelGenerationError("vocoder returned non-finite audio")
        return wav

    def generate_voice_design(
        self, text: str, description: str, language: str, gen: GenerationConfig
    ) -> np.ndarray:
        if self.model_type != "voice_design":
            raise InvalidModelLayoutError("loaded bundle is not a voice_design model")
        input_ids = self.tokenizer.encode(
            f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
        )
        instruct_ids = self.tokenizer.encode(f"<|im_start|>user\n{description}<|im_end|>\n")
        e = self.emb
        proj = lambda ids: _text_project(ids, e)
        codec = e["talker_codec_embedding"]
        tts_pad = proj([self.config["tts_pad_token_id"]])[0]
        tts_bos = proj([self.config["tts_bos_token_id"]])[0]
        tts_eos = proj([self.config["tts_eos_token_id"]])[0]
        codec_pad = codec[self.config["codec_pad_id"]]
        codec_bos = codec[self.config["codec_bos_id"]]
        parts: list[np.ndarray] = [proj(instruct_ids), proj(input_ids[:3])]
        for cid in self._codec_prefix(language):
            parts.append((tts_pad + codec[cid]).reshape(1, -1))
        parts.append((tts_bos + codec_pad).reshape(1, -1))
        for tid in input_ids[3:-5]:
            parts.append((proj([tid])[0] + codec_pad).reshape(1, -1))
        parts.append((tts_eos + codec_pad).reshape(1, -1))
        parts.append((tts_pad + codec_bos).reshape(1, -1))
        prefill = np.concatenate(parts, axis=0)[np.newaxis, :, :].astype(np.float32)
        return self._decode_codes(prefill, tts_pad.reshape(1, -1), gen)

    def encode_reference(self, audio_24k: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.model_type != "base":
            raise InvalidModelLayoutError("loaded bundle is not a base model")
        from .audio import compute_log_mel

        if len(audio_24k) > 240_000:
            raise ValueError(
                "WaveKat Base MVP accepts at most 10 seconds of reference audio; select a matching excerpt"
            )
        mel = compute_log_mel(audio_24k)
        speaker = (
            self._session("speaker_encoder", root=True)
            .run(None, {"mels": mel})[0]
            .astype(np.float32)
        )
        padded = np.zeros(240_000, dtype=np.float32)
        padded[: len(audio_24k)] = audio_24k
        full = self._session("tokenizer_encoder", root=True).run(
            None, {"waveform": padded[np.newaxis, :]}
        )[0]
        frames = int(np.ceil(len(audio_24k) / 1920))
        return speaker, np.asarray(full[:, :, :frames], dtype=np.int64)

    def generate_clone(
        self,
        text: str,
        transcript: str,
        language: str,
        speaker: np.ndarray,
        ref_codes: np.ndarray,
        gen: GenerationConfig,
    ) -> np.ndarray:
        if self.model_type != "base":
            raise InvalidModelLayoutError("loaded bundle is not a base model")
        input_ids = self.tokenizer.encode(
            f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
        )
        ref_ids = self.tokenizer.encode(f"<|im_start|>assistant\n{transcript}<|im_end|>\n")
        e = self.emb
        proj = lambda ids: _text_project(ids, e)
        codec = e["talker_codec_embedding"]
        cp_embs = e["cp_codec_embeddings"]
        hidden = int(self.config["talker_hidden_size"])
        groups = int(self.config["talker_num_code_groups"])
        tts_pad = proj([self.config["tts_pad_token_id"]])[0]
        tts_bos = proj([self.config["tts_bos_token_id"]])[0]
        tts_eos = proj([self.config["tts_eos_token_id"]])[0]
        codec_pad = codec[self.config["codec_pad_id"]]
        codec_bos = codec[self.config["codec_bos_id"]]
        parts: list[np.ndarray] = [proj(input_ids[:3])]
        for cid in self._codec_prefix(language):
            parts.append((tts_pad + codec[cid]).reshape(1, -1))
        parts.append((tts_pad + np.asarray(speaker, dtype=np.float32).reshape(-1)).reshape(1, -1))
        parts.append((tts_bos + codec_pad).reshape(1, -1))
        combined = list(ref_ids[3:-2]) + list(input_ids[3:-5])
        text_embed = np.concatenate([proj(combined), tts_eos.reshape(1, -1)], axis=0)
        ref_frames = int(ref_codes.shape[2])
        frame_embed = np.zeros((ref_frames, hidden), dtype=np.float32)
        for f in range(ref_frames):
            frame_embed[f] += codec[int(ref_codes[0, 0, f])]
            for g in range(groups - 1):
                frame_embed[f] += cp_embs[g][int(ref_codes[0, g + 1, f])]
        codec_embed = np.concatenate([codec_bos.reshape(1, -1), frame_embed], axis=0)
        parts.append(np.concatenate([text_embed + codec_pad, codec_embed + tts_pad], axis=0))
        prefill = np.concatenate(parts, axis=0)[np.newaxis, :, :].astype(np.float32)
        # Decode target frames using the same loop, but run the vocoder here with ref+generated
        generated = self._generate_code_matrix(prefill, tts_pad.reshape(1, -1), gen)
        all_codes = (
            np.concatenate([ref_codes[0].T, generated], axis=0).T[np.newaxis, :, :].astype(np.int64)
        )
        wav = (
            self._session("vocoder")
            .run(None, {"codes": all_codes})[0]
            .reshape(-1)
            .astype(np.float32)
        )
        total_frames = all_codes.shape[2]
        cut = int(ref_frames / max(total_frames, 1) * len(wav))
        wav = wav[cut:]
        if not np.all(np.isfinite(wav)):
            raise ModelGenerationError("vocoder returned non-finite audio")
        return wav

    def _generate_code_matrix(
        self, prefill_embeds: np.ndarray, trailing_hidden: np.ndarray, gen: GenerationConfig
    ) -> np.ndarray:
        # Same autoregressive core as _decode_codes, returning frames before vocoding.
        prefill = self._session("talker_prefill")
        decode = self._session("talker_decode")
        cp = self._session("code_predictor")
        cfg = self.config
        nl, ng = int(cfg["talker_num_layers"]), int(cfg["talker_num_code_groups"])
        cpl, cph, cpd = (
            int(cfg["cp_num_layers"]),
            int(cfg["cp_num_kv_heads"]),
            int(cfg["cp_head_dim"]),
        )
        vs, eos = int(cfg["talker_vocab_size"]), int(cfg["codec_eos_token_id"])
        codec, cp_embs = self.emb["talker_codec_embedding"], self.emb["cp_codec_embeddings"]
        T = prefill_embeds.shape[1]
        out = prefill.run(
            None,
            {
                "inputs_embeds": prefill_embeds,
                "attention_mask": np.ones((1, T), np.int64),
                "position_ids": np.arange(T, dtype=np.int64).reshape(1, 1, T).repeat(3, axis=0),
            },
        )
        logits, hidden = out[0], out[1]
        kv = out[2:]
        keys = np.stack([kv[i * 2] for i in range(nl)])
        values = np.stack([kv[i * 2 + 1] for i in range(nl)])
        suppress = np.zeros(vs, bool)
        suppress[max(0, vs - 1024) :] = True
        suppress[eos] = False
        rng = np.random.default_rng(gen.seed)
        seen: list[int] = []
        frames: list[list[int]] = []
        pos = T
        for step in range(gen.max_new_tokens):
            scores = logits[0, -1, :].astype(np.float64, copy=True)
            scores[suppress] = -np.inf
            if step < 2:
                scores[eos] = -np.inf
            if gen.repetition_penalty != 1.0 and seen:
                idx = np.asarray(seen, np.int64)
                s = scores[idx]
                scores[idx] = np.where(
                    s > 0, s / gen.repetition_penalty, s * gen.repetition_penalty
                )
            g0 = _sample(scores, gen, rng)
            if g0 == eos:
                break
            seen.append(g0)
            frame = [g0]
            cp_input = np.concatenate([hidden[0, -1:, :], codec[g0].reshape(1, -1)], axis=0)[
                None, :, :
            ].astype(np.float32)
            ck = np.zeros((cpl, 1, cph, 0, cpd), np.float32)
            cv = np.zeros_like(ck)
            for g in range(ng - 1):
                co = cp.run(
                    None,
                    {
                        "inputs_embeds": cp_input,
                        "generation_steps": np.array([g], np.int64),
                        "past_keys": ck,
                        "past_values": cv,
                    },
                )
                ck, cv = co[1], co[2]
                tok = _sample(co[0][0, -1, :], gen, rng)
                frame.append(tok)
                cp_input = cp_embs[g][tok].reshape(1, 1, -1).astype(np.float32)
            frames.append(frame)
            nxt = codec[g0].astype(np.float32, copy=True)
            for g in range(ng - 1):
                nxt += cp_embs[g][frame[g + 1]]
            nxt += trailing_hidden[0]
            de = decode.run(
                None,
                {
                    "inputs_embeds": nxt.reshape(1, 1, -1).astype(np.float32),
                    "attention_mask": np.ones((1, pos + 1), np.int64),
                    "position_ids": np.array([[[pos]]], np.int64).repeat(3, axis=0),
                    "past_keys": keys,
                    "past_values": values,
                },
            )
            logits, hidden, keys, values = de[0], de[1], de[2], de[3]
            pos += 1
        if not frames:
            raise ModelGenerationError("model produced no codec frames")
        return np.asarray(frames, dtype=np.int64)
