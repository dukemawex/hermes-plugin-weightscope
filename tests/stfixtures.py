"""Tiny, deterministic synthetic safetensors fixtures (no torch / safetensors dependency).

Also used by scripts/make_fixtures.py for the demo.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

_NP = {"F32": "<f4", "F16": "<f2", "F64": "<f8", "I32": "<i4", "I64": "<i8", "U8": "u1", "I8": "i1"}


def to_bf16_bytes(arr: np.ndarray) -> bytes:
    """float32 -> bfloat16 (round-to-nearest-even) raw little-endian bytes."""
    u = np.ascontiguousarray(arr, dtype="<f4").view("<u4").astype(np.uint64)
    rounded = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype("<u2")
    return rounded.tobytes()


def encode(tensors: dict, metadata: dict | None = None) -> bytes:
    """tensors: name -> (dtype, ndarray). Returns safetensors bytes."""
    header, blobs, off = {}, [], 0
    if metadata:
        header["__metadata__"] = metadata
    for name, (dtype, arr) in tensors.items():
        arr = np.asarray(arr)
        raw = to_bf16_bytes(arr) if dtype == "BF16" else np.ascontiguousarray(arr, dtype=_NP[dtype]).tobytes()
        header[name] = {"dtype": dtype, "shape": list(arr.shape), "data_offsets": [off, off + len(raw)]}
        blobs.append(raw)
        off += len(raw)
    hb = json.dumps(header, separators=(",", ":")).encode()
    hb += b" " * (-len(hb) % 8)  # spec-style alignment padding
    return struct.pack("<Q", len(hb)) + hb + b"".join(blobs)


def write(path: Path, tensors: dict, metadata: dict | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encode(tensors, metadata))
    return path


def base_model(seed: int = 0, n_layers: int = 4, d: int = 16, vocab: int = 32) -> dict:
    rng = np.random.default_rng(seed)
    t = {"model.embed_tokens.weight": ("F32", rng.standard_normal((vocab, d)).astype(np.float32))}
    for i in range(n_layers):
        p = f"model.layers.{i}"
        t[f"{p}.self_attn.q_proj.weight"] = ("F32", rng.standard_normal((d, d)).astype(np.float32) * 0.1)
        t[f"{p}.self_attn.v_proj.weight"] = ("F32", rng.standard_normal((d, d)).astype(np.float32) * 0.1)
        t[f"{p}.mlp.up_proj.weight"] = ("F32", rng.standard_normal((2 * d, d)).astype(np.float32) * 0.1)
        t[f"{p}.input_layernorm.weight"] = ("F32", np.ones(d, dtype=np.float32))
    t["model.norm.weight"] = ("F32", np.ones(d, dtype=np.float32))
    t["lm_head.weight"] = ("F32", rng.standard_normal((vocab, d)).astype(np.float32))
    return t


def finetuned(base: dict, layer: int = 2, scale: float = 0.05, seed: int = 1) -> dict:
    """Copy of base where only model.layers.<layer> MLP + q_proj change."""
    rng = np.random.default_rng(seed)
    out = {k: (dt, a.copy()) for k, (dt, a) in base.items()}
    for name in (f"model.layers.{layer}.mlp.up_proj.weight", f"model.layers.{layer}.self_attn.q_proj.weight"):
        dt, a = out[name]
        out[name] = (dt, (a + rng.standard_normal(a.shape).astype(np.float32) * scale).astype(np.float32))
    return out


def as_dtype(model: dict, dtype: str) -> dict:
    return {k: (dtype, a) for k, (_, a) in model.items()}


def write_sharded(root: Path, model: dict, n_shards: int = 2) -> Path:
    """Write HF-style shards + model.safetensors.index.json."""
    root.mkdir(parents=True, exist_ok=True)
    names = list(model)
    per = -(-len(names) // n_shards)
    weight_map = {}
    for s in range(n_shards):
        fname = f"model-{s + 1:05d}-of-{n_shards:05d}.safetensors"
        part = {n: model[n] for n in names[s * per:(s + 1) * per]}
        write(root / fname, part)
        weight_map.update({n: fname for n in part})
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 0}, "weight_map": weight_map}))
    return root
