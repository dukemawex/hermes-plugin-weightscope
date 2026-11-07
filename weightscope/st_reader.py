"""Minimal, hardened safetensors reader (header parsing + chunked tensor reads).

Written from the public safetensors format description:
  [8-byte little-endian u64 N][N bytes UTF-8 JSON header][data buffer]
Each header entry: {"dtype": str, "shape": [int, ...], "data_offsets": [start, end]}
with offsets relative to the start of the data buffer. An optional "__metadata__"
entry maps str -> str.

Nothing here loads a whole checkpoint or even a whole tensor: tensor data is read
with seek-and-read in bounded chunks (default 1M elements).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Iterator

import numpy as np

# dtype name -> (numpy dtype string, itemsize). BF16 is read as uint16 and widened.
SUPPORTED_DTYPES: dict[str, tuple[str, int]] = {
    "F64": ("<f8", 8),
    "F32": ("<f4", 4),
    "F16": ("<f2", 2),
    "BF16": ("<u2", 2),
    "I64": ("<i8", 8),
    "I32": ("<i4", 4),
    "I16": ("<i2", 2),
    "I8": ("i1", 1),
    "U64": ("<u8", 8),
    "U32": ("<u4", 4),
    "U16": ("<u2", 2),
    "U8": ("u1", 1),
    "BOOL": ("?", 1),
}
# Known dtypes we can size-check but do not decode (reported as unsupported in diffs).
SIZED_UNSUPPORTED: dict[str, int] = {"F8_E4M3": 1, "F8_E5M2": 1, "F8_E8M0": 1, "C64": 8}

INDEX_NAME = "model.safetensors.index.json"
MiB = 1024 * 1024
GiB = 1024 * MiB


class WeightScopeError(Exception):
    """Clean, user-facing error (bad path, corrupt file, limit exceeded)."""


@dataclass(frozen=True)
class Limits:
    max_header_bytes: int = 100 * MiB
    max_index_bytes: int = 50 * MiB
    max_tensors: int = 200_000
    max_total_bytes: int = 64 * GiB  # sum of tensor data bytes per checkpoint
    max_files: int = 1024
    chunk_elems: int = 1 << 20  # elements per streamed chunk on the numpy backend (~8 MB as float64)
    device_budget_mib: int = 256  # torch backend: approx. device working memory per chunk pair
    allow_external_symlinks: bool = False


@dataclass
class TensorInfo:
    name: str
    dtype: str
    shape: tuple[int, ...]
    file: Path
    abs_offset: int  # absolute byte offset in file
    nbytes: int

    @property
    def numel(self) -> int:
        return math.prod(self.shape) if self.shape else 1

    @property
    def supported(self) -> bool:
        return self.dtype in SUPPORTED_DTYPES


@dataclass
class FileInfo:
    path: Path
    size: int
    header_bytes: int
    n_tensors: int
    metadata: dict = field(default_factory=dict)


@dataclass
class Checkpoint:
    root: Path
    layout: str  # "file" | "directory" | "sharded-index"
    files: list[FileInfo]
    tensors: dict[str, TensorInfo]
    warnings: list[str] = field(default_factory=list)

    @property
    def total_bytes(self) -> int:
        return sum(t.nbytes for t in self.tensors.values())


def _is_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def parse_file(path: Path, limits: Limits) -> tuple[FileInfo, list[TensorInfo]]:
    """Parse and validate one .safetensors header. Raises WeightScopeError."""
    try:
        with open(path, "rb") as fh:
            size = os.fstat(fh.fileno()).st_size
            if size < 8:
                raise WeightScopeError(f"{path.name}: truncated file ({size} bytes, need >= 8 for header length)")
            n = int.from_bytes(fh.read(8), "little")
            if n == 0:
                raise WeightScopeError(f"{path.name}: empty header")
            if n > limits.max_header_bytes:
                raise WeightScopeError(
                    f"{path.name}: header length {n} exceeds limit {limits.max_header_bytes} bytes")
            if 8 + n > size:
                raise WeightScopeError(f"{path.name}: truncated file (header claims {n} bytes, file has {size - 8})")
            raw = fh.read(n)
    except OSError as exc:
        raise WeightScopeError(f"{path.name}: cannot read file: {exc.strerror or exc}") from None
    try:
        header = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WeightScopeError(f"{path.name}: header is not valid UTF-8 JSON ({type(exc).__name__})") from None
    if not isinstance(header, dict):
        raise WeightScopeError(f"{path.name}: header must be a JSON object")

    metadata = header.pop("__metadata__", None) or {}
    if not isinstance(metadata, dict):
        raise WeightScopeError(f"{path.name}: __metadata__ must be an object")
    if len(header) > limits.max_tensors:
        raise WeightScopeError(f"{path.name}: {len(header)} tensors exceeds limit {limits.max_tensors}")

    data_start = 8 + n
    data_len = size - data_start
    tensors: list[TensorInfo] = []
    for name, entry in header.items():
        if not isinstance(entry, dict):
            raise WeightScopeError(f"{path.name}: entry {name!r} is not an object")
        dtype = entry.get("dtype")
        shape = entry.get("shape")
        offs = entry.get("data_offsets")
        if not isinstance(dtype, str):
            raise WeightScopeError(f"{path.name}: tensor {name!r} has no string dtype")
        if not isinstance(shape, list) or not all(_is_int(d) and d >= 0 for d in shape):
            raise WeightScopeError(f"{path.name}: tensor {name!r} has an invalid shape")
        if (not isinstance(offs, list) or len(offs) != 2 or not all(_is_int(o) for o in offs)
                or offs[0] < 0 or offs[1] < offs[0]):
            raise WeightScopeError(f"{path.name}: tensor {name!r} has invalid data_offsets")
        start, end = offs
        if end > data_len:
            raise WeightScopeError(
                f"{path.name}: tensor {name!r} data_offsets [{start}, {end}] exceed data region "
                f"({data_len} bytes) - file truncated or corrupt")
        numel = math.prod(shape) if shape else 1
        itemsize = SUPPORTED_DTYPES.get(dtype, (None, SIZED_UNSUPPORTED.get(dtype)))[1]
        if itemsize is not None and numel * itemsize != end - start:
            raise WeightScopeError(
                f"{path.name}: tensor {name!r} size mismatch: {dtype} x {shape} = {numel * itemsize} bytes, "
                f"offsets span {end - start}")
        tensors.append(TensorInfo(name, dtype, tuple(shape), path, data_start + start, end - start))

    # Reject overlapping byte ranges (zero-length tensors cannot overlap).
    spans = sorted((t.abs_offset, t.abs_offset + t.nbytes, t.name) for t in tensors if t.nbytes)
    for (s0, e0, n0), (s1, e1, n1) in zip(spans, spans[1:]):
        if s1 < e0:
            raise WeightScopeError(f"{path.name}: tensors {n0!r} and {n1!r} have overlapping data ranges")

    meta_out = {str(k)[:64]: str(v)[:200] for k, v in list(metadata.items())[:20]}
    return FileInfo(path, size, n, len(tensors), meta_out), tensors


def _confined(base: Path, rel: str, limits: Limits, what: str) -> Path:
    """Resolve *rel* inside *base*; reject absolute paths, traversal and escaping symlinks."""
    if not isinstance(rel, str) or not rel or "\x00" in rel:
        raise WeightScopeError(f"{what}: invalid shard filename {rel!r}")
    if (PurePosixPath(rel).is_absolute() or PureWindowsPath(rel).is_absolute() or PureWindowsPath(rel).drive
            or "\\" in rel or ".." in PurePosixPath(rel).parts):
        raise WeightScopeError(f"{what}: shard path {rel!r} escapes the checkpoint directory")
    candidate = base / rel
    resolved = candidate.resolve()
    if not limits.allow_external_symlinks:
        try:
            resolved.relative_to(base.resolve())
        except ValueError:
            raise WeightScopeError(
                f"{what}: shard {rel!r} resolves outside the checkpoint directory (symlink); "
                f"set allow_external_symlinks to permit this") from None
    if not resolved.is_file():
        raise WeightScopeError(f"{what}: shard file {rel!r} not found")
    return resolved


def open_checkpoint(path: str | os.PathLike, limits: Limits | None = None) -> Checkpoint:
    """Open a .safetensors file, a directory of shards, or a directory with an index."""
    limits = limits or Limits()
    p = Path(os.path.expanduser(str(path)))
    if not p.exists():
        raise WeightScopeError(f"path not found: {path}")
    warnings: list[str] = []
    if p.is_file():
        if p.name.endswith(".index.json"):
            return _open_index(p, limits)
        files = [p.resolve()]
        layout, root = "file", p
    elif p.is_dir():
        index = p / INDEX_NAME
        if index.is_file():
            return _open_index(index, limits)
        names = sorted(c.name for c in p.iterdir() if c.name.endswith(".safetensors"))
        if not names:
            raise WeightScopeError(f"no .safetensors files in directory: {path}")
        files = [_confined(p, n, limits, "directory") for n in names]
        layout, root = "directory", p
    else:
        raise WeightScopeError(f"not a file or directory: {path}")
    return _assemble(root, layout, files, limits, warnings, weight_map=None)


def _open_index(index: Path, limits: Limits) -> Checkpoint:
    try:
        if index.stat().st_size > limits.max_index_bytes:
            raise WeightScopeError(f"{index.name}: index exceeds {limits.max_index_bytes} bytes")
        data = json.loads(index.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WeightScopeError(f"{index.name}: cannot parse index ({type(exc).__name__})") from None
    wm = data.get("weight_map") if isinstance(data, dict) else None
    if not isinstance(wm, dict) or not wm:
        raise WeightScopeError(f"{index.name}: missing or empty weight_map")
    if len(wm) > limits.max_tensors:
        raise WeightScopeError(f"{index.name}: {len(wm)} tensors exceeds limit {limits.max_tensors}")
    base = index.parent
    if not all(isinstance(v, str) for v in wm.values()):
        raise WeightScopeError(f"{index.name}: weight_map values must be shard filenames")
    shard_names = sorted(set(wm.values()))
    files = [_confined(base, s, limits, index.name) for s in shard_names]
    by_rel = dict(zip(shard_names, files))
    weight_map = {k: by_rel[v] for k, v in wm.items()}
    return _assemble(base, "sharded-index", files, limits, [], weight_map=weight_map)


def _assemble(root, layout, files, limits, warnings, weight_map) -> Checkpoint:
    if len(files) > limits.max_files:
        raise WeightScopeError(f"{len(files)} files exceeds limit {limits.max_files}")
    infos: list[FileInfo] = []
    tensors: dict[str, TensorInfo] = {}
    total = 0
    for f in files:
        info, ts = parse_file(f, limits)
        infos.append(info)
        for t in ts:
            if t.name in tensors:
                raise WeightScopeError(f"tensor {t.name!r} appears in more than one file")
            tensors[t.name] = t
            total += t.nbytes
        if len(tensors) > limits.max_tensors:
            raise WeightScopeError(f"{len(tensors)} tensors exceeds limit {limits.max_tensors}")
        if total > limits.max_total_bytes:
            raise WeightScopeError(
                f"checkpoint tensor data exceeds max_total_bytes ({limits.max_total_bytes} bytes); raise the limit to proceed")
    if weight_map is not None:
        for name, shard in weight_map.items():
            t = tensors.get(name)
            if t is None:
                raise WeightScopeError(f"index lists tensor {name!r} but no shard contains it")
            if t.file != shard:
                raise WeightScopeError(f"index maps {name!r} to {shard.name} but it is stored in {t.file.name}")
        extra = set(tensors) - set(weight_map)
        if extra:
            warnings.append(f"{len(extra)} tensor(s) in shards are not listed in the index")
    return Checkpoint(Path(root), layout, infos, tensors, warnings)


def iter_raw_chunks(t: TensorInfo, chunk_elems: int) -> Iterator[np.ndarray]:
    """Yield the tensor's flattened values in their stored dtype (BF16 as raw uint16), bounded chunks.

    Seek-and-read with numpy (no whole-tensor buffer); backends convert/upcast each chunk themselves.
    """
    if t.dtype not in SUPPORTED_DTYPES:
        raise WeightScopeError(f"dtype {t.dtype} is not supported for numeric comparison")
    n = t.numel
    if n == 0:
        return
    chunk_elems = max(1, int(chunk_elems))
    dt = np.dtype(SUPPORTED_DTYPES[t.dtype][0])
    try:
        fh = open(t.file, "rb")
    except OSError as exc:
        raise WeightScopeError(f"cannot open {t.file.name}: {exc.strerror or exc}") from None
    with fh:
        fh.seek(t.abs_offset)
        for i in range(0, n, chunk_elems):
            count = min(chunk_elems, n - i)
            part = np.fromfile(fh, dtype=dt, count=count)
            if part.size != count:
                raise WeightScopeError(f"short read in tensor {t.name!r} ({t.file.name} changed or truncated)")
            yield part


def bf16_to_f32(raw_u16: np.ndarray) -> np.ndarray:
    """bfloat16 bit patterns (uint16) -> float32 via a 16-bit shift."""
    return (raw_u16.astype(np.uint32) << 16).view(np.float32)


def iter_chunks(t: TensorInfo, chunk_elems: int) -> Iterator[np.ndarray]:
    """Yield float64 chunks of a tensor's flattened values (bounded memory)."""
    for part in iter_raw_chunks(t, chunk_elems):
        if t.dtype == "BF16":
            part = bf16_to_f32(part)
        yield part.astype(np.float64)


def sha256_file(path: Path, bufsize: int = 4 * MiB) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(bufsize):
            h.update(chunk)
    return h.hexdigest()
