"""Checkpoint summary and per-tensor / per-layer diff metrics (numpy only, streamed)."""

from __future__ import annotations

import math
import re
import time
from collections import defaultdict

from .backends import Backend, NumpyBackend, resolve_backend
from .st_reader import (Checkpoint, Limits, SUPPORTED_DTYPES, TensorInfo, WeightScopeError,
                        iter_raw_chunks, open_checkpoint, sha256_file)

INTERPRETATION_NOTE = (
    "These are numeric differences between stored weights only. A weight change does NOT establish a "
    "backdoor, a behaviour change, a safety status, or tampering, and an unchanged tensor does not prove "
    "identical behaviour. Use these numbers to decide where to look, then test behaviour separately."
)
LIMITATIONS = [
    "safetensors only (no .bin/.pt pickles, GGUF, ONNX)",
    "quantized/packed formats (FP8, GPTQ/AWQ int packing, bitsandbytes) are listed but not decoded",
    "numpy (CPU) by default; GPU use needs PyTorch installed in the Hermes environment (--backend torch)",
    "integer tensors are compared as float64 (very large 64-bit values lose precision)",
]
TOP_K_MAX = 100
LIST_MAX = 200
RANK_KEYS = ("relative", "l2", "max_abs")

_NUM_SEG = re.compile(r"^(.*?\.\d+)(?:\.|$)")


def layer_key(name: str) -> str:
    """'model.layers.3.mlp.up_proj.weight' -> 'model.layers.3'; 'lm_head.weight' -> 'lm_head'."""
    m = _NUM_SEG.match(name)
    if m:
        return m.group(1)
    return name.rsplit(".", 1)[0] if "." in name else name


def _f(x, digits: int = 6):
    """JSON-safe rounded float (None for NaN/Inf)."""
    if x is None:
        return None
    x = float(x)
    if not math.isfinite(x):
        return None
    return float(f"{x:.{digits}g}")


def _clamp(v, lo, hi, default):
    try:
        v = int(v)
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, v))


def _capped(items: list, cap: int) -> dict:
    return {"count": len(items), "items": items[:cap], "truncated": len(items) > cap}


def _ckpt_brief(ck: Checkpoint) -> dict:
    return {
        "path": str(ck.root),
        "layout": ck.layout,
        "files": len(ck.files),
        "tensors": len(ck.tensors),
        "parameters": sum(t.numel for t in ck.tensors.values()),
        "bytes": ck.total_bytes,
    }


# ── summary ──────────────────────────────────────────────────────────────────

def summarize(path, *, limits: Limits | None = None, max_list: int = 50, hash_files: bool = True) -> dict:
    t0 = time.monotonic()
    limits = limits or Limits()
    max_list = _clamp(max_list, 0, LIST_MAX, 50)
    ck = open_checkpoint(path, limits)
    by_dtype: dict[str, dict] = defaultdict(lambda: {"tensors": 0, "parameters": 0, "bytes": 0})
    for t in ck.tensors.values():
        d = by_dtype[t.dtype]
        d["tensors"] += 1
        d["parameters"] += t.numel
        d["bytes"] += t.nbytes
    files = []
    for fi in ck.files:
        entry = {"file": fi.path.name, "bytes": fi.size, "header_bytes": fi.header_bytes, "tensors": fi.n_tensors}
        if hash_files:
            entry["sha256"] = sha256_file(fi.path)
        if fi.metadata:
            entry["metadata"] = fi.metadata
        files.append(entry)
    names = sorted(ck.tensors)
    tensors = [{"name": n, "dtype": ck.tensors[n].dtype, "shape": list(ck.tensors[n].shape),
                "parameters": ck.tensors[n].numel} for n in names]
    largest = sorted(ck.tensors.values(), key=lambda t: (-t.numel, t.name))[:min(10, max_list or 10)]
    unsupported = sorted(n for n, t in ck.tensors.items() if not t.supported)
    return {
        "ok": True,
        "tool": "ckpt_summary",
        "checkpoint": _ckpt_brief(ck),
        "dtypes": dict(sorted(by_dtype.items())),
        "files": files[:max_list] if max_list else files[:20],
        "files_truncated": len(files) > (max_list or 20),
        "largest_tensors": [{"name": t.name, "shape": list(t.shape), "parameters": t.numel} for t in largest],
        "tensors": _capped(tensors, max_list),
        "undecoded_dtype_tensors": _capped(unsupported, max_list),
        "layers": len({layer_key(n) for n in names}),
        "warnings": ck.warnings,
        "note": "File hashes identify exact bytes; they say nothing about model behaviour or safety.",
        "elapsed_s": round(time.monotonic() - t0, 3),
    }


# ── diff ─────────────────────────────────────────────────────────────────────

class _Acc:
    __slots__ = ("sdd", "saa", "sbb", "sab", "maxd", "changed", "compared", "nf_a", "nf_b", "numel")

    def __init__(self):
        self.sdd = self.saa = self.sbb = self.sab = self.maxd = 0.0
        self.changed = self.compared = self.nf_a = self.nf_b = self.numel = 0


def compare_tensor(ta: TensorInfo, tb: TensorInfo, tol: float, chunk_elems: int,
                   backend: Backend | None = None) -> _Acc:
    """Stream both tensors chunk by chunk through *backend*; sums accumulate on the host in float64."""
    backend = backend or NumpyBackend()
    acc = _Acc()
    acc.numel = ta.numel
    for ra, rb in zip(iter_raw_chunks(ta, chunk_elems), iter_raw_chunks(tb, chunk_elems)):
        st = backend.chunk_stats(ra, ta.dtype, rb, tb.dtype, tol)
        acc.sdd += st.sdd
        acc.saa += st.saa
        acc.sbb += st.sbb
        acc.sab += st.sab
        acc.maxd = max(acc.maxd, st.maxd)
        acc.changed += st.changed
        acc.compared += st.compared
        acc.nf_a += st.nf_a
        acc.nf_b += st.nf_b
    return acc


def _rel(sdd: float, saa: float):
    """(relative change, zero_baseline flag). Relative change is None when the base norm is 0 but B differs."""
    l2, base = math.sqrt(sdd), math.sqrt(saa)
    if base > 0:
        return l2 / base, False
    return (0.0 if l2 == 0 else None), True


def _cosine(acc: _Acc):
    if acc.saa > 0 and acc.sbb > 0:
        return max(-1.0, min(1.0, acc.sab / math.sqrt(acc.saa * acc.sbb)))
    return None


def _rank_value(rec: dict, rank_by: str) -> float:
    if rank_by == "l2":
        return rec["l2_diff"] or 0.0
    if rank_by == "max_abs":
        return rec["max_abs_diff"] or 0.0
    if rec["rel_change"] is None:
        return math.inf if (rec["l2_diff"] or 0) > 0 else 0.0
    return rec["rel_change"]


def diff(path_a, path_b, *, limits: Limits | None = None, top_k: int = 10, top_layers: int = 10,
         tolerance: float = 0.0, rank_by: str = "relative", max_list: int = 25,
         backend: str | Backend | None = "auto", device: str | None = None) -> dict:
    t0 = time.monotonic()
    limits = limits or Limits()
    top_k = _clamp(top_k, 1, TOP_K_MAX, 10)
    top_layers = _clamp(top_layers, 1, TOP_K_MAX, 10)
    max_list = _clamp(max_list, 1, LIST_MAX, 25)
    if rank_by not in RANK_KEYS:
        raise WeightScopeError(f"rank_by must be one of {', '.join(RANK_KEYS)}")
    try:
        tolerance = float(tolerance)
    except (TypeError, ValueError):
        raise WeightScopeError("tolerance must be a number") from None
    if not math.isfinite(tolerance) or tolerance < 0:
        raise WeightScopeError("tolerance must be a finite number >= 0")

    be = backend if isinstance(backend, Backend) else resolve_backend(backend, device)
    chunk = be.chunk_elems(limits)
    a = open_checkpoint(path_a, limits)
    b = open_checkpoint(path_b, limits)
    warnings = [f"A: {w}" for w in a.warnings] + [f"B: {w}" for w in b.warnings]
    if {f.path for f in a.files} == {f.path for f in b.files}:
        warnings.append("A and B resolve to the same file(s); every tensor will be identical")

    names_a, names_b = set(a.tensors), set(b.tensors)
    only_a = sorted(names_a - names_b)
    only_b = sorted(names_b - names_a)
    common = sorted(names_a & names_b)

    shape_mm, dtype_mm, unsupported, nonfinite, records = [], [], [], [], []
    layers: dict[str, dict] = defaultdict(lambda: {"sdd": 0.0, "saa": 0.0, "tensors": 0, "changed": 0,
                                                    "maxd": 0.0, "params": 0})
    params_compared = params_changed = 0
    tot_sdd = tot_saa = 0.0
    identical = 0
    for name in common:
        ta, tb = a.tensors[name], b.tensors[name]
        if ta.shape != tb.shape:
            shape_mm.append({"name": name, "shape_a": list(ta.shape), "shape_b": list(tb.shape)})
            continue
        if ta.dtype != tb.dtype:
            dtype_mm.append({"name": name, "dtype_a": ta.dtype, "dtype_b": tb.dtype})
        if not (ta.supported and tb.supported):
            unsupported.append({"name": name, "dtype_a": ta.dtype, "dtype_b": tb.dtype})
            continue
        acc = compare_tensor(ta, tb, tolerance, chunk, be)
        rel, zero_base = _rel(acc.sdd, acc.saa)
        rec = {
            "name": name,
            "shape": list(ta.shape),
            "dtype": ta.dtype if ta.dtype == tb.dtype else f"{ta.dtype}->{tb.dtype}",
            "l2_diff": _f(math.sqrt(acc.sdd)),
            "base_l2": _f(math.sqrt(acc.saa)),
            "rel_change": _f(rel),
            "max_abs_diff": _f(acc.maxd),
            "cosine": _f(_cosine(acc), 9),
            "frac_changed": _f(acc.changed / acc.compared) if acc.compared else None,
        }
        if zero_base:
            rec["zero_baseline"] = True
        if acc.nf_a or acc.nf_b:
            rec["nonfinite_a"], rec["nonfinite_b"] = acc.nf_a, acc.nf_b
            nonfinite.append({"name": name, "nonfinite_a": acc.nf_a, "nonfinite_b": acc.nf_b,
                              "of": acc.numel})
        changed = acc.sdd > 0 or acc.nf_a != acc.nf_b
        identical += 0 if changed else 1
        if changed:
            records.append(rec)
        lk = layers[layer_key(name)]
        lk["sdd"] += acc.sdd
        lk["saa"] += acc.saa
        lk["tensors"] += 1
        lk["changed"] += 1 if changed else 0
        lk["maxd"] = max(lk["maxd"], acc.maxd)
        lk["params"] += acc.numel
        params_compared += acc.compared
        params_changed += acc.changed
        tot_sdd += acc.sdd
        tot_saa += acc.saa

    records.sort(key=lambda r: (-_rank_value(r, rank_by), r["name"]))
    layer_recs = []
    for key, v in layers.items():
        rel, zero_base = _rel(v["sdd"], v["saa"])
        rec = {"layer": key, "tensors": v["tensors"], "changed_tensors": v["changed"], "parameters": v["params"],
               "l2_diff": _f(math.sqrt(v["sdd"])), "base_l2": _f(math.sqrt(v["saa"])),
               "rel_change": _f(rel), "max_abs_diff": _f(v["maxd"])}
        if zero_base:
            rec["zero_baseline"] = True
        if v["changed"]:
            layer_recs.append(rec)
    layer_recs.sort(key=lambda r: (-_rank_value(r, rank_by), r["layer"]))
    tot_rel, _ = _rel(tot_sdd, tot_saa)

    return {
        "ok": True,
        "tool": "ckpt_diff",
        "a": _ckpt_brief(a),
        "b": _ckpt_brief(b),
        "settings": {"tolerance": tolerance, "rank_by": rank_by, "top_k": top_k, "top_layers": top_layers},
        "execution": be.info(limits),
        "totals": {
            "common_tensors": len(common),
            "compared_tensors": identical + len(records),
            "changed_tensors": len(records),
            "identical_tensors": identical,
            "changed_layers": len(layer_recs),
            "compared_parameters": params_compared,
            "changed_parameters": params_changed,
            "frac_parameters_changed": _f(params_changed / params_compared) if params_compared else None,
            "global_l2_diff": _f(math.sqrt(tot_sdd)),
            "global_rel_change": _f(tot_rel),
        },
        "top_tensors": records[:top_k],
        "top_layers": layer_recs[:top_layers],
        "only_in_a": _capped(only_a, max_list),
        "only_in_b": _capped(only_b, max_list),
        "shape_mismatches": _capped(shape_mm, max_list),
        "dtype_mismatches": _capped(dtype_mm, max_list),
        "undecoded_dtypes": _capped(unsupported, max_list),
        "nonfinite": _capped(nonfinite, max_list),
        "warnings": warnings,
        "interpretation": INTERPRETATION_NOTE,
        "limitations": LIMITATIONS,
        "elapsed_s": round(time.monotonic() - t0, 3),
    }


def supported_dtypes() -> list[str]:
    return sorted(SUPPORTED_DTYPES)
