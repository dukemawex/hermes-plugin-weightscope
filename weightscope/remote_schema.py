"""Strict validation of remote-worker responses (protocol weightscope-remote/1).

Everything a worker returns is untrusted: unknown keys are rejected, every number must be finite and in
range, strings and lists are length-capped. Mirrors weightscope/docs/remote-worker.schema.json.
"""

from __future__ import annotations

import math

from .st_reader import WeightScopeError

PROTOCOL = "weightscope-remote/1"
MAX_TOP = 100          # top_tensors / top_layers
MAX_ITEMS = 200        # items in capped blocks, tensors/files lists
MAX_WARNINGS = 50
MAX_NAME = 1024        # tensor / layer / path strings
MAX_SHORT = 200        # device fields, dtype strings, errors
MAX_DIMS = 16
MAX_INT = 10 ** 15
LAYOUTS = {"file", "directory", "sharded-index"}
RANK_KEYS = {"relative", "l2", "max_abs"}
DEVICE_KEYS = {"backend", "device", "device_name", "compute_dtype", "accumulate_dtype", "torch_version",
               "cuda_version", "numpy_version", "worker", "chunk_elems"}
BLOCKS = ("only_in_a", "only_in_b", "shape_mismatches", "dtype_mismatches", "undecoded_dtypes", "nonfinite")


class _Invalid(Exception):
    pass


def _fail(path, msg):
    raise _Invalid(f"{path}: {msg}")


def _keys(obj, path, required, optional=()):
    if not isinstance(obj, dict):
        _fail(path, "expected object")
    missing = [k for k in required if k not in obj]
    if missing:
        _fail(path, f"missing {', '.join(missing)}")
    extra = set(obj) - set(required) - set(optional)
    if extra:
        _fail(path, f"unexpected key(s) {', '.join(sorted(map(str, extra))[:5])}")


def _str(v, path, cap=MAX_NAME, choices=None):
    if not isinstance(v, str):
        _fail(path, "expected string")
    if len(v) > cap:
        _fail(path, f"string longer than {cap}")
    if choices is not None and v not in choices:
        _fail(path, f"unexpected value {v[:40]!r}")
    return v


def _int(v, path, lo=0, hi=MAX_INT):
    if not isinstance(v, int) or isinstance(v, bool):
        _fail(path, "expected integer")
    if not lo <= v <= hi:
        _fail(path, f"integer out of range [{lo}, {hi}]")
    return v


def _num(v, path, lo=0.0, hi=math.inf, nullable=True):
    if v is None and nullable:
        return v
    if not isinstance(v, (int, float)) or isinstance(v, bool):
        _fail(path, "expected number")
    if not math.isfinite(v):
        _fail(path, "non-finite number")
    if not lo <= v <= hi:
        _fail(path, f"number out of range [{lo}, {hi}]")
    return v


def _bool(v, path):
    if not isinstance(v, bool):
        _fail(path, "expected boolean")


def _list(v, path, cap):
    if not isinstance(v, list):
        _fail(path, "expected array")
    if len(v) > cap:
        _fail(path, f"more than {cap} items")
    return v


def _shape(v, path):
    for i, d in enumerate(_list(v, path, MAX_DIMS)):
        _int(d, f"{path}[{i}]")


def _brief(v, path):
    _keys(v, path, ("path", "layout", "files", "tensors", "parameters", "bytes"))
    _str(v["path"], f"{path}.path")
    _str(v["layout"], f"{path}.layout", choices=LAYOUTS)
    for k in ("files", "tensors", "parameters", "bytes"):
        _int(v[k], f"{path}.{k}")


def _block(v, path, item_check):
    _keys(v, path, ("count", "items", "truncated"))
    _int(v["count"], f"{path}.count")
    items = _list(v["items"], f"{path}.items", MAX_ITEMS)
    _bool(v["truncated"], f"{path}.truncated")
    if v["count"] < len(items):
        _fail(path, "count smaller than items")
    for i, it in enumerate(items):
        item_check(it, f"{path}.items[{i}]")


def _name_item(it, p):
    _str(it, p)


def _shape_mm(it, p):
    _keys(it, p, ("name", "shape_a", "shape_b"))
    _str(it["name"], f"{p}.name")
    _shape(it["shape_a"], f"{p}.shape_a")
    _shape(it["shape_b"], f"{p}.shape_b")


def _dtype_mm(it, p):
    _keys(it, p, ("name", "dtype_a", "dtype_b"))
    _str(it["name"], f"{p}.name")
    _str(it["dtype_a"], f"{p}.dtype_a", 32)
    _str(it["dtype_b"], f"{p}.dtype_b", 32)


def _nonfinite(it, p):
    _keys(it, p, ("name", "nonfinite_a", "nonfinite_b", "of"))
    _str(it["name"], f"{p}.name")
    for k in ("nonfinite_a", "nonfinite_b", "of"):
        _int(it[k], f"{p}.{k}")


_BLOCK_ITEMS = {"only_in_a": _name_item, "only_in_b": _name_item, "shape_mismatches": _shape_mm,
                "dtype_mismatches": _dtype_mm, "undecoded_dtypes": _dtype_mm, "nonfinite": _nonfinite}


def _tensor_rec(r, p):
    _keys(r, p, ("name", "shape", "dtype", "l2_diff", "base_l2", "rel_change", "max_abs_diff", "cosine",
                 "frac_changed"), ("zero_baseline", "nonfinite_a", "nonfinite_b"))
    _str(r["name"], f"{p}.name")
    _shape(r["shape"], f"{p}.shape")
    _str(r["dtype"], f"{p}.dtype", 32)
    for k in ("l2_diff", "base_l2", "rel_change", "max_abs_diff"):
        _num(r[k], f"{p}.{k}")
    _num(r["cosine"], f"{p}.cosine", -1.0, 1.0)
    _num(r["frac_changed"], f"{p}.frac_changed", 0.0, 1.0)
    if "zero_baseline" in r:
        _bool(r["zero_baseline"], f"{p}.zero_baseline")
    for k in ("nonfinite_a", "nonfinite_b"):
        if k in r:
            _int(r[k], f"{p}.{k}")


def _layer_rec(r, p):
    _keys(r, p, ("layer", "tensors", "changed_tensors", "parameters", "l2_diff", "base_l2", "rel_change",
                 "max_abs_diff"), ("zero_baseline",))
    _str(r["layer"], f"{p}.layer")
    for k in ("tensors", "changed_tensors", "parameters"):
        _int(r[k], f"{p}.{k}")
    for k in ("l2_diff", "base_l2", "rel_change", "max_abs_diff"):
        _num(r[k], f"{p}.{k}")
    if "zero_baseline" in r:
        _bool(r["zero_baseline"], f"{p}.zero_baseline")


def _warnings(v, p):
    for i, w in enumerate(_list(v, p, MAX_WARNINGS)):
        _str(w, f"{p}[{i}]", 500)


def _diff_result(r, p="result"):
    _keys(r, p, ("a", "b", "settings", "totals", "top_tensors", "top_layers", *BLOCKS, "warnings", "elapsed_s"))
    _brief(r["a"], f"{p}.a")
    _brief(r["b"], f"{p}.b")
    s = r["settings"]
    _keys(s, f"{p}.settings", ("tolerance", "rank_by", "top_k", "top_layers"))
    _num(s["tolerance"], f"{p}.settings.tolerance", nullable=False)
    _str(s["rank_by"], f"{p}.settings.rank_by", choices=RANK_KEYS)
    _int(s["top_k"], f"{p}.settings.top_k", 1, MAX_TOP)
    _int(s["top_layers"], f"{p}.settings.top_layers", 1, MAX_TOP)
    t = r["totals"]
    _keys(t, f"{p}.totals", ("common_tensors", "compared_tensors", "changed_tensors", "identical_tensors",
                             "changed_layers", "compared_parameters", "changed_parameters",
                             "frac_parameters_changed", "global_l2_diff", "global_rel_change"))
    for k in ("common_tensors", "compared_tensors", "changed_tensors", "identical_tensors", "changed_layers",
              "compared_parameters", "changed_parameters"):
        _int(t[k], f"{p}.totals.{k}")
    if t["changed_tensors"] + t["identical_tensors"] != t["compared_tensors"]:
        _fail(f"{p}.totals", "changed + identical != compared")
    if t["changed_parameters"] > t["compared_parameters"]:
        _fail(f"{p}.totals", "changed_parameters > compared_parameters")
    _num(t["frac_parameters_changed"], f"{p}.totals.frac_parameters_changed", 0.0, 1.0)
    _num(t["global_l2_diff"], f"{p}.totals.global_l2_diff")
    _num(t["global_rel_change"], f"{p}.totals.global_rel_change")
    for i, rec in enumerate(_list(r["top_tensors"], f"{p}.top_tensors", MAX_TOP)):
        _tensor_rec(rec, f"{p}.top_tensors[{i}]")
    for i, rec in enumerate(_list(r["top_layers"], f"{p}.top_layers", MAX_TOP)):
        _layer_rec(rec, f"{p}.top_layers[{i}]")
    for b in BLOCKS:
        _block(r[b], f"{p}.{b}", _BLOCK_ITEMS[b])
    _warnings(r["warnings"], f"{p}.warnings")
    _num(r["elapsed_s"], f"{p}.elapsed_s", nullable=False)


def _summary_result(r, p="result"):
    _keys(r, p, ("checkpoint", "dtypes", "files", "files_truncated", "largest_tensors", "tensors",
                 "undecoded_dtype_tensors", "layers", "warnings", "note", "elapsed_s"))
    _brief(r["checkpoint"], f"{p}.checkpoint")
    if not isinstance(r["dtypes"], dict) or len(r["dtypes"]) > 64:
        _fail(f"{p}.dtypes", "expected object with at most 64 dtypes")
    for k, v in r["dtypes"].items():
        _str(k, f"{p}.dtypes key", 32)
        _keys(v, f"{p}.dtypes.{k}", ("tensors", "parameters", "bytes"))
        for f in ("tensors", "parameters", "bytes"):
            _int(v[f], f"{p}.dtypes.{k}.{f}")
    for i, f in enumerate(_list(r["files"], f"{p}.files", MAX_ITEMS)):
        q = f"{p}.files[{i}]"
        _keys(f, q, ("file", "bytes", "header_bytes", "tensors"), ("sha256", "metadata"))
        _str(f["file"], f"{q}.file")
        for k in ("bytes", "header_bytes", "tensors"):
            _int(f[k], f"{q}.{k}")
        if "sha256" in f:
            h = _str(f["sha256"], f"{q}.sha256", 64)
            if len(h) != 64 or any(c not in "0123456789abcdef" for c in h):
                _fail(f"{q}.sha256", "expected 64 lowercase hex chars")
        if "metadata" in f:
            if not isinstance(f["metadata"], dict) or len(f["metadata"]) > 20:
                _fail(f"{q}.metadata", "expected object with at most 20 keys")
            for mk, mv in f["metadata"].items():
                _str(mk, f"{q}.metadata key", 64)
                _str(mv, f"{q}.metadata.{mk}", 200)
    _bool(r["files_truncated"], f"{p}.files_truncated")
    for i, t in enumerate(_list(r["largest_tensors"], f"{p}.largest_tensors", MAX_ITEMS)):
        _keys(t, f"{p}.largest_tensors[{i}]", ("name", "shape", "parameters"))
        _str(t["name"], f"{p}.largest_tensors[{i}].name")
        _shape(t["shape"], f"{p}.largest_tensors[{i}].shape")
        _int(t["parameters"], f"{p}.largest_tensors[{i}].parameters")

    def _tensor_item(it, q):
        _keys(it, q, ("name", "dtype", "shape", "parameters"))
        _str(it["name"], f"{q}.name")
        _str(it["dtype"], f"{q}.dtype", 32)
        _shape(it["shape"], f"{q}.shape")
        _int(it["parameters"], f"{q}.parameters")

    _block(r["tensors"], f"{p}.tensors", _tensor_item)
    _block(r["undecoded_dtype_tensors"], f"{p}.undecoded_dtype_tensors", _name_item)
    _int(r["layers"], f"{p}.layers")
    _warnings(r["warnings"], f"{p}.warnings")
    _str(r["note"], f"{p}.note", 500)
    _num(r["elapsed_s"], f"{p}.elapsed_s", nullable=False)


def _device(d, p="device"):
    if not isinstance(d, dict):
        _fail(p, "expected object")
    extra = set(d) - DEVICE_KEYS
    if extra:
        _fail(p, f"unexpected key(s) {', '.join(sorted(map(str, extra))[:5])}")
    for k in ("backend", "device"):
        if k not in d:
            _fail(p, f"missing {k}")
    for k, v in d.items():
        if k == "chunk_elems":
            _int(v, f"{p}.{k}")
        elif v is not None:
            _str(v, f"{p}.{k}", MAX_SHORT)


def validate_envelope(obj, kind: str) -> dict:
    """Validate a decoded response. kind: 'diff' | 'summary' | 'health'. Returns obj or raises WeightScopeError."""
    try:
        if not isinstance(obj, dict):
            _fail("response", "expected object")
        if obj.get("protocol") != PROTOCOL:
            _fail("response.protocol", f"expected {PROTOCOL!r}")
        if obj.get("ok") is False:
            _keys(obj, "response", ("ok", "protocol", "error"))
            _str(obj["error"], "response.error", 1000)
            return obj
        if obj.get("ok") is not True:
            _fail("response.ok", "expected boolean")
        if kind == "health":
            _keys(obj, "response", ("ok", "protocol", "device"))
        else:
            _keys(obj, "response", ("ok", "protocol", "result", "device"))
            (_diff_result if kind == "diff" else _summary_result)(obj["result"])
        _device(obj["device"])
        return obj
    except _Invalid as exc:
        raise WeightScopeError(f"remote response failed validation ({exc})") from None
