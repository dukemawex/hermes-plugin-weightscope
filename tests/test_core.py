"""Unit tests for the safetensors reader and diff metrics (synthetic fixtures only)."""

import json
import math
import os
import struct

import numpy as np
import pytest

import stfixtures as F
from weightscope.analysis import diff, layer_key, summarize
from weightscope.st_reader import Limits, WeightScopeError, iter_chunks, open_checkpoint
from weightscope.tools import make_handlers


@pytest.fixture
def pair(tmp_path):
    base = F.base_model()
    a = F.write(tmp_path / "base.safetensors", base)
    b = F.write(tmp_path / "ft.safetensors", F.finetuned(base, layer=2))
    return base, a, b


# ── metrics ──────────────────────────────────────────────────────────────────

def test_changed_layer_ranks_top(pair):
    _, a, b = pair
    r = diff(a, b)
    assert r["ok"]
    assert r["totals"]["changed_tensors"] == 2
    assert {t["name"] for t in r["top_tensors"]} == {
        "model.layers.2.mlp.up_proj.weight", "model.layers.2.self_attn.q_proj.weight"}
    assert r["top_layers"][0]["layer"] == "model.layers.2"
    assert len(r["top_layers"]) == 1
    for rank_by in ("l2", "max_abs"):
        assert diff(a, b, rank_by=rank_by)["top_layers"][0]["layer"] == "model.layers.2"
    assert "does NOT establish a backdoor" in r["interpretation"]


def test_changed_layer_ranks_top_amid_background_noise(tmp_path):
    """Every tensor drifts slightly; layer 1 gets a real update and must still rank first."""
    base = F.base_model()
    rng = np.random.default_rng(7)
    noisy = {k: (dt, (a + rng.standard_normal(a.shape).astype(np.float32) * 1e-4).astype(np.float32))
             for k, (dt, a) in base.items()}
    cand = F.finetuned(noisy, layer=1, scale=0.05)
    a = F.write(tmp_path / "a.safetensors", base)
    b = F.write(tmp_path / "b.safetensors", cand)
    r = diff(a, b, top_k=3)
    assert r["totals"]["changed_tensors"] == len(base)
    assert r["top_layers"][0]["layer"] == "model.layers.1"
    assert {t["name"] for t in r["top_tensors"][:2]} == {
        "model.layers.1.mlp.up_proj.weight", "model.layers.1.self_attn.q_proj.weight"}


def test_metrics_match_numpy(pair):
    base, a, b = pair
    ft = F.finetuned(base, layer=2)
    name = "model.layers.2.mlp.up_proj.weight"
    x = base[name][1].astype(np.float64).ravel()
    y = ft[name][1].astype(np.float64).ravel()
    rec = next(t for t in diff(a, b)["top_tensors"] if t["name"] == name)
    d = y - x
    assert rec["l2_diff"] == pytest.approx(np.linalg.norm(d), rel=1e-5)
    assert rec["rel_change"] == pytest.approx(np.linalg.norm(d) / np.linalg.norm(x), rel=1e-5)
    assert rec["max_abs_diff"] == pytest.approx(np.abs(d).max(), rel=1e-5)
    assert rec["cosine"] == pytest.approx(x @ y / (np.linalg.norm(x) * np.linalg.norm(y)), rel=1e-6)
    assert rec["frac_changed"] == pytest.approx(np.mean(np.abs(d) > 0))
    r_tol = diff(a, b, tolerance=0.1)
    rec_tol = next(t for t in r_tol["top_tensors"] if t["name"] == name)
    assert rec_tol["frac_changed"] == pytest.approx(np.mean(np.abs(d) > 0.1), abs=1e-6)


def test_chunked_equals_unchunked(pair):
    _, a, b = pair
    full = diff(a, b)
    tiny = diff(a, b, limits=Limits(chunk_elems=7))
    assert full["top_tensors"] == tiny["top_tensors"]
    assert full["totals"] == tiny["totals"]


def test_identical_checkpoints(pair):
    _, a, _ = pair
    r = diff(a, a)
    assert r["totals"]["changed_tensors"] == 0
    assert r["top_tensors"] == []
    assert any("same file" in w for w in r["warnings"])


def test_sharded_index_matches_single_file(tmp_path, pair):
    base, a, b = pair
    sharded_b = F.write_sharded(tmp_path / "ft_sharded", F.finetuned(base, layer=2), n_shards=3)
    ck = open_checkpoint(sharded_b)
    assert ck.layout == "sharded-index" and len(ck.files) == 3
    r1, r2 = diff(a, b), diff(a, sharded_b)
    assert r1["top_tensors"] == r2["top_tensors"]
    # index file path also accepted
    r3 = diff(a, sharded_b / "model.safetensors.index.json")
    assert r3["totals"] == r1["totals"]


def test_directory_without_index(tmp_path):
    base = F.base_model()
    d = tmp_path / "dir"
    names = list(base)
    F.write(d / "part1.safetensors", {n: base[n] for n in names[:5]})
    F.write(d / "part2.safetensors", {n: base[n] for n in names[5:]})
    ck = open_checkpoint(d)
    assert ck.layout == "directory" and len(ck.tensors) == len(base)


def test_only_in_and_shape_and_dtype_mismatch(tmp_path):
    base = F.base_model()
    other = dict(base)
    other.pop("lm_head.weight")
    other["extra.bias"] = ("F32", np.zeros(3, np.float32))
    other["model.norm.weight"] = ("F32", np.ones(20, np.float32))  # shape mismatch
    other["model.layers.0.input_layernorm.weight"] = ("F16", base["model.layers.0.input_layernorm.weight"][1])
    a = F.write(tmp_path / "a.safetensors", base)
    b = F.write(tmp_path / "b.safetensors", other)
    r = diff(a, b)
    assert r["only_in_a"]["items"] == ["lm_head.weight"]
    assert r["only_in_b"]["items"] == ["extra.bias"]
    assert r["shape_mismatches"]["items"] == [{"name": "model.norm.weight", "shape_a": [16], "shape_b": [20]}]
    assert r["dtype_mismatches"]["items"][0]["dtype_b"] == "F16"
    # ones are exactly representable in F16: the cast tensor is numerically identical
    assert r["totals"]["changed_tensors"] == 0


def test_bf16_and_f16_decoding(tmp_path):
    vals = np.array([0.0, 1.0, -2.5, 3.140625, 1e-3, 65504.0], dtype=np.float32)
    p = F.write(tmp_path / "x.safetensors", {"bf": ("BF16", vals), "hf": ("F16", vals), "i8": ("I8", np.arange(-3, 3))})
    ck = open_checkpoint(p)
    bf = np.concatenate(list(iter_chunks(ck.tensors["bf"], 4)))
    hf = np.concatenate(list(iter_chunks(ck.tensors["hf"], 4)))
    i8 = np.concatenate(list(iter_chunks(ck.tensors["i8"], 4)))
    np.testing.assert_allclose(bf, vals, rtol=2 ** -8)
    np.testing.assert_allclose(hf, vals, rtol=1e-3)
    assert list(i8) == [-3, -2, -1, 0, 1, 2]


def test_bf16_vs_f32_diff(tmp_path):
    base = F.base_model()
    a = F.write(tmp_path / "f32.safetensors", base)
    b = F.write(tmp_path / "bf16.safetensors", F.as_dtype(base, "BF16"))
    r = diff(a, b)
    assert r["dtype_mismatches"]["count"] == len(base)
    # bf16 rounding error is small relative to weights
    assert 0 < r["totals"]["global_rel_change"] < 0.01


def test_nan_inf_excluded_and_counted(tmp_path):
    x = np.arange(1, 9, dtype=np.float32)
    y = x.copy()
    y[0] = np.nan
    y[1] = np.inf
    y[2] += 1.0
    a = F.write(tmp_path / "a.safetensors", {"w": ("F32", x)})
    b = F.write(tmp_path / "b.safetensors", {"w": ("F32", y)})
    r = diff(a, b)
    rec = r["top_tensors"][0]
    assert rec["nonfinite_a"] == 0 and rec["nonfinite_b"] == 2
    assert rec["l2_diff"] == pytest.approx(1.0)
    assert rec["max_abs_diff"] == pytest.approx(1.0)
    assert r["nonfinite"]["count"] == 1
    json.dumps(r, allow_nan=False)  # strict JSON: no NaN/Inf leaks


def test_nonfinite_only_difference_counts_as_changed(tmp_path):
    x = np.ones(4, np.float32)
    y = x.copy()
    y[3] = np.nan
    a = F.write(tmp_path / "a.safetensors", {"w": ("F32", x)})
    b = F.write(tmp_path / "b.safetensors", {"w": ("F32", y)})
    r = diff(a, b)
    assert r["totals"]["changed_tensors"] == 1


def test_zero_baseline(tmp_path):
    a = F.write(tmp_path / "a.safetensors", {"z": ("F32", np.zeros(4, np.float32)),
                                             "same": ("F32", np.zeros(2, np.float32))})
    b = F.write(tmp_path / "b.safetensors", {"z": ("F32", np.array([0, 0, 3, 4], np.float32)),
                                             "same": ("F32", np.zeros(2, np.float32))})
    r = diff(a, b)
    rec = r["top_tensors"][0]
    assert rec["name"] == "z" and rec["zero_baseline"] is True
    assert rec["rel_change"] is None and rec["l2_diff"] == pytest.approx(5.0)
    assert rec["cosine"] is None
    assert r["totals"]["changed_tensors"] == 1 and r["totals"]["identical_tensors"] == 1
    json.dumps(r, allow_nan=False)


def test_top_k_and_list_caps(tmp_path):
    a_t = {f"t.{i}.w": ("F32", np.ones(2, np.float32)) for i in range(300)}
    b_t = {f"t.{i}.w": ("F32", np.full(2, 1 + i / 100, np.float32)) for i in range(300)}
    b_t.update({f"new.{i}": ("F32", np.ones(1, np.float32)) for i in range(250)})
    a = F.write(tmp_path / "a.safetensors", a_t)
    b = F.write(tmp_path / "b.safetensors", b_t)
    r = diff(a, b, top_k=5000, top_layers=0)
    assert len(r["top_tensors"]) == 100 and r["settings"]["top_k"] == 100
    assert r["top_tensors"][0]["name"] == "t.299.w"
    assert len(r["top_layers"]) == 1
    assert r["only_in_b"]["count"] == 250 and len(r["only_in_b"]["items"]) == 25 and r["only_in_b"]["truncated"]
    assert len(json.dumps(r)) < 60_000


def test_layer_key():
    assert layer_key("model.layers.3.mlp.up_proj.weight") == "model.layers.3"
    assert layer_key("transformer.h.11.attn.c_attn.weight") == "transformer.h.11"
    assert layer_key("model.embed_tokens.weight") == "model.embed_tokens"
    assert layer_key("lm_head.weight") == "lm_head"
    assert layer_key("bias") == "bias"


def test_summary(pair):
    base, a, _ = pair
    s = summarize(a)
    assert s["checkpoint"]["tensors"] == len(base)
    assert s["checkpoint"]["parameters"] == sum(x.size for _, x in base.values())
    assert s["dtypes"]["F32"]["tensors"] == len(base)
    import hashlib
    assert s["files"][0]["sha256"] == hashlib.sha256(a.read_bytes()).hexdigest()
    assert "sha256" not in summarize(a, hash_files=False)["files"][0]


# ── hardening ────────────────────────────────────────────────────────────────

def _raw(header: dict, data: bytes = b"") -> bytes:
    hb = json.dumps(header).encode()
    return struct.pack("<Q", len(hb)) + hb + data


@pytest.mark.parametrize("blob, msg", [
    (b"", "truncated"),
    (b"\x01\x02", "truncated"),
    (struct.pack("<Q", 0), "empty header"),
    (struct.pack("<Q", 50) + b"{}", "truncated"),
    (struct.pack("<Q", 5) + b"{nope", "not valid UTF-8 JSON"),
    (struct.pack("<Q", 4) + b"\xff\xfe\xfd\xfc", "not valid UTF-8 JSON"),
    (struct.pack("<Q", 2) + b"[]", "must be a JSON object"),
    (_raw({"w": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}}, b"\0" * 4), "exceed data region"),
    (_raw({"w": {"dtype": "F32", "shape": [3], "data_offsets": [0, 8]}}, b"\0" * 8), "size mismatch"),
    (_raw({"w": {"dtype": "F32", "shape": [-1], "data_offsets": [0, 4]}}, b"\0" * 4), "invalid shape"),
    (_raw({"w": {"dtype": "F32", "shape": [1], "data_offsets": [4, 0]}}, b"\0" * 4), "invalid data_offsets"),
    (_raw({"w": {"dtype": "F32", "shape": [1], "data_offsets": [0, True]}}, b"\0" * 4), "invalid data_offsets"),
    (_raw({"w": {"dtype": 5, "shape": [1], "data_offsets": [0, 4]}}, b"\0" * 4), "dtype"),
    (_raw({"w": "x"}), "not an object"),
    (_raw({"a": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]},
           "b": {"dtype": "F32", "shape": [2], "data_offsets": [4, 12]}}, b"\0" * 12), "overlapping"),
])
def test_corrupt_files_give_clean_errors(tmp_path, blob, msg):
    p = tmp_path / "bad.safetensors"
    p.write_bytes(blob)
    with pytest.raises(WeightScopeError, match=msg):
        open_checkpoint(p)


def test_truncated_real_file(tmp_path, pair):
    _, a, _ = pair
    data = a.read_bytes()
    t = tmp_path / "trunc.safetensors"
    t.write_bytes(data[:-100])
    with pytest.raises(WeightScopeError, match="exceed data region"):
        open_checkpoint(t)


def test_oversize_header_rejected_without_reading(tmp_path):
    p = tmp_path / "huge.safetensors"
    p.write_bytes(struct.pack("<Q", 200 * 1024 * 1024) + b"{}")
    with pytest.raises(WeightScopeError, match="exceeds limit"):
        open_checkpoint(p)
    p2 = tmp_path / "small_limit.safetensors"
    F.write(p2, F.base_model())
    with pytest.raises(WeightScopeError, match="exceeds limit"):
        open_checkpoint(p2, Limits(max_header_bytes=64))


def test_tensor_count_and_total_bytes_limits(pair):
    _, a, _ = pair
    with pytest.raises(WeightScopeError, match="tensors exceeds limit"):
        open_checkpoint(a, Limits(max_tensors=3))
    with pytest.raises(WeightScopeError, match="max_total_bytes"):
        open_checkpoint(a, Limits(max_total_bytes=1000))


@pytest.mark.parametrize("shard", ["../outside.safetensors", "/etc/passwd", "sub/../../x.safetensors",
                                   "..\\x.safetensors", "C:\\x.safetensors", ""])
def test_index_path_traversal_rejected(tmp_path, shard):
    d = tmp_path / "ck"
    d.mkdir()
    F.write(tmp_path / "outside.safetensors", {"w": ("F32", np.ones(2, np.float32))})
    (d / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"w": shard}}))
    with pytest.raises(WeightScopeError, match="escapes|invalid shard"):
        open_checkpoint(d)


def test_index_symlink_escape_rejected_unless_allowed(tmp_path):
    d = tmp_path / "ck"
    d.mkdir()
    target = F.write(tmp_path / "elsewhere" / "real.safetensors", {"w": ("F32", np.ones(2, np.float32))})
    os.symlink(target, d / "model-00001-of-00001.safetensors")
    (d / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"w": "model-00001-of-00001.safetensors"}}))
    with pytest.raises(WeightScopeError, match="outside the checkpoint directory"):
        open_checkpoint(d)
    assert "w" in open_checkpoint(d, Limits(allow_external_symlinks=True)).tensors


def test_index_inconsistencies(tmp_path):
    base = F.base_model()
    d = F.write_sharded(tmp_path / "ck", base, n_shards=2)
    idx = json.loads((d / "model.safetensors.index.json").read_text())
    idx["weight_map"]["ghost.weight"] = "model-00001-of-00002.safetensors"
    (d / "model.safetensors.index.json").write_text(json.dumps(idx))
    with pytest.raises(WeightScopeError, match="no shard contains it"):
        open_checkpoint(d)
    (d / "model.safetensors.index.json").write_text("{not json")
    with pytest.raises(WeightScopeError, match="cannot parse index"):
        open_checkpoint(d)
    (d / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"w": "missing.safetensors"}}))
    with pytest.raises(WeightScopeError, match="not found"):
        open_checkpoint(d)


def test_duplicate_tensor_across_files(tmp_path):
    d = tmp_path / "dup"
    F.write(d / "a.safetensors", {"w": ("F32", np.ones(2, np.float32))})
    F.write(d / "b.safetensors", {"w": ("F32", np.ones(2, np.float32))})
    with pytest.raises(WeightScopeError, match="more than one file"):
        open_checkpoint(d)


def test_undecoded_dtype_listed_not_compared(tmp_path):
    hdr = {"q": {"dtype": "F8_E4M3", "shape": [4], "data_offsets": [0, 4]},
           "w": {"dtype": "F32", "shape": [1], "data_offsets": [4, 8]}}
    a = tmp_path / "a.safetensors"
    a.write_bytes(_raw(hdr, b"\1\2\3\4" + np.float32(1).tobytes()))
    r = diff(a, a)
    assert r["undecoded_dtypes"]["items"][0]["name"] == "q"
    assert summarize(a)["undecoded_dtype_tensors"]["count"] == 1


# ── handlers (JSON string, never raise) ──────────────────────────────────────

def test_handlers_return_json_and_never_raise(pair, tmp_path):
    _, a, b = pair
    summ, dif = make_handlers()
    ok = json.loads(dif({"a": str(a), "b": str(b), "top_k": 3}, task_id="t1", session_id="s"))
    assert ok["ok"] and len(ok["top_tensors"]) == 2
    assert json.loads(summ({"path": str(a)}))["ok"]
    for bad in ({}, {"a": 5, "b": None}, {"a": str(a), "b": str(tmp_path / "missing")},
                {"a": str(a), "b": str(b), "rank_by": "bogus"}, {"a": str(a), "b": str(b), "tolerance": "x"},
                {"a": str(a), "b": str(b), "tolerance": -1}, None, "not a dict"):
        res = json.loads(dif(bad))
        assert res["ok"] is False and res["error"]
    assert json.loads(summ({"path": str(tmp_path)}))["ok"] is False


def test_relative_paths_resolve_against_session_cwd(pair):
    _, a, b = pair
    _, dif = make_handlers(get_cwd=lambda: str(a.parent))
    res = json.loads(dif({"a": a.name, "b": b.name}))
    assert res["ok"] and res["totals"]["changed_tensors"] == 2
