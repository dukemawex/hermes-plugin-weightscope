"""Backend tests: numpy default, optional torch (CPU wheel here), auto-fallback, device errors.

No GPU exists on the development box: GPU tests are skipped unless torch reports CUDA/MPS.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import stfixtures as F
from weightscope import backends as B
from weightscope.analysis import diff
from weightscope.st_reader import Limits, WeightScopeError
from weightscope.tools import make_handlers

try:
    import torch  # noqa: F811
except Exception:  # pragma: no cover - exercised in the no-torch venv
    torch = None

needs_torch = pytest.mark.skipif(torch is None, reason="torch not installed")
HAS_CUDA = bool(torch is not None and torch.cuda.is_available())
HAS_MPS = bool(torch is not None and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def mixed_pair(tmp_path):
    """Covers F32, BF16, F16, I32, U8, NaN/Inf, zero baseline, and a dtype mismatch."""
    base = F.base_model(n_layers=3)
    ft = F.finetuned(base, layer=1, scale=0.05)
    rng = np.random.default_rng(3)
    extra_a = {
        "bf.w": ("BF16", rng.standard_normal(300).astype(np.float32)),
        "hf.w": ("F16", rng.standard_normal(70).astype(np.float32)),
        "i32.w": ("I32", np.arange(-50, 50, dtype=np.int32)),
        "u8.w": ("U8", np.arange(0, 200, dtype=np.uint8)),
        "nan.w": ("F32", np.arange(1, 11, dtype=np.float32)),
        "zero.w": ("F32", np.zeros(5, np.float32)),
        "mix.w": ("F32", rng.standard_normal(40).astype(np.float32)),
    }
    extra_b = {k: (dt, a.copy()) for k, (dt, a) in extra_a.items()}
    extra_b["bf.w"] = ("BF16", extra_a["bf.w"][1] + 0.01)
    extra_b["i32.w"] = ("I32", extra_a["i32.w"][1] + 3)
    nb = extra_a["nan.w"][1].copy()
    nb[0], nb[1], nb[4] = np.nan, np.inf, 9.5
    extra_b["nan.w"] = ("F32", nb)
    extra_b["zero.w"] = ("F32", np.array([0, 3, 0, 4, 0], np.float32))
    extra_b["mix.w"] = ("BF16", extra_a["mix.w"][1])
    a = F.write(tmp_path / "a.safetensors", {**base, **extra_a})
    b = F.write(tmp_path / "b.safetensors", {**ft, **extra_b})
    return a, b


def _assert_parity(r1, r2, rel):
    assert [t["name"] for t in r1["top_tensors"]] == [t["name"] for t in r2["top_tensors"]]
    for x, y in zip(r1["top_tensors"], r2["top_tensors"]):
        for k in ("l2_diff", "base_l2", "rel_change", "max_abs_diff", "cosine", "frac_changed"):
            if x[k] is None or y[k] is None:
                assert x[k] is None and y[k] is None, (x["name"], k)
            else:
                assert y[k] == pytest.approx(x[k], rel=rel, abs=1e-12), (x["name"], k, x[k], y[k])
        assert x.get("nonfinite_b") == y.get("nonfinite_b")
    for k in ("changed_tensors", "identical_tensors", "compared_parameters", "changed_parameters"):
        assert r1["totals"][k] == r2["totals"][k]
    assert r2["totals"]["global_rel_change"] == pytest.approx(r1["totals"]["global_rel_change"], rel=rel)


# ── numpy / selection logic (no torch needed) ────────────────────────────────

def test_numpy_backend_metadata(mixed_pair):
    r = diff(*mixed_pair, backend="numpy", top_k=100)
    assert r["execution"]["backend"] == "numpy" and r["execution"]["device"] == "cpu"
    assert r["execution"]["compute_dtype"] == "float64"
    assert r["execution"]["accumulate_dtype"].startswith("float64")
    assert r["top_layers"][0]["layer"] == "zero"  # zero-baseline change ranks first under relative
    assert "model.layers.1" in [x["layer"] for x in r["top_layers"]]


def test_auto_falls_back_to_numpy_without_torch(monkeypatch, mixed_pair):
    monkeypatch.setattr(B, "_import_torch", lambda: (None, "torch not importable (ImportError: test)"))
    be = B.resolve_backend("auto")
    assert be.name == "numpy" and "not importable" in be.fallback_reason
    r = diff(*mixed_pair, backend="auto")
    assert r["execution"]["backend"] == "numpy" and "fallback_reason" in r["execution"]
    with pytest.raises(WeightScopeError, match="backend 'torch' requested but torch not importable"):
        B.resolve_backend("torch")
    with pytest.raises(WeightScopeError, match="device 'cuda' requested"):
        B.resolve_backend("auto", "cuda")


def test_backend_and_device_validation():
    assert B.resolve_backend("auto", "cpu").name == "numpy"
    assert B.resolve_backend("NumPy").name == "numpy"
    with pytest.raises(WeightScopeError, match="backend must be one of"):
        B.resolve_backend("cupy")
    with pytest.raises(WeightScopeError, match="numpy backend runs on cpu only"):
        B.resolve_backend("numpy", "cuda")


def test_tool_reports_backend_errors_as_json(mixed_pair):
    _, dif = make_handlers()
    a, b = mixed_pair
    res = json.loads(dif({"a": str(a), "b": str(b), "backend": "numpy", "device": "cuda:0"}))
    assert res["ok"] is False and "cpu only" in res["error"]
    res = json.loads(dif({"a": str(a), "b": str(b), "backend": "numpy"}))
    assert res["ok"] and res["execution"]["backend"] == "numpy"


def test_backend_setting_used_when_no_arg(mixed_pair):
    a, b = mixed_pair
    _, dif = make_handlers(get_setting=lambda k, d=None: {"backend": "numpy", "record_history": False}.get(k, d))
    assert json.loads(dif({"a": str(a), "b": str(b)}))["execution"]["backend"] == "numpy"


def test_plugin_import_does_not_import_torch():
    code = ("import sys; sys.path.insert(0, sys.argv[1]); import weightscope, weightscope.backends, "
            "weightscope.analysis; print('torch' in sys.modules)")
    out = subprocess.run([sys.executable, "-c", code, str(ROOT)], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


def test_works_when_torch_is_unimportable(tmp_path):
    """Simulate a Hermes env without torch (sys.modules['torch'] = None blocks the import)."""
    base = F.base_model()
    a = F.write(tmp_path / "a.safetensors", base)
    b = F.write(tmp_path / "b.safetensors", F.finetuned(base))
    code = ("import sys, json; sys.modules['torch'] = None; sys.path.insert(0, sys.argv[1]);"
            "from weightscope.analysis import diff; from weightscope.st_reader import WeightScopeError;"
            "r = diff(sys.argv[2], sys.argv[3]); print(json.dumps(r['execution']));"
            "\ntry:\n    diff(sys.argv[2], sys.argv[3], backend='torch')\nexcept WeightScopeError as e:\n    print('ERR', e)")
    out = subprocess.run([sys.executable, "-c", code, str(ROOT), str(a), str(b)], capture_output=True, text=True,
                         check=True)
    first, second = out.stdout.strip().splitlines()
    comp = json.loads(first)
    assert comp["backend"] == "numpy" and "torch not importable" in comp["fallback_reason"]
    assert second.startswith("ERR backend 'torch' requested but torch not importable")


# ── torch on CPU ─────────────────────────────────────────────────────────────

@needs_torch
def test_torch_cpu_parity_with_numpy(mixed_pair):
    r_np = diff(*mixed_pair, backend="numpy", top_k=100)
    r_t = diff(*mixed_pair, backend="torch", device="cpu", top_k=100)
    assert r_t["execution"]["backend"] == "torch" and r_t["execution"]["device"] == "cpu"
    assert r_t["execution"]["compute_dtype"] == "float64"
    _assert_parity(r_np, r_t, rel=1e-9)
    assert r_t["dtype_mismatches"] == r_np["dtype_mismatches"]
    assert r_t["nonfinite"] == r_np["nonfinite"]


@needs_torch
def test_torch_small_device_budget_matches(mixed_pair):
    lim = Limits(device_budget_mib=16)
    be = B.resolve_backend("torch", "cpu")
    assert be.chunk_elems(lim) == 16 * 1024 * 1024 // (4 * 8 + 3 + 16)
    # force tiny transfer batches through the chunk_elems floor
    be.chunk_elems = lambda limits: 7
    r_small = diff(*mixed_pair, backend=be, top_k=100)
    _assert_parity(diff(*mixed_pair, backend="numpy", top_k=100), r_small, rel=1e-9)


@needs_torch
def test_torch_float32_compute_path_close(mixed_pair):
    """The MPS code path computes in float32; exercise that math on CPU (MPS itself is not available here)."""
    be = B.resolve_backend("torch", "cpu")
    be._cdt, be.compute_dtype = torch.float32, "float32"
    r = diff(*mixed_pair, backend=be, top_k=100)
    assert r["execution"]["compute_dtype"] == "float32"
    _assert_parity(diff(*mixed_pair, backend="numpy", top_k=100), r, rel=1e-4)


@needs_torch
def test_torch_bf16_transferred_natively():
    be = B.resolve_backend("torch", "cpu")
    raw = np.array([0x3F80, 0xC020, 0x7FC0], dtype=np.uint16)  # 1.0, -2.5, NaN in bfloat16
    t = torch.from_numpy(raw.view(np.int16)).view(torch.bfloat16)
    assert t.dtype == torch.bfloat16
    vals = be._to_device(raw, "BF16")
    assert vals.dtype == torch.float64 and vals[:2].tolist() == [1.0, -2.5] and torch.isnan(vals[2])


@needs_torch
def test_auto_without_gpu_picks_numpy():
    if HAS_CUDA or HAS_MPS:
        pytest.skip("a GPU is present")
    be = B.resolve_backend("auto")
    assert be.name == "numpy" and "no GPU" in be.fallback_reason


@needs_torch
def test_auto_prefers_cuda_when_reported(monkeypatch):
    """Selection logic only (mocked availability); real CUDA execution is not tested here."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(B, "_torch_device", lambda t, d: torch.device(d))
    be = B.resolve_backend("auto")
    assert be.name == "torch" and be.device == "cuda" and be.compute_dtype == "float64"


@needs_torch
@pytest.mark.parametrize("device, msg", [
    ("cuda", "CUDA is not available"), ("cuda:3", "CUDA is not available"),
    ("mps", "MPS is not available"), ("not-a-device", "invalid torch device"),
])
def test_unavailable_device_clean_error(device, msg, mixed_pair):
    if (device.startswith("cuda") and HAS_CUDA) or (device == "mps" and HAS_MPS):
        pytest.skip("device exists here")
    with pytest.raises(WeightScopeError, match=msg):
        B.resolve_backend("torch", device)
    _, dif = make_handlers()
    res = json.loads(dif({"a": str(mixed_pair[0]), "b": str(mixed_pair[1]), "backend": "torch", "device": device}))
    assert res["ok"] is False and msg in res["error"]
