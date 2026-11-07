"""Pluggable compute backends for the per-chunk diff statistics.

The parser stays numpy-based (seek-and-read of raw chunks). A backend turns a pair of raw chunks into
partial sums; partial sums are always accumulated across chunks on the host in Python float64.

* ``numpy``  - default, always available, CPU, float64 element math.
* ``torch``  - optional, lazily imported (never a declared dependency). Devices: ``cuda[:N]``, ``mps``,
  ``cpu`` (or any device string torch accepts). BF16 chunks are moved to the device as bfloat16 and
  upcast there. Element math runs in float64 where the device supports it (CPU, CUDA), float32 on MPS.

``auto`` picks torch only when torch imports AND a GPU (CUDA or MPS) is available; otherwise numpy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .st_reader import Limits, WeightScopeError, bf16_to_f32

BACKEND_CHOICES = ("auto", "numpy", "torch")
HOST_ACCUMULATE = "float64 (host, across chunks)"


@dataclass
class ChunkStats:
    sdd: float = 0.0
    saa: float = 0.0
    sbb: float = 0.0
    sab: float = 0.0
    maxd: float = 0.0
    changed: int = 0
    compared: int = 0
    nf_a: int = 0
    nf_b: int = 0


class Backend:
    name = "base"
    device = "cpu"
    compute_dtype = "float64"
    fallback_reason: str | None = None

    def chunk_elems(self, limits: Limits) -> int:
        return max(1, int(limits.chunk_elems))

    def chunk_stats(self, raw_a: np.ndarray, dtype_a: str, raw_b: np.ndarray, dtype_b: str,
                    tol: float) -> ChunkStats:
        raise NotImplementedError

    def info(self, limits: Limits) -> dict:
        out = {"backend": self.name, "device": self.device, "compute_dtype": self.compute_dtype,
               "accumulate_dtype": HOST_ACCUMULATE, "chunk_elems": self.chunk_elems(limits)}
        if self.fallback_reason:
            out["fallback_reason"] = self.fallback_reason
        return out


# ── numpy ────────────────────────────────────────────────────────────────────

def _np_f64(raw: np.ndarray, dtype: str) -> np.ndarray:
    if dtype == "BF16":
        raw = bf16_to_f32(raw)
    return raw.astype(np.float64)


class NumpyBackend(Backend):
    name = "numpy"

    def chunk_stats(self, raw_a, dtype_a, raw_b, dtype_b, tol):
        st = ChunkStats()
        with np.errstate(over="ignore", invalid="ignore"):
            ca, cb = _np_f64(raw_a, dtype_a), _np_f64(raw_b, dtype_b)
            fa, fb = np.isfinite(ca), np.isfinite(cb)
            st.nf_a = int(ca.size - np.count_nonzero(fa))
            st.nf_b = int(cb.size - np.count_nonzero(fb))
            m = fa & fb
            if not m.all():
                ca, cb = ca[m], cb[m]
            if ca.size == 0:
                return st
            d = cb - ca
            ad = np.abs(d)
            st.sdd, st.saa, st.sbb, st.sab = float(d @ d), float(ca @ ca), float(cb @ cb), float(ca @ cb)
            st.maxd = float(ad.max())
            st.changed = int(np.count_nonzero(ad > tol))
            st.compared = int(ca.size)
        return st


# ── torch (optional) ─────────────────────────────────────────────────────────

def _import_torch():
    """Lazy torch import. Returns (module, None) or (None, reason)."""
    try:
        import torch  # noqa: PLC0415 - optional dependency, imported only when requested
    except Exception as exc:  # ImportError, or OSError from broken CUDA libraries
        return None, f"torch not importable ({type(exc).__name__}: {exc})"
    return torch, None


def _mps_available(torch) -> bool:
    try:
        return bool(torch.backends.mps.is_available())
    except Exception:
        return False


class TorchBackend(Backend):
    name = "torch"

    def __init__(self, torch, device):
        self.torch = torch
        self._dev = device
        self.device = str(device)
        # MPS has no float64; everything else torch ships (CPU, CUDA, ROCm-as-cuda) does.
        self._cdt = torch.float32 if device.type == "mps" else torch.float64
        self.compute_dtype = "float32" if device.type == "mps" else "float64"

    def chunk_elems(self, limits: Limits) -> int:
        # ~2 inputs + diff + |diff| in compute dtype, plus finite masks and the raw transfer buffers.
        itemsize = 4 if self.compute_dtype == "float32" else 8
        per_elem = 4 * itemsize + 3 + 2 * 8
        return max(1024, int(limits.device_budget_mib) * 1024 * 1024 // per_elem)

    def _to_device(self, raw: np.ndarray, dtype: str):
        torch = self.torch
        if dtype == "BF16":  # native bfloat16: ship 2 bytes/elem, upcast on the device
            t = torch.from_numpy(raw.view(np.int16)).view(torch.bfloat16)
        elif dtype in ("U16", "U32"):
            t = torch.from_numpy(raw.astype(np.int64))
        elif dtype == "U64":
            t = torch.from_numpy(raw.astype(np.float64))
        else:
            t = torch.from_numpy(raw)
        return t.to(self._dev, non_blocking=False).to(self._cdt)

    def chunk_stats(self, raw_a, dtype_a, raw_b, dtype_b, tol):
        torch = self.torch
        st = ChunkStats()
        try:
            with torch.no_grad():
                ca, cb = self._to_device(raw_a, dtype_a), self._to_device(raw_b, dtype_b)
                fa, fb = torch.isfinite(ca), torch.isfinite(cb)
                m = fa & fb
                ca, cb = ca[m], cb[m]
                d = cb - ca
                ad = d.abs()
                n = int(ca.numel())
                if n:
                    vals = torch.stack([(d * d).sum(), (ca * ca).sum(), (cb * cb).sum(), (ca * cb).sum(),
                                        ad.max(), (ad > tol).sum().to(self._cdt)])
                    sdd, saa, sbb, sab, maxd, changed = vals.double().cpu().tolist()  # one host sync
                    st.sdd, st.saa, st.sbb, st.sab, st.maxd = sdd, saa, sbb, sab, maxd
                    st.changed = int(changed)
                st.compared = n
                st.nf_a = int(fa.numel() - int(fa.sum().item()))
                st.nf_b = int(fb.numel() - int(fb.sum().item()))
        except RuntimeError as exc:  # e.g. CUDA out of memory
            raise WeightScopeError(f"torch backend failed on {self.device}: {exc}. "
                                   f"Try a smaller device_budget_mib or --backend numpy") from None
        return st


def _torch_device(torch, device: str | None):
    """Resolve and verify a torch device; raise WeightScopeError if it is not usable."""
    if not device:
        if torch.cuda.is_available():
            device = "cuda"
        elif _mps_available(torch):
            device = "mps"
        else:
            device = "cpu"
    try:
        dev = torch.device(device)
    except Exception:
        raise WeightScopeError(f"invalid torch device {device!r}") from None
    if dev.type == "cuda":
        if not torch.cuda.is_available():
            raise WeightScopeError(f"device {device!r} requested but CUDA is not available to torch")
        if dev.index is not None and dev.index >= torch.cuda.device_count():
            raise WeightScopeError(f"device {device!r} requested but only {torch.cuda.device_count()} CUDA device(s) exist")
    elif dev.type == "mps" and not _mps_available(torch):
        raise WeightScopeError(f"device {device!r} requested but MPS is not available")
    try:  # final probe for anything else (xpu, privateuse backends, ...)
        torch.zeros(1, device=dev)
    except Exception as exc:
        raise WeightScopeError(f"device {device!r} is not usable by torch ({type(exc).__name__})") from None
    return dev


def resolve_backend(backend: str | None = "auto", device: str | None = None) -> Backend:
    backend = (backend or "auto").strip().lower()
    device = (device or "").strip() or None
    if backend not in BACKEND_CHOICES:
        raise WeightScopeError(f"backend must be one of {', '.join(BACKEND_CHOICES)}")
    if backend == "numpy":
        if device not in (None, "cpu"):
            raise WeightScopeError(f"numpy backend runs on cpu only (got device {device!r})")
        return NumpyBackend()
    if backend == "torch":
        torch, why = _import_torch()
        if torch is None:
            raise WeightScopeError(f"backend 'torch' requested but {why}. Install PyTorch into the Hermes "
                                   f"environment or use --backend numpy")
        return TorchBackend(torch, _torch_device(torch, device))
    # auto
    if device == "cpu":
        return NumpyBackend()
    torch, why = _import_torch()
    if torch is None:
        if device:
            raise WeightScopeError(f"device {device!r} requested but {why}")
        nb = NumpyBackend()
        nb.fallback_reason = why
        return nb
    if device:  # explicit accelerator under auto: must be real
        return TorchBackend(torch, _torch_device(torch, device))
    if torch.cuda.is_available():
        return TorchBackend(torch, _torch_device(torch, "cuda"))
    if _mps_available(torch):
        return TorchBackend(torch, _torch_device(torch, "mps"))
    nb = NumpyBackend()
    nb.fallback_reason = "torch is installed but no GPU (CUDA/MPS) is available"
    return nb

