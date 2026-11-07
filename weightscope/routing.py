"""Route a diff/summary to local GPU, a configured remote worker, or CPU numpy.

auto order: local torch GPU (CUDA/MPS) -> configured remote endpoint -> CPU numpy (only when no remote is
configured). A configured remote that fails is an error; there is no silent CPU fallback.

Every result carries:
* ``routing``   - what this process decided and why (mode: local | remote | cpu-fallback).
* ``execution`` - where it ran. Local: detected by this process. Remote: device info AS REPORTED BY THE
  WORKER, which this process cannot verify.
"""

from __future__ import annotations

from .analysis import INTERPRETATION_NOTE, LIMITATIONS, diff, summarize
from .backends import BACKEND_CHOICES, Backend, NumpyBackend, TorchBackend, _import_torch, _mps_available, \
    _torch_device, resolve_backend
from .remote import RemoteClient, RemoteSettings
from .st_reader import Limits, WeightScopeError

ROUTE_CHOICES = (*BACKEND_CHOICES, "remote")
REMOTE_SOURCE = "remote worker (self-reported; not verified by this machine)"


def _local_gpu_backend() -> tuple[Backend | None, str]:
    torch, why = _import_torch()
    if torch is None:
        return None, why
    if torch.cuda.is_available():
        return TorchBackend(torch, _torch_device(torch, "cuda")), "local CUDA GPU detected"
    if _mps_available(torch):
        return TorchBackend(torch, _torch_device(torch, "mps")), "local MPS GPU detected"
    return None, "torch is installed but no local GPU (CUDA/MPS) is available"


def plan(backend: str | None, device: str | None, remote: RemoteSettings | None):
    """Return ('local', Backend, routing) or ('remote', None, routing)."""
    backend = (backend or "auto").strip().lower()
    if backend not in ROUTE_CHOICES:
        raise WeightScopeError(f"backend must be one of {', '.join(ROUTE_CHOICES)}")
    if backend == "remote":
        if remote is None or not remote.url:
            raise WeightScopeError("backend 'remote' requested but no remote_url is configured")
        return "remote", None, {"mode": "remote", "requested": "remote", "reason": "backend 'remote' requested"}
    if backend in ("numpy", "torch"):
        be = resolve_backend(backend, device)
        return "local", be, {"mode": "local", "requested": backend, "reason": f"backend '{backend}' requested"}
    # auto
    if device:  # an explicit local device under auto keeps the old meaning: run it here or fail
        be = resolve_backend("auto", device)
        return "local", be, {"mode": "local", "requested": "auto", "reason": f"device {device!r} requested"}
    gpu, why = _local_gpu_backend()
    if gpu is not None:
        return "local", gpu, {"mode": "local", "requested": "auto", "reason": why}
    if remote is not None and remote.url:
        return "remote", None, {"mode": "remote", "requested": "auto",
                                "reason": f"no local GPU ({why}); remote endpoint configured"}
    nb = NumpyBackend()
    return "local", nb, {"mode": "cpu-fallback", "requested": "auto",
                         "reason": f"no local GPU ({why}) and no remote endpoint configured"}


def _local_execution(info: dict, be: Backend) -> dict:
    out = {"source": "local (this process)", **info}
    out.pop("fallback_reason", None)
    if isinstance(be, TorchBackend) and be.device.startswith("cuda"):
        try:
            out["device_name"] = be.torch.cuda.get_device_name(be._dev)
            out["torch_version"] = be.torch.__version__
            out["cuda_version"] = be.torch.version.cuda
        except Exception:
            pass
    return out


def run_diff(a_local, b_local, a_raw, b_raw, *, limits: Limits, backend=None, device=None,
             remote: RemoteSettings | None = None, top_k=10, top_layers=10, tolerance=0.0,
             rank_by="relative") -> dict:
    """a_local/b_local: paths resolved on this machine; a_raw/b_raw: the strings as given (sent to a worker)."""
    where, be, routing = plan(backend, device, remote)
    if where == "local":
        res = diff(a_local, b_local, limits=limits, top_k=top_k, top_layers=top_layers, tolerance=tolerance,
                   rank_by=rank_by, backend=be)
        res["execution"] = _local_execution(res.pop("execution"), be)
        res["routing"] = routing
        return res
    client = RemoteClient(remote)
    routing["endpoint"] = client.endpoint
    params = {"top_k": top_k, "top_layers": top_layers, "tolerance": tolerance, "rank_by": rank_by}
    if device and (backend or "").lower() == "remote":
        params["device"] = device
    env = client.diff(a_raw, b_raw, params)
    out = {"ok": True, "tool": "ckpt_diff", **env["result"]}
    out["routing"] = routing
    out["execution"] = {"source": REMOTE_SOURCE, "endpoint": client.endpoint, "reported": env["device"]}
    out["interpretation"] = INTERPRETATION_NOTE
    out["limitations"] = LIMITATIONS
    return out


def run_summary(path_local, path_raw, *, limits: Limits, backend=None, remote: RemoteSettings | None = None,
                max_list=50, hash_files=True) -> dict:
    b = (backend or "auto").strip().lower()
    if b not in (*ROUTE_CHOICES, "local"):
        raise WeightScopeError(f"backend must be one of {', '.join(ROUTE_CHOICES)}")
    go_remote = False
    if b == "remote":
        if remote is None or not remote.url:
            raise WeightScopeError("backend 'remote' requested but no remote_url is configured")
        go_remote, routing = True, {"mode": "remote", "requested": "remote", "reason": "backend 'remote' requested"}
    elif b == "auto" and remote is not None and remote.url:
        gpu, why = _local_gpu_backend()  # same order as diff so both tools see the same machine
        if gpu is None:
            go_remote, routing = True, {"mode": "remote", "requested": "auto",
                                        "reason": f"no local GPU ({why}); remote endpoint configured"}
        else:
            routing = {"mode": "local", "requested": "auto", "reason": why}
    else:
        routing = {"mode": "local", "requested": b, "reason": "summary reads files on this machine"}
    if not go_remote:
        res = summarize(path_local, limits=limits, max_list=max_list, hash_files=hash_files)
        res["routing"] = routing
        res["execution"] = {"source": "local (this process)", "backend": "numpy", "device": "cpu"}
        return res
    client = RemoteClient(remote)
    routing["endpoint"] = client.endpoint
    env = client.summary(path_raw, {"max_list": max_list, "hash_files": bool(hash_files)})
    out = {"ok": True, "tool": "ckpt_summary", **env["result"]}
    out["routing"] = routing
    out["execution"] = {"source": REMOTE_SOURCE, "endpoint": client.endpoint, "reported": env["device"]}
    return out


def remote_health(remote: RemoteSettings | None) -> dict:
    if remote is None or not remote.url:
        raise WeightScopeError("no remote_url is configured")
    client = RemoteClient(remote)
    env = client.health()
    return {"ok": True, "endpoint": client.endpoint, "protocol": env["protocol"],
            "execution": {"source": REMOTE_SOURCE, "reported": env["device"]}}
