#!/usr/bin/env python3
"""WeightScope reference remote worker (protocol weightscope-remote/1). NOT deployed anywhere by this repo.

Shipped inside the plugin for convenience only: the plugin never imports, registers or starts it. Run it yourself
on the machine that holds the checkpoints (from the plugin directory: python3 worker/remote_worker.py ...).

Runs the same numpy/torch backends as the plugin behind three endpoints:
  GET  /v1/health    POST /v1/diff    POST /v1/summary
Checkpoints are REFERENCES to files already on the worker, confined to --root directories. Nothing is uploaded.

Required: at least one --root, and a bearer token in the env var named by --token-env.
Binding to a non-loopback address requires TLS (--tls-cert/--tls-key) or --behind-tls-proxy (for platforms
that terminate HTTPS in front of the pod, e.g. a RunPod HTTPS proxy port).

Example (on the GPU machine):
  export WEIGHTSCOPE_WORKER_TOKEN=$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')
  python3 worker/remote_worker.py --root /workspace/models --host 0.0.0.0 --port 8765 --behind-tls-proxy
"""

from __future__ import annotations

import argparse
import hmac
import json
import os
import ssl
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

_PKG_DIR = Path(__file__).resolve().parents[1]  # the plugin directory (the weightscope package)
if "weightscope" not in sys.modules:
    import importlib.util as _ilu
    _spec = _ilu.spec_from_file_location("weightscope", _PKG_DIR / "__init__.py",
                                         submodule_search_locations=[str(_PKG_DIR)])
    _mod = _ilu.module_from_spec(_spec)
    sys.modules["weightscope"] = _mod
    _spec.loader.exec_module(_mod)  # defines register(); the worker never calls it

from weightscope.analysis import diff, summarize  # noqa: E402
from weightscope.backends import Backend, resolve_backend  # noqa: E402
from weightscope.remote_schema import PROTOCOL  # noqa: E402
from weightscope.st_reader import GiB, Limits, WeightScopeError  # noqa: E402

MAX_REQUEST_BYTES = 64 * 1024
WORKER_ID = "weightscope-reference-worker/0.1"
_STRIP_DIFF = ("ok", "tool", "execution", "interpretation", "limitations")
_STRIP_SUMMARY = ("ok", "tool")


def resolve_ref(ref, roots: list[Path]) -> str:
    """Map a request reference to a real path inside an approved root, or raise."""
    if not isinstance(ref, dict) or len(ref) != 1:
        raise WeightScopeError("checkpoint reference must be {\"path\": ...} or {\"ref\": ...}")
    if "ref" in ref:
        raise WeightScopeError("object references are not supported by the reference worker; use a path")
    value = ref.get("path")
    if not isinstance(value, str) or not value or "\x00" in value or len(value) > 4096:
        raise WeightScopeError("invalid checkpoint path")
    p = Path(value)
    if not p.is_absolute():
        raise WeightScopeError("checkpoint path must be absolute on the worker")
    real = p.resolve()
    for root in roots:
        if real == root or root in real.parents:
            return str(real)
    raise WeightScopeError("checkpoint path is outside the worker's approved roots")


def device_info(be: Backend, limits: Limits) -> dict:
    info = be.info(limits)
    out = {k: info[k] for k in ("backend", "device", "compute_dtype", "accumulate_dtype", "chunk_elems")}
    out["worker"] = WORKER_ID
    try:
        import numpy
        out["numpy_version"] = numpy.__version__
    except Exception:
        pass
    torch = getattr(be, "torch", None)
    if torch is not None:
        out["torch_version"] = str(torch.__version__)
        out["cuda_version"] = torch.version.cuda
        if be.device.startswith("cuda"):
            try:
                out["device_name"] = torch.cuda.get_device_name(be._dev)
            except Exception:
                pass
    return out


def build_server(host: str, port: int, roots: list[str], token: str, backend: str = "auto",
                 device: str | None = None, limits: Limits | None = None,
                 ssl_context: ssl.SSLContext | None = None) -> ThreadingHTTPServer:
    if not roots:
        raise SystemExit("at least one --root is required")
    if not token or len(token) < 16:
        raise SystemExit("a worker token of at least 16 characters is required")
    root_paths = [Path(r).resolve() for r in roots]
    for r in root_paths:
        if not r.is_dir():
            raise SystemExit(f"root is not a directory: {r}")
    limits = limits or Limits()
    resolve_backend(backend, device)  # fail fast on an unusable device
    expected = f"Bearer {token}".encode()

    class Handler(BaseHTTPRequestHandler):
        server_version = "weightscope-worker"
        sys_version = ""

        def log_message(self, fmt, *args):  # request line + status only; headers (token) are never logged
            sys.stderr.write(f"[worker] {self.address_string()} {fmt % args}\n")

        def _send(self, code: int, obj: dict):
            body = json.dumps(obj, separators=(",", ":")).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _error(self, code: int, msg: str):
            self._send(code, {"ok": False, "protocol": PROTOCOL, "error": msg[:1000]})

        def _authorized(self) -> bool:
            got = (self.headers.get("Authorization") or "").encode()
            if hmac.compare_digest(got, expected):
                return True
            self._error(401, "unauthorized")
            return False

        def _backend(self, params: dict) -> Backend:
            dev = params.get("device") or device
            return resolve_backend(backend, dev if isinstance(dev, str) else None)

        def do_GET(self):
            if not self._authorized():
                return
            if self.path != "/v1/health":
                return self._error(404, "not found")
            try:
                be = resolve_backend(backend, device)
                self._send(200, {"ok": True, "protocol": PROTOCOL, "device": device_info(be, limits)})
            except WeightScopeError as exc:
                self._error(503, str(exc))

        def do_POST(self):
            if not self._authorized():
                return
            if self.path not in ("/v1/diff", "/v1/summary"):
                return self._error(404, "not found")
            length = self.headers.get("Content-Length")
            if not length or not length.isdigit() or int(length) > MAX_REQUEST_BYTES:
                return self._error(413, f"request body required, at most {MAX_REQUEST_BYTES} bytes")
            try:
                req = json.loads(self.rfile.read(int(length)).decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                return self._error(400, "malformed JSON")
            if not isinstance(req, dict) or req.get("protocol") != PROTOCOL:
                return self._error(400, f"protocol must be {PROTOCOL}")
            params = req.get("params") or {}
            if not isinstance(params, dict):
                return self._error(400, "params must be an object")
            try:
                be = self._backend(params)
                if self.path == "/v1/diff":
                    a = resolve_ref(req.get("a"), root_paths)
                    b = resolve_ref(req.get("b"), root_paths)
                    res = diff(a, b, limits=limits, backend=be, top_k=params.get("top_k", 10),
                               top_layers=params.get("top_layers", 10), tolerance=params.get("tolerance", 0.0),
                               rank_by=params.get("rank_by", "relative"))
                    strip = _STRIP_DIFF
                else:
                    path = resolve_ref(req.get("checkpoint"), root_paths)
                    res = summarize(path, limits=limits, max_list=params.get("max_list", 50),
                                    hash_files=params.get("hash_files", True) is not False)
                    strip = _STRIP_SUMMARY
            except WeightScopeError as exc:
                return self._error(400, str(exc))
            except Exception as exc:
                sys.stderr.write(f"[worker] internal error: {type(exc).__name__}\n")
                return self._error(500, "internal worker error")
            result = {k: v for k, v in res.items() if k not in strip}
            self._send(200, {"ok": True, "protocol": PROTOCOL, "result": result, "device": device_info(be, limits)})

    srv = ThreadingHTTPServer((host, port), Handler)
    if ssl_context is not None:
        srv.socket = ssl_context.wrap_socket(srv.socket, server_side=True)
    return srv


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", action="append", required=True, help="Approved checkpoint root (repeatable)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--token-env", default="WEIGHTSCOPE_WORKER_TOKEN", help="NAME of the env var holding the token")
    ap.add_argument("--backend", choices=["auto", "numpy", "torch"], default="auto")
    ap.add_argument("--device", default=None)
    ap.add_argument("--max-total-gib", type=float, default=256)
    ap.add_argument("--device-budget-mib", type=int, default=1024)
    ap.add_argument("--tls-cert")
    ap.add_argument("--tls-key")
    ap.add_argument("--behind-tls-proxy", action="store_true",
                    help="Allow plain HTTP on a non-loopback bind because an HTTPS proxy terminates TLS in front")
    a = ap.parse_args(argv)
    token = os.environ.get(a.token_env, "")
    if not token:
        ap.error(f"set the worker token in ${a.token_env}")
    loopback = a.host in ("127.0.0.1", "localhost", "::1")
    ctx = None
    if a.tls_cert and a.tls_key:
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(a.tls_cert, a.tls_key)
    elif not loopback and not a.behind_tls_proxy:
        ap.error("non-loopback bind needs --tls-cert/--tls-key or --behind-tls-proxy")
    limits = Limits(max_total_bytes=int(a.max_total_gib * GiB), device_budget_mib=a.device_budget_mib)
    srv = build_server(a.host, a.port, a.root, token, a.backend, a.device, limits, ctx)
    sys.stderr.write(f"[worker] serving {PROTOCOL} on {a.host}:{a.port} ({'https' if ctx else 'http'}); "
                     f"roots: {', '.join(a.root)}\n")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
