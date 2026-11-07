"""Offline remote-routing CONTRACT tests.

Every "worker" here is an in-process HTTP server on 127.0.0.1 (the reference worker or a scripted fake).
They prove routing, protocol, validation, auth and limit behaviour only. They do NOT show real GPU or
RunPod execution: the reference worker runs the numpy backend on this machine's CPU.
"""

import json
import logging
import os
import secrets
from pathlib import Path

import pytest

import stfixtures as F
from remote_fixture import Scripted, Running, free_port, load_worker_module
from weightscope import routing
from weightscope.backends import NumpyBackend
from weightscope.remote import RemoteClient, RemoteSettings, read_token, validate_url
from weightscope.remote_schema import MAX_ITEMS, MAX_TOP, PROTOCOL
from weightscope.st_reader import Limits, WeightScopeError
from weightscope.tools import make_handlers

TOKEN_ENV = "WS_TEST_REMOTE_TOKEN"


@pytest.fixture
def token(monkeypatch):
    tok = "tok-" + secrets.token_urlsafe(24)
    monkeypatch.setenv(TOKEN_ENV, tok)
    return tok


@pytest.fixture
def ckdir(tmp_path):
    root = tmp_path / "worker-root"
    base = F.base_model()
    F.write(root / "base.safetensors", base)
    F.write_sharded(root / "ft", F.finetuned(base, layer=2), n_shards=2)
    F.write(tmp_path / "outside" / "secret.safetensors", base)
    return root


@pytest.fixture
def worker(ckdir, token):
    mod = load_worker_module()
    srv = mod.build_server("127.0.0.1", 0, [str(ckdir)], token, backend="numpy")
    run = Running(srv)
    yield run
    run.close()


@pytest.fixture
def scripted():
    s = Scripted()
    yield s
    s.close()


def settings(url, **kw):
    kw.setdefault("allow_http_localhost", True)
    kw.setdefault("timeout_s", 5)
    return RemoteSettings(url=url, token_env=TOKEN_ENV, **kw)


def no_local_gpu(monkeypatch):
    monkeypatch.setattr(routing, "_local_gpu_backend", lambda: (None, "no local GPU (test)"))


def _diff(remote, ckdir, backend="remote", a="base.safetensors", b="ft", **kw):
    a_raw, b_raw = str(ckdir / a), str(ckdir / b)
    return routing.run_diff(a_raw, b_raw, a_raw, b_raw, limits=Limits(), backend=backend, remote=remote, **kw)


# ── valid diff response used by schema tests ──────────────────────────────────

def _valid_envelope(worker, ckdir):
    remote = settings(worker.url)
    client = RemoteClient(remote)
    return client.diff(str(ckdir / "base.safetensors"), str(ckdir / "ft"),
                       {"top_k": 5, "top_layers": 5, "tolerance": 0.0, "rank_by": "relative"})


# ── round trip through the reference worker ──────────────────────────────────

def test_remote_diff_roundtrip_reference_worker(worker, ckdir, monkeypatch):
    no_local_gpu(monkeypatch)
    res = _diff(settings(worker.url), ckdir)
    assert res["ok"] and res["routing"]["mode"] == "remote"
    assert res["routing"]["endpoint"] == worker.url
    assert res["top_layers"][0]["layer"] == "model.layers.2"
    ex = res["execution"]
    assert ex["source"].startswith("remote worker (self-reported; not verified")
    assert ex["reported"]["backend"] == "numpy" and ex["reported"]["worker"].startswith("weightscope-reference-worker")
    assert "does NOT establish" in res["interpretation"]
    # same numbers as running locally
    local = _diff(None, ckdir, backend="numpy")
    assert local["top_tensors"] == res["top_tensors"] and local["totals"] == res["totals"]


def test_remote_summary_and_health(worker, ckdir):
    remote = settings(worker.url)
    s = routing.run_summary(str(ckdir / "ft"), str(ckdir / "ft"), limits=Limits(), backend="remote", remote=remote)
    assert s["routing"]["mode"] == "remote" and s["checkpoint"]["files"] == 2
    assert "reported" in s["execution"]
    h = routing.remote_health(remote)
    assert h["ok"] and h["protocol"] == PROTOCOL and h["execution"]["reported"]["backend"] == "numpy"


# ── auto routing order ───────────────────────────────────────────────────────

def test_auto_prefers_local_gpu_over_remote(monkeypatch, ckdir, scripted):
    """Mocked local GPU: auto stays local and never contacts the configured remote."""
    monkeypatch.setattr(routing, "_local_gpu_backend", lambda: (NumpyBackend(), "local CUDA GPU detected (mocked)"))
    res = _diff(settings(scripted.url), ckdir, backend="auto")
    assert res["routing"] == {"mode": "local", "requested": "auto", "reason": "local CUDA GPU detected (mocked)"}
    assert res["execution"]["source"] == "local (this process)"
    assert scripted.requests == []


def test_auto_uses_remote_when_no_local_gpu(monkeypatch, worker, ckdir):
    no_local_gpu(monkeypatch)
    res = _diff(settings(worker.url), ckdir, backend="auto")
    assert res["routing"]["mode"] == "remote" and "remote endpoint configured" in res["routing"]["reason"]


def test_auto_cpu_fallback_only_without_remote(monkeypatch, ckdir):
    no_local_gpu(monkeypatch)
    res = _diff(None, ckdir, backend="auto")
    assert res["routing"]["mode"] == "cpu-fallback"
    assert "no remote endpoint configured" in res["routing"]["reason"]
    assert res["execution"]["backend"] == "numpy" and res["execution"]["source"] == "local (this process)"


def test_auto_with_failing_remote_errors_instead_of_cpu_fallback(monkeypatch, ckdir, token):
    no_local_gpu(monkeypatch)
    with pytest.raises(WeightScopeError, match="connection refused"):
        _diff(settings(f"http://127.0.0.1:{free_port()}"), ckdir, backend="auto")


# ── explicit backends ────────────────────────────────────────────────────────

def test_explicit_numpy_ignores_configured_remote(ckdir, scripted):
    res = _diff(settings(scripted.url), ckdir, backend="numpy")
    assert res["routing"]["mode"] == "local" and scripted.requests == []


def test_explicit_remote_without_url_errors(ckdir):
    with pytest.raises(WeightScopeError, match="no remote_url is configured"):
        _diff(None, ckdir, backend="remote")


def test_explicit_torch_cpu_stays_local(ckdir, scripted):
    pytest.importorskip("torch")
    res = _diff(settings(scripted.url), ckdir, backend="torch", device="cpu")
    assert res["routing"]["mode"] == "local" and res["execution"]["backend"] == "torch"
    assert scripted.requests == []


# ── config / transport failures ──────────────────────────────────────────────

def test_missing_and_bad_token_env(monkeypatch, scripted):
    monkeypatch.delenv(TOKEN_ENV, raising=False)
    with pytest.raises(WeightScopeError, match=f"remote token variable {TOKEN_ENV} is not set"):
        RemoteClient(settings(scripted.url))
    with pytest.raises(WeightScopeError, match="UPPER_SNAKE"):
        read_token("lower-case; rm -rf")
    assert scripted.requests == []


def test_timeout(scripted, token):
    scripted.respond(200, {"ok": True}, delay=3)
    with pytest.raises(WeightScopeError, match="timed out after 1s"):
        RemoteClient(settings(scripted.url, timeout_s=1)).diff("/x", "/y", {})


def test_connection_refused(token):
    with pytest.raises(WeightScopeError, match="connection refused"):
        RemoteClient(settings(f"http://127.0.0.1:{free_port()}")).diff("/x", "/y", {})


@pytest.mark.parametrize("status, match", [(401, "rejected the token"), (403, "rejected the token"),
                                           (500, "returned HTTP 500")])
def test_http_errors_redact_token(scripted, token, status, match):
    scripted.respond(status, lambda h: f"echo: {h.headers.get('Authorization')}".encode())
    with pytest.raises(WeightScopeError, match=match) as ei:
        RemoteClient(settings(scripted.url)).diff("/x", "/y", {})
    assert token not in str(ei.value)
    assert len(scripted.requests) == 1  # POST is never retried


def test_redirect_refused(scripted, token):
    scripted.respond(302, b"", headers={"Location": "https://evil.example/steal"})
    with pytest.raises(WeightScopeError, match="redirect"):
        RemoteClient(settings(scripted.url)).diff("/x", "/y", {})
    assert len(scripted.requests) == 1


def test_malformed_json(scripted, token):
    scripted.respond(200, b"{not json")
    with pytest.raises(WeightScopeError, match="malformed JSON"):
        RemoteClient(settings(scripted.url)).diff("/x", "/y", {})


@pytest.mark.parametrize("declare_length", [True, False])
def test_oversized_response(scripted, token, declare_length):
    body = json.dumps({"ok": True, "pad": "x" * 20000}).encode()
    headers = None if declare_length else {"_no_length": "1"}
    scripted.respond(200, body, headers=headers)
    with pytest.raises(WeightScopeError, match="too large|exceeds"):
        RemoteClient(settings(scripted.url, max_response_bytes=4096)).diff("/x", "/y", {})


def _mutations():
    def unknown_key(e): e["result"]["surprise"] = 1
    def bad_cosine(e): e["result"]["top_tensors"][0]["cosine"] = 2.0
    def neg_l2(e): e["result"]["top_tensors"][0]["l2_diff"] = -1
    def too_many(e): e["result"]["top_tensors"] = e["result"]["top_tensors"][:1] * (MAX_TOP + 1)
    def too_many_items(e): e["result"]["only_in_a"] = {"count": MAX_ITEMS + 1, "items": ["x"] * (MAX_ITEMS + 1),
                                                       "truncated": False}
    def wrong_protocol(e): e["protocol"] = "other/9"
    def bad_totals(e): e["result"]["totals"]["identical_tensors"] += 5
    def string_number(e): e["result"]["totals"]["global_l2_diff"] = "1.0"
    def device_extra(e): e["device"]["gpu_verified"] = "yes"
    def missing_device(e): del e["device"]
    def bool_as_int(e): e["result"]["totals"]["changed_tensors"] = True
    def huge_name(e): e["result"]["top_tensors"][0]["name"] = "n" * 5000
    return [unknown_key, bad_cosine, neg_l2, too_many, too_many_items, wrong_protocol, bad_totals,
            string_number, device_extra, missing_device, bool_as_int, huge_name]


@pytest.mark.parametrize("mutate", _mutations(), ids=lambda f: f.__name__)
def test_schema_invalid_responses_rejected(worker, ckdir, scripted, token, mutate):
    env = _valid_envelope(worker, ckdir)
    mutate(env)
    scripted.respond(200, env)
    with pytest.raises(WeightScopeError, match="failed validation"):
        RemoteClient(settings(scripted.url)).diff("/x", "/y", {})


def test_nan_in_response_rejected(worker, ckdir, scripted, token):
    env = _valid_envelope(worker, ckdir)
    raw = json.dumps(env).replace('"global_l2_diff": ', '"global_l2_diff": NaN, "_x": ', 1)
    scripted.respond(200, raw)
    with pytest.raises(WeightScopeError, match="failed validation"):
        RemoteClient(settings(scripted.url)).diff("/x", "/y", {})


@pytest.mark.parametrize("url, match", [
    ("http://example.com", "must use https"),
    ("ftp://127.0.0.1/x", "must use https"),
    ("https://user:pw@example.com", "must not contain credentials"),
    ("https://example.com/?token=abc", "query string"),
    ("https://", "no host"),
])
def test_url_policy(url, match):
    with pytest.raises(WeightScopeError, match=match):
        validate_url(url, allow_http_localhost=True)


def test_http_localhost_needs_explicit_flag():
    with pytest.raises(WeightScopeError, match="remote_allow_http_localhost"):
        validate_url("http://127.0.0.1:9", allow_http_localhost=False)
    assert validate_url("https://pod-8765.proxy.example/base/", False)[0] == "https://pod-8765.proxy.example/base"


def test_health_retries_bounded_and_not_on_auth(scripted, token, worker):
    good = RemoteClient(settings(worker.url)).health()
    scripted.respond(503, b"busy").respond(503, b"busy").respond(200, good)
    assert RemoteClient(settings(scripted.url)).health()["ok"] is True
    assert len(scripted.requests) == 3 and all(r[0] == "GET" for r in scripted.requests)


def test_health_gives_up_after_three(scripted, token):
    scripted.respond(500, b"down")
    with pytest.raises(WeightScopeError, match="HTTP 500"):
        RemoteClient(settings(scripted.url)).health()
    assert len(scripted.requests) == 3


def test_health_no_retry_on_401(scripted, token):
    scripted.respond(401, b"no")
    with pytest.raises(WeightScopeError, match="rejected the token"):
        RemoteClient(settings(scripted.url)).health()
    assert len(scripted.requests) == 1


def test_request_shape_and_no_upload(scripted, token, worker, ckdir):
    """Requests carry references only: the body is tiny JSON, never tensor bytes."""
    scripted.respond(200, _valid_envelope(worker, ckdir))
    RemoteClient(settings(scripted.url)).diff(str(ckdir / "base.safetensors"), "ref:bucket/model-v2", {"top_k": 3})
    method, path, headers, body = scripted.requests[0]
    req = json.loads(body)
    assert method == "POST" and path == "/v1/diff" and len(body) < 2048
    assert req == {"protocol": PROTOCOL, "a": {"path": str(ckdir / "base.safetensors")},
                   "b": {"ref": "bucket/model-v2"}, "params": {"top_k": 3}}
    assert headers["Authorization"] == f"Bearer {token}"


# ── token never leaks ────────────────────────────────────────────────────────

def test_token_never_in_tool_output_or_logs(scripted, token, ckdir, caplog, capsys, monkeypatch):
    no_local_gpu(monkeypatch)
    caplog.set_level(logging.DEBUG)
    scripted.respond(500, lambda h: f"oops {h.headers.get('Authorization')}".encode())
    cfg = {"remote_url": scripted.url, "remote_token_env": TOKEN_ENV, "remote_allow_http_localhost": True,
           "record_history": False}
    _, dif = make_handlers(get_setting=lambda k, d=None: cfg.get(k, d))
    out = dif({"a": str(ckdir / "base.safetensors"), "b": str(ckdir / "ft")})
    res = json.loads(out)
    assert res["ok"] is False and "HTTP 500" in res["error"] and "[REDACTED]" in res["error"]
    captured = capsys.readouterr()
    for text in (out, caplog.text, captured.out, captured.err):
        assert token not in text


# ── tool args ────────────────────────────────────────────────────────────────

def test_tool_backend_remote_arg(worker, ckdir, monkeypatch):
    cfg = {"remote_url": worker.url, "remote_token_env": TOKEN_ENV, "remote_allow_http_localhost": True,
           "record_history": False}
    summ, dif = make_handlers(get_setting=lambda k, d=None: cfg.get(k, d))
    res = json.loads(dif({"a": str(ckdir / "base.safetensors"), "b": str(ckdir / "ft"), "backend": "remote"}))
    assert res["ok"] and res["routing"]["mode"] == "remote"
    s = json.loads(summ({"path": str(ckdir / "base.safetensors"), "backend": "remote"}))
    assert s["ok"] and s["routing"]["mode"] == "remote"
    # without a configured URL the tool cannot pick one (no remote_url tool arg by design)
    _, dif2 = make_handlers(get_setting=lambda k, d=None: {"record_history": False}.get(k, d))
    err = json.loads(dif2({"a": "x", "b": "y", "backend": "remote", "remote_url": "https://attacker.example"}))
    assert err["ok"] is False and "no remote_url is configured" in err["error"]


# ── reference worker hardening ───────────────────────────────────────────────

@pytest.mark.parametrize("ref, msg", [
    ("OUTSIDE", "outside the worker's approved roots"),
    ("relative/base.safetensors", "must be absolute"),
    ("TRAVERSAL", "outside the worker's approved roots"),
    ("SYMLINK", "outside the worker's approved roots"),
    ("ref:bucket/x", "object references are not supported"),
])
def test_worker_path_allowlist(worker, ckdir, ref, msg):
    outside = ckdir.parent / "outside" / "secret.safetensors"
    if ref == "OUTSIDE":
        ref = str(outside)
    elif ref == "TRAVERSAL":
        ref = str(ckdir / ".." / "outside" / "secret.safetensors")
    elif ref == "SYMLINK":
        link = ckdir / "link.safetensors"
        os.symlink(outside, link)
        ref = str(link)
    with pytest.raises(WeightScopeError, match=msg):
        RemoteClient(settings(worker.url)).diff(str(ckdir / "base.safetensors"), ref, {})


def test_worker_rejects_wrong_token(worker, monkeypatch):
    monkeypatch.setenv(TOKEN_ENV, "wrong-token-value-123456")
    with pytest.raises(WeightScopeError, match="rejected the token"):
        RemoteClient(settings(worker.url)).health()


def test_worker_requires_roots_and_token(ckdir):
    mod = load_worker_module()
    with pytest.raises(SystemExit):
        mod.build_server("127.0.0.1", 0, [], "x" * 32)
    with pytest.raises(SystemExit):
        mod.build_server("127.0.0.1", 0, [str(ckdir)], "short")
    with pytest.raises(SystemExit):
        mod.main(["--root", str(ckdir), "--host", "0.0.0.0", "--token-env", "WS_UNSET_TOKEN_VAR_X"])


def test_worker_refuses_public_plain_http(ckdir, monkeypatch):
    mod = load_worker_module()
    monkeypatch.setenv("WS_WORKER_TOK", "y" * 32)
    with pytest.raises(SystemExit):
        mod.main(["--root", str(ckdir), "--host", "0.0.0.0", "--token-env", "WS_WORKER_TOK"])


def test_schema_file_matches_validator_caps():
    schema = json.loads((Path(__file__).resolve().parent.parent / "weightscope" / "docs" / "remote-worker.schema.json").read_text())
    defs = schema["$defs"]
    assert defs["diffResult"]["properties"]["top_tensors"]["maxItems"] == MAX_TOP
    assert defs["diffResult"]["properties"]["top_layers"]["maxItems"] == MAX_TOP
    assert defs["cappedBlock"]["properties"]["items"]["maxItems"] == MAX_ITEMS
    assert defs["envelopeBase"]["properties"]["protocol"]["const"] == PROTOCOL


def test_schema_file_covers_reference_worker_output(worker, ckdir):
    """Without a jsonschema dependency: every key the reference worker emits is declared in the schema doc."""
    schema = json.loads((Path(__file__).resolve().parent.parent / "weightscope" / "docs" / "remote-worker.schema.json").read_text())
    defs = schema["$defs"]
    env = _valid_envelope(worker, ckdir)
    assert set(env) == set(defs["diffResponse"]["properties"])
    assert set(env["result"]) == set(defs["diffResult"]["properties"])
    assert set(env["result"]["totals"]) == set(defs["diffResult"]["properties"]["totals"]["properties"])
    assert set(env["device"]) <= set(defs["device"]["properties"])
    for rec in env["result"]["top_tensors"]:
        assert set(rec) <= set(defs["tensorRecord"]["properties"])
    s = RemoteClient(settings(worker.url)).summary(str(ckdir / "ft"), {"max_list": 5})
    assert set(s["result"]) == set(defs["summaryResult"]["properties"])


def test_plugin_never_imports_or_registers_worker():
    """The worker ships inside the package but is not wired into register() or any plugin module."""
    import subprocess
    import sys
    pkg = Path(__file__).resolve().parent.parent / "weightscope"
    for py in pkg.glob("*.py"):
        assert "remote_worker" not in py.read_text(), py
    code = ("import sys, weightscope\n"
            "class C:\n"
            "  def __getattr__(self, n): return lambda *a, **k: None\n"
            "weightscope.register(C())\n"
            "assert not [m for m in sys.modules if 'remote_worker' in m]\n"
            "assert 'http.server' not in sys.modules, 'plugin import must not pull in a server'\n")
    r = subprocess.run([sys.executable, "-c", code], cwd=pkg.parent, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_worker_script_runs_from_plugin_dir(tmp_path):
    """Run as a script from a copied plugin dir with a non-standard name (as an installed copy might be)."""
    import shutil
    import subprocess
    import sys
    dst = tmp_path / "plugins" / "weightscope-copy"
    shutil.copytree(Path(__file__).resolve().parent.parent / "weightscope", dst,
                    ignore=shutil.ignore_patterns("__pycache__"))
    r = subprocess.run([sys.executable, "worker/remote_worker.py", "--help"], cwd=dst, capture_output=True, text=True,
                       env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": ""})
    assert r.returncode == 0, r.stderr
    assert "--root" in r.stdout and "--behind-tls-proxy" in r.stdout
