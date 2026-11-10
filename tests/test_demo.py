"""Unit tests for the published-model demo: fetch consent, cache, SHA checks, no live network."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request

import pytest

import stfixtures as F
from weightscope.demo import (
    DEMO_BASE,
    DEMO_INSTRUCT,
    TOTAL_BYTES,
    _HostLimitedRedirect,
    _allowed_host,
    _resolve_url,
    default_cache_dir,
    download_spec,
    ensure_files,
    render_demo,
    run_demo,
    write_layer_heatmap,
)
from weightscope.st_reader import WeightScopeError


def _pair(tmp: Path):
    """Two tiny safetensors files, advertised like the real catalog."""
    base = F.base_model(seed=0)
    inst = F.finetuned(base, layer=2)
    base_bytes = F.encode(base)
    inst_bytes = F.encode(inst)
    catalog = (
        {
            "id": "base",
            "label": "toy-base",
            "repo": "Example/Toy-Base",
            "revision": "aaa111",
            "filename": "model.safetensors",
            "sha256": hashlib.sha256(base_bytes).hexdigest(),
            "size": len(base_bytes),
        },
        {
            "id": "instruct",
            "label": "toy-instruct",
            "repo": "Example/Toy-Instruct",
            "revision": "bbb222",
            "filename": "model.safetensors",
            "sha256": hashlib.sha256(inst_bytes).hexdigest(),
            "size": len(inst_bytes),
        },
    )
    blobs = {
        _resolve_url(catalog[0]): base_bytes,
        _resolve_url(catalog[1]): inst_bytes,
    }
    return catalog, blobs


class _FakeResp:
    def __init__(self, data: bytes, url: str):
        self._buf = io.BytesIO(data)
        self._url = url

    def geturl(self):
        return self._url

    def read(self, n: int = -1):
        return self._buf.read(n)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeOpener:
    def __init__(self, mapping: dict[str, bytes], redirects: dict[str, str] | None = None):
        self.mapping = mapping
        self.redirects = redirects or {}
        self.calls: list[str] = []

    def open(self, req, timeout=None):
        url = req.full_url if hasattr(req, "full_url") else str(req)
        self.calls.append(url)
        final = self.redirects.get(url, url)
        if final not in self.mapping:
            raise HTTPError(url, 404, "not in fake mapping", hdrs=None, fp=None)
        return _FakeResp(self.mapping[final], final)


class BoomOpener:
    def open(self, req, timeout=None):
        raise AssertionError(f"unexpected network call to {getattr(req, 'full_url', req)}")


def test_allowed_hosts():
    assert _allowed_host("huggingface.co")
    assert _allowed_host("cdn-lfs.huggingface.co")
    assert _allowed_host("cas-bridge.xethub.hf.co")
    assert _allowed_host("hf.co")
    assert not _allowed_host("evil.example")
    assert not _allowed_host("huggingface.co.evil.example")
    assert not _allowed_host("")


def test_redirect_handler_refuses_off_hf():
    handler = _HostLimitedRedirect()
    req = Request("https://huggingface.co/org/repo/resolve/abc/model.safetensors")
    with pytest.raises(WeightScopeError, match="refused redirect to host"):
        handler.redirect_request(req, fp=None, code=302, msg="Found", headers={},
                                 newurl="https://evil.example/model.safetensors")
    with pytest.raises(WeightScopeError, match="non-https"):
        handler.redirect_request(req, fp=None, code=302, msg="Found", headers={},
                                 newurl="http://huggingface.co/model.safetensors")
    # allowed official CDN host is not rejected here (super() may still need a well-formed request)
    out = handler.redirect_request(req, fp=None, code=302, msg="Found", headers={},
                                   newurl="https://cas-bridge.xethub.hf.co/blob/xyz")
    assert out is not None
    assert out.full_url.startswith("https://cas-bridge.xethub.hf.co/")


def test_refuses_without_fetch_and_makes_no_network(tmp_path):
    pair, blobs = _pair(tmp_path)
    cache = tmp_path / "cache"
    with pytest.raises(WeightScopeError, match="--fetch"):
        ensure_files(pair, cache_dir=cache, fetch=False, opener=BoomOpener())
    assert not any(cache.rglob("*.safetensors"))


def test_fetch_downloads_verifies_and_reuses_cache(tmp_path):
    pair, blobs = _pair(tmp_path)
    cache = tmp_path / "cache"
    opener = FakeOpener(blobs)
    files = ensure_files(pair, cache_dir=cache, fetch=True, opener=opener)
    assert len(opener.calls) == 2
    assert all(p.path.is_file() for p in files)
    assert files[0].downloaded and files[1].downloaded
    # second call is fully offline
    files2 = ensure_files(pair, cache_dir=cache, fetch=False, opener=BoomOpener())
    assert all(not p.downloaded for p in files2)
    assert files2[0].path == files[0].path


def test_sha_mismatch_on_download(tmp_path):
    pair, blobs = _pair(tmp_path)
    bad = dict(blobs)
    url = _resolve_url(pair[0])
    bad[url] = b"this is not the pinned file" + b"\x00" * 16
    # size won't match either unless we pad — force same length so SHA is the failing check
    bad[url] = (b"\x00" * pair[0]["size"])
    cache = tmp_path / "cache"
    with pytest.raises(WeightScopeError, match="SHA-256"):
        ensure_files(pair, cache_dir=cache, fetch=True, opener=FakeOpener(bad))
    assert not (cache / "Example--Toy-Base" / "aaa111" / "model.safetensors").exists()
    assert list(cache.rglob("*.part")) == []


def test_corrupt_cache_requires_fetch(tmp_path):
    pair, blobs = _pair(tmp_path)
    cache = tmp_path / "cache"
    ensure_files(pair, cache_dir=cache, fetch=True, opener=FakeOpener(blobs))
    victim = cache / "Example--Toy-Base" / "aaa111" / "model.safetensors"
    victim.write_bytes(victim.read_bytes()[:-8] + b"\x00" * 8)
    with pytest.raises(WeightScopeError, match="re-run with --fetch"):
        ensure_files(pair, cache_dir=cache, fetch=False, opener=BoomOpener())
    # --fetch replaces it
    files = ensure_files(pair, cache_dir=cache, fetch=True, opener=FakeOpener(blobs))
    assert files[0].downloaded


def test_run_demo_diff_and_identical_control(tmp_path):
    pair, blobs = _pair(tmp_path)
    cache = tmp_path / "cache"
    heatmap = tmp_path / "out.svg"
    result = run_demo(fetch=True, cache_dir=cache, pair=pair, opener=FakeOpener(blobs),
                      heatmap=True, heatmap_path=heatmap, top=5, progress=None)
    assert result["ok"]
    assert result["summary"]["ok"]
    diff = result["diff"]
    assert diff["ok"] and diff["totals"]["changed_tensors"] >= 1
    assert diff["top_layers"][0]["layer"] == "model.layers.2"
    control = result["control"]
    assert control["ok"]
    assert control["totals"]["changed_tensors"] == 0
    assert control["totals"]["identical_tensors"] == control["totals"]["compared_tensors"]
    assert heatmap.is_file() and b"<svg" in heatmap.read_bytes()
    text = render_demo(result, top=5)
    assert "0 changed" in text
    assert "numeric differences between stored weights only" in text
    assert "Control: base vs itself" in text
    # no live network on the cached second run
    result2 = run_demo(fetch=False, cache_dir=cache, pair=pair, opener=BoomOpener(),
                       heatmap=False)
    assert result2["diff"]["totals"]["changed_tensors"] == diff["totals"]["changed_tensors"]
    assert result2["pair"]["base"]["downloaded"] is False


def test_heatmap_handles_empty_layers(tmp_path):
    path = write_layer_heatmap({"top_layers": []}, tmp_path / "empty.svg")
    assert "No layer rows" in path.read_text()


def test_cli_and_slash_require_fetch(tmp_path, capsys, monkeypatch):
    from weightscope.cli import make_cli_handler, make_slash_handler, setup_argparse
    import argparse

    cache = tmp_path / "cache"
    monkeypatch.setenv("WEIGHTSCOPE_DEMO_CACHE", str(cache))
    parser = argparse.ArgumentParser()
    setup_argparse(parser)
    args = parser.parse_args(["demo"])  # no --fetch
    # argparse dest for cache-dir
    if getattr(args, "cache_dir", None) is None:
        args.cache_dir = str(cache)
    else:
        args.cache_dir = str(cache)
    rc = make_cli_handler(None)(args)
    captured = capsys.readouterr()
    assert rc == 1
    assert "--fetch" in captured.err or "--fetch" in captured.out

    msg = make_slash_handler(None, None)("demo")
    assert msg.startswith("weightscope error:") and "--fetch" in msg


def test_cli_demo_with_mocked_fetch(tmp_path, capsys, monkeypatch):
    from weightscope import demo as demo_mod
    from weightscope.cli import make_cli_handler, make_slash_handler, setup_argparse
    import argparse

    pair, blobs = _pair(tmp_path)
    cache = tmp_path / "cache"
    opener = FakeOpener(blobs)
    monkeypatch.setattr(demo_mod, "DEMO_PAIR", pair)
    monkeypatch.setattr(demo_mod, "_opener", lambda: opener)

    parser = argparse.ArgumentParser()
    setup_argparse(parser)
    args = parser.parse_args(["demo", "--fetch", "--cache-dir", str(cache), "--no-heatmap"])
    rc = make_cli_handler(None)(args)
    out = capsys.readouterr()
    assert rc == 0, out.err
    assert "Control: base vs itself" in out.out
    assert "0 changed" in out.out
    assert "numeric differences between stored weights only" in out.out

    # slash has no --cache-dir; point it at the cache the CLI just filled (offline, no network).
    monkeypatch.setenv("WEIGHTSCOPE_DEMO_CACHE", str(cache))
    msg = make_slash_handler(None, None)("demo --no-heatmap")
    assert "0 changed" in msg
    assert not msg.startswith("weightscope error:")


def test_pinned_catalog_sizes_are_consistent():
    assert DEMO_BASE["size"] + DEMO_INSTRUCT["size"] == TOTAL_BYTES
    assert TOTAL_BYTES == 2 * 269_060_552
    assert len(DEMO_BASE["sha256"]) == 64 and len(DEMO_BASE["revision"]) == 40
    assert DEMO_BASE["filename"] == "model.safetensors"
    assert DEMO_INSTRUCT["filename"] == "model.safetensors"


def test_default_cache_dir_prefers_explicit_and_hermes_home(tmp_path, monkeypatch):
    explicit = tmp_path / "explicit"
    assert default_cache_dir(explicit) == explicit.resolve()
    monkeypatch.delenv("WEIGHTSCOPE_DEMO_CACHE", raising=False)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hhome"))
    # plugin_storage may or may not be importable; either way the result is under a plugin-data dir
    got = default_cache_dir()
    assert "weightscope" in str(got) and "demo-cache" in str(got)
