"""Published-model demo: SmolLM2-135M (base) vs Instruct, plus an identical-file control.

Network is used only when the caller passes fetch=True (CLI `--fetch`, slash `demo --fetch`)
AND a pinned file is missing or fails its SHA-256 check. Cached, verified files are reused
offline. Model weights are never shipped in this repository.
"""

from __future__ import annotations

import hashlib
import os
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from .render import render_diff, render_summary
from .routing import run_diff, run_summary
from .st_reader import WeightScopeError
from .tools import limits_from

# Pinned to immutable Hugging Face revisions + LFS SHA-256 (verified 2026-11-10).
# Sizes are exact byte lengths of model.safetensors.
DEMO_BASE = {
    "id": "base",
    "label": "SmolLM2-135M (base)",
    "repo": "HuggingFaceTB/SmolLM2-135M",
    "revision": "93efa2f097d58c2a74874c7e644dbc9b0cee75a2",
    "filename": "model.safetensors",
    "sha256": "80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1",
    "size": 269_060_552,
}
DEMO_INSTRUCT = {
    "id": "instruct",
    "label": "SmolLM2-135M-Instruct",
    "repo": "HuggingFaceTB/SmolLM2-135M-Instruct",
    "revision": "12fd25f77366fa6b3b4b768ec3050bf629380bac",
    "filename": "model.safetensors",
    "sha256": "5af571cbf074e6d21a03528d2330792e532ca608f24ac70a143f6b369968ab8c",
    "size": 269_060_552,
}
DEMO_PAIR = (DEMO_BASE, DEMO_INSTRUCT)
TOTAL_BYTES = DEMO_BASE["size"] + DEMO_INSTRUCT["size"]
USER_AGENT = "weightscope/0.1.1 (+https://github.com/dukemawex/hermes-plugin-weightscope)"
ALLOWED_HOSTS = {"huggingface.co", "hf.co"}
ALLOWED_HOST_SUFFIXES = (".huggingface.co", ".hf.co")
CONNECT_TIMEOUT_S = 30
READ_TIMEOUT_S = 600

HONESTY = (
    "These are numeric differences between stored weights only. A weight change does NOT "
    "establish a backdoor, a behaviour change, a safety status, or tampering, and an unchanged "
    "tensor does not prove identical behaviour. Use these numbers to decide where to look, then "
    "test behaviour separately."
)


@dataclass(frozen=True)
class PreparedFile:
    spec: dict
    path: Path
    downloaded: bool


def default_cache_dir(explicit: str | os.PathLike | None = None) -> Path:
    """Cache lives outside the plugin install tree so updates cannot delete the weights."""
    if explicit:
        return Path(explicit).expanduser().resolve()
    env = os.environ.get("WEIGHTSCOPE_DEMO_CACHE")
    if env:
        return Path(env).expanduser().resolve()
    try:
        from plugins.plugin_storage import plugin_data_dir
        return Path(plugin_data_dir("weightscope")) / "demo-cache"
    except Exception:
        pass
    home = os.environ.get("HERMES_HOME")
    if home:
        return Path(home).expanduser().resolve() / "plugin-data" / "weightscope" / "demo-cache"
    return Path.home() / ".cache" / "weightscope" / "demo-cache"


def cache_path(cache_dir: Path, spec: dict) -> Path:
    safe_repo = spec["repo"].replace("/", "--")
    return cache_dir / safe_repo / spec["revision"] / spec["filename"]


def _human_bytes(n: int) -> str:
    x = float(n)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if x < 1024 or unit == "GiB":
            return f"{int(x)} B" if unit == "B" else f"{x:.1f} {unit}"
        x /= 1024
    return str(n)


def _allowed_host(host: str) -> bool:
    host = (host or "").lower().rstrip(".")
    if not host:
        return False
    if host in ALLOWED_HOSTS:
        return True
    return any(host.endswith(sfx) for sfx in ALLOWED_HOST_SUFFIXES)


def _resolve_url(spec: dict) -> str:
    return f"https://huggingface.co/{spec['repo']}/resolve/{spec['revision']}/{spec['filename']}"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


class _HostLimitedRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        parts = urllib.parse.urlparse(newurl)
        if parts.scheme != "https":
            raise WeightScopeError(f"demo download refused redirect to non-https URL ({parts.scheme})")
        if parts.username or parts.password:
            raise WeightScopeError("demo download refused a redirect that contains credentials")
        if not _allowed_host(parts.hostname or ""):
            raise WeightScopeError(
                f"demo download refused redirect to host {parts.hostname!r} "
                "(only huggingface.co / *.huggingface.co / hf.co / *.hf.co are allowed)"
            )
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _opener():
    ctx = ssl.create_default_context()
    https = urllib.request.HTTPSHandler(context=ctx)
    return urllib.request.build_opener(https, _HostLimitedRedirect())


def download_spec(spec: dict, dest: Path, opener=None, progress=None) -> None:
    """HTTPS GET of a pinned Hugging Face file; SHA-256 must match before the file is kept."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    url = _resolve_url(spec)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    opener = opener or _opener()
    received = 0
    expected = int(spec["size"])
    try:
        with opener.open(req, timeout=READ_TIMEOUT_S) as resp:
            final = resp.geturl() or url
            host = urllib.parse.urlparse(final).hostname or ""
            if urllib.parse.urlparse(final).scheme != "https" or not _allowed_host(host):
                raise WeightScopeError(f"demo download landed on disallowed URL host {host!r}")
            hasher = hashlib.sha256()
            with tmp.open("wb") as out:
                while True:
                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
                    hasher.update(chunk)
                    received += len(chunk)
                    if received > expected + 1024:
                        raise WeightScopeError(
                            f"demo download of {spec['repo']} exceeded the pinned size "
                            f"({_human_bytes(expected)})"
                        )
                    if progress is not None:
                        progress(spec, received, expected)
        if received != expected:
            raise WeightScopeError(
                f"demo download of {spec['repo']} was {received} bytes, expected {expected}"
            )
        digest = hasher.hexdigest()
        if digest != spec["sha256"]:
            raise WeightScopeError(
                f"demo download of {spec['repo']} failed SHA-256 check "
                f"(got {digest}, expected {spec['sha256']})"
            )
        tmp.replace(dest)
    except WeightScopeError:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise
    except urllib.error.HTTPError as exc:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise WeightScopeError(f"demo download of {spec['repo']} failed: HTTP {exc.code}") from None
    except urllib.error.URLError as exc:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise WeightScopeError(f"demo download of {spec['repo']} failed: {exc.reason}") from None
    except TimeoutError:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise WeightScopeError(f"demo download of {spec['repo']} timed out") from None


def ensure_files(pair=DEMO_PAIR, cache_dir: Path | None = None, fetch: bool = False,
                 opener=None, progress=None) -> list[PreparedFile]:
    cache_dir = default_cache_dir(cache_dir)
    missing = []
    prepared: list[PreparedFile] = []
    for spec in pair:
        path = cache_path(cache_dir, spec)
        downloaded = False
        if path.is_file() and path.stat().st_size == spec["size"]:
            digest = sha256_file(path)
            if digest == spec["sha256"]:
                prepared.append(PreparedFile(spec, path, False))
                continue
            # Corrupt or stale cache: only replace when the caller consented to fetch.
            if not fetch:
                raise WeightScopeError(
                    f"cached file {path} failed SHA-256 check; re-run with --fetch to download it again"
                )
            path.unlink()
        elif path.exists() and not fetch:
            raise WeightScopeError(
                f"cached file {path} has the wrong size; re-run with --fetch to download it again"
            )
        if not fetch:
            missing.append(spec)
            continue
        download_spec(spec, path, opener=opener, progress=progress)
        downloaded = True
        prepared.append(PreparedFile(spec, path, downloaded))
    if missing:
        names = ", ".join(s["repo"] for s in missing)
        raise WeightScopeError(
            "weightscope demo is offline by default and will not download checkpoints unless you "
            f"pass --fetch (slash: /weightscope demo --fetch). That flag consents to a one-time "
            f"HTTPS download of {len(missing)} pinned file(s) ({_human_bytes(sum(s['size'] for s in missing))}) "
            f"from huggingface.co: {names}. Files are cached at {cache_dir} and are not part of the "
            "plugin repository. After they are cached, `demo` runs fully offline."
        )
    return prepared


def write_layer_heatmap(diff_result: dict, dest: Path, title: str = "WeightScope layer heatmap") -> Path:
    """Simple SVG bar chart of per-layer relative change. No external dependencies."""
    layers = list(diff_result.get("top_layers") or [])
    # Prefer a stable model-order listing when the full layer list is present.
    all_layers = diff_result.get("layers") or layers
    rows = all_layers or layers
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        dest.write_text(
            f'<svg xmlns="http://www.w3.org/2000/svg" width="800" height="80">'
            f'<text x="16" y="44" font-family="monospace" font-size="14">No layer rows to plot</text></svg>\n',
            encoding="utf-8",
        )
        return dest

    def rel(row):
        v = row.get("rel_change")
        return 0.0 if v is None else float(v)

    max_rel = max((rel(r) for r in rows), default=0.0) or 1.0
    width, left, bar_h, gap, right = 900, 280, 16, 6, 40
    height = 56 + len(rows) * (bar_h + gap)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>text{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;fill:#0f172a}</style>",
        f'<rect width="{width}" height="{height}" fill="#f8fafc"/>',
        f'<text x="16" y="24" font-size="16" font-weight="700">{_xml(title)}</text>',
        '<text x="16" y="42" font-size="11" fill="#475569">relative change ||B-A|| / ||A|| per layer group — numeric only, not a safety verdict</text>',
    ]
    y = 56
    bar_max = width - left - right
    for row in rows:
        name = str(row.get("layer") or row.get("name") or "?")
        value = rel(row)
        bw = max(1, int(bar_max * (value / max_rel))) if value else 0
        parts.append(f'<text x="{left - 8}" y="{y + bar_h - 4}" font-size="11" text-anchor="end">{_xml(name)}</text>')
        parts.append(f'<rect x="{left}" y="{y}" width="{bar_max}" height="{bar_h}" fill="#e2e8f0" rx="2"/>')
        if bw:
            parts.append(f'<rect x="{left}" y="{y}" width="{bw}" height="{bar_h}" fill="#2563eb" rx="2"/>')
        parts.append(f'<text x="{left + bar_max + 6}" y="{y + bar_h - 4}" font-size="11">{value:.4g}</text>')
        y += bar_h + gap
    parts.append("</svg>\n")
    dest.write_text("\n".join(parts), encoding="utf-8")
    return dest


def _xml(text: str) -> str:
    return (text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            .replace('"', "&quot;").replace("'", "&apos;"))


def _progress_stderr(spec, received, expected):
    pct = (100 * received / expected) if expected else 0
    print(f"  downloading {spec['repo']}  {_human_bytes(received)} / {_human_bytes(expected)}  ({pct:5.1f}%)",
          file=sys.stderr, end="\r" if received < expected else "\n", flush=True)


def run_demo(*, fetch: bool = False, cache_dir=None, backend: str = "numpy",
             top: int = 8, heatmap: bool = True, heatmap_path=None,
             get_setting=None, pair=None, opener=None, progress=None,
             record=False) -> dict:
    """Fetch (optional) and run summary + base-vs-instruct diff + identical-file control."""
    t0 = time.perf_counter()
    pair = pair or DEMO_PAIR
    cache = default_cache_dir(cache_dir)
    files = ensure_files(pair, cache_dir=cache, fetch=fetch, opener=opener,
                         progress=progress)
    base, instruct = files[0], files[1]
    limits = limits_from(get_setting)
    summary = run_summary(str(base.path), str(base.path), limits=limits, backend="local",
                          hash_files=False)
    cmp_diff = run_diff(str(base.path), str(instruct.path), str(base.path), str(instruct.path),
                        limits=limits, backend=backend or "numpy", device=None, top_k=top,
                        top_layers=100, tolerance=0.0, rank_by="relative")
    control = run_diff(str(base.path), str(base.path), str(base.path), str(base.path),
                       limits=limits, backend=backend or "numpy", device=None, top_k=top,
                       top_layers=top, tolerance=0.0, rank_by="relative")
    svg = None
    if heatmap:
        dest = Path(heatmap_path) if heatmap_path else cache / "smollm2-135m-base-vs-instruct.svg"
        svg = str(write_layer_heatmap(
            cmp_diff, dest,
            title=f"{base.spec['label']} vs {instruct.spec['label']}",
        ))
    return {
        "ok": True,
        "pair": {
            "base": {**{k: base.spec[k] for k in ("label", "repo", "revision", "filename", "sha256", "size")},
                     "path": str(base.path), "downloaded": base.downloaded},
            "instruct": {**{k: instruct.spec[k] for k in ("label", "repo", "revision", "filename", "sha256", "size")},
                         "path": str(instruct.path), "downloaded": instruct.downloaded},
        },
        "cache_dir": str(cache),
        "total_download_bytes": TOTAL_BYTES,
        "summary": summary,
        "diff": cmp_diff,
        "control": control,
        "heatmap": svg,
        "interpretation": HONESTY,
        "elapsed_s": round(time.perf_counter() - t0, 3),
    }


def render_demo(result: dict, top: int = 8) -> str:
    if not result.get("ok"):
        return f"weightscope error: {result.get('error')}"
    base, inst = result["pair"]["base"], result["pair"]["instruct"]
    lines = [
        "WeightScope demo: SmolLM2-135M (base) vs SmolLM2-135M-Instruct",
        "Pinned public checkpoints from Hugging Face (downloaded only with --fetch; then reused offline).",
        f"  base:     {base['repo']} @ {base['revision'][:12]}  "
        f"{_human_bytes(base['size'])}  sha256={base['sha256'][:12]}…",
        f"  instruct: {inst['repo']} @ {inst['revision'][:12]}  "
        f"{_human_bytes(inst['size'])}  sha256={inst['sha256'][:12]}…",
        f"  cache:    {result['cache_dir']}",
        f"  source:   {('fresh download' if base['downloaded'] or inst['downloaded'] else 'cache (offline)')}",
        "",
        "== 1. Summary (base, hashing skipped — SHA-256 already verified at fetch) ==",
        render_summary(result["summary"]),
        "",
        "== 2. Diff: base vs Instruct ==",
        render_diff(result["diff"], top=top),
        "",
        "== 3. Control: base vs itself (expect 0 changed) ==",
        render_diff(result["control"], top=top),
    ]
    if result.get("heatmap"):
        lines += ["", f"Layer heatmap (SVG): {result['heatmap']}"]
    lines += ["", "Note: " + result["interpretation"], f"(demo wall time {result['elapsed_s']}s)"]
    return "\n".join(lines)

