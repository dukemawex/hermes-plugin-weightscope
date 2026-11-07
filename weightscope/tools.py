"""Tool handlers: always return a JSON string, never raise, accept **kwargs."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Callable

from . import history
from .remote import DEFAULT_TOKEN_ENV, RemoteSettings
from .routing import remote_health, run_diff, run_summary
from .st_reader import GiB, MiB, Limits, WeightScopeError

logger = logging.getLogger(__name__)

SETTING_DEFAULTS = {
    "max_total_gib": 64,
    "max_tensors": 200_000,
    "max_header_mib": 100,
    "allow_external_symlinks": False,
    "record_history": True,
    "backend": "auto",
    "device": "",
    "device_budget_mib": 256,
    "remote_url": "",
    "remote_token_env": DEFAULT_TOKEN_ENV,
    "remote_timeout_s": 120,
    "remote_max_response_mib": 4,
    "remote_allow_http_localhost": False,
}


def _setting(get_setting, key):
    if get_setting is None:
        return SETTING_DEFAULTS[key]
    try:
        v = get_setting(key, SETTING_DEFAULTS[key])
    except Exception:
        return SETTING_DEFAULTS[key]
    return SETTING_DEFAULTS[key] if v is None else v


def remote_from(get_setting, url=None, token_env=None, allow_http_localhost=None) -> RemoteSettings | None:
    """Remote settings, or None when no endpoint is configured (=> fully offline)."""
    url = url if url else _setting(get_setting, "remote_url")
    if not url:
        return None
    def num(key, lo, hi):
        try:
            return max(lo, min(hi, float(_setting(get_setting, key))))
        except (TypeError, ValueError):
            return float(SETTING_DEFAULTS[key])
    return RemoteSettings(
        url=str(url),
        token_env=str(token_env or _setting(get_setting, "remote_token_env") or DEFAULT_TOKEN_ENV),
        timeout_s=num("remote_timeout_s", 1, 3600),
        max_response_bytes=int(num("remote_max_response_mib", 0.01, 64) * MiB),
        allow_http_localhost=bool(allow_http_localhost if allow_http_localhost is not None
                                  else _setting(get_setting, "remote_allow_http_localhost")),
    )


def limits_from(get_setting: Callable[[str, object], object] | None, **overrides) -> Limits:
    def val(key):
        if key in overrides and overrides[key] is not None:
            return overrides[key]
        if get_setting is None:
            return SETTING_DEFAULTS[key]
        try:
            return get_setting(key, SETTING_DEFAULTS[key])
        except Exception:
            return SETTING_DEFAULTS[key]

    def num(key, lo, hi):
        try:
            return max(lo, min(hi, float(val(key))))
        except (TypeError, ValueError):
            return SETTING_DEFAULTS[key]

    return Limits(
        max_header_bytes=int(num("max_header_mib", 1, 1024) * MiB),
        max_tensors=int(num("max_tensors", 1, 10_000_000)),
        max_total_bytes=int(num("max_total_gib", 0.001, 4096) * GiB),
        allow_external_symlinks=bool(val("allow_external_symlinks")),
        device_budget_mib=int(num("device_budget_mib", 16, 65536)),
    )


def backend_choice(get_setting, backend=None, device=None) -> tuple[str, str | None]:
    """Explicit arg > plugin setting > default ('auto', no device)."""
    b = backend if backend else _setting(get_setting, "backend")
    d = device if device else _setting(get_setting, "device")
    return str(b or "auto"), (str(d) if d else None)


def resolve_path(p, base_dir: str | None) -> str:
    if not isinstance(p, str) or not p.strip():
        raise WeightScopeError("path must be a non-empty string")
    p = os.path.expanduser(p.strip())
    if not os.path.isabs(p) and base_dir:
        p = str(Path(base_dir) / p)
    return p


def _err(tool: str, msg: str) -> str:
    return json.dumps({"ok": False, "tool": tool, "error": msg,
                       "hint": "weightscope reads safetensors (single file, shard directory, or model.safetensors.index.json) "
                               "on this machine, or on a configured remote worker with backend 'remote'."})


def make_handlers(get_setting=None, get_cwd: Callable[[], str] | None = None, surface: str = "tool"):
    def _cwd():
        if get_cwd is None:
            return None
        try:
            return get_cwd() or None
        except Exception:
            return None

    def ckpt_summary(args: dict, **kwargs) -> str:
        try:
            args = args if isinstance(args, dict) else {}
            raw = args.get("path")
            res = run_summary(resolve_path(raw, _cwd()), raw, limits=limits_from(get_setting),
                              backend=args.get("backend") or "auto", remote=remote_from(get_setting),
                              max_list=args.get("max_list", 50), hash_files=args.get("hash_files", True) is not False)
            return json.dumps(res, separators=(",", ":"))
        except WeightScopeError as exc:
            return _err("ckpt_summary", str(exc))
        except Exception as exc:  # never raise into the agent loop
            logger.debug("ckpt_summary failed", exc_info=True)
            return _err("ckpt_summary", f"unexpected {type(exc).__name__}: {exc}")

    def ckpt_diff(args: dict, **kwargs) -> str:
        try:
            args = args if isinstance(args, dict) else {}
            cwd = _cwd()
            be, dev = backend_choice(get_setting, args.get("backend"), args.get("device"))
            a, b = args.get("a"), args.get("b")
            res = run_diff(resolve_path(a, cwd), resolve_path(b, cwd), a, b,
                           limits=limits_from(get_setting), backend=be, device=dev, remote=remote_from(get_setting),
                           top_k=args.get("top_k", 10), top_layers=args.get("top_layers", 10),
                           tolerance=args.get("tolerance", 0.0), rank_by=args.get("rank_by") or "relative")
            if get_setting is not None and limits_from_bool(get_setting, "record_history"):
                history.record(res, surface)
            return json.dumps(res, separators=(",", ":"))
        except WeightScopeError as exc:
            return _err("ckpt_diff", str(exc))
        except Exception as exc:
            logger.debug("ckpt_diff failed", exc_info=True)
            return _err("ckpt_diff", f"unexpected {type(exc).__name__}: {exc}")

    return ckpt_summary, ckpt_diff


def limits_from_bool(get_setting, key) -> bool:
    try:
        return bool(get_setting(key, SETTING_DEFAULTS[key]))
    except Exception:
        return bool(SETTING_DEFAULTS[key])
