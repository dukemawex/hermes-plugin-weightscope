"""`hermes weightscope ...` CLI tree and the `/weightscope` slash command."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys

from . import history
from .render import render_diff, render_health, render_recent, render_summary
from .routing import remote_health, run_diff, run_summary
from .st_reader import WeightScopeError
from .tools import backend_choice, limits_from, limits_from_bool, remote_from, resolve_path

USAGE = ("Usage:\n  /weightscope summary <path>\n  /weightscope diff <base> <candidate> [--top N] "
         "[--rank-by relative|l2|max_abs] [--tolerance X] [--backend auto|numpy|torch|remote] [--device D]\n"
         "  /weightscope recent\n  /weightscope demo [--fetch]   # published SmolLM2-135M pair; --fetch consents to download")


def setup_argparse(sub: argparse.ArgumentParser) -> None:
    subs = sub.add_subparsers(dest="weightscope_command")
    s = subs.add_parser("summary", help="Inventory a safetensors checkpoint (dtypes, shapes, params, hashes)")
    s.add_argument("path")
    s.add_argument("--json", action="store_true", help="Print the JSON the tool returns")
    s.add_argument("--no-hash", action="store_true", help="Skip SHA-256 hashing of files")
    s.add_argument("--max-list", type=int, default=50)
    s.add_argument("--backend", choices=["auto", "local", "remote"], default=None,
                   help="'remote' reads PATH on the configured remote worker")
    d = subs.add_parser("diff", help="Compare two safetensors checkpoints (A = base, B = candidate)")
    d.add_argument("a")
    d.add_argument("b")
    d.add_argument("--json", action="store_true", help="Print the JSON the tool returns")
    d.add_argument("--top", type=int, default=10, help="Top-k tensors/layers (max 100)")
    d.add_argument("--rank-by", choices=["relative", "l2", "max_abs"], default="relative")
    d.add_argument("--tolerance", type=float, default=0.0)
    d.add_argument("--backend", choices=["auto", "numpy", "torch", "remote"], default=None,
                   help="Where to compute (default: plugin setting, normally auto = local GPU, else remote worker "
                        "if configured, else CPU numpy). With 'remote', A and B are paths on the worker")
    d.add_argument("--device", default=None, help="torch device: cuda, cuda:N, mps, cpu (remote: worker device)")
    d.add_argument("--device-budget-mib", type=int, default=None,
                   help="torch backend: approx. device memory per chunk pair (default 256)")
    h = subs.add_parser("remote-health", help="Check the configured remote worker (GET /v1/health)")
    h.add_argument("--json", action="store_true")
    r = subs.add_parser("recent", help="Show recently recorded diffs (from this plugin's data dir)")
    r.add_argument("--limit", type=int, default=10)
    r.add_argument("--json", action="store_true")
    demo = subs.add_parser(
        "demo",
        help="Published-model walkthrough: SmolLM2-135M vs Instruct, plus an identical-file control",
    )
    demo.add_argument("--fetch", action="store_true",
                      help="Consent to download the pinned public checkpoints from Hugging Face if they "
                           "are not already cached (~513 MiB total). Offline by default; no download without this flag.")
    demo.add_argument("--json", action="store_true")
    demo.add_argument("--backend", choices=["auto", "numpy", "torch"], default="numpy")
    demo.add_argument("--cache-dir", default=None, help="Directory for the pinned demo checkpoints")
    demo.add_argument("--heatmap", default=None, help="Write a per-layer SVG heatmap to this path")
    demo.add_argument("--no-heatmap", action="store_true")
    demo.add_argument("--top", type=int, default=8, help="Top-k tensors shown in the printed tables")
    for p in (s, d, h):
        p.add_argument("--remote-url", default=None, help="Remote worker base URL (https; overrides the setting)")
        p.add_argument("--remote-token-env", default=None,
                       help="NAME of the env var / Hermes secret holding the worker token (default WEIGHTSCOPE_REMOTE_TOKEN)")
        p.add_argument("--remote-allow-http-localhost", action="store_true", default=None,
                       help="Allow http:// for a worker on localhost (testing)")
    for p in (s, d):
        p.add_argument("--max-total-gib", type=float, default=None, help="Per-checkpoint data limit")
        p.add_argument("--allow-external-symlinks", action="store_true", default=None,
                       help="Allow shard symlinks that resolve outside the checkpoint dir (e.g. HF cache)")


def make_cli_handler(get_setting=None):
    def handle(args) -> int:
        cmd = getattr(args, "weightscope_command", None)
        try:
            if cmd == "summary":
                res = run_summary(os.path.abspath(args.path), args.path, limits=_limits(get_setting, args),
                                  backend=args.backend or "auto", remote=_remote(get_setting, args),
                                  max_list=args.max_list, hash_files=not args.no_hash)
                print(json.dumps(res, indent=2) if args.json else render_summary(res))
                return 0
            if cmd == "diff":
                be, dev = backend_choice(get_setting, args.backend, args.device)
                res = run_diff(os.path.abspath(args.a), os.path.abspath(args.b), args.a, args.b,
                               limits=_limits(get_setting, args), backend=be, device=dev,
                               remote=_remote(get_setting, args), top_k=args.top, top_layers=args.top,
                               tolerance=args.tolerance, rank_by=args.rank_by)
                if get_setting is not None and limits_from_bool(get_setting, "record_history"):
                    history.record(res, "cli")
                print(json.dumps(res, indent=2) if args.json else render_diff(res))
                return 0
            if cmd == "remote-health":
                res = remote_health(_remote(get_setting, args))
                print(json.dumps(res, indent=2) if args.json else render_health(res))
                return 0
            if cmd == "recent":
                rows = history.recent(args.limit)
                print(json.dumps(rows, indent=2) if args.json else render_recent(rows))
                return 0
            if cmd == "demo":
                from .demo import _progress_stderr, render_demo, run_demo
                res = run_demo(fetch=bool(args.fetch), cache_dir=args.cache_dir,
                               backend=args.backend or "numpy", top=args.top,
                               heatmap=not args.no_heatmap, heatmap_path=args.heatmap,
                               get_setting=get_setting,
                               progress=_progress_stderr if args.fetch else None)
                if get_setting is not None and limits_from_bool(get_setting, "record_history"):
                    history.record(res["diff"], "demo")
                print(json.dumps(res, indent=2) if args.json else render_demo(res, top=args.top))
                return 0
            print("usage: hermes weightscope {summary PATH | diff A B | remote-health | recent | demo [--fetch]} [--json]")
            return 2
        except WeightScopeError as exc:
            if getattr(args, "json", False):
                print(json.dumps({"ok": False, "error": str(exc)}))
            else:
                print(f"weightscope error: {exc}", file=sys.stderr)
            return 1
    return handle


def _remote(get_setting, args):
    return remote_from(get_setting, url=getattr(args, "remote_url", None),
                       token_env=getattr(args, "remote_token_env", None),
                       allow_http_localhost=getattr(args, "remote_allow_http_localhost", None))


def _limits(get_setting, args):
    return limits_from(get_setting, max_total_gib=getattr(args, "max_total_gib", None),
                       allow_external_symlinks=getattr(args, "allow_external_symlinks", None),
                       device_budget_mib=getattr(args, "device_budget_mib", None))


def make_slash_handler(get_setting=None, get_cwd=None):
    def handle(raw_args: str = "", **kwargs) -> str:
        try:
            argv = shlex.split(raw_args or "")
        except ValueError as exc:
            return f"weightscope: cannot parse arguments ({exc})\n{USAGE}"
        if not argv or argv[0] in ("help", "-h", "--help"):
            return USAGE
        cwd = None
        if get_cwd is not None:
            try:
                cwd = get_cwd() or None
            except Exception:
                cwd = None
        try:
            if argv[0] == "summary" and len(argv) == 2:
                res = run_summary(resolve_path(argv[1], cwd), argv[1], limits=limits_from(get_setting),
                                  remote=remote_from(get_setting), max_list=20)
                return render_summary(res)
            if argv[0] == "diff" and len(argv) >= 3:
                opts = {"top": 10, "rank-by": "relative", "tolerance": 0.0, "backend": None, "device": None}
                rest = argv[3:]
                while rest:
                    key = rest.pop(0).lstrip("-")
                    if key not in opts or not rest:
                        return f"weightscope: unknown or incomplete option --{key}\n{USAGE}"
                    opts[key] = rest.pop(0)
                be, dev = backend_choice(get_setting, opts["backend"], opts["device"])
                res = run_diff(resolve_path(argv[1], cwd), resolve_path(argv[2], cwd), argv[1], argv[2],
                               limits=limits_from(get_setting), backend=be, device=dev,
                               remote=remote_from(get_setting), top_k=opts["top"], top_layers=opts["top"],
                               tolerance=opts["tolerance"], rank_by=opts["rank-by"])
                if get_setting is not None and limits_from_bool(get_setting, "record_history"):
                    history.record(res, "slash")
                return render_diff(res)
            if argv[0] == "recent":
                return render_recent(history.recent(10))
            if argv[0] == "demo":
                from .demo import render_demo, run_demo
                rest = argv[1:]
                fetch = any(a in ("--fetch", "fetch") for a in rest)
                no_heatmap = "--no-heatmap" in rest
                res = run_demo(fetch=fetch, get_setting=get_setting, heatmap=not no_heatmap)
                if get_setting is not None and limits_from_bool(get_setting, "record_history"):
                    history.record(res["diff"], "slash")
                return render_demo(res)
            return USAGE
        except WeightScopeError as exc:
            return f"weightscope error: {exc}"
        except Exception as exc:
            return f"weightscope error: unexpected {type(exc).__name__}: {exc}"
    return handle
