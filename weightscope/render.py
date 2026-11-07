"""Plain-text rendering for the CLI and the /weightscope slash command."""

from __future__ import annotations


def _num(x, fmt="{:.4g}"):
    return "n/a" if x is None else fmt.format(x)


def _human_bytes(n: int) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if n < 1024 or unit == "TiB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.2f} {unit}"
        n /= 1024
    return str(n)


def render_error(res: dict) -> str:
    return f"weightscope error: {res.get('error')}"


def render_summary(res: dict) -> str:
    if not res.get("ok"):
        return render_error(res)
    ck = res["checkpoint"]
    out = [f"Checkpoint: {ck['path']}  ({ck['layout']}, {ck['files']} file(s))",
           f"Tensors: {ck['tensors']:,}   Parameters: {ck['parameters']:,}   Data: {_human_bytes(ck['bytes'])}   "
           f"Layer groups: {res['layers']}", "dtypes:"]
    for dt, v in res["dtypes"].items():
        out.append(f"  {dt:<8} {v['tensors']:>6} tensors  {v['parameters']:>14,} params")
    out.append("files:")
    for f in res["files"]:
        h = f.get("sha256", "(not hashed)")
        out.append(f"  {f['file']}  {_human_bytes(f['bytes'])}  sha256={h}")
    out.append("largest tensors:")
    for t in res["largest_tensors"][:5]:
        out.append(f"  {t['name']}  {t['shape']}  {t['parameters']:,}")
    if res["undecoded_dtype_tensors"]["count"]:
        out.append(f"undecoded dtypes (not comparable): {res['undecoded_dtype_tensors']['count']} tensor(s)")
    out += [f"warning: {w}" for w in res["warnings"]]
    out.append(f"note: {res['note']}")
    if res.get("routing"):
        out.append(render_route(res))
    return "\n".join(out)


def render_diff(res: dict, top: int | None = None) -> str:
    if not res.get("ok"):
        return render_error(res)
    a, b, t = res["a"], res["b"], res["totals"]
    out = [f"A (base): {a['path']}  [{a['tensors']} tensors, {a['parameters']:,} params]",
           f"B:        {b['path']}  [{b['tensors']} tensors, {b['parameters']:,} params]",
           f"Compared {t['compared_tensors']} tensors: {t['changed_tensors']} changed, {t['identical_tensors']} identical; "
           f"{t['changed_layers']} layer group(s) changed",
           f"Global ||B-A|| = {_num(t['global_l2_diff'])}   relative = {_num(t['global_rel_change'])}   "
           f"params changed = {_num(t['frac_parameters_changed'], '{:.2%}')}",
           "", f"Top changed tensors (rank by {res['settings']['rank_by']}):",
           f"  {'#':>2}  {'rel_change':>10}  {'l2_diff':>10}  {'max_abs':>10}  {'cosine':>10}  {'frac>tol':>8}  name"]
    rows = res["top_tensors"][: top or len(res["top_tensors"])]
    for i, r in enumerate(rows, 1):
        rel = "zero-base" if r.get("zero_baseline") and r["rel_change"] is None else _num(r["rel_change"])
        out.append(f"  {i:>2}  {rel:>10}  {_num(r['l2_diff']):>10}  {_num(r['max_abs_diff']):>10}  "
                   f"{_num(r['cosine'], '{:.6f}'):>10}  {_num(r['frac_changed'], '{:.1%}'):>8}  {r['name']}")
    if not rows:
        out.append("  (no numeric differences found)")
    if res["top_layers"]:
        out += ["", "Top changed layer groups:"]
        for r in res["top_layers"][: top or len(res["top_layers"])]:
            out.append(f"  {r['layer']:<40} rel={_num(r['rel_change']):>10}  "
                       f"changed {r['changed_tensors']}/{r['tensors']} tensors")
    for key, label in (("only_in_a", "only in A"), ("only_in_b", "only in B"),
                       ("shape_mismatches", "shape mismatches"), ("dtype_mismatches", "dtype mismatches"),
                       ("undecoded_dtypes", "undecoded dtypes (skipped)"), ("nonfinite", "tensors with NaN/Inf")):
        block = res[key]
        if block["count"]:
            items = [i if isinstance(i, str) else _brief(i) for i in block["items"][:5]]
            more = f" (+{block['count'] - len(items)} more)" if block["count"] > len(items) else ""
            out.append(f"{label}: {block['count']}: " + "; ".join(items) + more)
    out += [f"warning: {w}" for w in res["warnings"]]
    out += ["", "Note: " + res["interpretation"], f"({res['elapsed_s']}s)", render_route(res)]
    return "\n".join(out)


def _brief(item: dict) -> str:
    name = item.get("name", "?")
    if "shape_a" in item:
        return f"{name} {item['shape_a']} vs {item['shape_b']}"
    if "nonfinite_a" in item:
        return f"{name} (A:{item['nonfinite_a']} B:{item['nonfinite_b']} of {item['of']})"
    if "dtype_a" in item:
        return f"{name} {item['dtype_a']} vs {item['dtype_b']}"
    return name


def render_recent(rows: list[dict]) -> str:
    if not rows:
        return "No recorded weightscope diffs yet."
    out = ["Recent diffs (newest first, times UTC):"]
    for r in rows:
        out.append(f"  {r['created_at_utc']}  [{r['surface']}]  changed {r['changed_tensors']}/{r['compared_tensors']}"
                   f"  rel={_num(r['global_rel_change'])}  top={r['top_tensor']}\n      A={r['a']}\n      B={r['b']}")
    return "\n".join(out)


def render_route(res: dict) -> str:
    rt, ex = res.get("routing") or {}, res.get("execution") or {}
    lines = []
    if rt:
        lines.append(f"routing: {rt.get('mode')} ({rt.get('reason')})")
    if "reported" in ex:
        rep = ex["reported"]
        dev = rep.get("device_name") or rep.get("device")
        lines.append(f"execution: {ex.get('endpoint')} reports backend={rep.get('backend')} device={dev} "
                     f"[worker-reported, not verified here]")
    elif ex:
        dev = ex.get("device_name") or ex.get("device")
        lines.append(f"execution: local backend={ex.get('backend')} device={dev} compute={ex.get('compute_dtype', 'n/a')}")
    return "\n".join(lines)


def render_health(res: dict) -> str:
    rep = res["execution"]["reported"]
    return (f"remote worker {res['endpoint']} is up ({res['protocol']}); worker reports backend={rep.get('backend')} "
            f"device={rep.get('device_name') or rep.get('device')} [worker-reported, not verified here]")
