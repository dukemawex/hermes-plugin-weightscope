#!/usr/bin/env python3
"""Speed/memory check: two ~512 MB BF16 checkpoints (one 256 MB tensor), diffed with WeightScope.

Usage: scripts/bench_synthetic.py [--backend auto|numpy|torch] [--device DEV] [--workdir DIR]
(writes ~1 GB of temporary files, deletes them after)
"""
import argparse
import json
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

SHAPES = {"embed.weight": (131072, 1024)}  # 128M params = 256 MB in BF16
SHAPES.update({f"layers.{i}.w": (4096, 4096) for i in range(8)})


def write(path: Path, modify: bool):
    hdr, off = {}, 0
    for n, s in SHAPES.items():
        nb = int(np.prod(s)) * 2
        hdr[n] = {"dtype": "BF16", "shape": list(s), "data_offsets": [off, off + nb]}
        off += nb
    hb = json.dumps(hdr).encode()
    hb += b" " * (-len(hb) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hb)) + hb)
        for i, (n, s) in enumerate(SHAPES.items()):
            a = np.random.default_rng(i).integers(0x3C00, 0x3F80, size=int(np.prod(s)), dtype=np.uint16)
            if modify and n == "layers.5.w":
                a[::7] += 1
            f.write(a.astype("<u2").tobytes())


ap = argparse.ArgumentParser()
ap.add_argument("--backend", default="numpy", choices=["auto", "numpy", "torch"])
ap.add_argument("--device", default="")
ap.add_argument("--workdir", default=None)
opts = ap.parse_args()
work = Path(tempfile.mkdtemp(dir=opts.workdir))
try:
    write(work / "a.safetensors", False)
    write(work / "b.safetensors", True)
    code = ("import json,sys,time; sys.path.insert(0, sys.argv[3]);"
            "from weightscope.analysis import diff; t=time.monotonic();"
            "r=diff(sys.argv[1], sys.argv[2], top_k=2, backend=sys.argv[4], device=sys.argv[5] or None);"
            "print(json.dumps({'diff_seconds': round(time.monotonic()-t, 2),"
            "'peak_rss_mib': [int(l.split()[1]) // 1024 for l in open('/proc/self/status') if l.startswith('VmHWM')][0],"
            "'changed_tensors': r['totals']['changed_tensors'], 'top_tensor': r['top_tensors'][0]['name'],"
            "'execution': r['execution']}))")
    out = subprocess.run([sys.executable, "-c", code, str(work / "a.safetensors"), str(work / "b.safetensors"),
                          str(Path(__file__).resolve().parent.parent), opts.backend, opts.device],
                         capture_output=True, text=True, check=True)
    res = json.loads(out.stdout)
    res["bytes_compared"] = (work / "a.safetensors").stat().st_size * 2
    res["largest_tensor_mib"] = 256
    print(json.dumps(res, indent=2))
finally:
    shutil.rmtree(work, ignore_errors=True)
