#!/usr/bin/env python3
"""Create tiny synthetic safetensors checkpoints for the WeightScope demo (no downloads)."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))
import stfixtures as F  # noqa: E402

out = Path(sys.argv[1] if len(sys.argv) > 1 else "demo-ckpts")
base = F.base_model(n_layers=6, d=32, vocab=64)
F.write(out / "base" / "model.safetensors", base, metadata={"format": "pt", "note": "synthetic demo"})
F.write_sharded(out / "finetuned", F.finetuned(base, layer=3, scale=0.05), n_shards=3)
F.write(out / "finetuned-bf16" / "model.safetensors", F.as_dtype(F.finetuned(base, layer=3, scale=0.05), "BF16"))
evil = out / "evil-index"
evil.mkdir(parents=True, exist_ok=True)
(evil / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"w": "../../../../etc/passwd"}}))
(out / "corrupt.safetensors").write_bytes((10**12).to_bytes(8, "little") + b"{")
print(f"wrote synthetic checkpoints under {out}/")
