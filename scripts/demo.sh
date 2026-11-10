#!/usr/bin/env bash
# WeightScope synthetic demo: local fixtures -> real `hermes weightscope` CLI. No model key, no network.
# For the published-model walkthrough (SmolLM2-135M vs Instruct, ~513 MiB download once), run:
#   hermes weightscope demo --fetch
# Usage: scripts/demo.sh            (expects .venv with Hermes v0.23.0 + numpy; see README)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PYTHON:-$ROOT/.venv/bin/python}"
HERMES="${HERMES:-$ROOT/.venv/bin/hermes}"
DEMO="$(mktemp -d "${TMPDIR:-/tmp}/weightscope-demo.XXXXXX")"
trap 'rm -rf "$DEMO"' EXIT

# Isolated Hermes profile so the demo never touches ~/.hermes.
export HERMES_HOME="$DEMO/hermes-home"
mkdir -p "$HERMES_HOME/plugins"
cp -r "$ROOT/weightscope" "$HERMES_HOME/plugins/weightscope"
rm -rf "$HERMES_HOME/plugins/weightscope/__pycache__"
printf 'plugins:\n  enabled:\n    - weightscope\n' > "$HERMES_HOME/config.yaml"

step() { printf '\n\033[1;36m$ %s\033[0m\n' "$*"; }
cd "$DEMO"
"$PY" "$ROOT/scripts/make_fixtures.py" ckpts >/dev/null

step hermes weightscope summary ckpts/finetuned --no-hash
"$HERMES" weightscope summary ckpts/finetuned --no-hash

step hermes weightscope diff ckpts/base ckpts/finetuned --top 5 --backend numpy
"$HERMES" weightscope diff ckpts/base ckpts/finetuned --top 5 --backend numpy

step hermes weightscope diff ckpts/finetuned ckpts/finetuned-bf16 --top 3 --tolerance 1e-3 --backend numpy   "# same weights, BF16 rounding only"
"$HERMES" weightscope diff ckpts/finetuned ckpts/finetuned-bf16 --top 3 --tolerance 1e-3 --backend numpy

if "$PY" -c "import torch" 2>/dev/null; then
  step hermes weightscope diff ckpts/base ckpts/finetuned --top 2 --backend torch --device cpu
  "$HERMES" weightscope diff ckpts/base ckpts/finetuned --top 2 --backend torch --device cpu
fi

step hermes weightscope summary ckpts/evil-index   "# index tries ../../etc/passwd"
"$HERMES" weightscope summary ckpts/evil-index || true

step hermes weightscope summary ckpts/corrupt.safetensors   "# header claims 1 TB"
"$HERMES" weightscope summary ckpts/corrupt.safetensors || true

step hermes weightscope recent
"$HERMES" weightscope recent
