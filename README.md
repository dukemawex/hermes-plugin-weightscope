# hermes-plugin-weightscope (development tree)

The Hermes plugin lives in [`weightscope/`](weightscope/README.md) (catalog `subdir: weightscope`); read that README
for what the plugin does, its metrics, hardening, disclosures and limitations.

This root holds what the plugin itself doesn't ship: tests, demo and benchmark scripts.

```
weightscope/               the plugin (plugin.yaml, __init__.py register(ctx), parser, analysis, CLI, README)
tests/test_core.py         parser + metrics + hardening unit tests (synthetic fixtures generated in-test)
tests/test_backends.py     numpy/torch(cpu) backend parity, auto-fallback, device errors
tests/test_hardware.py     CUDA/MPS parity, marked `hardware`, deselected by default (run: pytest -m hardware on a GPU box)
tests/test_remote.py       remote routing contract tests against in-process 127.0.0.1 servers (protocol only, CPU)
tests/remote_fixture.py    scripted fake worker + loader for the reference worker
tests/test_demo.py         published-model demo: --fetch consent, mocked HTTP, SHA cache, identical-file control
tests/test_hermes_integration.py   loads the plugin through Hermes's real PluginManager / tool registry / CLI
tests/stfixtures.py        tiny deterministic safetensors writer used by tests and the demo
scripts/demo.sh            < 90 s synthetic terminal demo, no model key, isolated HERMES_HOME
scripts/make_fixtures.py   writes the demo checkpoints
scripts/bench_synthetic.py [--backend numpy|torch|auto] [--device D]  ~1 GB speed / peak-memory check (temp files deleted afterwards)
```

## Dev setup (Hermes v0.23.0 from a local checkout)

```bash
uv venv .venv --python 3.14
UV_PROJECT_ENVIRONMENT=$PWD/.venv uv sync --frozen --python 3.14 --project /path/to/hermes-agent   # at tag v0.23.0
uv pip install --python .venv/bin/python pytest "numpy>=1.26,<3"
# optional, for the torch backend tests (CPU wheel):
uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cpu
```

## Checks

```bash
.venv/bin/python -m pytest -v                                   # unit + remote contract + Hermes integration tests
.venv/bin/python -m pytest -m hardware                          # CUDA/MPS parity, only meaningful on a GPU machine
.venv/bin/hermes plugins validate weightscope --install-deps    # catalog CI gate
(cd weightscope && ../.venv/bin/hermes plugins doctor . --ci)   # runtime contract check
scripts/demo.sh                                                 # synthetic demo (no network)
.venv/bin/hermes weightscope demo --fetch                       # published SmolLM2-135M pair (~513 MiB, once)
```

Weight differences are numeric only; they do not establish a backdoor, behaviour change, safety status or tampering.
