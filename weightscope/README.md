# WeightScope

Safetensors checkpoint inspection for [Hermes Agent](https://hermes-agent.nousresearch.com/docs/).
WeightScope answers the question "**which tensors and layers differ between these two checkpoints, and by how
much?**" without running a model or loading a whole checkpoint into memory. It is fully offline
by default. Network is used only if you configure a remote GPU worker (see [Remote GPU worker](#remote-gpu-worker)),
or if you explicitly run `hermes weightscope demo --fetch` to download the pinned public demo checkpoints
(see [Published-model demo](#published-model-demo)).

> **What the numbers mean, and what they don't.** WeightScope reports numeric differences between stored
> weights. A weight change does **not** establish a backdoor, a behaviour change, a safety status, or
> tampering, and an unchanged tensor does not prove identical behaviour. Use the output to decide *where to look*,
> then test behaviour separately.

## What it adds to Hermes

| Surface | Name | What it does |
|---|---|---|
| Tool | `ckpt_summary` | Tensor list, dtypes, shapes, parameter counts per dtype, largest tensors, file sizes, SHA-256 per file |
| Tool | `ckpt_diff` | Per-tensor and per-layer metrics, top-k most-changed, only-in-A / only-in-B, shape and dtype mismatches, NaN/Inf counts |
| CLI | `hermes weightscope summary PATH` / `diff A B` / `remote-health` / `recent` / `demo [--fetch]` | Same analysis in a terminal; `--json` prints exactly what the tool returns |
| Slash command | `/weightscope summary <path>` / `diff <a> <b>` / `recent` / `demo [--fetch]` | Deterministic table in any chat session, with no model tokens spent |

Both tools return compact JSON strings, never raise, and cap every list (top-k ≤ 100, name lists ≤ 25 by
default with `count` and `truncated` fields), so a huge model can't flood the context window.

## Install (local)

```bash
cp -r weightscope ~/.hermes/plugins/weightscope   # or: hermes plugins install <git url> once published
hermes plugins enable weightscope
```

Requires Hermes Agent `>= 0.23.0` and NumPy (`numpy>=1.26,<3`, declared in `plugin.yaml`).

GPU support is optional: install PyTorch (CUDA, ROCm or Apple MPS build) into the Hermes Python environment
yourself. The plugin never installs or imports torch unless the torch backend is selected or `auto` probes for it.

## Inputs

* a single `.safetensors` file
* a directory of `.safetensors` files (non-recursive)
* a directory containing `model.safetensors.index.json` (Hugging Face sharded layout), or the index file itself

Relative paths in tool and slash-command calls resolve against the session's working directory.

## Metrics (`ckpt_diff`, A = base, B = candidate)

For every tensor present in both with the same shape (values widened to float64, streamed in 1M-element chunks):

| Field | Definition |
|---|---|
| `l2_diff` | ‖B − A‖₂ |
| `base_l2` | ‖A‖₂ |
| `rel_change` | ‖B − A‖ / ‖A‖. When ‖A‖ = 0 it is `0` if B is also zero, otherwise `null` with `zero_baseline: true` (ranked first under `relative`) |
| `max_abs_diff` | max \|B − A\| |
| `cosine` | A·B / (‖A‖‖B‖); `null` when either norm is 0 |
| `frac_changed` | fraction of compared elements with \|B − A\| > `tolerance` (default 0 = any change) |
| `nonfinite_a/b` | NaN/Inf counts; those positions are excluded from every metric, and a tensor whose NaN/Inf pattern differs counts as changed |

Per-layer groups aggregate the squared sums of their tensors (`model.layers.3.*` → `model.layers.3`; names without
a numeric segment group by their module prefix, e.g. `lm_head`). Ranking: `relative` (default), `l2` or `max_abs`.
A dtype mismatch (e.g. F32 vs BF16) is reported and the values are still compared numerically.

## Compute backends (CPU and GPU)

The parser always stays in numpy (seek-and-read of bounded raw chunks). The per-chunk metric math runs on a
pluggable backend:

| Backend | Device | Element math | Notes |
|---|---|---|---|
| `numpy` (default fallback) | CPU | float64 | always available |
| `torch` | `cuda`, `cuda:N`, `mps`, `cpu` | float64 on CUDA/CPU, float32 on MPS (no float64 there) | BF16 chunks are shipped to the device as bfloat16 and upcast there |

Partial sums are always accumulated across chunks on the host in float64.

* `--backend auto` (default) picks, in order: torch on a local CUDA or MPS GPU → the configured remote worker
  → numpy on this CPU (only when no remote is configured). The decision and its reason are in `routing`.
* Pick explicitly with `hermes weightscope diff A B --backend torch --device cuda`, or the `backend` / `device`
  tool args, or the `backend` / `device` settings. Explicit arg > setting > `auto`.
* `device_budget_mib` (default 256, CLI `--device-budget-mib`) sets the transfer batch: chunk elements ≈
  budget / (bytes per element of working memory), so device memory stays bounded whatever the tensor size.
* An unavailable or invalid device (`cuda` without CUDA, `cuda:3` with one GPU, `mps` off macOS, a typo)
  is a clean error, never a silent fallback. A device out-of-memory error suggests a smaller budget.
* Every diff reports what ran in `execution`: `backend`, `device`, `compute_dtype`, `accumulate_dtype`, `chunk_elems`
  (plus `device_name` and torch/CUDA versions on CUDA).

Measured on CPU only (`scripts/bench_synthetic.py`, two 512 MB BF16 checkpoints, warm cache, 8 vCPU box):
numpy 2.6 s / 82 MiB peak RSS; torch on CPU 6.2 s / 530 MiB (most of it is the torch runtime).
Torch on CPU is slower than numpy here, so `auto` only chooses torch when there is a GPU. No GPU timings have been measured.

## Remote GPU worker

For checkpoints that live on a GPU machine, WeightScope can send the diff to a remote worker over HTTPS
(protocol and setup: [`docs/REMOTE_WORKER.md`](docs/REMOTE_WORKER.md); JSON schema: [`docs/remote-worker.schema.json`](docs/remote-worker.schema.json);
reference worker: [`worker/remote_worker.py`](worker/remote_worker.py), all shipped inside this plugin directory).

```bash
hermes weightscope remote-health                                   # uses remote_url from settings
hermes weightscope diff /models/base /models/candidate --backend remote
hermes weightscope diff A B --backend remote --remote-url https://worker.example --remote-token-env MY_TOKEN
```

* A/B are paths **on the worker**, inside its approved roots, or `ref:<id>` object references. Nothing is uploaded:
  checkpoints are too large and often too private to ship per call.
* Tool calls use `backend: "remote"` (or `auto`). The endpoint comes only from settings or CLI flags, never from
  tool arguments.
* A configured remote that fails gives an explicit error. `auto` does not quietly fall back to the CPU.
* Output keeps `routing` (`local` / `remote` / `cpu-fallback` + reason) apart from `execution`. For remote runs,
  `execution.reported` is the worker's self-reported device, labelled as not verified by this machine.
* Client safeguards: https only (plain http only for localhost with `remote_allow_http_localhost`), no redirects,
  bounded timeout and response size, strict schema validation, health-only bounded retry, and a token read by
  env-var/secret name that is redacted from every error.

## Hardening

* Header length capped (100 MiB by default), must be valid UTF-8 JSON object; every entry's dtype, shape and offsets are validated
* Offsets must lie within the file's data region, sizes must equal `dtype × shape`, overlapping ranges are rejected
* Truncated files, empty headers and short reads give clean errors, never tracebacks
* Index shard paths: absolute paths, `..`, backslashes and drive letters are rejected; symlinks are resolved and must stay
  inside the index directory unless `allow_external_symlinks` is set (needed for Hugging Face cache snapshots, whose files are symlinks into `blobs/`)
* Index/shard consistency: tensors listed in the index must be in the named shard; a tensor stored in two files is an error
* Configurable limits: tensor count (200k), tensor data per checkpoint (64 GiB), header size, file count
* Memory stays bounded by chunk size (see the benchmark numbers above)

## Settings (`plugins.entries.weightscope.settings` in config.yaml, or the Desktop settings form)

| Key | Default | Meaning |
|---|---|---|
| `max_total_gib` | 64 | Refuse checkpoints with more tensor data than this |
| `max_tensors` | 200000 | Refuse checkpoints with more tensors |
| `max_header_mib` | 100 | Maximum safetensors JSON header size |
| `allow_external_symlinks` | false | Allow shard symlinks resolving outside the checkpoint directory |
| `record_history` | true | Keep paths and headline numbers of recent diffs |
| `backend` | auto | `auto`, `numpy`, `torch` or `remote` |
| `device` | "" | torch device (`cuda`, `cuda:1`, `mps`, `cpu`); empty = automatic |
| `device_budget_mib` | 256 | torch backend: approximate device memory per chunk pair |
| `remote_url` | "" | HTTPS base URL of a remote worker; empty = fully offline |
| `remote_token_env` | WEIGHTSCOPE_REMOTE_TOKEN | **Name** of the env var / Hermes secret holding the bearer token |
| `remote_timeout_s` | 120 | Remote request timeout (1–3600 s) |
| `remote_max_response_mib` | 4 | Maximum remote response size |
| `remote_allow_http_localhost` | false | Allow `http://` to localhost test workers only |

## What it reads and writes (disclosure)

* **Reads** only the files at the paths you or the agent pass in (headers, tensor bytes and, for `ckpt_summary`, the whole
  file for hashing). Results contain statistics, never raw tensor values.
* **Writes** only `<hermes home>/plugin-data/weightscope/`: `data.db` for each diff (timestamp, the two paths, the tensor counts,
  the global relative change and the top tensor name; last 200 kept; disable with `record_history: false`), and, if you
  run `demo --fetch`, the pinned public checkpoints plus an SVG heatmap under `demo-cache/`.
* **Network: none by default.** With `remote_url` empty (the default) and without `demo --fetch`, WeightScope makes no
  network calls at all.
  When you set `remote_url` and a call is routed remotely (`backend: remote`, or `auto` with no local GPU),
  it sends HTTPS requests **only to that URL**: the checkpoint path strings or references, the diff parameters
  and an `Authorization: Bearer` header with the token from the env var/secret named in `remote_token_env`.
  It receives statistics only. No checkpoint bytes are uploaded. `remote-health` makes one GET.
  The only other network path is `hermes weightscope demo --fetch` (or `/weightscope demo --fetch`): an explicit
  consent flag that downloads the two pinned SmolLM2-135M `model.safetensors` files above (513.2 MiB total) over HTTPS
  from huggingface.co / official Hugging Face CDNs. No Hugging Face token is sent. Without `--fetch` the demo refuses
  and does not touch the network. Cached, SHA-256-verified files are reused offline.
* **Credentials:** only the optional remote-worker token, read by name through Hermes's secret scope (else the environment),
  never written to disk, logs, history or results. The published-model demo does not use a credential.
* **Reference worker:** `worker/remote_worker.py` and `docs/` ship in this directory as documentation and an optional
  server you can run yourself on a GPU machine. The plugin never imports, registers or starts the worker, and it opens
  no listening sockets. When you run the worker, it serves HTTP(S) on the host/port you choose and reads only files under the
  `--root` directories you pass.
* Torch, if you install it, is only imported, never downloaded. No shell commands, no background processes, no model calls.
* The published-model demo compares two public checkpoints; the numbers it prints are not a safety, quality or
  alignment verdict on SmolLM2 or the Instruct fine-tune.

## Limitations

* Remote: the worker's device report is self-reported and can't be verified by the client. The reference worker
  supports paths only (no `ref:` object store), and there is no upload
* safetensors only: no PyTorch `.bin`/`.pt` pickles, GGUF or ONNX
* Quantized or packed formats (FP8, GPTQ/AWQ int packing, bitsandbytes) are listed as undecoded and skipped; integer tensors
  are compared as plain numbers, which is meaningless for packed weights
* Large checkpoints are mostly I/O bound: expect minutes for multi-GB models on either backend. The slash command runs synchronously in the session host
* With torch installed, `auto` imports torch to probe for a GPU, which adds a few seconds per call; set `backend: numpy` to skip that
* CuPy is not supported (torch covers CUDA, ROCm and MPS with one code path)
* Integer tensors are compared as float64 (very large 64-bit values lose precision)
* Tensors are matched by exact name; renamed or re-sharded-and-renamed layers show up as only-in-A / only-in-B
* Metrics are global per tensor; they don't localise changes within a tensor (rows/heads)
* Under `plugins.isolation: host`, Hermes skips `register_cli_command`, so `hermes weightscope` is only available in the local CLI

## Published-model demo

`hermes weightscope demo --fetch` (slash: `/weightscope demo --fetch`) downloads two pinned public
[SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M) checkpoints from Hugging Face — the base model and
the Instruct fine-tune — then summarises the base, diffs base vs Instruct, diffs the base file against itself (the
control, which must report 0 changed tensors), and writes a per-layer SVG heatmap. `--fetch` is the consent flag:
without it the command refuses and makes no network calls. After the files are cached, later runs are fully offline
and do not need `--fetch`.

The weights are **not** in this repository. Each file is `model.safetensors` at a pinned revision and SHA-256:

| Checkpoint | Revision | Size | SHA-256 |
|---|---|---|---|
| `HuggingFaceTB/SmolLM2-135M` | `93efa2f097d58c2a74874c7e644dbc9b0cee75a2` | 256.6 MiB (269,060,552 bytes) | `80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1` |
| `HuggingFaceTB/SmolLM2-135M-Instruct` | `12fd25f77366fa6b3b4b768ec3050bf629380bac` | 256.6 MiB (269,060,552 bytes) | `5af571cbf074e6d21a03528d2330792e532ca608f24ac70a143f6b369968ab8c` |

Total download: **513.2 MiB**. HTTPS only, huggingface.co and official `*.huggingface.co` / `*.hf.co` CDNs, no API
token. Cache default: `<hermes home>/plugin-data/weightscope/demo-cache` (or `$WEIGHTSCOPE_DEMO_CACHE` / `--cache-dir`).
On this machine the first download took about 25 s; the analysis then took **4.4 s** wall time (numpy, CPU), of which
2.0 s was the Instruct diff and 1.9 s the identical-file control.

Measured 10 Nov 2026 (numpy / CPU) — these are the numbers the command actually printed:

```text
$ hermes weightscope demo --fetch --backend numpy --top 8

== 1. Summary (base) ==
Tensors: 272   Parameters: 134,515,008   Data: 256.57 MiB   Layer groups: 32
dtypes: BF16  272 tensors  134,515,008 params

== 2. Diff: base vs Instruct ==
Compared 272 tensors: 272 changed, 0 identical; 32 layer group(s) changed
Global ||B-A|| = 180.6   relative = 0.08575   params changed = 97.67%

Top changed tensors (rank by relative):
   #  rel_change     l2_diff     max_abs      cosine  frac>tol  name
   1      0.3278       4.849     0.06982    0.949790     99.6%  model.layers.0.self_attn.v_proj.weight
   2      0.3235      0.3096     0.04742    0.947746     99.7%  model.layers.0.input_layernorm.weight
   3      0.2331       8.427     0.07141    0.973707     99.3%  model.layers.0.self_attn.o_proj.weight
   4      0.1362        5.05     0.07446    0.990821     98.8%  model.layers.2.self_attn.v_proj.weight
   5      0.1241       10.71      0.1084    0.992337     98.4%  model.layers.28.self_attn.q_proj.weight

Top changed layer groups:
  model.embed_tokens                       rel=    0.1167  changed 1/1 tensors
  model.layers.25                          rel=   0.09547  changed 9/9 tensors
  model.layers.26                          rel=   0.09231  changed 9/9 tensors

== 3. Control: base vs itself (expect 0 changed) ==
Compared 272 tensors: 0 changed, 272 identical; 0 layer group(s) changed
Global ||B-A|| = 0   relative = 0   params changed = 0.00%
warning: A and B resolve to the same file(s); every tensor will be identical
```

A per-layer SVG heatmap is written next to the cache (or to `--heatmap PATH`).

## Example

```text
$ hermes weightscope diff ckpts/base ckpts/finetuned --top 5
Compared 27 tensors: 2 changed, 25 identical; 1 layer group(s) changed
Global ||B-A|| = 2.759   relative = 0.04112   params changed = 10.63%

Top changed tensors (rank by relative):
   #  rel_change     l2_diff     max_abs      cosine  frac>tol  name
   1      0.5008       2.279      0.1876    0.896996    100.0%  model.layers.3.mlp.up_proj.weight
   2      0.4821       1.556      0.1586    0.900774    100.0%  model.layers.3.self_attn.q_proj.weight
```

(synthetic fixtures from `scripts/make_fixtures.py`)

## License

MIT; see `LICENSE`. Provenance notes are in `NOTICE`.
