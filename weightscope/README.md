# WeightScope

Safetensors checkpoint inspection for [Hermes Agent](https://hermes-agent.nousresearch.com/docs/).
WeightScope answers the question "**which tensors and layers differ between these two checkpoints, and by how
much?**" without downloading anything, running a model or loading a whole checkpoint into memory. It is fully offline
by default. Network is used only if you configure a remote GPU worker (see [Remote GPU worker](#remote-gpu-worker)).

> **What the numbers mean, and what they don't.** WeightScope reports numeric differences between stored
> weights. A weight change does **not** establish a backdoor, a behaviour change, a safety status, or
> tampering, and an unchanged tensor does not prove identical behaviour. Use the output to decide *where to look*,
> then test behaviour separately.

## What it adds to Hermes

| Surface | Name | What it does |
|---|---|---|
| Tool | `ckpt_summary` | Tensor list, dtypes, shapes, parameter counts per dtype, largest tensors, file sizes, SHA-256 per file |
| Tool | `ckpt_diff` | Per-tensor and per-layer metrics, top-k most-changed, only-in-A / only-in-B, shape and dtype mismatches, NaN/Inf counts |
| CLI | `hermes weightscope summary PATH` / `diff A B` / `remote-health` / `recent` | Same analysis in a terminal; `--json` prints exactly what the tool returns |
| Slash command | `/weightscope summary <path>` / `diff <a> <b>` / `recent` | Deterministic table in any chat session, with no model tokens spent |

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
* **Writes** only `<hermes home>/plugin-data/weightscope/data.db`: for each diff, the timestamp, the two paths, the tensor counts,
  the global relative change and the top tensor name (last 200 kept; disable with `record_history: false`).
* **Network: none by default.** With `remote_url` empty (the default) WeightScope makes no network calls at all.
  When you set `remote_url` and a call is routed remotely (`backend: remote`, or `auto` with no local GPU),
  it sends HTTPS requests **only to that URL**: the checkpoint path strings or references, the diff parameters
  and an `Authorization: Bearer` header with the token from the env var/secret named in `remote_token_env`.
  It receives statistics only. No checkpoint bytes are uploaded. `remote-health` makes one GET.
* **Credentials:** only the remote token, read by name through Hermes's secret scope (else the environment),
  never written to disk, logs, history or results.
* **Reference worker:** `worker/remote_worker.py` and `docs/` ship in this directory as documentation and an optional
  server you can run yourself on a GPU machine. The plugin never imports, registers or starts the worker, and it opens
  no listening sockets. When you run the worker, it serves HTTP(S) on the host/port you choose and reads only files under the
  `--root` directories you pass.
* Torch, if you install it, is only imported, never downloaded. No shell commands, no background processes, no model calls.

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
