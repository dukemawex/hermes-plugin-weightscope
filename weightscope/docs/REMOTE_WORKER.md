# WeightScope remote worker protocol (`weightscope-remote/1`)

WeightScope is offline by default. When `remote_url` is set, `ckpt_diff` / `ckpt_summary` can run on a remote
GPU worker instead of this machine. This document is the contract between the plugin (client,
`weightscope/remote.py`) and any worker; `worker/remote_worker.py` (inside the plugin directory) is a small reference implementation. The plugin never
imports, registers or starts the worker; you run it yourself on the machine that holds the checkpoints.
The machine-readable schema is [`remote-worker.schema.json`](remote-worker.schema.json).

## Routing

| `backend` | What happens |
|---|---|
| `auto` (default) | 1. local CUDA/MPS GPU via torch → 2. configured remote worker → 3. CPU numpy, **only when no remote is configured** |
| `numpy` | local CPU, never contacts the remote |
| `torch` (+ `device`) | local torch device, never contacts the remote |
| `remote` | the configured worker; error if `remote_url` is empty |

If a remote is configured and chosen but fails (refused, timeout, 401, bad response), the call fails with an
explicit error. There is **no silent CPU fallback**, so a result never looks like a GPU run when it wasn't.

Each result separates:

* `routing`: what this process decided and why: `{"mode": "local" | "remote" | "cpu-fallback", "requested", "reason", "endpoint"?}`
* `execution`: where it ran. Local runs come from this process. Remote runs are
  `{"source": "remote worker (self-reported; not verified by this machine)", "endpoint", "reported": {...device block...}}`.
  The device block is whatever the worker says; the client checks its shape but can't verify the hardware.

## Endpoints

All requests carry `Authorization: Bearer <token>`. All bodies are JSON (`Content-Type: application/json`).

| Method | Path | Request | Success response |
|---|---|---|---|
| GET | `/v1/health` | none | `{"ok": true, "protocol": "weightscope-remote/1", "device": {...}}` |
| POST | `/v1/diff` | `{"protocol", "a": REF, "b": REF, "params": {top_k, top_layers, tolerance, rank_by, device?}}` | `{"ok": true, "protocol", "result": DIFF, "device": {...}}` |
| POST | `/v1/summary` | `{"protocol", "checkpoint": REF, "params": {max_list, hash_files}}` | `{"ok": true, "protocol", "result": SUMMARY, "device": {...}}` |

Errors: any HTTP status with `{"ok": false, "protocol": "weightscope-remote/1", "error": "<message>"}` (a 200 with
`ok: false` also counts as an error).

`result` is the same object the local tool returns, minus `ok`, `tool`, and for diffs `execution`,
`interpretation` and `limitations`. The client adds back its own interpretation note and limitations, so a
worker can't change them.

`device` keys (all optional except `backend`, `device`; unknown keys are rejected): `backend`, `device`,
`device_name`, `compute_dtype`, `accumulate_dtype`, `torch_version`, `cuda_version`, `numpy_version`, `worker`,
`chunk_elems`.

### Checkpoint references (REF)

* `{"path": "/abs/path/on/worker"}`: a file or directory **already on the worker**, inside its approved roots.
  In the plugin, any A/B string is sent as a `path` when the route is remote. It is a worker path and is
  never read locally.
* `{"ref": "<opaque id>"}`: an object-store reference (pass `ref:<id>` as the path). This is reserved: the
  protocol carries it, but the reference worker rejects it.

### Why there is no upload

Checkpoints are usually tens to hundreds of GB, and unreleased weights are sensitive. Uploading them per call
would be slow, expensive and a privacy risk, and would make the worker a general file sink. So the protocol
only passes references to data the worker operator has already placed in an approved location. Upload is
deliberately not implemented.

## Client validation (what the plugin enforces)

* URL: `https` only. Plain `http` works only for `localhost` / `127.0.0.1` / `::1` **and** only with
  `remote_allow_http_localhost: true` (or CLI `--remote-allow-http-localhost`). Credentials, query strings and
  fragments in the URL are rejected. Redirects are refused, so the token is never forwarded to another host.
* Token: read by **name** (`remote_token_env`, default `WEIGHTSCOPE_REMOTE_TOKEN`) through Hermes's
  `agent.secret_scope.get_secret` when it's available, else `os.environ`. It is never logged, never put in
  results, and redacted from every error message, including echoed server bodies. The token value never
  goes in config.
* Timeout `remote_timeout_s` (default 120 s, clamped 1–3600). Response size ≤ `remote_max_response_mib`
  (default 4 MiB), checked against `Content-Length` and while reading.
* Retries: only `GET /v1/health`, at most 3 attempts with short backoff, and none after a 401/403, malformed
  JSON or a validation failure. POSTs are never retried.
* Strict response validation (`weightscope/remote_schema.py`): exact protocol string, no unknown keys, typed
  fields, finite numbers in range (cosine ∈ [−1, 1], fractions ∈ [0, 1], norms ≥ 0, NaN/Infinity rejected),
  list caps (top lists ≤ 100, capped item lists ≤ 200, warnings ≤ 50, shapes ≤ 16 dims), string length caps,
  consistent totals (`changed + identical == compared`), and SHA-256 values that are 64 lowercase hex chars.
* The model-facing tools have **no** `remote_url` argument. The endpoint comes only from settings or CLI flags,
  so a prompt can't redirect the token to another host.

## Reference worker (`worker/remote_worker.py`)

Standard library HTTP server plus the plugin's own analysis code (numpy, or torch if installed on the worker).

* `--root DIR` (required, repeatable): approved roots. Paths must be absolute, and they are resolved
  (symlinks and `..`) before the check, so a symlink inside a root that points outside it is rejected.
* Token from the env var named by `--token-env` (default `WEIGHTSCOPE_WORKER_TOKEN`), ≥ 16 characters,
  compared in constant time on every endpoint, including health.
* Request body ≤ 64 KiB. Errors are returned as messages without tracebacks. The log shows request lines
  only, never headers.
* Binds to `127.0.0.1` by default. A non-loopback bind needs `--tls-cert/--tls-key` or `--behind-tls-proxy`
  (for platforms that terminate HTTPS in front of the process).

## Setting up a real endpoint (future; nothing here has been deployed)

On the GPU machine (e.g. a pod with CUDA and torch already installed):

```bash
git clone https://github.com/dukemawex/hermes-plugin-weightscope && cd hermes-plugin-weightscope/weightscope
pip install "numpy>=1.26,<3"                               # torch: use the image's CUDA build
export WEIGHTSCOPE_WORKER_TOKEN="$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')"
python3 worker/remote_worker.py --root /workspace/models --host 0.0.0.0 --port 8765 \
    --backend torch --device cuda --behind-tls-proxy           # HTTPS terminated by the platform proxy
# or, serving TLS itself:
python3 worker/remote_worker.py --root /workspace/models --host 0.0.0.0 --port 8765 \
    --tls-cert cert.pem --tls-key key.pem
```

Put the checkpoints under the approved root (e.g. `/workspace/models/base`, `/workspace/models/candidate`).

On the Hermes side:

```yaml
# ~/.hermes/config.yaml
plugins:
  entries:
    weightscope:
      settings:
        remote_url: https://<pod-id>-8765.proxy.runpod.net   # any HTTPS URL that reaches the worker
        remote_token_env: WEIGHTSCOPE_REMOTE_TOKEN
```

```bash
export WEIGHTSCOPE_REMOTE_TOKEN=<same value as WEIGHTSCOPE_WORKER_TOKEN>   # or store it as a Hermes secret
hermes weightscope remote-health
hermes weightscope diff /workspace/models/base /workspace/models/candidate --backend remote
```

A/B are **worker** paths. `remote-health` prints the worker-reported device. Treat it as the worker's claim.
