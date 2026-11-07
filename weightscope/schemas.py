"""Tool schemas (what the model reads)."""

_PATH_DESC = ("Path to a .safetensors file, a directory of .safetensors shards, or a directory / file "
              "with model.safetensors.index.json. Relative paths resolve against the session's working directory.")

CKPT_SUMMARY = {
    "name": "ckpt_summary",
    "description": (
        "Offline inventory of a local safetensors checkpoint: tensor names, dtypes, shapes, parameter "
        "counts per dtype, largest tensors, file sizes and SHA-256 hashes. Reads headers (and file bytes "
        "for hashing) only; no network, no model execution. Use before ckpt_diff or to identify a checkpoint."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": _PATH_DESC},
            "hash_files": {"type": "boolean", "description": "Compute SHA-256 of each file (default true; slow on very large files)."},
            "max_list": {"type": "integer", "description": "Max tensor names to list (default 50, max 200)."},
            "backend": {"type": "string", "enum": ["auto", "local", "remote"],
                        "description": "'remote' reads the checkpoint on the configured remote worker (path on the "
                                       "worker). Default auto follows the same routing as ckpt_diff. Usually omit."},
        },
        "required": ["path"],
    },
}

CKPT_DIFF = {
    "name": "ckpt_diff",
    "description": (
        "Offline numeric comparison of two local safetensors checkpoints (A = base, B = candidate), locally (CPU/GPU) or on a configured remote GPU worker. Streams "
        "tensors one at a time and reports, per tensor and per layer: L2 norm of B-A, change relative to "
        "A's norm, max abs difference, cosine similarity, fraction of elements changed above a tolerance; "
        "plus the top-k most-changed tensors/layers, tensors only in A or only in B, shape and dtype "
        "mismatches, and NaN/Inf counts (excluded from metrics). Results are numeric differences only: they "
        "do NOT establish a backdoor, behaviour change, safety status or tampering."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "a": {"type": "string", "description": "Base checkpoint. " + _PATH_DESC},
            "b": {"type": "string", "description": "Candidate checkpoint to compare against the base."},
            "top_k": {"type": "integer", "description": "Most-changed tensors to return (default 10, max 100)."},
            "top_layers": {"type": "integer", "description": "Most-changed layers to return (default 10, max 100)."},
            "tolerance": {"type": "number", "description": "Absolute per-element threshold for frac_changed (default 0 = any change)."},
            "rank_by": {"type": "string", "enum": ["relative", "l2", "max_abs"],
                        "description": "Ranking metric (default relative = ||B-A|| / ||A||)."},
            "backend": {"type": "string", "enum": ["auto", "numpy", "torch", "remote"],
                        "description": "Where to compute (default from settings, normally auto: local GPU, else the "
                                       "configured remote worker, else CPU numpy). With 'remote', a and b are paths "
                                       "on the WORKER (or 'ref:<id>'). Usually omit."},
            "device": {"type": "string", "description": "torch device such as cuda, cuda:1, mps or cpu (for 'remote', "
                                                        "the worker's device). Usually omit."},
        },
        "required": ["a", "b"],
    },
}
