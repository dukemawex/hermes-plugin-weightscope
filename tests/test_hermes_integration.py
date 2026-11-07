"""Integration tests against a real Hermes Agent v0.23.0 install (the venv this suite runs in).

The plugin is copied into an isolated HERMES_HOME, enabled through config.yaml and loaded by
Hermes's own PluginManager; tools are invoked through Hermes's tool registry, the slash command
through Hermes's plugin-command dispatcher, and the CLI through the real `hermes` entry point.
No model, provider key or network is used.
"""

import json
import os
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

import stfixtures as F

hermes_cli = pytest.importorskip("hermes_cli.plugins", reason="Hermes Agent is not installed in this venv")

ROOT = Path(__file__).resolve().parent.parent
PLUGIN_DIR = ROOT / "weightscope"
HERMES_BIN = Path(sys.executable).with_name("hermes")


def _make_home(tmp_path: Path) -> dict:
    home = tmp_path / "hermes-home"
    (home / "plugins").mkdir(parents=True)
    shutil.copytree(PLUGIN_DIR, home / "plugins" / "weightscope",
                    ignore=shutil.ignore_patterns("__pycache__"))
    (home / "config.yaml").write_text("plugins:\n  enabled:\n    - weightscope\n", encoding="utf-8")
    bundled = tmp_path / "bundled-plugins"
    bundled.mkdir()
    oshome = tmp_path / "os-home"
    oshome.mkdir()
    return {"HOME": str(oshome), "HERMES_HOME": str(home), "HERMES_BUNDLED_PLUGINS": str(bundled)}


@pytest.fixture
def ckpts(tmp_path):
    base = F.base_model()
    a = F.write(tmp_path / "ck" / "base.safetensors", base)
    b = F.write_sharded(tmp_path / "ck" / "finetuned", F.finetuned(base, layer=2), n_shards=2)
    return a, b


@pytest.fixture
def manager(tmp_path, monkeypatch):
    env = _make_home(tmp_path)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    from hermes_cli.plugins import PluginManager
    mgr = PluginManager()
    mgr.discover_and_load()
    yield mgr
    mgr.unload()


def test_plugin_loads_and_registers_through_plugin_manager(manager):
    loaded = manager._plugins["weightscope"]
    assert loaded.error is None, loaded.error
    assert loaded.enabled is True
    assert loaded.manifest.version == "0.1.0"
    assert sorted(loaded.tools_registered) == ["ckpt_diff", "ckpt_summary"]
    # provides_tools in plugin.yaml matches what register() actually registered
    import hermes_yaml as yaml
    manifest = yaml.safe_load((PLUGIN_DIR / "plugin.yaml").read_text())
    assert sorted(manifest["provides_tools"]) == sorted(loaded.tools_registered)
    assert "weightscope" in manager._cli_commands
    assert "weightscope" in manager._plugin_commands


def test_tools_dispatch_through_hermes_registry(manager, ckpts):
    from tools.registry import registry
    a, b = ckpts
    assert registry.get_toolset_for_tool("ckpt_diff") == "weightscope"
    schema = registry.get_schema("ckpt_diff")
    assert schema["parameters"]["required"] == ["a", "b"]

    raw = registry.dispatch("ckpt_diff", {"a": str(a), "b": str(b), "top_k": 5},
                            task_id="t-1", session_id="s-1", scope=manager.scope_key)
    assert isinstance(raw, str)
    res = json.loads(raw)
    assert res["ok"] is True
    assert res["b"]["layout"] == "sharded-index"
    assert res["top_layers"][0]["layer"] == "model.layers.2"
    assert res["top_tensors"][0]["name"].startswith("model.layers.2.")

    summ = json.loads(registry.dispatch("ckpt_summary", {"path": str(b)}, scope=manager.scope_key))
    assert summ["ok"] and summ["checkpoint"]["files"] == 2

    err = json.loads(registry.dispatch("ckpt_diff", {"a": str(a), "b": "/nonexistent/x"},
                                       scope=manager.scope_key))
    assert err["ok"] is False and "not found" in err["error"]


def test_slash_command_through_hermes_dispatch(manager, ckpts):
    from hermes_cli.plugin_command_dispatch import invoke_plugin_command
    a, b = ckpts
    handler = manager._plugin_commands["weightscope"]["handler"]
    out = invoke_plugin_command(handler, f"diff {a} {b} --top 3", session_id="s-1", platform="cli")
    assert "model.layers.2" in out and "does NOT establish" in out
    assert "Usage" in invoke_plugin_command(handler, "", platform="cli")
    assert "weightscope error" in invoke_plugin_command(handler, "summary /nonexistent.safetensors")


def _run_hermes(env_extra: dict, *args, cwd=None):
    env = {k: v for k, v in os.environ.items() if not k.startswith("HERMES_")}
    env.update(env_extra)
    return subprocess.run([str(HERMES_BIN), *args], capture_output=True, text=True, env=env, cwd=cwd,
                          timeout=120)


@pytest.mark.skipif(not HERMES_BIN.exists(), reason="hermes entry point not in this venv")
def test_cli_command_end_to_end(tmp_path, ckpts):
    env = _make_home(tmp_path)
    a, b = ckpts
    p = _run_hermes(env, "weightscope", "diff", str(a), str(b), "--json", "--top", "3")
    assert p.returncode == 0, p.stderr[-2000:]
    res = json.loads(p.stdout)
    assert res["top_layers"][0]["layer"] == "model.layers.2"

    p = _run_hermes(env, "weightscope", "summary", "base.safetensors", "--no-hash", cwd=a.parent)
    assert p.returncode == 0, p.stderr[-2000:]
    assert "Tensors: 19" in p.stdout

    p = _run_hermes(env, "weightscope", "diff", str(a), str(tmp_path / "missing"))
    assert p.returncode == 1 and "path not found" in p.stderr

    # the CLI diff above was recorded in the plugin's own data dir via plugins.plugin_storage.plugin_db
    db = Path(env["HERMES_HOME"]) / "plugin-data" / "weightscope" / "data.db"
    assert db.is_file()
    rows = sqlite3.connect(db).execute("SELECT surface, top_tensor FROM diffs").fetchall()
    assert rows and rows[0][0] == "cli" and rows[0][1].startswith("model.layers.2.")
    p = _run_hermes(env, "weightscope", "recent", "--json")
    assert p.returncode == 0 and json.loads(p.stdout)[0]["surface"] == "cli"


@pytest.mark.skipif(not HERMES_BIN.exists(), reason="hermes entry point not in this venv")
def test_plugin_not_loaded_when_disabled(tmp_path, ckpts):
    env = _make_home(tmp_path)
    Path(env["HERMES_HOME"], "config.yaml").write_text("plugins:\n  enabled: []\n", encoding="utf-8")
    p = _run_hermes(env, "weightscope", "summary", str(ckpts[0]))
    assert p.returncode != 0  # opt-in: no command without enabling


def test_plugin_settings_from_hermes_config_apply(tmp_path, monkeypatch, ckpts):
    """Limits and history toggles are read through ctx.get_config (plugins.entries.weightscope.settings)."""
    env = _make_home(tmp_path)
    Path(env["HERMES_HOME"], "config.yaml").write_text(
        "plugins:\n  enabled:\n    - weightscope\n  entries:\n    weightscope:\n      settings:\n"
        "        max_tensors: 3\n        record_history: false\n", encoding="utf-8")
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    from hermes_cli.plugins import PluginManager
    from tools.registry import registry
    mgr = PluginManager()
    mgr.discover_and_load()
    try:
        a, b = ckpts
        res = json.loads(registry.dispatch("ckpt_diff", {"a": str(a), "b": str(b)}, scope=mgr.scope_key))
        assert res["ok"] is False and "exceeds limit 3" in res["error"]
        p = _run_hermes(env, "weightscope", "diff", str(a), str(a))
        assert p.returncode == 1 and "exceeds limit 3" in p.stderr
    finally:
        mgr.unload()
    # a successful diff with record_history: false leaves no history DB behind
    Path(env["HERMES_HOME"], "config.yaml").write_text(
        "plugins:\n  enabled:\n    - weightscope\n  entries:\n    weightscope:\n      settings:\n"
        "        record_history: false\n", encoding="utf-8")
    p = _run_hermes(env, "weightscope", "diff", str(a), str(b))
    assert p.returncode == 0, p.stderr[-2000:]
    assert not (Path(env["HERMES_HOME"]) / "plugin-data" / "weightscope" / "data.db").exists()


@pytest.mark.skipif(not HERMES_BIN.exists(), reason="hermes entry point not in this venv")
def test_cli_backend_flags(tmp_path, ckpts):
    env = _make_home(tmp_path)
    a, b = ckpts
    p = _run_hermes(env, "weightscope", "diff", str(a), str(b), "--json", "--backend", "numpy")
    assert p.returncode == 0, p.stderr[-2000:]
    assert json.loads(p.stdout)["execution"]["backend"] == "numpy"
    p = _run_hermes(env, "weightscope", "diff", str(a), str(b), "--backend", "numpy", "--device", "cuda")
    assert p.returncode == 1 and "cpu only" in p.stderr
    try:
        import torch
    except Exception:
        torch = None
    if torch is None:
        p = _run_hermes(env, "weightscope", "diff", str(a), str(b), "--backend", "torch")
        assert p.returncode == 1 and "torch not importable" in p.stderr
        return
    p = _run_hermes(env, "weightscope", "diff", str(a), str(b), "--json", "--backend", "torch", "--device", "cpu")
    assert p.returncode == 0, p.stderr[-2000:]
    comp = json.loads(p.stdout)["execution"]
    assert comp["backend"] == "torch" and comp["device"] == "cpu" and comp["compute_dtype"] == "float64"
    if not torch.cuda.is_available():
        p = _run_hermes(env, "weightscope", "diff", str(a), str(b), "--backend", "torch", "--device", "cuda:0")
        assert p.returncode == 1 and "CUDA is not available" in p.stderr


def test_remote_route_through_hermes_registry_and_cli(tmp_path, monkeypatch):
    """Remote settings from Hermes config, tool dispatched by Hermes's registry, worker = in-process
    reference worker on 127.0.0.1 (CPU numpy). Proves the wiring/protocol only, not GPU execution."""
    from remote_fixture import Running, load_worker_module
    root = tmp_path / "worker-root"
    base = F.base_model()
    a = F.write(root / "base.safetensors", base)
    b = F.write_sharded(root / "ft", F.finetuned(base, layer=2), n_shards=2)
    token = "itest-" + os.urandom(16).hex()
    worker = Running(load_worker_module().build_server("127.0.0.1", 0, [str(root)], token, backend="numpy"))
    env = _make_home(tmp_path)
    Path(env["HERMES_HOME"], "config.yaml").write_text(
        "plugins:\n  enabled:\n    - weightscope\n  entries:\n    weightscope:\n      settings:\n"
        f"        remote_url: {worker.url}\n        remote_token_env: WS_ITEST_TOKEN\n"
        "        remote_allow_http_localhost: true\n        record_history: false\n", encoding="utf-8")
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("WS_ITEST_TOKEN", token)
    from hermes_cli.plugins import PluginManager
    from tools.registry import registry
    mgr = PluginManager()
    mgr.discover_and_load()
    try:
        raw = registry.dispatch("ckpt_diff", {"a": str(a), "b": str(b), "backend": "remote"}, scope=mgr.scope_key)
        assert token not in raw
        res = json.loads(raw)
        assert res["ok"], res
        assert res["routing"]["mode"] == "remote" and res["top_layers"][0]["layer"] == "model.layers.2"
        assert res["execution"]["reported"]["backend"] == "numpy"
    finally:
        mgr.unload()
    env["WS_ITEST_TOKEN"] = token
    if HERMES_BIN.exists():
        p = _run_hermes(env, "weightscope", "diff", str(a), str(b), "--json", "--backend", "remote")
        assert p.returncode == 0, p.stderr[-2000:]
        assert json.loads(p.stdout)["routing"]["mode"] == "remote"
        assert token not in p.stdout + p.stderr
        # explicit CLI flags override config; plain http still needs the localhost flag
        p = _run_hermes(env, "weightscope", "remote-health", "--remote-url", worker.url,
                        "--remote-token-env", "WS_ITEST_TOKEN", "--remote-allow-http-localhost")
        assert p.returncode == 0, p.stderr[-2000:]
        assert "worker-reported" in p.stdout and token not in p.stdout + p.stderr
        env["WS_ITEST_TOKEN"] = "wrong-token-000000000000"
        p = _run_hermes(env, "weightscope", "diff", str(a), str(b), "--backend", "remote")
        assert p.returncode == 1 and "rejected the token" in p.stderr
    worker.close()
