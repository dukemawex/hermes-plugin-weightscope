"""WeightScope: offline-by-default safetensors checkpoint summary and diff for Hermes Agent.

Registers two tools (ckpt_summary, ckpt_diff), the `hermes weightscope` CLI command
and the `/weightscope` slash command. Numpy on CPU by default; optional torch GPU backend
and optional remote worker. The only other network use is `weightscope demo --fetch`, which
downloads pinned public Hugging Face checkpoints after explicit consent. No model execution.
"""

from __future__ import annotations

import logging

from . import schemas
from .cli import make_cli_handler, make_slash_handler, setup_argparse
from .tools import make_handlers

logger = logging.getLogger(__name__)

TOOLSET = "weightscope"


def register(ctx):
    def get_setting(key, default=None):
        return ctx.get_config(key, default=default)

    def get_cwd():
        return ctx.session_cwd()

    ckpt_summary, ckpt_diff = make_handlers(get_setting, get_cwd, surface="tool")
    ctx.register_tool(name="ckpt_summary", toolset=TOOLSET, schema=schemas.CKPT_SUMMARY,
                      handler=ckpt_summary, emoji="🔬")
    ctx.register_tool(name="ckpt_diff", toolset=TOOLSET, schema=schemas.CKPT_DIFF,
                      handler=ckpt_diff, emoji="🔬")
    ctx.register_cli_command(name="weightscope", help="Safetensors checkpoint summary/diff (offline by default)",
                             setup_fn=setup_argparse, handler_fn=make_cli_handler(get_setting),
                             description="Compare two local safetensors checkpoints tensor by tensor; `demo --fetch` runs the published-model walkthrough.")
    ctx.register_command("weightscope", handler=make_slash_handler(get_setting, get_cwd),
                         description="Summarize or diff local safetensors checkpoints (offline by default)",
                         args_hint="summary <path> | diff <base> <candidate> | recent | demo [--fetch]")
