"""In-process HTTP fixtures on 127.0.0.1 for remote-routing CONTRACT tests.

These servers prove the client/protocol behaviour only (routing, validation, auth, limits). They run on
this machine's CPU; they do NOT demonstrate real GPU or RunPod execution.
"""

from __future__ import annotations

import importlib.util
import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_worker_module():
    spec = importlib.util.spec_from_file_location("ws_remote_worker", ROOT / "weightscope" / "worker" / "remote_worker.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Running:
    def __init__(self, srv):
        self.srv = srv
        self.thread = threading.Thread(target=srv.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{srv.server_address[1]}"

    def close(self):
        self.srv.shutdown()
        self.srv.server_close()


class Scripted:
    """A fake worker whose responses are scripted per test. Records every request (method, path, headers)."""

    def __init__(self):
        self.requests = []
        self.responses = []  # list of (status, body_bytes, headers dict, delay_s); last one repeats
        outer = self

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.0"

            def log_message(self, *a):
                pass

            def _handle(self):
                n = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(n) if n else b""
                outer.requests.append((self.command, self.path, dict(self.headers), body))
                idx = min(len(outer.requests) - 1, len(outer.responses) - 1)
                status, payload, headers, delay = outer.responses[idx]
                if delay:
                    time.sleep(delay)
                if callable(payload):
                    payload = payload(self)
                self.send_response(status)
                for k, v in (headers or {}).items():
                    if not k.startswith("_"):
                        self.send_header(k, v)
                if "Content-Length" not in (headers or {}) and not (headers or {}).get("_no_length"):
                    self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                try:
                    self.wfile.write(payload)
                except (BrokenPipeError, ConnectionResetError):
                    pass

            do_GET = do_POST = _handle

        self.running = Running(ThreadingHTTPServer(("127.0.0.1", 0), H))
        self.url = self.running.url

    def respond(self, status=200, body=b"{}", headers=None, delay=0.0):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode()
        elif isinstance(body, str):
            body = body.encode()
        self.responses.append((status, body, headers, delay))
        return self

    def close(self):
        self.running.close()


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port
