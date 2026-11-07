"""Remote GPU worker client (protocol weightscope-remote/1). Standard library only (urllib).

Network calls happen ONLY when a remote endpoint is configured and the remote route is chosen.
Security properties:
* HTTPS required; plain http only to localhost/127.0.0.1/::1 and only with remote_allow_http_localhost.
* The bearer token is read by NAME (Hermes secret scope, else os.environ), sent only to the configured
  endpoint, never logged, and redacted from every error message. Redirects are refused (a redirect could
  forward the Authorization header to another host).
* Bounded timeout, bounded response size, strict response validation (remote_schema.py).
* Only idempotent GET /v1/health is retried (bounded); POSTs are never retried.
"""

from __future__ import annotations

import json
import logging
import re
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass

from .remote_schema import PROTOCOL, validate_envelope
from .st_reader import MiB, WeightScopeError

logger = logging.getLogger(__name__)

DEFAULT_TOKEN_ENV = "WEIGHTSCOPE_REMOTE_TOKEN"
_ENV_NAME = re.compile(r"^[A-Z_][A-Z0-9_]{0,127}$")
_LOCALHOSTS = {"localhost", "127.0.0.1", "::1"}
HEALTH_ATTEMPTS = 3
MAX_REF_LEN = 4096


@dataclass(frozen=True)
class RemoteSettings:
    url: str
    token_env: str = DEFAULT_TOKEN_ENV
    timeout_s: float = 120.0
    max_response_bytes: int = 4 * MiB
    allow_http_localhost: bool = False


def redact(text: str, token: str | None) -> str:
    text = str(text)
    if token:
        text = text.replace(token, "[REDACTED]")
    return text


def validate_url(url: str, allow_http_localhost: bool) -> tuple[str, str]:
    """Return (base_url_without_trailing_slash, display_endpoint) or raise."""
    if not isinstance(url, str) or not url.strip():
        raise WeightScopeError("remote_url is empty")
    parts = urllib.parse.urlsplit(url.strip())
    if parts.username or parts.password:
        raise WeightScopeError("remote_url must not contain credentials; use remote_token_env")
    if parts.query or parts.fragment:
        raise WeightScopeError("remote_url must not contain a query string or fragment")
    host = (parts.hostname or "").lower()
    if not host:
        raise WeightScopeError("remote_url has no host")
    if parts.scheme == "https":
        pass
    elif parts.scheme == "http":
        if host not in _LOCALHOSTS:
            raise WeightScopeError("remote_url must use https (http is allowed only for localhost)")
        if not allow_http_localhost:
            raise WeightScopeError("http remote_url refused: set remote_allow_http_localhost for a localhost test worker")
    else:
        raise WeightScopeError("remote_url must use https")
    base = urllib.parse.urlunsplit((parts.scheme, parts.netloc, parts.path.rstrip("/"), "", ""))
    return base, base


def read_token(name: str) -> str:
    """Read the bearer token by env-var NAME via Hermes's secret scope (else os.environ)."""
    if not isinstance(name, str) or not _ENV_NAME.match(name):
        raise WeightScopeError("remote_token_env must be an UPPER_SNAKE environment variable name")
    value = None
    try:
        from agent.secret_scope import get_secret  # Hermes route (profile-aware), when running inside Hermes
        value = get_secret(name)
    except ImportError:
        import os
        value = os.environ.get(name)
    except Exception as exc:  # e.g. UnscopedSecretError under multiplexing
        raise WeightScopeError(f"cannot read remote token {name}: {type(exc).__name__}") from None
    if not value:
        raise WeightScopeError(f"remote token variable {name} is not set")
    if any(c in value for c in "\r\n\x00") or len(value) > 4096:
        raise WeightScopeError(f"remote token in {name} is not a valid header value")
    return value


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise urllib.error.HTTPError(req.full_url, code, "redirect refused by weightscope", headers, fp)


def checkpoint_ref(value) -> dict:
    """'ref:<id>' -> object reference; anything else -> a path ON THE WORKER (never read locally)."""
    if not isinstance(value, str) or not value.strip() or "\x00" in value or len(value) > MAX_REF_LEN:
        raise WeightScopeError("remote checkpoint reference must be a non-empty string")
    value = value.strip()
    if value.startswith("ref:"):
        return {"ref": value[4:]}
    return {"path": value}


class RemoteClient:
    def __init__(self, settings: RemoteSettings):
        self.settings = settings
        self.base, self.endpoint = validate_url(settings.url, settings.allow_http_localhost)
        self._token = read_token(settings.token_env)
        self._timeout = max(1.0, min(float(settings.timeout_s), 3600.0))
        self._max = max(1024, int(settings.max_response_bytes))
        self._opener = urllib.request.build_opener(_NoRedirect())

    def _err(self, msg: str) -> WeightScopeError:
        return WeightScopeError(redact(msg, self._token))

    def _request(self, method: str, path: str, body: dict | None):
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(self.base + path, data=data, method=method)
        req.add_header("Authorization", f"Bearer {self._token}")
        req.add_header("Accept", "application/json")
        req.add_header("User-Agent", "weightscope/0.1")
        if data is not None:
            req.add_header("Content-Type", "application/json")
        try:
            with self._opener.open(req, timeout=self._timeout) as resp:
                length = resp.headers.get("Content-Length")
                if length and length.isdigit() and int(length) > self._max:
                    raise self._err(f"remote response too large ({length} bytes > {self._max})")
                raw = resp.read(self._max + 1)
        except WeightScopeError:
            raise
        except urllib.error.HTTPError as exc:
            if exc.code in (401, 403):
                raise self._err(f"remote worker at {self.endpoint} rejected the token (HTTP {exc.code})") from None
            if 300 <= exc.code < 400:
                raise self._err(f"remote worker at {self.endpoint} tried to redirect (HTTP {exc.code}); refused") from None
            snippet = ""
            try:
                snippet = exc.read(300).decode("utf-8", "replace").strip()
            except Exception:
                pass
            raise self._err(f"remote worker at {self.endpoint} returned HTTP {exc.code}"
                            + (f": {snippet[:200]}" if snippet else "")) from None
        except (TimeoutError, socket.timeout):
            raise self._err(f"remote worker at {self.endpoint} timed out after {self._timeout:g}s") from None
        except urllib.error.URLError as exc:
            reason = exc.reason
            if isinstance(reason, (TimeoutError, socket.timeout)):
                raise self._err(f"remote worker at {self.endpoint} timed out after {self._timeout:g}s") from None
            if isinstance(reason, ConnectionRefusedError):
                raise self._err(f"cannot connect to remote worker at {self.endpoint}: connection refused") from None
            raise self._err(f"cannot reach remote worker at {self.endpoint}: {reason}") from None
        except (ConnectionError, OSError) as exc:
            raise self._err(f"cannot reach remote worker at {self.endpoint}: {type(exc).__name__}") from None
        if len(raw) > self._max:
            raise self._err(f"remote response exceeds {self._max} bytes")
        try:
            return json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise self._err("remote worker returned malformed JSON") from None

    def _checked(self, obj, kind):
        try:
            env = validate_envelope(obj, kind)
        except WeightScopeError as exc:
            raise self._err(str(exc)) from None
        if env["ok"] is False:
            raise self._err(f"remote worker error: {env['error'][:300]}")
        return env

    def health(self) -> dict:
        last = None
        for attempt in range(HEALTH_ATTEMPTS):  # idempotent: small bounded retry
            try:
                return self._checked(self._request("GET", "/v1/health", None), "health")
            except WeightScopeError as exc:
                last = exc
                msg = str(exc)
                if "rejected the token" in msg or "validation" in msg or "malformed" in msg:
                    break
                time.sleep(0.2 * (attempt + 1))
        raise last

    def diff(self, a: str, b: str, params: dict) -> dict:
        body = {"protocol": PROTOCOL, "a": checkpoint_ref(a), "b": checkpoint_ref(b), "params": params}
        return self._checked(self._request("POST", "/v1/diff", body), "diff")

    def summary(self, path: str, params: dict) -> dict:
        body = {"protocol": PROTOCOL, "checkpoint": checkpoint_ref(path), "params": params}
        return self._checked(self._request("POST", "/v1/summary", body), "summary")
