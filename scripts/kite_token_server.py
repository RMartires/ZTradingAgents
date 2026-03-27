#!/usr/bin/env python3
"""Local helper to exchange Kite Connect `request_token` for `access_token`.

Zerodha redirects here after login. Register this exact URL in your Kite app:
  http://127.0.0.1:8765/kite/callback

Usage:
  export KITE_API_KEY=... KITE_API_SECRET=...
  python scripts/kite_token_server.py

Then open:
  https://kite.zerodha.com/connect/login?v=3&api_key=YOUR_API_KEY

After login, the browser shows JSON with access_token — copy into KITE_ACCESS_TOKEN.

If Kite returns Invalid `checksum`: formula is correct per docs; use the API secret from the
same app as KITE_API_KEY, ensure no quotes/spaces in .env, and the same api_key as in the login URL.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from time import sleep
from urllib.parse import parse_qs, quote, unquote, urlparse

import requests

KITE_LOGIN_URL = "https://kite.zerodha.com/connect/login"
KITE_SESSION_URL = "https://api.kite.trade/session/token"


def _load_env_file(path: str) -> None:
    """Set env vars from a simple KEY=VALUE .env file when unset or empty."""
    if not os.path.isfile(path):
        return
    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            if line.startswith("export "):
                line = line[7:].strip()
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip("'").strip('"')
            # Do not skip when shell has `export KITE_API_SECRET=` (empty); .env should win.
            if key and not (os.environ.get(key) or "").strip():
                os.environ[key] = value


def _checksum(api_key: str, request_token: str, api_secret: str) -> str:
    # Kite docs: SHA-256 hex of (api_key + request_token + api_secret), no separators.
    # Strip each part so stray newlines from .env don't break the checksum.
    combined = (
        api_key.strip()
        + request_token.strip()
        + api_secret.strip()
    )
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()


def _log_checksum_validation(
    api_key: str, request_token: str, api_secret: str, checksum_hex: str
) -> None:
    """Log checksum and a shell one-liner to compare against the server (stderr)."""
    k = api_key.strip()
    t = request_token.strip()
    s = api_secret.strip()
    sys.stderr.write("\n--- kite_token_server: checksum validation ---\n")
    sys.stderr.write(f"checksum (sha256 hex): {checksum_hex}\n")
    sys.stderr.write(
        f"lengths after Python .strip(): api_key={len(k)} request_token={len(t)} "
        f"api_secret={len(s)} combined={len(k) + len(t) + len(s)}\n"
    )
    combined = k + t + s
    quoted = shlex.quote(combined)
    sys.stderr.write(
        "copy-paste (exact concatenation Python used; leaks secrets in your terminal scrollback):\n"
        f"  CHECKSUM=$(printf '%s' {quoted} | shasum -a 256 | awk '{{print $1}}') && echo \"$CHECKSUM\"\n"
    )
    if os.environ.get("KITE_OAUTH_DEBUG", "").strip().lower() in ("1", "true", "yes"):
        sys.stderr.write(
            "KITE_OAUTH_DEBUG: Python repr (secrets exposed — disable after use):\n"
            f"  api_key={k!r}\n  request_token={t!r}\n  api_secret={s!r}\n"
        )
    sys.stderr.write("---\n\n")


def exchange_request_token(api_key: str, api_secret: str, request_token: str) -> dict:
    """Exchange tokens. Prefer official kiteconnect client so the HTTP request matches Zerodha's SDK."""
    k = api_key.strip()
    t = request_token.strip()
    s = api_secret.strip()
    checksum_hex = _checksum(k, t, s)
    # _log_checksum_validation(api_key, request_token, api_secret, checksum_hex)

    try:
        from kiteconnect import KiteConnect
    except ImportError:
        KiteConnect = None  # type: ignore[misc, assignment]

    if KiteConnect is not None:
        try:
            session_data = KiteConnect(api_key=k).generate_session(t, s)
            return {"status": "success", "data": session_data}
        except Exception as e:
            msg = str(e)
            et = type(e).__name__
            err: dict = {"status": "error", "message": msg, "error_type": et, "data": None}
            code = getattr(e, "code", None)
            if code is not None:
                err["code"] = code
            if "checksum" in msg.lower():
                sys.stderr.write(
                    "hint: If your local checksum matches but Kite rejects: (1) request_token is "
                    "**single-use** — do not refresh the callback URL; do a fresh login. "
                    "(2) Confirm api_secret in .env is the one shown for this api_key in the "
                    "Kite Connect developer console.\n"
                )
            return err

    body = {"api_key": k, "request_token": t, "checksum": checksum_hex}
    r = requests.post(
        KITE_SESSION_URL,
        headers={"X-Kite-Version": "3"},
        data=body,
        timeout=30,
    )
    try:
        payload = r.json()
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Kite returned non-JSON (HTTP {r.status_code}): {r.text[:500]}") from e
    return payload


def make_handler(api_key: str, api_secret: str, callback_path: str):
    callback_path = callback_path.rstrip("/") or "/kite/callback"

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args) -> None:  # noqa: A003
            sys.stderr.write("%s - %s\n" % (self.address_string(), format % args))

        def _send(self, code: int, body: bytes, content_type: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            path = parsed.path.rstrip("/") or "/"

            if path == "/":
                login = f"{KITE_LOGIN_URL}?v=3&api_key={quote(api_key, safe='')}"
                html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Kite token helper</title></head>
<body>
  <p>Open the Kite login page, finish login, then you will land on the callback with JSON.</p>
  <p><a href="{login}">Log in with Zerodha (Kite Connect)</a></p>
  <p>Callback URL (must match your app settings): <code>http://127.0.0.1:{self.server.server_port}{callback_path}</code></p>
</body></html>"""
                self._send(200, html.encode("utf-8"), "text/html; charset=utf-8")
                return

            norm_callback = callback_path.rstrip("/") or "/kite/callback"
            if path != norm_callback:
                self._send(
                    404,
                    json.dumps({"error": "not_found", "path": path}).encode("utf-8"),
                    "application/json",
                )
                return

            qs = parse_qs(parsed.query)
            status = (qs.get("status") or [""])[0]
            request_token = (qs.get("request_token") or [""])[0]
            request_token = unquote(request_token).strip()

            if status != "success":
                err = (qs.get("message") or ["login_failed"])[0]
                body = json.dumps(
                    {
                        "error": "kite_login_not_successful",
                        "status": status or None,
                        "message": err,
                    },
                    indent=2,
                ).encode("utf-8")
                self._send(400, body, "application/json")
                return

            if not request_token:
                body = json.dumps({"error": "missing_request_token"}).encode("utf-8")
                self._send(400, body, "application/json")
                return

            try:
                payload = exchange_request_token(api_key, api_secret, request_token)
            except (requests.RequestException, RuntimeError) as e:
                body = json.dumps({"error": "kite_request_failed", "message": str(e)}).encode("utf-8")
                self._send(502, body, "application/json")
                return

            if payload.get("status") != "success":
                body = json.dumps(
                    {
                        "error": "kite_session_error",
                        "kite": payload,
                    },
                    indent=2,
                ).encode("utf-8")
                code = 401 if payload.get("error_type") == "TokenException" else 400
                self._send(code, body, "application/json")
                return

            data = payload.get("data") or {}
            access_token = data.get("access_token")
            out = {
                "access_token": access_token,
                "note": "Set KITE_ACCESS_TOKEN in your .env (and restart your app).",
            }
            self._send(200, json.dumps(out, indent=2).encode("utf-8"), "application/json")

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description="Kite Connect request_token → access_token helper.")
    parser.add_argument(
        "--host",
        default=os.getenv("KITE_OAUTH_HOST", "127.0.0.1"),
        help="Bind address (default 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("KITE_OAUTH_PORT", "8765")),
        help="Port (default 8765)",
    )
    parser.add_argument(
        "--callback-path",
        default=os.getenv("KITE_OAUTH_CALLBACK_PATH", "/kite/callback"),
        help="Path Zerodha redirects to (default /kite/callback)",
    )
    parser.add_argument(
        "--env-file",
        default=os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"),
        help="Load env vars from this file if not already set",
    )
    args = parser.parse_args()

    _load_env_file(args.env_file)

    api_key = os.getenv("KITE_API_KEY", "").strip()
    api_secret = os.getenv("KITE_API_SECRET", "").strip()
    if not api_key or not api_secret:
        print(
            "Set KITE_API_KEY and KITE_API_SECRET (e.g. in .env), then re-run.",
            file=sys.stderr,
        )
        sys.exit(1)

    handler = make_handler(api_key, api_secret, args.callback_path)
    server = HTTPServer((args.host, args.port), handler)
    cb = args.callback_path if args.callback_path.startswith("/") else f"/{args.callback_path}"
    print(f"Serving http://{args.host}:{args.port}/  — callback: http://{args.host}:{args.port}{cb}", file=sys.stderr)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.", file=sys.stderr)


if __name__ == "__main__":
    main()
