"""Optional HTTP basic auth for a deployment that is reachable from the network.

The app exposes two endpoints that are cheap to call and expensive to serve: a
file upload that accepts up to 200 MB and spawns a torch subprocess loading a
522 MB checkpoint, and an LLM summary that proxies to whatever provider is
configured. On a public URL and with no credential in front of them, the first
is a denial-of-service primitive and the second is an open proxy.

This guard is off unless ``BRIDGE_RNA_BASIC_AUTH`` is set, so a loopback
development run is unaffected and the test suite sees no change. It is applied
in ``wsgi.py`` rather than in ``build_app`` for that reason: the credential
belongs to a deployment, not to the application.
"""

from __future__ import annotations

import binascii
import base64
import hmac
import os

# Fly's health check has no credential to present, so one path is exempt. It
# reveals nothing: it returns a fixed string and touches no artifact.
HEALTH_PATH = "/healthz"

_REALM = "Bridge RNA"


def parse_credential(raw: str) -> tuple[str, str] | None:
    """Split a ``user:password`` setting, or return None if it is unusable.

    A blank user or a blank password is treated as unset rather than as a
    credential that anyone can guess, because a guard that accepts an empty
    password is worse than a documented absence of one.
    """
    if ":" not in raw:
        return None
    user, _, password = raw.partition(":")
    if not user or not password:
        return None
    return user, password


def credential_from_env(env: dict[str, str] | None = None) -> tuple[str, str] | None:
    """The configured credential, or None when the guard should stay off."""
    source = os.environ if env is None else env
    return parse_credential(str(source.get("BRIDGE_RNA_BASIC_AUTH", "")).strip())


def _decode_header(header: str) -> tuple[str, str] | None:
    """Username and password out of an ``Authorization: Basic ...`` header."""
    scheme, _, payload = header.partition(" ")
    if scheme.lower() != "basic" or not payload:
        return None
    try:
        decoded = base64.b64decode(payload, validate=True).decode("utf-8")
    except (binascii.Error, UnicodeDecodeError, ValueError):
        return None
    if ":" not in decoded:
        return None
    user, _, password = decoded.partition(":")
    return user, password


def check(header: str | None, expected: tuple[str, str]) -> bool:
    """Whether an Authorization header matches the expected credential.

    Both halves are compared with ``hmac.compare_digest`` and both comparisons
    always run, so the answer does not leak which half was wrong through timing.
    """
    if not header:
        return False
    presented = _decode_header(header)
    if presented is None:
        return False
    user_ok = hmac.compare_digest(presented[0], expected[0])
    password_ok = hmac.compare_digest(presented[1], expected[1])
    return user_ok and password_ok


def install_basic_auth(server, credential: tuple[str, str] | None = None) -> bool:
    """Guard every request on `server` behind basic auth. Returns whether it is on.

    `server` is the Flask application underneath Dash. With no credential
    configured this is a no-op and returns False, which is what a local run
    wants; the caller is responsible for deciding whether that is acceptable
    for the interface it is about to bind to.
    """
    from flask import Response, request

    expected = credential if credential is not None else credential_from_env()
    if expected is None:
        return False

    @server.before_request
    def _require_basic_auth():  # pragma: no cover - exercised via test client
        if request.path == HEALTH_PATH:
            return None
        if check(request.headers.get("Authorization"), expected):
            return None
        return Response(
            "Authentication required.\n",
            401,
            {"WWW-Authenticate": f'Basic realm="{_REALM}", charset="UTF-8"'},
        )

    return True
