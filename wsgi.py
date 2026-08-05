"""WSGI entry point for a hosted Bridge RNA.

`app.py` stays the local entry point and keeps using Flask's development
server. That server is single-threaded and explicitly not for anything shared,
so a deployment runs gunicorn against this module instead:

    gunicorn --workers 1 --threads 4 wsgi:application

One worker is deliberate rather than conservative. Peak resident memory was
measured at 1,852 MB on the real corpus with all six coordinate sets, the
tissue color-by and a cached retrieval live in one process; those are ordinary
heap pages that a second worker would not share. Concurrency comes from threads,
which is the right shape here because the work is numpy and file IO and both
release the GIL. See `docs/deployment.md`.
"""

from __future__ import annotations

import os
import sys

from app import build_app
from bridge_rna.auth import HEALTH_PATH, install_basic_auth

_dash_app = build_app()
application = _dash_app.server


@application.route(HEALTH_PATH)
def _healthz():
    """Liveness probe. Fixed string, touches no artifact, needs no credential."""
    return "ok\n", 200, {"Content-Type": "text/plain; charset=utf-8"}


_auth_on = install_basic_auth(application)

if not _auth_on:
    # Not fatal: a deployment may sit behind its own proxy or a private network.
    # But the upload endpoint spawns a torch subprocess for any caller, so an
    # unguarded public bind should be a decision rather than an oversight.
    print(
        "[bridge-rna] WARNING: BRIDGE_RNA_BASIC_AUTH is unset, so every endpoint "
        "is open, including the 200 MB file upload that spawns a torch "
        "subprocess. Set it unless something else is authenticating requests.",
        file=sys.stderr,
        flush=True,
    )
else:
    print("[bridge-rna] HTTP basic auth is enabled.", file=sys.stderr, flush=True)

print(
    f"[bridge-rna] WSGI application ready (pid {os.getpid()}).",
    file=sys.stderr,
    flush=True,
)
