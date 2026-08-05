"""Where the browser checks point: a local subprocess, or a deployment.

The three e2e scripts all did the same thing - start `app.py`, wait for it to
announce itself, drive it at 127.0.0.1. That is the right default, but it makes
"the app works" and "the deployed app works" the same claim, and they are not:
the deployment adds gunicorn, a container, a proxy, and a basic-auth guard, and
any of those can break a page the local run renders perfectly.

So the target is a parameter. With no flags this behaves exactly as before.
With `--base-url` it checks an app that is already running and starts nothing,
and with `--http-auth` it presents credentials to one behind the guard.
"""

from __future__ import annotations

import contextlib
import subprocess
import time


def add_target_args(ap) -> None:
    """Add --base-url and --http-auth to an e2e script's parser."""
    ap.add_argument(
        "--base-url",
        help="Check an already-running app at this URL instead of starting one. "
             "This is how a deployment is verified: the same checks, against "
             "the hosted URL rather than a local subprocess.")
    ap.add_argument(
        "--http-auth", metavar="USER:PASSWORD",
        help="HTTP basic credentials, for a deployment behind the "
             "BRIDGE_RNA_BASIC_AUTH guard.")


def credentials(args) -> dict[str, str] | None:
    """Playwright `http_credentials` for the target, or None when it is open."""
    if not getattr(args, "http_auth", None):
        return None
    user, _, password = args.http_auth.partition(":")
    return {"username": user, "password": password}


@contextlib.contextmanager
def target(args, py: str, repo, announce: str = "serving on", timeout: int = 120):
    """Yield the base URL to drive, starting a local app only if needed.

    Raises RuntimeError if a locally started server never announces itself,
    which is a clearer failure than a browser timing out against a dead port.
    """
    base_url = getattr(args, "base_url", None)
    if base_url:
        base = base_url.rstrip("/")
        print(f"    [server] checking the app already running at {base}")
        yield base
        return

    base = f"http://127.0.0.1:{args.port}"
    server = subprocess.Popen(
        [py, "app.py", "--port", str(args.port)], cwd=repo,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        # Wait for the server to announce itself rather than sleeping blind.
        t0 = time.time()
        while time.time() - t0 < timeout:
            line = server.stdout.readline()
            if not line:
                break
            print("    [server] " + line.rstrip(), flush=True)
            if announce in line:
                break
        else:
            raise RuntimeError("server never announced itself")
        yield base
    finally:
        server.terminate()
        try:
            server.wait(timeout=15)
        except subprocess.TimeoutExpired:
            server.kill()
