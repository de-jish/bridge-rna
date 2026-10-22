"""Keep reading a dev server's output for as long as it runs.

The browser scripts start `app.py` with its output on a pipe and read that
pipe only until the "serving on" line. The dev server logs every request to
it, and a pipe holds about 64 KB, so a few hundred callbacks into a run the
server blocked writing a log line and every request after that hung - a page
that never loaded, a search that timed out - always late in a run.

`drain_server_output` is called right after the start-up announcement and
reads the rest on a daemon thread. Request lines are dropped; anything else,
a traceback included, is still printed with the `    [server] ` prefix.
Werkzeug colors a request line by its status - a 304 arrives as
`"\\x1b[36mGET ...\\x1b[0m" 304` - so the match runs on the line with its
color codes stripped, or every cached asset would still be printed.
"""

from __future__ import annotations

import re
import subprocess
import threading

REQUEST_LOG = re.compile(r'"(GET|POST|HEAD) [^"]*" \d{3}')
ANSI_COLOR = re.compile(r"\x1b\[[\d;]*m")


def is_request_log(line: str) -> bool:
    """Whether `line` is one of the dev server's per-request log lines."""
    return bool(REQUEST_LOG.search(ANSI_COLOR.sub("", line)))


def drain_server_output(server: subprocess.Popen) -> threading.Thread:
    """Start a daemon thread that empties `server.stdout` until it closes."""
    def drain():
        for line in server.stdout:
            if not is_request_log(line):
                print("    [server] " + line.rstrip(), flush=True)

    thread = threading.Thread(target=drain, daemon=True)
    thread.start()
    return thread
