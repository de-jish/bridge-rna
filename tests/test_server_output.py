"""The browser scripts' server pipe must be drained, or the server blocks.

`server_output.drain_server_output` exists because a dev server writing
request logs into a pipe nobody reads stops at about 64 KB. These tests run
a stand-in server that writes far more than a pipe holds and assert that it
exits only when drained, and that the drain keeps what is worth reading.
"""

from __future__ import annotations

import subprocess
import sys

import server_output

# A real Werkzeug request line, in the format app.py's dev server writes.
REQUEST = ('127.0.0.1 - - [21/Oct/2026 22:07:17] '
           '"POST /_dash-update-component HTTP/1.1" 200 -')

# Writes ~400 KB of request lines - several times a pipe's capacity - then
# one line that is not a request, then exits.
FLOOD = f"""
import sys
for _ in range(5000):
    sys.stdout.write({REQUEST!r} + "\\n")
sys.stdout.write("Traceback: something worth seeing\\n")
sys.stdout.flush()
"""


def _start_flood() -> subprocess.Popen:
    return subprocess.Popen([sys.executable, "-c", FLOOD],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True)


def test_an_undrained_pipe_blocks_the_server():
    """The defect itself: unread, the writer cannot finish."""
    proc = _start_flood()
    try:
        try:
            proc.wait(timeout=3)
            blocked = False
        except subprocess.TimeoutExpired:
            blocked = True
        assert blocked, "the flood fit in the pipe; raise its size"
    finally:
        proc.kill()
        proc.communicate()


def test_a_drained_server_runs_to_completion_and_keeps_its_errors(capsys):
    proc = _start_flood()
    thread = server_output.drain_server_output(proc)
    assert proc.wait(timeout=30) == 0
    thread.join(timeout=30)
    assert not thread.is_alive()
    out = capsys.readouterr().out
    assert out == "    [server] Traceback: something worth seeing\n"


def test_request_lines_are_recognised_and_nothing_else_is():
    assert server_output.is_request_log(REQUEST)
    assert server_output.is_request_log(
        '127.0.0.1 - - [21/Oct/2026 22:07:17] "GET /map HTTP/1.1" 200 -')
    # Werkzeug colors by status; these are the exact bytes it writes for a
    # cached asset and a missing one.
    assert server_output.is_request_log(
        '127.0.0.1 - - [21/Oct/2026 22:49:56] '
        '"\x1b[36mGET /assets/00-fonts.css?m=1 HTTP/1.1\x1b[0m" 304 -')
    assert server_output.is_request_log(
        '127.0.0.1 - - [21/Oct/2026 22:49:56] '
        '"\x1b[33mGET /nope HTTP/1.1\x1b[0m" 404 -')
    assert not server_output.is_request_log(
        "[bridge-rna] serving on http://127.0.0.1:8000")
    assert not server_output.is_request_log(
        "\x1b[31m\x1b[1mWARNING: This is a development server.\x1b[0m")
    assert not server_output.is_request_log(
        'Traceback (most recent call last): "POST" failed')
