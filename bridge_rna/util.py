"""Small shared helpers.

These sit at the bottom of the import graph: every other module uses them and
this one imports nothing from the package.
"""

from __future__ import annotations

import re
from typing import Any

import pandas as pd


def _find_first_existing(paths: list) -> Any:
    for p in paths:
        if p.exists():
            return p
    return None


def _safe_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value).strip()


def _last_nonempty_line(text: str) -> str:
    """Return the last non-empty line of a subprocess error blob.

    For a Python traceback this is the actual exception line
    (e.g. "RuntimeError: Requested OSDR sample not found") rather than the
    surrounding stack frames, which keeps the UI status message clean.
    """
    for line in reversed(_safe_str(text).splitlines()):
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


class RetrievalError(RuntimeError):
    """Retrieval failure carrying a clean message plus the full raw detail.

    ``str(err)`` is the short one-line message shown in the status banner;
    ``err.detail`` holds the full traceback/output for the collapsible panel.
    """

    def __init__(self, message: str, detail: str = "") -> None:
        super().__init__(message)
        self.detail = detail or message
def _first_non_empty(row: pd.Series, candidates: list[str]) -> str:
    for col in candidates:
        if col in row.index:
            value = _safe_str(row[col])
            if value:
                return value
    return ""


def _extract_gse(value: str) -> str:
    m = re.search(r"(GSE\d+)", value or "", flags=re.IGNORECASE)
    return m.group(1).upper() if m else ""
def _format_count(value: int | None) -> str:
    return f"{value:,}" if isinstance(value, int) else "—"
