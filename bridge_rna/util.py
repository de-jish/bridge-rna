"""Small shared helpers.

These sit at the bottom of the import graph: every other module uses them and
this one imports nothing from the package.
"""

from __future__ import annotations

import re
from pathlib import Path
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


def counts_separator(path: str | Path) -> str:
    """Tab for a `.tsv` or `.txt` counts file, gzipped or not; comma otherwise."""
    suffixes = {x.lower() for x in Path(path).suffixes}
    return "\t" if suffixes & {".tsv", ".txt"} else ","


def text_counts_reason(column: str, values: pd.Series) -> str:
    """One plain sentence if a sample column holds text where counts belong, else "".

    Empty cells are not text and are left to the preprocessing. True and False
    are words, although pandas reads them as booleans that would pass as 1 and 0.
    Shared by the upload loader and `precompute/embed_upload.py`, so the two can
    never describe the same bad file differently.
    """
    if pd.api.types.is_bool_dtype(values):
        text = values
    else:
        text = values[values.notna() & pd.to_numeric(values, errors="coerce").isna()]
    if not len(text):
        return ""
    # Quoted on one short line: the banner shows the reason's first line.
    example = " ".join(str(text.iloc[0]).split())
    if len(example) > 40:
        example = example[:40] + "..."
    return (f"Column '{column}' holds text where counts were expected (for example "
            f"'{example}'). Expected mouse Ensembl gene IDs in the first column "
            "and numeric counts in each sample column.")


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
    return f"{value:,}" if isinstance(value, int) else "Unavailable"
