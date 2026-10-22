"""Pure data structures for a retrieved ARCHS4 neighborhood.

The neighborhood **is** the retrieved matches. Until 2026-10-22 it was a
separate, fixed top-250 prefix of the same scan, stored beside the hits: with
k free to reach 1,000 the map drew two overlapping sets of rings that disagreed
in number (250 teal circles, k white rings) and the explorer described 250
samples while the table listed k. Joshua asked for one set. The retrieval view
now stores only an availability marker, and the map builds this payload from
the stored hits, so the explorer, the map marks and the table count the same
rows by construction.
"""

from __future__ import annotations

from collections import Counter
import math
import re
from typing import Any

import numpy as np
import pandas as pd

from .util import _safe_str

ROW_TEXT_FIELDS = (
    "gsm", "gse", "title", "source_name", "characteristics",
    "tissue", "species",
)
STALE_EVIDENCE_REASON = (
    "This saved retrieval predates the neighborhood explorer. Run this "
    "retrieval again to explore its matches."
)
UNSUPPORTED_EVIDENCE_MODES = frozenset(("demo", "precomputed"))


def _index(value: Any) -> int | None:
    if isinstance(value, (bool, np.bool_)):
        return None
    if isinstance(value, str):
        text = value.strip()
        if not re.fullmatch(r"[+-]?\d+", text):
            return None
        try:
            return int(text)
        except (TypeError, ValueError, OverflowError):
            return None
    try:
        if pd.isna(value) or not math.isfinite(value):
            return None
        integer = int(value)
        return integer if value == integer else None
    except (TypeError, ValueError, OverflowError):
        return None


def _text(value: Any) -> str:
    """Normalize either NumPy or pandas scalar missing text to an empty string."""
    try:
        return "" if pd.isna(value) else _safe_str(value)
    except (TypeError, ValueError):
        return _safe_str(value)


def build_payload(frame: pd.DataFrame | None, *, label: str = "") -> dict:
    """Serialize ranked retrieval rows, every one of them, into a JSON-safe payload.

    Rank is the row's position, so `frame` must be in rank order, which stored
    hits are.
    """
    rows = []
    source = frame if frame is not None else pd.DataFrame()
    source = source.reset_index(drop=True)
    for rank, (_, raw) in enumerate(source.iterrows(), 1):
        row = {field: _text(raw.get(field)) for field in ROW_TEXT_FIELDS}
        score = pd.to_numeric(raw.get("score"), errors="coerce")
        row.update({
            "rank": rank,
            "score": None if pd.isna(score) or not np.isfinite(score) else float(score),
            "archs4_index": _index(raw.get("archs4_index")),
        })
        rows.append(row)
    return {
        "available": True,
        "reason": "",
        "label": _safe_str(label),
        "depth_requested": len(rows),
        "depth_returned": len(rows),
        "metric": "cosine",
        "space": "embedding-512d",
        "hits": rows,
    }


def marker_payload(label: str = "") -> dict:
    """What the retrieval view stores: the matches can be explored, and as what.

    No rows. The map builds them from the stored hits with `build_payload`, so
    the hits-store carries each match once rather than twice.
    """
    return {"available": True, "reason": "", "label": _safe_str(label),
            "from_hits": True}


def unavailable_payload(reason: str) -> dict:
    """Return the stable empty state when retrieval results are unavailable."""
    return {
        "available": False, "reason": _safe_str(reason), "label": "",
        "depth_requested": 0, "depth_returned": 0,
        "metric": "cosine", "space": "embedding-512d", "hits": [],
    }


def evidence_unavailable_reason(mode: str = "") -> str:
    """Explain whether another run can produce exact evidence for this mode."""
    normalized = _safe_str(mode).lower()
    if normalized in UNSUPPORTED_EVIDENCE_MODES:
        return (
            f"The {normalized} mode cannot place its matches on the map, so "
            "there is nothing to explore. Rerunning that same mode will not "
            "add it. "
            "Use a cached OSDR sample, uploaded counts, or cohort retrieval "
            "instead."
        )
    return STALE_EVIDENCE_REASON


def _hits(payload: dict | None) -> list[dict]:
    if not isinstance(payload, dict) or not payload.get("available"):
        return []
    return [hit for hit in payload.get("hits", []) if isinstance(hit, dict)]


def _categories(hits: list[dict], field: str) -> dict:
    values = [_safe_str(hit.get(field)) for hit in hits]
    counts = Counter(value for value in values if value)
    covered = sum(counts.values())
    items = [
        {"label": label, "count": count, "share": count / covered}
        for label, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ]
    return {"covered": covered, "items": items}


def _score_summary(hits: list[dict]) -> dict:
    scores = [hit.get("score") for hit in hits if hit.get("score") is not None]
    numeric = pd.to_numeric(scores, errors="coerce")
    values = np.asarray(numeric[~pd.isna(numeric)], dtype=float)
    if not len(values):
        return {"count": 0, "median": None, "minimum": None, "maximum": None}
    return {
        "count": int(len(values)),
        "median": float(np.median(values)),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
    }


def summarize(payload: dict | None) -> dict:
    """Summarize metadata coverage, studies, and cosine scores."""
    hits = _hits(payload)
    available = bool(isinstance(payload, dict) and payload.get("available"))
    tissue = _categories(hits, "tissue")
    species = _categories(hits, "species")
    groups = study_groups(payload)
    assigned_groups = [group for group in groups if group["gse"]]
    depth = len(hits)

    if tissue["covered"]:
        leading = tissue["items"][0]
        sentence = (
            f"{leading['label']} is the leading tissue: {leading['count']} of "
            f"{tissue['covered']} samples with tissue metadata.")
    else:
        sentence = (
            f"No tissue metadata is available for the returned depth of {depth} "
            "samples.")

    return {
        "available": available,
        "reason": _safe_str(payload.get("reason")) if isinstance(payload, dict) else "",
        "depth": depth,
        "tissue": tissue,
        "species": species,
        "study_count": len(assigned_groups),
        "top_three_study_samples": sum(
            group["sample_count"] for group in assigned_groups[:3]),
        "score": _score_summary(hits),
        "sentence": sentence,
    }


def study_groups(payload: dict | None) -> list[dict]:
    """Group returned rows by study, retaining blank GSE rows as one group."""
    groups: dict[str, list[dict]] = {}
    for hit in _hits(payload):
        groups.setdefault(_safe_str(hit.get("gse")), []).append(hit)

    summaries = []
    for gse, rows in groups.items():
        best_rank = min(int(row.get("rank", 0)) for row in rows)
        display_gse = gse or "No GSE recorded"
        tissue = _categories(rows, "tissue")
        leading_tissue = tissue["items"][0] if tissue["items"] else None
        title = next((_safe_str(row.get("title")) for row in rows
                      if _safe_str(row.get("title"))), "")
        summaries.append({
            "gse": gse,
            "display_gse": display_gse,
            "sample_count": len(rows),
            "best_rank": best_rank,
            "title": title,
            "dominant_tissue": (leading_tissue["label"]
                                if leading_tissue else ""),
            "dominant_tissue_count": (leading_tissue["count"]
                                      if leading_tissue else 0),
            "tissue_covered": tissue["covered"],
        })
    return sorted(
        summaries,
        key=lambda group: (-group["sample_count"], group["best_rank"],
                           group["display_gse"]),
    )


def sample_rows(payload: dict | None, query: str = "") -> list[dict]:
    """Return rank-ordered rows matching an optional case-insensitive query."""
    needle = _safe_str(query).lower()
    rows = []
    for hit in _hits(payload):
        haystack = " ".join(_safe_str(hit.get(field)).lower()
                             for field in ROW_TEXT_FIELDS)
        if not needle or needle in haystack:
            rows.append(dict(hit))
    return sorted(rows, key=lambda row: int(row.get("rank", 0)))
