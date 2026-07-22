"""The NASA OSDR side: the sample table and per-study summaries.

`load_osdr_samples` builds `sample_id` as "<accession>|<sample name>", which is
the same key `precompute/embed_osdr.py` writes as `sample_key`. That agreement
is what lets a retrieval and a point on the manifold refer to the same sample
without a translation table, and it is pinned by a test.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd

from .util import _safe_str

try:
    from osdr_metadata import get_study_summary
except Exception:  # the summary is enrichment, never a hard requirement
    get_study_summary = None

_OSDR_STUDY_SUMMARY_CACHE: dict[str, dict[str, str]] = {}


def _fetch_osdr_study_summary(study_id: str) -> dict[str, str]:
    """Fetch OSDR study summary via osdr_metadata.py with in-process caching."""
    sid = _safe_str(study_id)
    if not sid:
        return {}

    cached = _OSDR_STUDY_SUMMARY_CACHE.get(sid)
    if cached is not None:
        return cached

    if get_study_summary is None:
        _OSDR_STUDY_SUMMARY_CACHE[sid] = {}
        return {}

    try:
        summary = get_study_summary(sid)
        out = {
            "dataset_id": _safe_str(summary.get("dataset_id", sid)),
            "study_title": _safe_str(summary.get("study_title", "")),
            "study_description": _safe_str(summary.get("study_description", "")),
            "study_publication_title": _safe_str(summary.get("study_publication_title", "")),
            "study_protocol_description": _safe_str(summary.get("study_protocol_description", "")),
        }
        _OSDR_STUDY_SUMMARY_CACHE[sid] = out
        return out
    except Exception:
        _OSDR_STUDY_SUMMARY_CACHE[sid] = {}
        return {}
def load_osdr_samples(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")

    out = pd.DataFrame()
    out["sample_name"] = df.get("id.sample name", "")
    out["study_id"] = df.get("id.accession", "")
    out["tissue"] = df.get("study.characteristics.material type", "")
    out["condition"] = df.get("study.factor value.spaceflight", "")
    out["strain"] = df.get("study.characteristics.strain", "")
    out["sex"] = df.get("study.characteristics.sex", "")
    out["duration"] = df.get("study.parameter value.duration", "")
    out["counts_path"] = df.get("counts_path", "")
    out["sample_id"] = out["study_id"].astype(str) + "|" + out["sample_name"].astype(str)

    keep = out["sample_name"].astype(str).str.len() > 0
    out = out[keep].drop_duplicates(subset=["sample_id"]).reset_index(drop=True)
    return out
def _eligible_osdr_count(df: pd.DataFrame) -> int | None:
    """OSDR samples the app can actually retrieve.

    This used to count "mouse counts present + a spaceflight condition
    present", which is only the demo script's *first* filter and overstated the
    number by 55: those 55 pass that filter but their name matches no column in
    their counts matrix, so retrieval still fails. The header would then have
    said 2,163 were retrievable while the picker disabled 55 of them.

    The honest figure is the one `sample_tier` calls anything but unavailable,
    and it is the same 2,108 with the cache or without it - the cached and
    subprocess tiers cover exactly the same samples, differing only in speed -
    which is also the count of OSDR points on the map.
    """
    try:
        from .retrieval import TIER_UNAVAILABLE, sample_tier

        return int(sum(
            sample_tier(_safe_str(r["sample_id"]), _safe_str(r["sample_name"]),
                        _safe_str(r.get("counts_path")), _safe_str(r.get("condition")))
            != TIER_UNAVAILABLE
            for _, r in df.iterrows()))
    except Exception:
        return None
