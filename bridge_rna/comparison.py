"""Exact paired cosine evidence and optional source-attributed study context."""
from pathlib import Path
import sqlite3

import numpy as np

from .retrieval import _topk_cosine_matrix


def paired_rankings(index_vecs, query_a, query_b, depth=250):
    """Score one fixed reference universe for both arms in a single index pass.

    Retaining two float32 score rows costs ~7.5 MB on the shipped corpus. Exact
    ranks are available even outside the other arm's prefix. Ties break by index.
    This is the existing spherical-centroid cosine, with no difference estimator.
    """
    q = np.asarray([query_a, query_b], dtype=np.float32)
    if q.ndim != 2 or not np.isfinite(q).all() or np.any(np.linalg.norm(q, axis=1) == 0):
        raise ValueError("Comparison queries must be finite, nonzero vectors.")
    if int(depth) < 1 or len(index_vecs) == 0:
        raise ValueError("Comparison evidence needs a positive depth and a nonempty reference.")
    scores = np.empty((2, len(index_vecs)), dtype=np.float32)
    _topk_cosine_matrix(index_vecs, q, min(depth, len(index_vecs)), score_sink=scores)
    if not np.isfinite(scores).all():
        raise ValueError("Nonfinite reference scores cannot be ranked.")
    order = np.argsort(-scores, axis=1, kind="stable")
    ranks = np.empty_like(order)
    for arm in range(2):
        ranks[arm, order[arm]] = np.arange(1, len(index_vecs) + 1)
    candidates = np.union1d(order[0, :depth], order[1, :depth])
    rows = [dict(archs4_index=int(i), rank_a=int(ranks[0, i]), rank_b=int(ranks[1, i]),
                 score_a=float(scores[0, i]), score_b=float(scores[1, i])) for i in candidates]
    rows.sort(key=lambda r: (min(r["rank_a"], r["rank_b"]), r["rank_a"], r["archs4_index"]))
    return dict(rows=rows, order_a=order[0, :depth], order_b=order[1, :depth],
                scores=scores, reference_count=len(index_vecs))


def evidence_groups(rows, depth=30):
    groups = {}
    for row in rows:
        if min(row["rank_a"], row["rank_b"]) <= int(depth):
            groups.setdefault(row.get("gse") or "", []).append(row)
    return [dict(gse=gse, rows=members) for gse, members in sorted(
        groups.items(), key=lambda item: (min(min(r["rank_a"], r["rank_b"])
                                            for r in item[1]), item[0]))]


EVIDENCE_PAGE_SIZE = 30


def evidence_page(rows, depth, page):
    """Page the study-grouped evidence without dropping or recomputing candidates."""
    ordered = [r for g in evidence_groups(rows, depth) for r in g["rows"]]
    start = max(0, int(page)) * EVIDENCE_PAGE_SIZE
    return evidence_groups(ordered[start:start + EVIDENCE_PAGE_SIZE], depth), len(ordered)


def study_context(rows, path):
    """Optional local SQLite, always read-only; identity mismatch refuses context.

    Every sample remains one row; all study memberships are carried as provenance.
    An absent/mismatched sidecar never changes the search population or ranking.
    """
    empty = dict(available=False, reason="Local study context is not configured.",
                 studies={}, memberships={})
    if not path or not rows:
        return empty
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        return {**empty, "reason": "The configured study-context database is unavailable."}
    try:
        with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=5) as db:
            db.execute("PRAGMA query_only=ON")
            marks = ",".join("?" for _ in rows)
            keys = [r["gsm"] for r in rows]
            identities = dict(db.execute(f"SELECT gsm,global_index FROM samples WHERE gsm IN ({marks})", keys))
            if any(identities.get(r["gsm"]) != r["archs4_index"] for r in rows):
                return {**empty, "reason": "Study context refused: sample/index identity mismatch."}
            memberships = {}
            for gsm, gse in db.execute(f"SELECT gsm,gse FROM sample_studies WHERE gsm IN ({marks}) ORDER BY gse", keys):
                memberships.setdefault(gsm, []).append(gse)
            gses = sorted({g for values in memberships.values() for g in values}
                          | {r["gse"] for r in rows if r.get("gse")})
            studies = {}
            for start in range(0, len(gses), 500):
                batch = gses[start:start + 500]
                for gse, title, summary, design in db.execute(
                    "SELECT gse,title,summary,overall_design FROM studies WHERE gse IN ("
                    + ",".join("?" for _ in batch) + ")", batch):
                    studies[gse] = dict(title=title or "", summary=summary or "", design=design or "")
            return dict(available=True, reason="", studies=studies, memberships=memberships)
    except (sqlite3.Error, OSError, ValueError):
        return {**empty, "reason": "Study context could not be read; sample annotations remain available."}
