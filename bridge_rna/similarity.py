"""Identical profiles among retrieved hits: one profile under two GSM accessions.

Pure numpy. It opens no memmap and imports no Dash, so it is tested on small
hand-built matrices; `retrieval.reference_vectors` supplies the real vectors.
"""

from __future__ import annotations

import numpy as np

# One expression profile deposited under two GSM accessions. Measured over the
# top 1,000 hits of nine queries on 2026-10-21: every one of 305 hit pairs below
# this cosine distance sat in two *different* GEO series (re-deposits,
# "[re-analysis]" copies, one animal filed under two tissue labels), while all
# 14 pairs between 1e-6 and 1e-5 were biological replicates inside one series.
# A 0.9999 cosine threshold would have been wrong by four orders of magnitude:
# one query had 13,810 hit pairs above it. See docs/retrieval-results.md.
IDENTICAL_PROFILE_DISTANCE = 1e-6


def cosine_matrix(unit: np.ndarray) -> np.ndarray:
    """Pairwise cosine of unit-norm rows, symmetric, with an exact unit diagonal."""
    unit = np.asarray(unit, dtype=np.float64)
    if unit.size == 0:
        return np.zeros((0, 0), dtype=np.float64)
    cos = np.clip(unit @ unit.T, -1.0, 1.0)
    cos = (cos + cos.T) / 2.0
    np.fill_diagonal(cos, 1.0)
    return cos


def identical_groups(unit: np.ndarray,
                     distance: float = IDENTICAL_PROFILE_DISTANCE) -> list[list[int]]:
    """Positions whose profiles are identical, as groups of two or more.

    Groups are connected components of the "closer than `distance`" relation,
    each sorted ascending, ordered by their first position. Callers pass rows
    in rank order, so a group's first position is its best-ranked accession.
    """
    cos = cosine_matrix(unit)
    n = cos.shape[0]
    if n < 2:
        return []
    close = (1.0 - cos) < float(distance)
    np.fill_diagonal(close, False)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j in zip(*np.nonzero(np.triu(close, 1))):
        a, b = find(int(i)), find(int(j))
        if a != b:
            parent[max(a, b)] = min(a, b)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return sorted((g for g in groups.values() if len(g) > 1), key=lambda g: g[0])
