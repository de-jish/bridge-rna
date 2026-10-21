"""Pairwise structure among retrieved hits: identical profiles and clustering.

Pure numpy. It opens no memmap and imports no Dash, so it is tested on small
hand-built matrices; `retrieval.reference_vectors` supplies the real vectors.
The serving app must not import scipy or sklearn (`test_app` pins that), which
is why average linkage is written out here rather than borrowed.
"""

from __future__ import annotations

from typing import NamedTuple

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


class Merge(NamedTuple):
    """One agglomeration step, in scipy's linkage convention.

    `left` and `right` are cluster ids: below n a leaf position, at or above n
    the cluster formed by merge `id - n`.
    """

    left: int
    right: int
    height: float
    size: int


def average_linkage(distance: np.ndarray) -> list[Merge]:
    """UPGMA over a symmetric distance matrix: n - 1 merges, lowest first.

    Ties break toward the lower pair of cluster ids, so the result is
    deterministic. O(n^3), which is nothing at the 50 rows the similarity plot
    draws.
    """
    d = np.array(distance, dtype=np.float64)
    n = d.shape[0]
    if n < 2:
        return []
    active = {i: (i, 1) for i in range(n)}  # slot -> (cluster id, size)
    np.fill_diagonal(d, np.inf)
    merges: list[Merge] = []
    for step in range(n - 1):
        slots = sorted(active)
        sub = d[np.ix_(slots, slots)]
        flat = int(np.argmin(sub))
        a_pos, b_pos = divmod(flat, len(slots))
        a, b = sorted((slots[a_pos], slots[b_pos]))
        (ida, na), (idb, nb) = active[a], active[b]
        height = float(max(d[a, b], 0.0))
        left, right = sorted((ida, idb))
        merges.append(Merge(left, right, height, na + nb))
        # The merged cluster takes slot a; its distances are size-weighted means.
        for s in slots:
            if s in (a, b):
                continue
            merged = (d[a, s] * na + d[b, s] * nb) / (na + nb)
            d[a, s] = d[s, a] = merged
        d[b, :] = d[:, b] = np.inf
        del active[b]
        active[a] = (n + step, na + nb)
    return merges


def leaf_order(merges: list[Merge], n: int) -> list[int]:
    """Leaf positions left to right, as a dendrogram of `merges` draws them."""
    if n == 0:
        return []
    if not merges:
        return list(range(n))
    children = {n + i: (m.left, m.right) for i, m in enumerate(merges)}
    order: list[int] = []
    stack = [n + len(merges) - 1]
    while stack:
        node = stack.pop()
        if node < n:
            order.append(node)
        else:
            left, right = children[node]
            stack.extend((right, left))
    return order


def dendrogram_segments(merges: list[Merge], n: int
                        ) -> list[tuple[float, float, float, float]]:
    """Line segments `(x0, y0, x1, y1)` for a dendrogram over `leaf_order`.

    x is the leaf's position in that order (0 to n - 1); y is merge height.
    Each merge contributes the two legs and the crossbar of one bracket.
    """
    order = leaf_order(merges, n)
    x = {leaf: float(i) for i, leaf in enumerate(order)}
    y = {leaf: 0.0 for leaf in range(n)}
    segments: list[tuple[float, float, float, float]] = []
    for i, m in enumerate(merges):
        node = n + i
        xl, xr = x[m.left], x[m.right]
        yl, yr = y[m.left], y[m.right]
        segments.append((xl, yl, xl, m.height))
        segments.append((xr, yr, xr, m.height))
        segments.append((xl, m.height, xr, m.height))
        x[node] = (xl + xr) / 2.0
        y[node] = m.height
    return segments
