"""Stratified sampling and viewport level-of-detail for the ARCHS4 background.

All 940k ARCHS4 points cannot be live WebGL glyphs at once (the plan caps live
glyphs near 100k), so the background is a stratified sample over the full
corpus, drawn atop a datashader-style density raster of all points. On zoom the
sample is recomputed over just the visible window, so fine structure appears
instead of the same sparse dots enlarging.
"""

from __future__ import annotations

import numpy as np


def stratified_archs4_sample(
    species: np.ndarray,
    n_target: int,
    seed: int = 0,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Indices into the ARCHS4 block, ~proportional across species.

    `species` is the per-ARCHS4 species_id array. `mask`, if given, restricts
    the candidate pool (e.g. to a viewport). Returns sorted indices.
    """
    n = len(species)
    pool = np.arange(n) if mask is None else np.where(mask)[0]
    if len(pool) <= n_target:
        return np.sort(pool)

    rng = np.random.default_rng(seed)
    sp = species[pool]
    out = []
    classes, class_counts = np.unique(sp, return_counts=True)
    total = len(pool)
    for cls, cnt in zip(classes, class_counts):
        take = max(1, int(round(n_target * cnt / total)))
        members = pool[sp == cls]
        take = min(take, len(members))
        out.append(rng.choice(members, size=take, replace=False))
    return np.sort(np.concatenate(out))


def viewport_mask(coords_xy: np.ndarray, bounds: tuple[float, float, float, float]) -> np.ndarray:
    """Boolean mask of points inside (xmin, xmax, ymin, ymax)."""
    xmin, xmax, ymin, ymax = bounds
    x, y = coords_xy[:, 0], coords_xy[:, 1]
    return (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)
