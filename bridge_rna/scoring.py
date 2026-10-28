"""What a match score means: plain cosine, or cosine after whitening.

The only module that knows what whitening is. Every scan in
`bridge_rna/retrieval.py` takes a `scoring` key from here and asks this module
for the query-side operator; nothing else learns the transform.

**Whitening** maps an L2-normalised embedding `x` to `W'(x - mu)`, where `mu`
is the ARCHS4 mean and `W = V diag(1 / sqrt(max(lambda, floor * lambda_max)))`
from the eigendecomposition of the ARCHS4 covariance. The whitened score is
the cosine between two whitened vectors. It was chosen by measurement
(docs/matching-benchmark.md): at a floor of 1e-4 it raised per-tissue
retrieval accuracy from 0.644 to 0.693 within ARCHS4 and from 0.956 to 0.979
for OSDR queries, and lowered no tissue by more than 1.5 points.

**It costs what a cosine scan costs.** With `u = unit(W'(q - mu))` and
`v = W u` per query, and `r_x = |W'(x - mu)|` stored once per reference row,

    score(x) = (x - mu) . v / r_x

so a scan stays one pass over the memmap: centre the block, one product,
divide by the stored norms. The block is centred *before* the product because
the normalised embeddings sit close to their mean, and `x . v - mu . v` in
float32 would lose most of the digits that carry the ranking.

The fitted transform lives in `cache/whitening.npz`, written by
`precompute/build_whitening.py`. It is optional: without it, or if it does not
describe the index on disk, whitening is reported unavailable and every search
scores by plain cosine.

Design: docs/whitened-matching.md.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np

COSINE = "cosine"
WHITENED = "whitened"
SCORINGS = (WHITENED, COSINE)

LABELS = {WHITENED: "Whitened", COSINE: "Plain cosine"}

# What the score is called wherever a number is shown. Defined in
# `results.py`, which must stay pure stdlib, and re-exported here.
from .results import SCORE_NAMES, score_name  # noqa: E402

# The eigenvalue floor, relative to the largest eigenvalue. Measured, not
# chosen: below 1e-4 accuracy plateaus, above it the gain shrinks (1e-2, the
# value PR #3 used, gave up half of it).
FLOOR = 1e-4

# Whether whitening is the default when it is available. Joshua's rule
# (2026-10-28): only once `precompute/validate_cohorts.py --scoring whitened`
# passes. It passed on 2026-10-28, all six checks over all 204 cohorts:
# leave-one-out top-5 0.799 (0.747 under cosine), +0.275 over the
# structure-free null, +0.041 over the within-study null, knee at k = 5.
# Machines without cache/whitening.npz still default to plain cosine.
WHITENED_BY_DEFAULT = True

BUILD_HINT = "Run precompute/build_whitening.py to enable whitened matching."


@dataclass(frozen=True)
class Whitening:
    mu: np.ndarray          # (d,) float32, the ARCHS4 mean of normalised rows
    W: np.ndarray           # (d, d) float64, the whitening map
    row_norms: np.ndarray   # (n,) float32, |W'(x - mu)| per reference row
    meta: dict[str, Any]


@lru_cache(maxsize=1)
def load_whitening() -> Whitening | None:
    """The fitted transform, or None when it is absent or describes another index.

    Cached for the life of the process, like the memmap itself. A transform
    whose row count or dimension does not match the index on disk is refused
    outright: its stored norms would divide one sample's score by another's.
    """
    from manifold import paths as mpaths

    path = mpaths.WHITENING_NPZ
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as z:
            mu = z["mu"].astype(np.float32)
            W = z["W"].astype(np.float64)
            row_norms = z["row_norms"].astype(np.float32)
            meta = json.loads(str(z["meta"]))
    except Exception as exc:  # noqa: BLE001 - a broken artifact must not break search
        print(f"[scoring] ignoring {path}: {exc}", file=sys.stderr, flush=True)
        return None

    try:
        from .retrieval import _load_archs4_index

        vecs, _, d = _load_archs4_index()
        n = int(vecs.shape[0])
    except Exception:  # noqa: BLE001 - no index means nothing to whiten against
        return None
    # Every check runs inside the guard: a malformed optional artifact must
    # fall back to plain cosine, never stop the view from loading.
    try:
        _check(mu, W, row_norms, meta, vecs, n, d)
    except Exception as exc:  # noqa: BLE001
        print(f"[scoring] ignoring {path}: {exc} Re-run precompute/build_whitening.py.",
              file=sys.stderr, flush=True)
        return None
    return Whitening(mu=mu, W=W, row_norms=row_norms, meta=meta)


def _check(mu, W, row_norms, meta, vecs, n: int, d: int) -> None:
    if not isinstance(meta, dict):
        raise ValueError("its record is not a mapping.")
    if mu.shape != (d,) or W.shape != (d, d) or row_norms.shape != (n,):
        raise ValueError(f"shapes {mu.shape}, {W.shape}, {row_norms.shape} do not fit "
                         f"an index of {n} rows of dimension {d}.")
    if int(meta.get("index_bytes", -1)) != int(vecs.nbytes):
        raise ValueError("it was built against an index of another size.")
    if meta.get("index_fingerprint") != index_fingerprint(vecs):
        raise ValueError("it was built against a different embedding memmap.")
    if not (np.isfinite(mu).all() and np.isfinite(W).all()):
        raise ValueError("its mean or matrix is not finite.")
    if not (np.isfinite(row_norms).all() and (row_norms > 0).all()):
        raise ValueError("it has non-finite or zero row norms.")


FINGERPRINT_ROWS = 4096


def index_fingerprint(vecs) -> str:
    """A content identity for the embedding memmap, cheap enough to check at load.

    Sampled, so it catches a replaced, rebuilt or reordered index - the ways an
    index actually changes - but not an edit confined to rows it skips. The
    complete check is the whole-file SHA-256 recorded as `index_sha256`, which
    `deploy/preflight.py` verifies on every apply; hashing 963 MB at every
    process start is the cost this split avoids.

    The stored row norms are positional, so a transform built for one index and
    applied to another that merely has the same shape would divide each row's
    score by some other row's norm. Hashing every byte of the 963 MB memmap at
    each process start would cost seconds; hashing the shape and 4,096 rows at
    evenly spaced positions costs milliseconds and changes if the rows are
    replaced, reordered, or the index is rebuilt.
    """
    import hashlib

    n, d = int(vecs.shape[0]), int(vecs.shape[1])
    h = hashlib.sha256(f"{n}x{d}:{vecs.dtype}".encode())
    rows = np.unique(np.linspace(0, n - 1, min(FINGERPRINT_ROWS, n)).astype(np.int64))
    h.update(np.ascontiguousarray(vecs[rows]).tobytes())
    return h.hexdigest()


def transform_id(w: "Whitening | None" = None) -> str:
    """A short name for one fitted transform, recorded with every whitened result."""
    w = w if w is not None else load_whitening()
    return str((w.meta or {}).get("transform_id", "")) if w is not None else ""


def available() -> bool:
    return load_whitening() is not None


def default_scoring() -> str:
    return WHITENED if WHITENED_BY_DEFAULT and available() else COSINE


def resolve(value: Any) -> str:
    """The scoring a search actually runs: what was asked for, if it can be.

    An unknown value, or whitening on a machine without the transform, falls
    back to plain cosine rather than failing the search; the payload records
    what ran, so the interface never claims whitening it did not do.
    """
    value = str(value or "").strip().lower()
    if value == WHITENED and available():
        return WHITENED
    if value == COSINE:
        return COSINE
    return default_scoring() if value not in SCORINGS else COSINE


def query_operator(Q: np.ndarray, w: Whitening) -> np.ndarray:
    """`(d, m)` float32 columns `v_j = W unit(W'(q_j - mu))` for unit queries `Q`."""
    q = np.asarray(Q, dtype=np.float64)
    if q.ndim == 1:
        q = q.reshape(1, -1)
    q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    z = (q - w.mu.astype(np.float64)) @ w.W                 # (m, d) whitened queries
    z /= np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-12)
    return np.ascontiguousarray((w.W @ z.T).astype(np.float32))


def control_options() -> list[dict]:
    """The Matching control's options. Whitened is shown disabled, not hidden,
    when the transform is not on this machine, with the command that builds it."""
    ok = available()
    return [
        {"label": LABELS[WHITENED] if ok else f"{LABELS[WHITENED]} (unavailable)",
         "value": WHITENED, "disabled": not ok},
        {"label": LABELS[COSINE], "value": COSINE},
    ]
