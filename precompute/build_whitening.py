#!/usr/bin/env python3
"""Fit the whitening transform on the ARCHS4 embedding and write cache/whitening.npz.

    .venv/bin/python precompute/build_whitening.py [--floor 1e-4] [--check-queries 8]

Two passes over the 963 MB memmap, about a minute each: the first accumulates
the mean and covariance of the L2-normalised rows in float64, the second
stores each row's whitened norm `|W'(x - mu)|`, normalised exactly as the
app's scan normalises (float32, `x / (|x| + 1e-12)`), so the stored norm and
the scan's numerator are computed from the same vector.

Then it checks itself, because the app ranks by a reformulation rather than by
the transform: for real OSDR queries it scores every reference row both ways -
the scan's `(x - mu) . v / r_x` in float32 and an explicit float64
`cos(W'(x - mu), W'(q - mu))` - and fails unless the scores agree within
`--tolerance` and the top 100 agree exactly, in order. The record it writes
says what it fitted on and what it measured.

Design: docs/whitened-matching.md. Measurement that chose it:
docs/matching-benchmark.md.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from bridge_rna import retrieval as R  # noqa: E402
from bridge_rna import scoring  # noqa: E402
from manifold import paths  # noqa: E402

BLOCK = 50_000


def unit32(x: np.ndarray) -> np.ndarray:
    """The scan's own normalisation, operation for operation."""
    x = np.asarray(x, dtype=np.float32)
    x /= (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)
    return x


def fit(vecs) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n, d = vecs.shape
    total = np.zeros(d)
    outer = np.zeros((d, d))
    for start in range(0, n, BLOCK):
        x = unit32(vecs[start:start + BLOCK]).astype(np.float64)
        total += x.sum(axis=0)
        outer += x.T @ x
    mu = total / n
    cov = outer / n - np.outer(mu, mu)
    lam, V = np.linalg.eigh(cov)
    order = np.argsort(lam)[::-1]
    return mu, lam[order], V[:, order]


def row_norms(vecs, mu32: np.ndarray, W: np.ndarray) -> np.ndarray:
    """`|W'(x - mu)|` from exactly the centred vector the scan multiplies.

    The scan normalises in float32 and subtracts `mu` in float32
    (`retrieval._topk_cosine_matrix`); doing the subtraction in float64 here
    would hand each row a norm for a slightly different vector than its
    numerator. Only the product and the norm are taken in float64.
    """
    n = vecs.shape[0]
    out = np.empty(n, dtype=np.float32)
    for start in range(0, n, BLOCK):
        xc = unit32(vecs[start:start + BLOCK])
        xc -= mu32
        out[start:start + len(xc)] = np.linalg.norm(xc.astype(np.float64) @ W, axis=1)
    return out


def check(vecs, w: scoring.Whitening, queries: np.ndarray, tol: float) -> dict:
    """The app's own scan against the explicit float64 transform, every row.

    Runs `retrieval._topk_cosine_matrix` itself - singly, as a sample search
    does, and as one padded matrix product, as a cohort search does - so what
    is checked is the arithmetic that ranks, not a reimplementation of it.

    The contract checked is the one plain cosine has always had: the scan's
    float32 arithmetic is the definition of the ranking. Against float64 the
    scores agree to `tol` and the top 100 are identical; deeper near-ties can
    order differently in either scoring, which is rounding, not disagreement.
    """
    n = vecs.shape[0]
    q32 = unit32(queries)
    zq = (q32.astype(np.float64) - w.mu.astype(np.float64)) @ w.W
    zq /= np.linalg.norm(zq, axis=1, keepdims=True)
    exact = np.empty((len(queries), n), dtype=np.float64)
    for start in range(0, n, BLOCK):
        xc = unit32(vecs[start:start + BLOCK])
        xc -= w.mu
        zx = xc.astype(np.float64) @ w.W
        zx /= np.linalg.norm(zx, axis=1, keepdims=True)
        exact[:, start:start + len(xc)] = zq @ zx.T

    err, top_same, padded_same = 0.0, 0, 0
    for j in range(len(queries)):
        idx, score = R._topk_cosine_from_memmap(vecs, queries[j], 100,
                                                scoring=scoring.WHITENED)
        err = max(err, float(np.abs(score - exact[j, idx]).max()))
        expected = np.lexsort((np.arange(n), -exact[j]))[:100]
        top_same += bool(np.array_equal(idx, expected))
        pidx, _ = R._topk_cosine_matrix(vecs, queries[j:j + 1], 100,
                                        preserve_matrix_product=True,
                                        scoring=scoring.WHITENED)
        padded_same += bool(np.array_equal(pidx[0][:20], idx[:20]))
    m = len(queries)
    return {"max_abs_error": err, "top100_identical": top_same,
            "padded_top20_identical": padded_same, "queries": m,
            "passed": err <= tol and top_same == m and padded_same == m}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--floor", type=float, default=scoring.FLOOR)
    ap.add_argument("--check-queries", type=int, default=8)
    ap.add_argument("--tolerance", type=float, default=1e-4)
    args = ap.parse_args()

    vecs, _, d = R._load_archs4_index()
    n = int(vecs.shape[0])
    t0 = time.time()
    mu, lam, V = fit(vecs)
    floor = args.floor * lam[0]
    floored = int((lam < floor).sum())
    W = V * (1.0 / np.sqrt(np.maximum(lam, floor)))
    print(f"fitted on all {n:,} rows in {time.time() - t0:.0f}s; PC1 {lam[0] / lam.sum():.1%}; "
          f"{floored} of {d} directions at the floor {args.floor:g}", flush=True)

    mu32 = mu.astype(np.float32)
    t0 = time.time()
    norms = row_norms(vecs, mu32, W)
    print(f"row norms in {time.time() - t0:.0f}s; median {np.median(norms):.2f}, "
          f"range {norms.min():.2f} to {norms.max():.2f}", flush=True)

    import hashlib

    meta = {
        "method": "whitening", "floor": args.floor, "floored_directions": floored,
        "fitted_rows": n, "dimension": d, "index_bytes": int(vecs.nbytes),
        "index_fingerprint": scoring.index_fingerprint(vecs),
        "normalisation": "float32 x/(|x|+1e-12), centred in float32",
        "pc1_share": float(lam[0] / lam.sum()),
        "transform_id": hashlib.sha256(mu32.tobytes() + W.tobytes()).hexdigest()[:12],
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    # Write beside the real path and check *that* file through the app's own
    # loader and scan; only a passing artifact replaces the live one.
    out = paths.WHITENING_NPZ
    tmp = out.with_name(out.stem + ".candidate.npz")
    np.savez(tmp, mu=mu32, W=W, row_norms=norms, eigenvalues=lam,
             meta=np.array(json.dumps(meta)))
    paths.WHITENING_NPZ = tmp
    scoring.load_whitening.cache_clear()
    w = scoring.load_whitening()
    if w is None:
        tmp.unlink(missing_ok=True)
        raise SystemExit("FAILED: the app's loader refused the artifact just built.")

    cached = R._cached_osdr_embeddings()
    if cached is None:
        tmp.unlink(missing_ok=True)
        raise SystemExit("OSDR embeddings are needed for the self-check: cache is missing")
    osdr = cached[0]
    pick = np.random.default_rng(20261028).choice(len(osdr), args.check_queries, replace=False)
    t0 = time.time()
    result = check(vecs, w, osdr[pick].astype(np.float64), args.tolerance)
    print(f"self-check in {time.time() - t0:.0f}s: max |score error| {result['max_abs_error']:.2e}, "
          f"top 100 identical for {result['top100_identical']}/{result['queries']} queries, "
          f"padded cohort scan top 20 identical for "
          f"{result['padded_top20_identical']}/{result['queries']}", flush=True)
    if not result["passed"]:
        tmp.unlink(missing_ok=True)
        print("FAILED: the scan does not reproduce the transform; nothing replaced.")
        return 1
    meta["self_check"] = result
    np.savez(tmp, mu=mu32, W=W, row_norms=norms, eigenvalues=lam,
             meta=np.array(json.dumps(meta)))
    tmp.replace(out)
    print(f"wrote {out.relative_to(REPO)} ({out.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
