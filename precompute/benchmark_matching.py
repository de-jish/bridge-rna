#!/usr/bin/env python3
"""Which linear transform of the embedding makes nearest neighbours share tissue?

Rebuilds, on this corpus and against a null, the per-tissue comparison Joshua
shared on 2026-10-27 (raw cosine, all-but-the-top, whitening). Plan, metric
and decision rule: docs/matching-benchmark.md. Like validate_cohorts.py it
opens the real 963 MB memmap and is not part of pytest.

    .venv/bin/python precompute/benchmark_matching.py [--per-tissue 200] [--methods raw,abtt3]

Every method is a centring plus a 512 x m matrix applied to L2-normalised
vectors, fitted on ARCHS4 rows disjoint from the queries, followed by cosine.
All queries of both benchmarks are scored together in one streaming pass over
the memmap per method, keeping a running top-k with the query's own GEO series
masked out for the within-ARCHS4 benchmark.

Writes docs/validation/matching-benchmark.csv (per method x benchmark x tissue)
and prints the summary the decision rule reads.
"""
from __future__ import annotations

import argparse
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from bridge_rna import cohorts as C  # noqa: E402
from bridge_rna import retrieval as R  # noqa: E402
from manifold import data  # noqa: E402

K = 10
REPEAT_K = 5
BLOCK = 16_384
FIT_ROWS = 100_000
SEED = 20261027
UNLABELLED = {"Other", "Unknown", ""}
OUT = REPO / "docs" / "validation" / "matching-benchmark.csv"


def unit(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


# --- Transforms ---------------------------------------------------------------

def fit_stats(vecs, rows: np.ndarray):
    x = unit(np.asarray(vecs[np.sort(rows)], dtype=np.float64))
    mu = x.mean(axis=0)
    cov = np.cov(x - mu, rowvar=False)
    lam, v = np.linalg.eigh(cov)
    order = np.argsort(lam)[::-1]
    return mu, lam[order], v[:, order]


def build_methods(mu, lam, v, only: set[str] | None):
    d = len(mu)
    methods: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    methods["raw"] = (np.zeros(d), np.eye(d))
    for D in (1, 2, 3, 5, 10, 20):
        u = v[:, :D]
        methods[f"abtt{D}"] = (mu, np.eye(d) - u @ u.T)
    for f in (1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6):
        scale = 1.0 / np.sqrt(np.maximum(lam, f * lam[0]))
        methods[f"white_floor{f:g}"] = (mu, v * scale)
    for k in (64, 128, 256):
        methods[f"white_top{k}"] = (mu, v[:, :k] / np.sqrt(lam[:k]))
    if only:
        methods = {k: m for k, m in methods.items() if k in only}
    return {k: (m.astype(np.float32), w.astype(np.float32)) for k, (m, w) in methods.items()}


def transform(x: np.ndarray, mu: np.ndarray, w: np.ndarray) -> np.ndarray:
    return unit((unit(x) - mu) @ w)


# --- One pass ------------------------------------------------------------------

def top_k(vecs, queries: np.ndarray, q_series: np.ndarray, series: np.ndarray,
          mu, w, k: int = K) -> np.ndarray:
    """Top-k reference rows per query under (mu, w); a query whose series is
    not -1 never receives a row from that series. Ties go to the lower row."""
    q = transform(queries, mu, w)
    nq, n = len(q), len(vecs)
    best_s = np.full((nq, k), -np.inf, dtype=np.float32)
    best_i = np.zeros((nq, k), dtype=np.int64)
    masked = q_series >= 0
    for start in range(0, n, BLOCK):
        end = min(start + BLOCK, n)
        x = transform(np.asarray(vecs[start:end], dtype=np.float32), mu, w)
        s = q @ x.T
        blk_series = series[start:end]
        if masked.any():
            same = q_series[masked, None] == blk_series[None, :]
            sub = s[masked]
            sub[same] = -np.inf
            s[masked] = sub
        # The app's own tie-exact selection: a plain argpartition keeps an
        # arbitrary member of a tie at the cut, and tied rows (one profile filed
        # under two accessions) can carry different tissue labels.
        part = R._tie_exact_top(s, k)
        cand_s = np.concatenate([best_s, np.take_along_axis(s, part, 1)], axis=1)
        cand_i = np.concatenate([best_i, part + start], axis=1)
        order = np.lexsort((cand_i, -cand_s), axis=1)[:, :k]
        best_s = np.take_along_axis(cand_s, order, 1)
        best_i = np.take_along_axis(cand_i, order, 1)
    return best_i


# --- Metrics -------------------------------------------------------------------

def label_metrics(q_tissue: np.ndarray, hits: np.ndarray, ref_tissue: np.ndarray):
    """Per query: kNN accuracy (majority labelled neighbour tissue equals the
    query's) and precision (share of the k neighbours with the query's tissue)."""
    acc = np.zeros(len(q_tissue))
    prec = np.zeros(len(q_tissue))
    for i, (t, row) in enumerate(zip(q_tissue, hits)):
        labels = [ref_tissue[j] for j in row]
        prec[i] = sum(l == t for l in labels) / len(labels)
        voted = [l for l in labels if l not in UNLABELLED]
        if voted:
            vals, counts = np.unique(voted, return_counts=True)
            winners = set(vals[counts == counts.max()])
            acc[i] = 1.0 if t in winners and len(winners) == 1 else (
                1.0 / len(winners) if t in winners else 0.0)
    return acc, prec


def repeatability(hits_b: np.ndarray, osdr_keys: list[str], cohorts, study_of: dict):
    """Mean top-5 overlap between replicate pairs in one cohort, and between
    random pairs from the same study but different cohorts."""
    pos = {k: i for i, k in enumerate(osdr_keys)}
    sets = [set(r[:REPEAT_K]) for r in hits_b]
    within, cohort_of = [], {}
    for c in cohorts:
        idx = [pos[m] for m in c.members if m in pos]
        for m in idx:
            cohort_of[m] = c.cohort_id
        for a, b in combinations(idx, 2):
            within.append(len(sets[a] & sets[b]) / REPEAT_K)
    rng = np.random.default_rng(SEED)
    by_study: dict[str, list[int]] = {}
    for key, i in pos.items():
        by_study.setdefault(study_of[key], []).append(i)
    across = []
    for members in by_study.values():
        if len(members) < 2:
            continue
        for _ in range(min(200, len(members) * 4)):
            a, b = rng.choice(members, 2, replace=False)
            if cohort_of.get(a) != cohort_of.get(b):
                across.append(len(sets[a] & sets[b]) / REPEAT_K)
    return float(np.mean(within)), float(np.mean(across)), len(within)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-tissue", type=int, default=200)
    ap.add_argument("--methods", default="")
    ap.add_argument("--fit-seed", type=int, default=SEED,
                    help="seed for the fitting rows only; queries keep theirs")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--min-n", type=int, default=30,
                    help="tissues with fewer queries are reported but not summarised")
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)

    vecs, _meta, _d = R._load_archs4_index()
    n = len(vecs)
    meta = data.archs4_metadata()
    if meta is None:
        raise SystemExit("cache/archs4_metadata.parquet is required: run fetch_archs4_meta.py")
    ref_tissue = meta["tissue"].fillna("Unknown").astype(str).to_numpy()
    series = pd.factorize(meta["series_id"].fillna("").astype(str))[0].astype(np.int64)
    series[meta["series_id"].fillna("").astype(str).to_numpy() == ""] = -2  # never equals a query

    # Benchmark A queries: up to N per labelled tissue, fixed seed.
    a_rows = []
    for t in sorted(set(ref_tissue) - UNLABELLED):
        rows = np.flatnonzero(ref_tissue == t)
        a_rows.append(rng.choice(rows, min(args.per_tissue, len(rows)), replace=False))
    a_rows = np.sort(np.concatenate(a_rows))
    # Benchmark B queries: every OSDR sample with a labelled tissue.
    cached = R._cached_osdr_embeddings()
    osdr_vecs, key_pos = cached
    osdr_keys = sorted(key_pos, key=key_pos.get)
    osdr_tissue = np.asarray(data.osdr_tissue()).astype(str)
    b_keep = np.flatnonzero(~np.isin(osdr_tissue, list(UNLABELLED)))

    pool = np.setdiff1d(np.arange(n), a_rows)
    fit_rows = np.random.default_rng(args.fit_seed).choice(pool, FIT_ROWS, replace=False)
    t0 = time.time()
    mu, lam, v = fit_stats(vecs, fit_rows)
    print(f"fitted on {FIT_ROWS:,} ARCHS4 rows in {time.time() - t0:.0f}s; "
          f"PC1 {lam[0] / lam.sum():.1%}, top-10 {lam[:10].sum() / lam.sum():.1%}", flush=True)
    only = set(filter(None, args.methods.split(",")))
    methods = build_methods(mu, lam, v, only)

    queries = np.vstack([np.asarray(vecs[a_rows], dtype=np.float32), osdr_vecs])
    q_series = np.concatenate([series[a_rows], np.full(len(osdr_vecs), -1)])
    a_tissue = ref_tissue[a_rows]
    n_a = len(a_rows)
    print(f"benchmark A: {n_a:,} ARCHS4 queries over {len(set(a_tissue))} tissues; "
          f"benchmark B: {len(b_keep):,} labelled OSDR queries", flush=True)

    cohorts = C.build_cohorts()
    study_of = dict(zip(data.osdr_metadata()["sample_key"].astype(str),
                        data.osdr_metadata()["study"].astype(str)))
    shuffled_a = rng.permutation(a_tissue)
    shuffled_b = rng.permutation(osdr_tissue[b_keep])

    records = []
    for name, (m, w) in methods.items():
        t0 = time.time()
        hits = top_k(vecs, queries, q_series, series, m, w)
        hits_a, hits_b = hits[:n_a], hits[n_a:]
        for bench, q_t, h, null_t in (("A", a_tissue, hits_a, shuffled_a),
                                      ("B", osdr_tissue[b_keep], hits_b[b_keep], shuffled_b)):
            acc, prec = label_metrics(q_t, h, ref_tissue)
            null_acc, _ = label_metrics(null_t, h, ref_tissue)
            df = pd.DataFrame({"tissue": q_t, "acc": acc, "prec": prec, "null": null_acc})
            for t, g in df.groupby("tissue"):
                records.append(dict(method=name, bench=bench, tissue=t, n=len(g),
                                    knn_acc=g.acc.mean(), precision=g.prec.mean(),
                                    null_acc=g.null.mean()))
        within, across, pairs = repeatability(hits_b, osdr_keys, cohorts, study_of)
        records.append(dict(method=name, bench="C", tissue="replicates", n=pairs,
                            knn_acc=within, precision=across, null_acc=np.nan))
        print(f"  {name:<16} {time.time() - t0:5.0f}s  replicate top-5 overlap {within:.3f}"
              f" (same study, other cohort {across:.3f})", flush=True)

    out = pd.DataFrame.from_records(records)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False, float_format="%.4f")
    summarize(out, args.min_n)
    print(f"\nwrote {out_path}")
    return 0


def summarize(out: pd.DataFrame, min_n: int = 30) -> None:
    for bench in ("A", "B"):
        sub = out[(out.bench == bench) & (out.n >= min_n)]
        if sub.empty or "raw" not in set(sub.method):
            continue
        piv = sub.pivot(index="tissue", columns="method", values="knn_acc")
        base = piv["raw"]
        rows = []
        for m in piv.columns:
            delta = piv[m] - base
            rows.append(dict(method=m, mean_acc=piv[m].mean(), mean_delta=delta.mean(),
                             worst_drop=delta.min(), tissues_down_5=int((delta < -0.05).sum()),
                             null=sub[sub.method == m].null_acc.mean()))
        table = pd.DataFrame(rows).sort_values("mean_acc", ascending=False)
        print(f"\n== benchmark {bench}: per-tissue kNN accuracy, tissues with n >= {min_n} ==")
        print(table.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    c = out[out.bench == "C"].set_index("method")
    print("\n== benchmark C: replicate top-5 overlap ==")
    print(c[["knn_acc", "precision"]].rename(columns={"knn_acc": "within_cohort",
                                                       "precision": "same_study_other_cohort"})
          .sort_values("within_cohort", ascending=False).to_string(float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    sys.exit(main())
