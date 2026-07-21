"""Objective validation of the Phase 2/4 artifacts produced by build_projections.py.

Run this after every projection build. It exits nonzero on failure, so it can
gate a rebuild rather than relying on someone eyeballing a scatter plot.

Three groups of checks:

1. Structural - row counts agree across every artifact, coordinates are finite
   and non-degenerate, density rasters exist, and the identity table lines up
   with the OSDR metadata it is joined to positionally.

2. Invariant 2 - PC1 must land well below the 57.8% recorded in REFERENCE.md
   section 4. That figure was measured *before* L2 normalization, where PC1 is
   the sequencing-depth axis. A normalized build must be far below it; landing
   near it is evidence normalization was silently skipped.

3. Cross-corpus mixing - how separated OSDR and ARCHS4 are in the 512-d space.
   This is the honesty check behind the app's whole premise. A raw "OSDR mostly
   neighbours OSDR" number conflates replicate structure with a technical batch
   effect, so the result is stratified by study and by tissue. Tissue is the
   dominant axis of variation in bulk expression, so OSDR samples that share
   neither study nor tissue but still neighbour each other cannot be explained
   by biology.

    python precompute/validate_artifacts.py            # structural + invariant
    python precompute/validate_artifacts.py --mixing   # also stream the memmap
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from manifold import paths  # noqa: E402

# 57.8% is the pre-normalization PC1 (REFERENCE.md section 4). See invariant 2.
PC1_PRENORM_PCT = 57.8
PC1_CEILING_PCT = 50.0

_failures: list[str] = []
_warnings: list[str] = []


def check(cond: bool, msg: str) -> bool:
    print(("  OK   " if cond else "  FAIL ") + msg)
    if not cond:
        _failures.append(msg)
    return cond


def warn(msg: str) -> None:
    print("  WARN " + msg)
    _warnings.append(msg)


def validate_structure() -> tuple[int, np.ndarray]:
    print("=== 1. projection_stats.json ===")
    stats = json.loads(paths.PROJECTION_STATS_JSON.read_text())
    n_a4, n_osdr, total = stats["n_archs4"], stats["n_osdr"], stats["total"]
    print(f"  n_archs4={n_a4}  n_osdr={n_osdr}  total={total}")
    check(n_a4 + n_osdr == total, "row counts sum to the total")

    pc1 = stats["pca_pc1_pct"]
    print(f"  PC1 = {pc1:.1f}%   cumulative over 50 PCs = {stats['pca_cum_pct']:.1f}%")
    check(
        pc1 < PC1_CEILING_PCT,
        f"invariant 2: PC1 {pc1:.1f}% is well below the {PC1_PRENORM_PCT}% "
        "pre-normalization figure, so L2 normalization was applied",
    )

    print("\n=== 2. coordinate parquets ===")
    for name, p in [("pca2", paths.COORDS_PCA2), ("pca3", paths.COORDS_PCA3),
                    ("umap2", paths.COORDS_UMAP2), ("umap3", paths.COORDS_UMAP3)]:
        if not p.exists():
            check(False, f"{name}: {p.name} exists")
            continue
        a = pd.read_parquet(p).to_numpy(dtype=np.float64)
        check(len(a) == total, f"{name}: {len(a)} rows == {total}")
        check(bool(np.isfinite(a).all()), f"{name}: all values finite")
        check(bool((a.std(axis=0) > 1e-6).all()),
              f"{name}: every axis has real spread {np.round(a.std(axis=0), 2)}")

    print("\n=== 3. density rasters ===")
    for nm in ("pca2", "umap2"):
        p = paths.DENSITY_DIR / f"{nm}.png"
        sz = p.stat().st_size if p.exists() else 0
        check(p.exists() and sz > 5000, f"density/{nm}.png present and non-trivial ({sz} B)")

    print("\n=== 4. identity table alignment ===")
    meta = pd.read_parquet(paths.POINTS_META_PARQUET)
    check(len(meta) == total, f"points_meta rows {len(meta)} == {total}")
    dataset = meta["dataset"].to_numpy()
    is_osdr = dataset == 1
    check(int(is_osdr.sum()) == n_osdr, f"points_meta marks {int(is_osdr.sum())} OSDR points")
    check(bool((dataset[:n_a4] == 0).all()), f"first {n_a4} rows are ARCHS4")
    check(bool((dataset[n_a4:] == 1).all()), "trailing rows are OSDR")
    osdr_meta = pd.read_parquet(paths.OSDR_METADATA_PARQUET)
    check(
        len(osdr_meta) == n_osdr,
        f"osdr_metadata rows {len(osdr_meta)} == {n_osdr} (joined positionally)",
    )

    print("\n=== 5. OSDR is not collapsed to a single blob ===")
    u2 = pd.read_parquet(paths.COORDS_UMAP2).to_numpy(dtype=np.float64)
    ratio = float(np.linalg.norm(u2[is_osdr].std(axis=0)) / np.linalg.norm(u2.std(axis=0)))
    print(f"  OSDR umap2 spread / corpus spread = {ratio:.3f}")
    check(ratio > 0.05, f"OSDR occupies a real region of the map (ratio {ratio:.3f})")
    return total, is_osdr


def _osdr_neighbours(n_a4: int, total: int, k: int):
    """Exact top-k neighbours of every OSDR sample across the whole corpus.

    Brute force, not an ANN index. There are only 2,108 queries, so one
    streaming pass over the 963 MB memmap answers them exactly in well under a
    minute - and an exact answer is what an honesty check should rest on. The
    approximate index this replaced cost 2.07 GB on disk and was the app's
    single largest artifact despite being read by nothing but this function.

    Returns (labels, cosines), both (n_osdr, k), sorted by descending cosine.
    """
    manifest = json.loads(paths.ARCHS4_MANIFEST.read_text())
    mm = np.memmap(paths.ARCHS4_MMAP, dtype=np.float16, mode="r",
                   shape=(int(manifest["total_samples"]), 512))

    emb = np.load(paths.OSDR_EMBEDDINGS_NPY).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    n = len(emb)

    best_cos = np.full((n, 0), 0.0, dtype=np.float32)
    best_lab = np.full((n, 0), 0, dtype=np.int64)

    def merge(cos, lab):
        """Keep the running top-k after folding in one block of candidates."""
        nonlocal best_cos, best_lab
        cos = np.concatenate([best_cos, cos], axis=1)
        lab = np.concatenate([best_lab, lab], axis=1)
        take = min(k, cos.shape[1])
        part = np.argpartition(-cos, take - 1, axis=1)[:, :take]
        rows = np.arange(n)[:, None]
        cos, lab = cos[rows, part], lab[rows, part]
        order = np.argsort(-cos, axis=1)
        best_cos, best_lab = cos[rows, order], lab[rows, order]

    block = 50_000
    for start in range(0, n_a4, block):
        stop = min(start + block, n_a4)
        chunk = np.asarray(mm[start:stop], dtype=np.float32)
        chunk /= np.linalg.norm(chunk, axis=1, keepdims=True)
        merge(emb @ chunk.T, np.broadcast_to(
            np.arange(start, stop), (n, stop - start)))
    merge(emb @ emb.T, np.broadcast_to(
        np.arange(n_a4, n_a4 + n), (n, n)))
    return best_lab, best_cos, emb


def validate_mixing(total: int) -> None:
    """Stratified cross-corpus mixing. Streams the ARCHS4 memmap, so it is opt-in."""
    print("\n=== 6. cross-corpus mixing in 512-d, stratified ===")
    stats = json.loads(paths.PROJECTION_STATS_JSON.read_text())
    n_a4 = stats["n_archs4"]

    meta = pd.read_parquet(paths.OSDR_METADATA_PARQUET)
    study = meta["study"].astype(str).to_numpy()
    tissue = meta["tissue"].astype(str).to_numpy()

    K = 51  # one self-match plus 50
    print(f"  computing exact top-{K} neighbours for every OSDR sample "
          f"over {total:,} points")
    labels, cosines, emb = _osdr_neighbours(n_a4, total, K)
    n = len(emb)
    # Downstream code was written against hnswlib's cosine *distance*.
    dists = 1.0 - cosines

    cnt = {("same", "same"): 0, ("same", "diff"): 0,
           ("diff", "same"): 0, ("diff", "diff"): 0}
    n_a4_hits = 0
    cos_osdr, cos_a4 = [], []
    for i in range(n):
        keep = labels[i] != (n_a4 + i)
        lab, cos = labels[i][keep][:K - 1], 1.0 - dists[i][keep][:K - 1]
        is_o = lab >= n_a4
        n_a4_hits += int((~is_o).sum())
        if is_o.any():
            cos_osdr.append(cos[is_o].mean())
            for j in lab[is_o] - n_a4:
                cnt[("same" if study[j] == study[i] else "diff",
                     "same" if tissue[j] == tissue[i] else "diff")] += 1
        if (~is_o).any():
            cos_a4.append(cos[~is_o].mean())

    tot = sum(cnt.values()) + n_a4_hits
    same_study = cnt[("same", "same")] + cnt[("same", "diff")]
    cross_study = cnt[("diff", "same")] + cnt[("diff", "diff")]
    print(f"  same-study  OSDR : {same_study/tot*100:5.1f}%")
    print(f"  cross-study OSDR : {cross_study/tot*100:5.1f}%")
    print(f"  ARCHS4           : {n_a4_hits/tot*100:5.1f}%")

    # Chance model, averaged over SAMPLES. Summing over studies instead would
    # transpose these two magnitudes, because a query in a large study has far
    # more same-study partners than one in a small study.
    sizes = pd.Series(study).value_counts()
    size_of = pd.Series(study).map(sizes).to_numpy()
    exp_same = float(np.mean(size_of - 1) / (total - 1))
    exp_cross = float(np.mean(n - size_of) / (total - 1))
    assert abs((exp_same + exp_cross) - (n - 1) / (total - 1)) < 1e-12, "chance model must partition"

    print("\n  enrichment over chance:")
    print(f"    same-study  expected {exp_same*100:.4f}%  ->  "
          f"{(same_study/tot)/exp_same:6.0f}x   (replicate structure, expected)")
    print(f"    cross-study expected {exp_cross*100:.4f}%  ->  "
          f"{(cross_study/tot)/exp_cross:6.0f}x   (corpus batch effect)")

    # Tissue-controlled: biology cannot explain different-tissue clustering.
    n_dsdt = sum(int(((study != study[i]) & (tissue != tissue[i])).sum()) for i in range(n))
    exp_dsdt = n_dsdt / n / (total - 1)
    obs_dsdt = cnt[("diff", "diff")] / tot
    ratio = obs_dsdt / exp_dsdt
    print(f"\n  different study AND different tissue:")
    print(f"    observed {obs_dsdt*100:.3f}%  expected {exp_dsdt*100:.5f}%  ->  {ratio:.0f}x")

    print("\n  cosine geometry:")
    print(f"    OSDR -> its OSDR neighbours   : {np.mean(cos_osdr):.4f}")
    print(f"    OSDR -> its ARCHS4 neighbours : {np.mean(cos_a4):.4f}")
    print(f"    gap                           : {np.mean(cos_osdr)-np.mean(cos_a4):.4f}")

    if ratio > 50:
        warn(
            f"OSDR neighbours sharing neither study nor tissue are {ratio:.0f}x over "
            "chance. Biology cannot explain cross-tissue clustering, so a technical "
            "batch effect is present and the app's cross-dataset warning is load-bearing."
        )
    elif ratio > 10:
        warn(f"moderate tissue-controlled batch effect ({ratio:.0f}x over chance)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mixing", action="store_true",
                    help="Also run the cross-corpus mixing analysis (streams the "
                         "963 MB ARCHS4 memmap; needs BRIDGE_RNA_ROOT).")
    args = ap.parse_args()

    total, _ = validate_structure()
    if args.mixing:
        validate_mixing(total)

    print("\n" + "=" * 62)
    for w in _warnings:
        print("WARN: " + w)
    if _failures:
        for f in _failures:
            print("FAIL: " + f)
        print(f"VALIDATION FAILED ({len(_failures)} checks)")
        sys.exit(1)
    print(f"ALL VALIDATION CHECKS PASSED ({len(_warnings)} warning(s))")


if __name__ == "__main__":
    main()
