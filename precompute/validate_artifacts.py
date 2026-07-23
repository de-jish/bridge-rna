"""Objective validation of the Phase 2/4 artifacts produced by build_projections.py.

Run this after every projection build. It exits nonzero on failure, so it can
gate a rebuild rather than relying on someone eyeballing a scatter plot.

Four groups of checks:

1. Structural - row counts agree across every artifact, coordinates are finite
   and non-degenerate, and the identity table lines up with the OSDR metadata it
   is joined to positionally.

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

4. Projection quality - does the picture still stand for the 512-d space it was
   reduced from? Structural checks pass for any set of finite numbers, so they
   cannot tell a good projection from a scrambled one. This scores each
   coordinate set on local fidelity (are a point's 512-d neighbours still its
   neighbours on screen?) and on biological fidelity (do its neighbours share
   its tissue?), each against a null that says what the number would be if the
   map carried no information at all.

    python precompute/validate_artifacts.py             # structural + invariant
    python precompute/validate_artifacts.py --mixing    # also stream the memmap
    python precompute/validate_artifacts.py --quality   # also score the maps
    python precompute/validate_artifacts.py --quality --compare DIR
        # ... and score a second set of coordinate parquets alongside, which is
        # how a candidate build is compared against the shipped one.
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

# Every coordinate set the build can produce, in one place. `validate_structure`
# and `validate_quality` both walk this, so a new projection is registered once
# rather than in two lists that can drift apart.
_COORD_PATHS = [("pca2", paths.COORDS_PCA2), ("pca3", paths.COORDS_PCA3),
                ("umap2", paths.COORDS_UMAP2), ("umap3", paths.COORDS_UMAP3),
                ("tsne2", paths.COORDS_TSNE2), ("tsne3", paths.COORDS_TSNE3)]

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
    reported = stats.get("pca_components_reported", 50)
    print(f"  PC1 = {pc1:.1f}%   cumulative over {reported} PCs = "
          f"{stats['pca_cum_pct']:.1f}%")
    check(
        pc1 < PC1_CEILING_PCT,
        f"invariant 2: PC1 {pc1:.1f}% is well below the {PC1_PRENORM_PCT}% "
        "pre-normalization figure, so L2 normalization was applied",
    )
    # The PCA is an exact eigendecomposition of the whole corpus, so the full
    # spectrum is present and must sum to 1. A short or renormalized spectrum
    # means the build fell back to a truncated fit without saying so.
    spectrum = np.asarray(stats["pca_explained_variance_ratio"], dtype=np.float64)
    check(len(spectrum) == 512,
          f"the full 512-component spectrum was recorded ({len(spectrum)} present)")
    check(abs(spectrum.sum() - 1.0) < 1e-9,
          f"explained variance sums to 1 ({spectrum.sum():.9f}), so the fit was exact")

    print("\n=== 2. coordinate parquets ===")
    for name, p in _COORD_PATHS:
        method = name[:-1]  # "umap2" -> "umap"
        # A stage that was skipped never wrote its `<method>_fit` marker, so its
        # coordinates are legitimately absent and saying so is not a failure. A
        # stage that *ran* and left no parquet is a real one. PCA has no skip
        # flag, so it is always expected.
        expected = method == "pca" or f"{method}_fit" in stats
        if not p.exists():
            if expected:
                check(False, f"{name}: {p.name} exists")
            else:
                print(f"  SKIP {name}: not built (no {method}_fit in the record)")
            continue
        a = pd.read_parquet(p).to_numpy(dtype=np.float64)
        check(len(a) == total, f"{name}: {len(a)} rows == {total}")
        check(bool(np.isfinite(a).all()), f"{name}: all values finite")
        check(bool((a.std(axis=0) > 1e-6).all()),
              f"{name}: every axis has real spread {np.round(a.std(axis=0), 2)}")

    print("\n=== 3. identity table alignment ===")
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

    print("\n=== 4. OSDR is not collapsed to a single blob ===")
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
    print("\n=== 5. cross-corpus mixing in 512-d, stratified ===")
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


# --- 6. Projection quality ---------------------------------------------------

# Metrics are computed on a sample, because an exact k-NN in 512-d over the
# whole corpus is a 942,563^2 problem. 60,000 is the size the UMAP settings were
# originally chosen against (REFERENCE.md section 4), so the numbers here are
# comparable to the ones recorded there.
QUALITY_SAMPLE = 60_000
QUALITY_K = 15          # local fidelity: recall of the 512-d k-NN
QUALITY_PURITY_K = 25   # biological fidelity: tissue agreement among neighbours
# Relative regression under --compare that counts as real rather than as noise.
REGRESSION_TOLERANCE = 0.02


def _sample_vectors(sample: np.ndarray, n_a4: int) -> np.ndarray:
    """L2-normalized 512-d vectors for the sampled global indices."""
    manifest = json.loads(paths.ARCHS4_MANIFEST.read_text())
    mm = np.memmap(paths.ARCHS4_MMAP, dtype=np.float16, mode="r",
                   shape=(int(manifest["total_samples"]), 512))
    out = np.empty((len(sample), 512), dtype=np.float32)
    a4 = sample < n_a4
    if a4.any():
        out[a4] = np.asarray(mm[sample[a4]], dtype=np.float32)
    if (~a4).any():
        osdr = np.load(paths.OSDR_EMBEDDINGS_NPY).astype(np.float32)
        out[~a4] = osdr[sample[~a4] - n_a4]
    out /= np.linalg.norm(out, axis=1, keepdims=True)
    return out


def _exact_knn(x: np.ndarray, k: int, block: int = 2048) -> np.ndarray:
    """Top-k neighbours of every row of x, excluding self, by cosine on unit rows.

    Brute force in blocks. Approximate neighbours would put the measurement
    error and the thing being measured in the same place.
    """
    n = len(x)
    out = np.empty((n, k), dtype=np.int32)
    for s in range(0, n, block):
        e = min(s + block, n)
        sim = x[s:e] @ x.T
        sim[np.arange(e - s), np.arange(s, e)] = -np.inf  # drop self
        part = np.argpartition(-sim, k, axis=1)[:, :k]
        rows = np.arange(e - s)[:, None]
        out[s:e] = part[rows, np.argsort(-sim[rows, part], axis=1)]
    return out


def _knn_euclidean(coords: np.ndarray, k: int, block: int = 2048) -> np.ndarray:
    """Top-k neighbours in a low-dimensional coordinate space."""
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="auto").fit(coords)
    _, idx = nn.kneighbors(coords, n_neighbors=k + 1)
    return idx[:, 1:].astype(np.int32)  # column 0 is the point itself


def _recall(truth: np.ndarray, got: np.ndarray) -> float:
    k = truth.shape[1]
    return float(np.mean([
        len(set(truth[i].tolist()) & set(got[i].tolist())) / k
        for i in range(len(truth))]))


def _purity(neighbours: np.ndarray, labels: np.ndarray, known: np.ndarray) -> float:
    """Fraction of a point's neighbours carrying its own label, over known points."""
    if not known.any():
        return float("nan")
    same = (labels[neighbours] == labels[:, None])
    return float(same[known].mean())


def validate_quality(total: int, compare: Path | None) -> None:
    from manifold import tissue as tissue_map

    print(f"\n=== 6. projection quality (sample of {QUALITY_SAMPLE:,}) ===")
    stats = json.loads(paths.PROJECTION_STATS_JSON.read_text())
    n_a4 = stats["n_archs4"]

    rng = np.random.default_rng(20260722)
    sample = np.sort(rng.choice(total, size=min(QUALITY_SAMPLE, total),
                                replace=False))
    print(f"  reading {len(sample):,} 512-d vectors")
    x = _sample_vectors(sample, n_a4)
    print(f"  computing exact {QUALITY_K}-NN in 512-d (brute force)")
    truth = _exact_knn(x, QUALITY_K)
    truth25 = _exact_knn(x, QUALITY_PURITY_K)

    # Tissue labels over the sample, in the shared vocabulary both corpora use.
    labels = np.full(len(sample), tissue_map.UNKNOWN, dtype=object)
    if paths.ARCHS4_METADATA_PARQUET.exists():
        a4_meta = pd.read_parquet(paths.ARCHS4_METADATA_PARQUET, columns=["tissue"])
        a4_lab = a4_meta["tissue"].astype(str).to_numpy()
        m = sample < min(n_a4, len(a4_lab))
        labels[m] = a4_lab[sample[m]]
    osdr_meta = pd.read_parquet(paths.OSDR_METADATA_PARQUET, columns=["tissue"])
    o = sample >= n_a4
    if o.any():
        labels[o] = [tissue_map.canonical_tissue(v)
                     for v in osdr_meta["tissue"].astype(str).to_numpy()[sample[o] - n_a4]]
    labels = np.asarray(labels, dtype=object)
    known = ~np.isin(labels, [tissue_map.UNKNOWN, tissue_map.OTHER])
    print(f"  {int(known.sum()):,} of {len(sample):,} sampled points carry a "
          f"named tissue bucket")

    # Nulls. Purity against shuffled labels says what agreement looks like when
    # the map means nothing; recall against a random neighbour set says the same
    # for local fidelity. Without them a purity of 0.6 is uninterpretable.
    null_labels = labels.copy()
    rng.shuffle(null_labels)
    null_purity = _purity(truth25, null_labels, known)
    null_recall = QUALITY_K / (len(sample) - 1)
    reference_purity = _purity(truth25, labels, known)
    print(f"  nulls: permuted-label purity {null_purity:.4f}, "
          f"random-neighbour recall {null_recall:.6f}")
    print(f"  512-d reference purity (the ceiling a projection is chasing): "
          f"{reference_purity:.4f}")

    sets = [("cache", {n: p for n, p in _COORD_PATHS})]
    if compare is not None:
        sets.append((compare.name, {n: compare / p.name for n, p in _COORD_PATHS}))

    # "Purity clears N times the null" is the wrong shape of test, because the
    # null moves with how concentrated the label distribution is: on a corpus
    # where one bucket dominates, a permuted label already agrees half the time
    # and no projection can be three times that. The question a projection
    # should answer is what *share of the recoverable structure* survived the
    # reduction, so both metrics are reported against the same two anchors -
    # the null below, and the 512-d space above, which is the ceiling.
    span = reference_purity - null_purity

    print(f"\n  {'coords':<10} {'set':<18} {'kNN recall':>11} {'purity':>9} "
          f"{'share of recoverable':>21}")
    results: dict[tuple[str, str], tuple[float, float, float]] = {}
    for set_name, paths_by_name in sets:
        for name, path in paths_by_name.items():
            if not Path(path).exists():
                continue
            coords = pd.read_parquet(path).to_numpy(dtype=np.float32)[sample]
            recall = _recall(truth, _knn_euclidean(coords, QUALITY_K))
            purity = _purity(_knn_euclidean(coords, QUALITY_PURITY_K), labels, known)
            share = (purity - null_purity) / span if span > 1e-6 else float("nan")
            results[(set_name, name)] = (recall, purity, share)
            print(f"  {name:<10} {set_name:<18} {recall:>11.4f} {purity:>9.4f} "
                  f"{share:>20.1%}")

    for name in dict(_COORD_PATHS):
        other = next((k for k in results if k[1] == name and k[0] != "cache"), None)
        if other is None or ("cache", name) not in results:
            continue
        new_r, new_p, new_s = results[("cache", name)]
        old_r, old_p, old_s = results[other]
        d_recall = (new_r - old_r) / max(old_r, 1e-9)
        d_purity = (new_p - old_p) / max(old_p, 1e-9)
        print(f"\n  {name}: vs {other[0]} - recall {old_r:.4f} -> {new_r:.4f} "
              f"({d_recall * 100:+.1f}%), purity {old_p:.4f} -> {new_p:.4f} "
              f"({d_purity * 100:+.1f}%)")
        # Only a *material* regression on both metrics is worth a warning. Seed
        # to seed variation on these metrics was measured at 0.001-0.002
        # absolute, so a fraction of a percent is noise, and a warning that
        # fires on noise is a warning people learn to skip.
        if d_recall < -REGRESSION_TOLERANCE and d_purity < -REGRESSION_TOLERANCE:
            warn(f"{name} is materially worse than {other[0]} on both metrics "
                 f"(recall {d_recall * 100:+.1f}%, purity {d_purity * 100:+.1f}%)")

    if span <= 1e-6:
        warn("the 512-d reference purity does not clear the permuted null, so "
             "there is no recoverable tissue structure to score a projection "
             "against on this corpus")
        return

    # The two neighbour embeddings are gated; PCA is not. PCA-2D is knowingly a
    # weak view - PC1 alone is 41% of the variance, so it is mostly one axis
    # plus noise - and it is kept as a fast linear sanity layer, not as the map.
    # Failing a build because the crudest projection is crude would be a gate
    # nobody could act on. UMAP and t-SNE are both offered as *the* map, so a
    # build where either stopped standing for the 512-d space is a real failure.
    for (set_name, name), (recall, purity, share) in sorted(results.items()):
        if set_name != "cache":
            continue
        gated = name.startswith(("umap", "tsne"))
        report = check if gated else (
            lambda cond, msg: None if cond else warn(msg + " (not gated: PCA is "
                                                     "a linear sanity layer)"))
        report(share > 0.25,
               f"{name}: keeps {share:.1%} of the tissue structure recoverable in "
               f"512-d (purity {purity:.4f} against null {null_purity:.4f} and "
               f"ceiling {reference_purity:.4f})")
        report(recall > null_recall * 20,
               f"{name}: {QUALITY_K}-NN recall {recall:.4f} is well above the "
               f"random-neighbour rate ({null_recall:.6f})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mixing", action="store_true",
                    help="Also run the cross-corpus mixing analysis (streams the "
                         "963 MB ARCHS4 memmap; needs BRIDGE_RNA_ROOT).")
    ap.add_argument("--quality", action="store_true",
                    help="Also score every coordinate set against the 512-d space "
                         "it was reduced from (reads the memmap; a few minutes).")
    ap.add_argument("--compare", type=Path, default=None,
                    help="Directory of coordinate parquets to score alongside the "
                         "cache, for comparing a candidate build with the shipped "
                         "one. Only meaningful with --quality.")
    args = ap.parse_args()

    total, _ = validate_structure()
    if args.mixing:
        validate_mixing(total)
    if args.quality:
        validate_quality(total, args.compare)

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
