#!/usr/bin/env python3
"""Phase 2+4: build the joint projection artifacts the serving app loads.

Produces, for the union of ARCHS4 (940,455) and OSDR (2,108) embeddings:
  - cache/coords_pca2.parquet, coords_pca3.parquet    (exact full-corpus PCA)
  - cache/coords_umap2.parquet, coords_umap3.parquet  (full-corpus UMAP)
  - cache/points_meta.parquet, cache/archs4_geo.parquet
  - cache/projection_stats.json                       (spectrum, extents, timings)

Global point order is fixed as [all ARCHS4 in global_index order, then all
OSDR in row order]; every artifact shares this order so the app can index
positionally. Row i < N_ARCHS4 is ARCHS4 memmap[i]; row i >= N_ARCHS4 is OSDR
npy[i - N_ARCHS4].

Design notes:
  * L2-normalize before any reduction. Raw ARCHS4 norms span 6.7-25.5 and PC1
    would otherwise be a magnitude axis (REFERENCE.md section 4).
  * Both reductions are fit on **every** point, not on a subsample. PCA gets
    there by accumulating an exact 512x512 second-moment matrix in one pass;
    UMAP gets there by fitting the 942,563-point graph directly. Neither is a
    landmark approximation any more - see fit_exact_pca and run_umap for the
    measurements that made the direct route affordable.
  * The k-nearest-neighbour graph is built once and reused for the 2-d and the
    3-d embedding, because the graph does not depend on the output dimension.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from manifold import paths, preflight  # noqa: E402

EMB_DIM = 512


def log(msg: str) -> None:
    print(f"[proj] {msg}", flush=True)


def open_archs4() -> tuple[np.memmap, pd.DataFrame]:
    manifest = json.loads(paths.ARCHS4_MANIFEST.read_text())
    n = int(manifest["total_samples"])
    d = int(manifest["embedding_dim"])
    assert d == EMB_DIM, f"unexpected embedding dim {d}"
    mm = np.memmap(paths.ARCHS4_MMAP, dtype=np.float16, mode="r", shape=(n, d))
    loc = pd.read_parquet(paths.ARCHS4_LOCATIONS)
    loc = loc.sort_values("global_index").reset_index(drop=True)
    assert len(loc) == n, "sample_locations length mismatch"
    return mm, loc


def l2_normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return (x / n).astype(np.float32)


def chunks(n: int, size: int):
    for s in range(0, n, size):
        yield s, min(s + size, n)


def stream_corpus(mm, n_archs4: int, osdr_norm: np.ndarray, batch: int):
    """Yield (start, stop, block) over the whole corpus in fixed global order.

    Blocks are L2-normalized float32. The ARCHS4 half is read straight off the
    memmap, so the 1.93 GB normalized corpus is never materialized by a caller
    that only needs one pass over it.
    """
    for s, e in chunks(n_archs4, batch):
        yield s, e, l2_normalize(np.asarray(mm[s:e], dtype=np.float32))
    total = n_archs4 + len(osdr_norm)
    if len(osdr_norm):
        yield n_archs4, total, osdr_norm


# --- PCA: exact, over every point -------------------------------------------

def fit_exact_pca(stream, total: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact PCA of the whole corpus from one streaming pass.

    This is not an approximation of a full-corpus fit, it *is* the full-corpus
    fit. PCA needs nothing from the data beyond its mean and its second moment,
    and both are sums, so one pass accumulating

        s   = sum_i x_i                    (512,)
        G   = sum_i x_i x_i^T              (512, 512)

    in float64 determines the covariance exactly:

        C = (G - n * mu mu^T) / (n - 1),   mu = s / n

    and ``eigh(C)`` then yields the same components and the same
    ``explained_variance_ratio_`` as ``sklearn.decomposition.PCA`` fit on the
    materialized 942,563 x 512 matrix, to within float64 round-off. The previous
    build fit ``IncrementalPCA`` on a 60,000-point subsample instead, which was
    an approximation adopted for a cost that turns out not to exist: the whole
    accumulation is ~250 GFLOP of BLAS matrix multiply and finishes in seconds.

    Returns (components, explained_variance_ratio, mean), components ordered by
    descending eigenvalue with all 512 present, so the full spectrum can be
    recorded rather than the leading 50.
    """
    t0 = time.time()
    gram = np.zeros((EMB_DIM, EMB_DIM), dtype=np.float64)
    acc = np.zeros(EMB_DIM, dtype=np.float64)
    seen = 0
    for i, (s, e, block) in enumerate(stream):
        b64 = block.astype(np.float64)
        gram += b64.T @ b64
        acc += b64.sum(axis=0)
        seen += len(block)
        if i % 5 == 0:
            log(f"  pca accumulate {e}/{total} ({time.time()-t0:.0f}s)")
    assert seen == total, f"stream yielded {seen} rows, expected {total}"

    mean = acc / seen
    cov = (gram - seen * np.outer(mean, mean)) / (seen - 1)
    cov = (cov + cov.T) / 2.0  # kill asymmetry from round-off before eigh
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    evals, evecs = evals[order], evecs[:, order]
    components = evecs.T  # (512, 512), row k is component k

    # Deterministic signs, matching sklearn's svd_flip: make the entry of
    # largest absolute value positive in every component. Without this an
    # eigensolver is free to return -v for v, and a rebuild would mirror the map
    # for no reason anyone could see in the data.
    flip = np.sign(components[np.arange(EMB_DIM),
                              np.argmax(np.abs(components), axis=1)])
    components *= flip[:, None]

    evals = np.clip(evals, 0.0, None)
    evr = evals / evals.sum()
    log(f"exact PCA over {seen} points in {time.time()-t0:.0f}s; "
        f"PC1 {evr[0]*100:.1f}%, cum50 {evr[:50].sum()*100:.1f}%")
    return components, evr, mean


def transform_pca(stream, total: int, components: np.ndarray,
                  mean: np.ndarray) -> np.ndarray:
    """Project the corpus onto the leading components in a second pass."""
    k = len(components)
    out = np.empty((total, k), dtype=np.float32)
    comp = np.ascontiguousarray(components.T.astype(np.float32))  # (512, k)
    mu = mean.astype(np.float32)
    t0 = time.time()
    for s, e, block in stream:
        out[s:e] = (block - mu) @ comp
    log(f"PCA transform of {total} points into {k}-d done in {time.time()-t0:.0f}s")
    return out


# --- UMAP: fit on every point, one shared neighbour graph --------------------

def load_normalized_corpus(mm, n_archs4: int, osdr_norm: np.ndarray,
                           batch: int) -> np.ndarray:
    """Materialize the full L2-normalized corpus as one C-contiguous float32 array.

    1.93 GB at the real corpus size. UMAP needs random access to every row while
    it fits, so unlike the PCA passes this one cannot stream; it is the single
    largest allocation in the build and it is why peak RSS is worth watching.
    """
    total = n_archs4 + len(osdr_norm)
    x = np.empty((total, EMB_DIM), dtype=np.float32)
    t0 = time.time()
    for s, e, block in stream_corpus(mm, n_archs4, osdr_norm, batch):
        x[s:e] = block
    log(f"materialized {total} x {EMB_DIM} normalized corpus "
        f"({x.nbytes/1e9:.2f} GB) in {time.time()-t0:.0f}s")
    return x


def build_knn(x: np.ndarray, n_neighbors: int, seed: int, n_jobs: int):
    """The k-nearest-neighbour graph, built once for both output dimensions.

    UMAP's graph depends on the input space and on ``n_neighbors``, never on
    ``n_components``, so the 2-d and 3-d embeddings can share one graph. Building
    it here rather than letting each ``UMAP.fit`` build its own halves the
    neighbour search for the build as a whole and guarantees the two maps are
    layouts of the *same* graph rather than of two independent approximations
    of it.

    ``n_jobs=1`` is the default because NN-descent's heap updates race under
    threads: the same seed gives a slightly different graph run to run at
    ``n_jobs=-1``, and this is the artifact every downstream coordinate derives
    from. This is also what UMAP would do internally, since ``random_state``
    forces ``n_jobs=1`` there (umap_.py:1952).
    """
    import pynndescent

    t0 = time.time()
    log(f"building k={n_neighbors} cosine neighbour graph over {len(x):,} points "
        f"(n_jobs={n_jobs})")
    index = pynndescent.NNDescent(
        x, n_neighbors=n_neighbors, metric="cosine", random_state=seed,
        low_memory=True, n_jobs=n_jobs, compressed=False, verbose=True,
    )
    knn_idx, knn_dist = index.neighbor_graph
    del index
    log(f"neighbour graph done in {time.time()-t0:.0f}s, shape {knn_idx.shape}")
    return np.ascontiguousarray(knn_idx), np.ascontiguousarray(knn_dist)


def umap_init_from_pca(pca_coords: np.ndarray, n_components: int,
                       seed: int) -> np.ndarray:
    """The initial layout the UMAP optimization starts from, taken from the PCA.

    UMAP's default ``init="spectral"`` is not merely slow at this scale, it is
    unusable. ``_spectral_layout`` sizes its Lanczos basis as
    ``max(2k+1, sqrt(n))`` (spectral.py:489), which at n=942,563 is 970 vectors,
    so ``eigsh`` allocates a 942,563 x 970 float64 basis: **7.31 GB**, on top of
    the 1.93 GB corpus and the graph. Measured on this 16 GB machine, that drove
    the build into 7.6 GB of swap and produced no progress in 25 minutes.
    ``init="tswspectral"`` uses the same formula and does not help.

    ``init="pca"`` is the documented alternative, but it would run a *second*
    PCA over the same 942,563 x 512 matrix and copy it to centre it. We already
    have the exact full-corpus PCA from the previous stage, so its coordinates
    are handed to UMAP directly, with the same treatment UMAP gives its own:
    scale the largest absolute coordinate to 10, then add a little noise. That
    is ``noisy_scale_coords`` (umap_.py:930), and the noise is what stops
    duplicate rows - GEO does contain resubmitted samples - from starting on top
    of each other.

    Using PCA rather than a spectral start is a real difference, not just a
    faster route to the same place: a spectral init is generally kinder to
    global structure. It is measured rather than assumed, by
    ``validate_artifacts.py --quality``.
    """
    coords = np.asarray(pca_coords[:, :n_components], dtype=np.float64)
    coords = coords * (10.0 / np.abs(coords).max())
    rng = np.random.RandomState(seed)
    return (coords + rng.normal(scale=1e-4, size=coords.shape)).astype(np.float32)


def run_umap(x: np.ndarray, knn, n_components: int, seed: int,
             n_neighbors: int, init: np.ndarray, densmap: bool = False,
             dens_lambda: float = 0.5) -> np.ndarray:
    """UMAP over every point in the corpus, from the shared neighbour graph.

    ``densmap=True`` swaps in densMAP, which adds a term penalizing the
    difference between each point's local radius in the input space and in the
    output. Plain UMAP does not preserve density at all - a tight cluster and a
    diffuse one can be drawn the same size - and on the 60,000-point evaluation
    densMAP raised density fidelity from 0.441 to 0.739 while giving up a little
    local fidelity. It was rejected for a year for a reason that no longer
    exists: ``umap-learn`` cannot ``.transform()`` new points into a densMAP
    embedding, which was fatal when the build was a landmark fit, and is
    irrelevant now that every point is fit directly.

    Two settings here were chosen by measurement rather than by default, and both
    matter more than they look. Scored against the original 512-d space on a
    60,000-point sample, with three seeds per configuration to establish that the
    differences are real (seed sd was 0.001-0.002 on both metrics):

        n_neighbors=30, euclidean on PCA-50   kNN recall 0.380   tissue purity 0.630
        n_neighbors=15, euclidean on PCA-50               0.417                0.638
        n_neighbors=30, cosine on raw 512-d               0.398                0.642
        n_neighbors=15, cosine on raw 512-d               0.426                0.646

    So the two changes compose, and together they buy 12% more local fidelity and
    a 25-NN tissue purity of 0.646 against a 0.073 permuted null. Reducing to
    PCA-50 first was discarding the 4.9% of variance those 50 components do not
    carry, and n_neighbors=30 was over-smoothing the local structure this map
    exists to show.

    There is no landmark fit and no ``.transform()`` step. The earlier build fit
    a 122,563-point subsample and pushed the remaining ~820k through
    ``.transform()``, which does not lay those points out at all: it places each
    one by a weighted average of where its landmark neighbours already sit, so
    820,000 of 942,563 points could only ever land inside the convex region the
    landmarks had already staked out. Fitting the whole corpus lets every point
    exert force on the layout it appears in. It cost hours in the original
    estimate and minutes in measurement.
    """
    import umap

    t0 = time.time()
    kind = "densMAP" if densmap else "UMAP"
    log(f"{kind}-{n_components}d fit on all {len(x):,} points, "
        f"n_neighbors={n_neighbors}, cosine on 512-d"
        + (f", dens_lambda={dens_lambda}" if densmap else ""))
    # Hand each fit its own copy of the graph. UMAP assigns the arrays through
    # without copying and then writes into them in place to disconnect far
    # neighbours (umap_.py:2647-2654), so a graph shared between the 2-d and the
    # 3-d fit would let the first quietly edit the second one's input. Nothing
    # should trip that write here - cosine disconnects at distance 2 and these
    # are 15 nearest neighbours of concentrated vectors - but 226 MB is a cheap
    # price for not depending on that.
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=0.1,
        metric="cosine",
        random_state=seed,
        precomputed_knn=(knn[0].copy(), knn[1].copy()),
        init=init,
        densmap=densmap,
        dens_lambda=dens_lambda if densmap else 2.0,
        verbose=True,
    )
    out = reducer.fit_transform(x).astype(np.float32)
    log(f"{kind}-{n_components}d done in {time.time()-t0:.0f}s")
    del reducer
    return out


def write_coords(coords: np.ndarray, path: Path) -> None:
    cols = ["x", "y", "z"][: coords.shape[1]]
    df = pd.DataFrame({c: coords[:, i].astype(np.float32) for i, c in enumerate(cols)})
    df.to_parquet(path, index=False)
    log(f"wrote {path.name} {coords.shape}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build joint PCA/UMAP projection artifacts.")
    ap.add_argument("--umap-neighbors", type=int, default=15,
                    help="UMAP n_neighbors. 15 beat the previous 30 on both local "
                         "fidelity and tissue purity by 8-37 seed standard "
                         "deviations; see run_umap.")
    ap.add_argument("--pca-report", type=int, default=50,
                    help="How many leading components to summarize in "
                         "projection_stats.json. The fit is always exact over "
                         "all 512; this only sets the cumulative figure that "
                         "invariant 2 is quoted against.")
    ap.add_argument("--batch", type=int, default=50000,
                    help="Rows per streaming block for the two PCA passes.")
    ap.add_argument("--knn-jobs", type=int, default=1,
                    help="Threads for the neighbour graph. The default of 1 is "
                         "reproducible; -1 is roughly 10x faster and gives a "
                         "slightly different graph run to run.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--archs4-limit", type=int, default=0, help="Debug: cap ARCHS4 rows.")
    ap.add_argument("--skip-umap", action="store_true")
    ap.add_argument("--densmap", action="store_true",
                    help="Fit densMAP instead of UMAP, writing to the same "
                         "coords_umap*.parquet names so a candidate cache can be "
                         "scored with validate_artifacts.py --quality --compare.")
    ap.add_argument("--dens-lambda", type=float, default=0.5,
                    help="densMAP density-preservation weight; 0.5 is what the "
                         "60,000-point evaluation scored.")
    args = ap.parse_args()

    paths.ensure_cache_dirs()

    preflight.require(
        [("ARCHS4 memmap", paths.ARCHS4_MMAP), ("ARCHS4 locations", paths.ARCHS4_LOCATIONS),
         ("OSDR embeddings", paths.OSDR_EMBEDDINGS_NPY)],
        "projection build",
    )

    mm, loc = open_archs4()
    n_archs4 = args.archs4_limit or len(loc)
    species = loc["species_id"].to_numpy()[:n_archs4].astype(np.int8)
    log(f"ARCHS4 rows: {n_archs4} (human {int((species==0).sum())}, mouse {int((species==1).sum())})")

    osdr_emb = np.load(paths.OSDR_EMBEDDINGS_NPY).astype(np.float32)
    osdr_norm = l2_normalize(osdr_emb)
    log(f"OSDR embeddings: {osdr_emb.shape}")

    # Gate: the embeddings and their metadata are indexed positionally by every
    # downstream artifact, so a length mismatch does not fail - it silently
    # attributes each OSDR point to a different sample's metadata.
    if paths.OSDR_METADATA_PARQUET.exists():
        n_meta = len(pd.read_parquet(paths.OSDR_METADATA_PARQUET))
        if n_meta != len(osdr_emb):
            raise SystemExit(
                f"ABORT: OSDR embeddings ({len(osdr_emb)}) and metadata ({n_meta}) "
                "disagree on row count. They are joined positionally, so every "
                "OSDR point would carry the wrong labels. Re-run "
                "precompute/embed_osdr.py."
            )

    total = n_archs4 + len(osdr_norm)
    stats: dict = {"n_archs4": int(n_archs4), "n_osdr": int(len(osdr_norm)),
                   "total": int(total)}

    def save_stats() -> None:
        """Persist after every stage, so a later failure does not lose the earlier work."""
        paths.PROJECTION_STATS_JSON.write_text(json.dumps(stats, indent=2))

    def stream():
        return stream_corpus(mm, n_archs4, osdr_norm, args.batch)

    # --- Identity table (dataset + src_index), fixed global order ----------
    dataset = np.concatenate([np.zeros(n_archs4, np.int8), np.ones(len(osdr_norm), np.int8)])
    src_index = np.concatenate([np.arange(n_archs4, dtype=np.int32),
                                np.arange(len(osdr_norm), dtype=np.int32)])
    species_full = np.concatenate([species, np.ones(len(osdr_norm), np.int8)])  # OSDR all mouse
    geo = loc["geo_accession"].astype(str).to_numpy()[:n_archs4]
    meta_df = pd.DataFrame({
        "dataset": dataset, "src_index": src_index, "species_id": species_full,
    })
    meta_df.to_parquet(paths.POINTS_META_PARQUET, index=False)
    # geo accessions for archs4 (for batch/study context), stored compactly.
    pd.DataFrame({"geo_accession": geo}).to_parquet(
        paths.ARCHS4_GEO_PARQUET, index=False)
    log("wrote points_meta.parquet + archs4_geo.parquet")

    # --- PCA: exact, over every point --------------------------------------
    t_pca = time.time()
    components, evr, mean = fit_exact_pca(stream(), total)
    k = min(args.pca_report, EMB_DIM)
    stats["pca_fit"] = "exact, full corpus (streaming second moment)"
    stats["pca_explained_variance_ratio"] = evr.tolist()
    stats["pca_components_reported"] = int(k)
    stats["pca_pc1_pct"] = float(evr[0] * 100)
    stats["pca_cum_pct"] = float(evr[:k].sum() * 100)

    pca3 = transform_pca(stream(), total, components[:3], mean)
    write_coords(pca3[:, :2], paths.COORDS_PCA2)
    write_coords(pca3, paths.COORDS_PCA3)
    stats["pca_seconds"] = round(time.time() - t_pca, 1)
    save_stats()
    del components

    # --- UMAP: fit on every point ------------------------------------------
    if not args.skip_umap:
        t_umap = time.time()
        stats["umap_fit"] = "full corpus, no landmark subsample"
        stats["umap_method"] = "densmap" if args.densmap else "umap"
        stats["umap_neighbors"] = int(args.umap_neighbors)
        stats["umap_metric"] = "cosine"
        stats["umap_input"] = "raw 512-d L2-normalized"
        stats["umap_init"] = "exact full-corpus PCA, scaled (not spectral)"
        if args.densmap:
            stats["dens_lambda"] = float(args.dens_lambda)

        x = load_normalized_corpus(mm, n_archs4, osdr_norm, args.batch)
        t_knn = time.time()
        knn = build_knn(x, args.umap_neighbors, args.seed, args.knn_jobs)
        stats["knn_seconds"] = round(time.time() - t_knn, 1)
        stats["knn_jobs"] = int(args.knn_jobs)
        save_stats()

        t = time.time()
        umap2 = run_umap(x, knn, 2, args.seed, args.umap_neighbors,
                         umap_init_from_pca(pca3, 2, args.seed),
                         densmap=args.densmap, dens_lambda=args.dens_lambda)
        stats["umap2_seconds"] = round(time.time() - t, 1)
        write_coords(umap2, paths.COORDS_UMAP2)
        save_stats()
        del umap2

        t = time.time()
        umap3 = run_umap(x, knn, 3, args.seed, args.umap_neighbors,
                         umap_init_from_pca(pca3, 3, args.seed),
                         densmap=args.densmap, dens_lambda=args.dens_lambda)
        stats["umap3_seconds"] = round(time.time() - t, 1)
        write_coords(umap3, paths.COORDS_UMAP3)
        save_stats()
        del umap3, knn, x
        stats["umap_seconds"] = round(time.time() - t_umap, 1)

    save_stats()
    log(f"ALL DONE. stats -> {paths.PROJECTION_STATS_JSON.name}")


if __name__ == "__main__":
    main()
