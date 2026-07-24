#!/usr/bin/env python3
"""Phase 2+4: build the joint projection artifacts the serving app loads.

Produces, for the union of ARCHS4 (940,455) and OSDR (2,108) embeddings:
  - cache/coords_pca2.parquet, coords_pca3.parquet    (exact full-corpus PCA)
  - cache/coords_umap2.parquet, coords_umap3.parquet  (full-corpus UMAP)
  - cache/coords_tsne2.parquet, coords_tsne3.parquet  (full-corpus t-SNE)
  - cache/points_meta.parquet, cache/archs4_geo.parquet
  - cache/projection_stats.json                       (spectrum, extents, timings)

Global point order is fixed as [all ARCHS4 in global_index order, then all
OSDR in row order]; every artifact shares this order so the app can index
positionally. Row i < N_ARCHS4 is ARCHS4 memmap[i]; row i >= N_ARCHS4 is OSDR
npy[i - N_ARCHS4].

Design notes:
  * L2-normalize before any reduction. Raw ARCHS4 norms span 6.7-26.4 and PC1
    would otherwise be a magnitude axis (REFERENCE.md section 4).
  * All three reductions are fit on **every** point, not on a subsample. PCA
    gets there by accumulating an exact 512x512 second-moment matrix in one
    pass; UMAP and t-SNE get there by fitting the 942,563-point graph directly.
    None is a landmark approximation - see fit_exact_pca, run_umap and run_tsne
    for the measurements that made the direct route affordable.
  * Each neighbour-based method builds its graph once and reuses it for the 2-d
    and the 3-d embedding, because the graph does not depend on the output
    dimension. UMAP and t-SNE need *different* graphs (k=n_neighbors against
    k=3*perplexity), so they get one each rather than sharing a padded one:
    slicing a k=90 graph down to k=30 is not the graph NN-descent would have
    built at k=30, and the graph is the artifact every coordinate derives from.
  * The three methods are independent stages. Each can be skipped, and
    ``save_stats`` merges into the existing record rather than replacing it, so
    rebuilding one method does not erase what the others measured.
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

# t-SNE's optimization schedule. These are openTSNE's own defaults, resolved to
# literals rather than left as its "auto" sentinels so the numbers this build
# actually ran are recorded in projection_stats.json and shown on the control
# rail. `early_exaggeration="auto"` resolves to 12 and `learning_rate="auto"`
# resolves to n_samples/exaggeration per phase, which is 78,547 while the
# exaggeration is on and 942,563 after it (tsne.py, __check_params). The
# learning rate is left as "auto" in the call so it tracks the corpus size
# instead of being pinned to today's row count.
TSNE_PERPLEXITY = 30
TSNE_EARLY_EXAGGERATION = 12
TSNE_EARLY_ITER = 250
TSNE_ITER = 500
TSNE_MOMENTUM = 0.8


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

    Two settings here were chosen by measurement rather than by default. Scored
    against the original 512-d space on a 60,000-point sample, with three seeds
    per configuration to establish that the differences are real (seed sd was
    0.001-0.002 on both metrics):

        n_neighbors=30, euclidean on PCA-50   kNN recall 0.380   tissue purity 0.630
        n_neighbors=15, euclidean on PCA-50               0.417                0.638
        n_neighbors=30, cosine on raw 512-d (shipped)     0.398                0.642
        n_neighbors=15, cosine on raw 512-d               0.426                0.646

    **Only the metric half of that experiment survived.** Dropping the PCA-50
    step was right and is permanent: it was discarding the 4.9% of variance
    those 50 components do not carry.

    **The `n_neighbors=15` half was reverted on 2026-07-23, and the table above
    is exactly why it should not be trusted: every row of it was fitted on a
    60,000-point subsample.** Rerun on the real 942,563-point corpus with
    `validate_artifacts.py --quality --compare`, 30 beats 15 on *both* metrics
    in both dimensionalities - umap2 recall 0.3955 to 0.4140 and purity 0.5838
    to 0.6014, umap3 recall 0.4596 to 0.4746 and purity 0.6169 to 0.6212 - so
    the ordering above simply inverts at full scale.

    The reason is that `n_neighbors` is a density parameter. It fixes how many
    neighbours define a point's neighbourhood, and what share of the manifold
    that covers depends on how many points are in it: fifteen out of 60,000 is
    roughly sixteen times as wide a view as fifteen out of 942,563. The integer
    cannot mean the same thing in both corpora, so it does not transfer.

    Take this as a standing caution rather than a settled number: any parameter
    that scales with corpus density has to be scored on the real corpus, and
    `--compare` against the previous cache is the way to do it. See
    `REFERENCE.md`, "n_neighbors back to 30: the subsample tuning did not
    transfer".

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


# --- t-SNE: fit on every point, from one shared affinity matrix -------------

def build_tsne_affinities(x: np.ndarray, perplexity: int, seed: int,
                          knn_jobs: int, n_jobs: int):
    """The perplexity-calibrated joint probabilities, built once for both dims.

    t-SNE's P matrix is the analogue of UMAP's fuzzy simplicial set: it depends
    on the input space and on the perplexity, never on the output dimension, so
    the 2-d and the 3-d embedding are layouts of the *same* affinities rather
    than of two independent approximations of it. That is the same argument
    ``build_knn`` makes for UMAP, and it matters more here, because P is the
    single largest allocation in this stage.

    The neighbour graph comes from ``build_knn``, the same NN-descent call UMAP
    uses, rather than from openTSNE's own index. Two reasons: the determinism
    story is then identical for both methods (``n_jobs=1``, one seed), and the
    graph is built by one code path that has already been reasoned about.

    The ``[:, 1:]`` slice is load-bearing and easy to get wrong. pynndescent
    returns each point's *self* match in column 0; openTSNE's own index strips
    it before returning (``nearest_neighbors.NNDescent.build`` ends with
    ``return indices[:, 1:], distances[:, 1:]``), and ``PrecomputedNeighbors``
    passes whatever it is handed straight through to the perplexity
    calibration. Leaving self in would hand every point a zero-distance
    neighbour, which is not what a perplexity of 30 means. So the graph is
    built at ``3 * perplexity + 1`` and sliced back to ``3 * perplexity``,
    which is the neighbour count openTSNE would have chosen itself.
    """
    from openTSNE import affinity, nearest_neighbors

    k = 3 * perplexity
    knn_idx, knn_dist = build_knn(x, k + 1, seed, knn_jobs)
    index = nearest_neighbors.PrecomputedNeighbors(
        np.ascontiguousarray(knn_idx[:, 1:]), np.ascontiguousarray(knn_dist[:, 1:]))
    del knn_idx, knn_dist

    t0 = time.time()
    log(f"calibrating perplexity-{perplexity} affinities over {len(x):,} points")
    aff = affinity.PerplexityBasedNN(
        knn_index=index, perplexity=perplexity, n_jobs=n_jobs,
        random_state=seed, verbose=True,
    )
    log(f"affinities done in {time.time()-t0:.0f}s, "
        f"P {aff.P.shape} with {aff.P.nnz:,} nonzeros "
        f"({aff.P.data.nbytes/1e9:.2f} GB of values)")
    return aff


def tsne_init_from_pca(pca_coords: np.ndarray, n_components: int,
                       seed: int) -> np.ndarray:
    """The initial layout t-SNE starts from, taken from the exact full-corpus PCA.

    Same source as ``umap_init_from_pca`` and for the same reason - the exact
    PCA is already built, so a second one would be wasted work - but scaled the
    opposite way, and the difference is not cosmetic. UMAP wants its init
    spread out (largest coordinate scaled to 10); t-SNE wants it collapsed
    almost to a point, ``std = 1e-4``, and openTSNE logs a warning above 1e-2
    because a large initial variance leaves the early-exaggeration phase with
    nothing to do and converges poorly. ``initialization.rescale`` is openTSNE's
    own helper for exactly this, and it is what ``initialization="pca"`` applies
    internally, so handing it our coordinates gives the same treatment the
    library would have given its own.

    A PCA start rather than a random one is the Kobak-Berens recommendation and
    is what makes t-SNE's global structure comparable to UMAP's here at all; a
    random init would make the two maps disagree about large-scale arrangement
    for reasons that have nothing to do with the data.

    The jitter is the same guard ``umap_init_from_pca`` documents: GEO contains
    resubmitted samples, so duplicate rows exist, and identical starting
    positions give a zero gradient between them forever. openTSNE's own
    ``initialization.pca`` adds jitter for this reason too.
    """
    from openTSNE import initialization

    coords = np.array(pca_coords[:, :n_components], dtype=np.float64)
    coords = initialization.rescale(coords)
    rng = np.random.RandomState(seed)
    coords += rng.normal(scale=1e-6, size=coords.shape)
    return np.ascontiguousarray(coords)


def run_tsne(affinities, init: np.ndarray, n_components: int, seed: int,
             n_jobs: int) -> np.ndarray:
    """t-SNE over every point in the corpus, from the shared affinity matrix.

    This drives openTSNE's low-level ``TSNEEmbedding`` rather than ``TSNE.fit``,
    but runs the identical two-phase schedule ``TSNE.fit`` runs - 250 iterations
    at exaggeration 12 and momentum 0.8, then 500 unexaggerated - because the
    only thing we need from the low level is the ability to hand *both* output
    dimensionalities the same precomputed affinities. Sharing them is safe:
    ``optimize`` applies exaggeration as ``P *= e`` and restores it with
    ``P /= e`` in a finally block, and the round trip was measured on this
    corpus's dtype at a relative error of 1.2e-16, which is float64 epsilon.
    Rebuilding P for the 3-d fit instead would cost about 2 GB for nothing.

    **The negative-gradient method is not a free choice, and it is why the 3-d
    build costs what it does.** openTSNE's interpolation accelerator (FIt-SNE)
    refuses more than two output dimensions outright - "Interpolation based
    t-SNE for >2 dimensions is currently unsupported (and generally a bad
    idea)" - so 2-d gets ``fft`` and 3-d must fall back to Barnes-Hut. FFT is
    linear in the point count; Barnes-Hut is n log n with a far larger
    constant, measured here at roughly 4x the per-iteration cost of FFT on
    50,000 points before the scaling difference is applied at all.

    ``learning_rate="auto"`` is passed through rather than resolved here so it
    tracks the corpus size: openTSNE reads it per phase as
    ``n_samples / exaggeration``, giving 78,547 while the exaggeration is on
    and 942,563 after it, which is the Belkina et al. scaling that keeps a
    corpus this size from needing far more iterations to settle.

    **``n_jobs`` does nothing on a stock macOS wheel, and that is measured, not
    assumed.** openTSNE parallelizes through OpenMP, and the PyPI macOS wheels
    are compiled without it: ``nm`` finds zero ``omp`` symbols and ``otool -L``
    no ``libomp`` linkage in ``_tsne``, ``kl_divergence`` or ``quad_tree``. So
    both fits run on one core no matter what ``--tsne-jobs`` says, which is the
    whole explanation for the 3-d stage's cost. Building openTSNE from source
    against ``libomp`` would recover roughly the core count, and it is
    deliberately not done here: threaded float summation makes the gradient
    order-dependent, and this is the artifact every coordinate derives from -
    the same argument that keeps ``--knn-jobs`` at 1 by default.
    """
    from openTSNE import TSNEEmbedding

    method = "fft" if n_components <= 2 else "bh"
    t0 = time.time()
    # Read back off the affinities rather than from the module constant: this
    # stage runs for hours and is read back from the log, so the one per-fit
    # line in it must not report a perplexity the run did not use.
    # `effective_perplexity_` is the one openTSNE actually calibrated to, which
    # it clamps against the neighbour count.
    perplexity = getattr(affinities, "effective_perplexity_",
                         getattr(affinities, "perplexity", TSNE_PERPLEXITY))
    log(f"t-SNE-{n_components}d fit on all {len(init):,} points, "
        f"perplexity={perplexity:g}, {method}, cosine on 512-d")
    emb = TSNEEmbedding(
        init, affinities,
        negative_gradient_method=method,
        learning_rate="auto",
        n_jobs=n_jobs,
        random_state=seed,
        verbose=True,
    )
    emb.optimize(n_iter=TSNE_EARLY_ITER, exaggeration=TSNE_EARLY_EXAGGERATION,
                 momentum=TSNE_MOMENTUM, inplace=True)
    emb.optimize(n_iter=TSNE_ITER, exaggeration=None,
                 momentum=TSNE_MOMENTUM, inplace=True)
    out = np.asarray(emb, dtype=np.float32)
    log(f"t-SNE-{n_components}d done in {time.time()-t0:.0f}s")
    del emb
    return out


def load_prior_stats(shape: dict) -> dict:
    """Start from the previous build record, so skipping a stage does not erase it.

    The three methods are independent stages and any of them can be skipped, so
    a run that rebuilds only t-SNE must not drop the UMAP settings the control
    rail reads back or the PCA spectrum ``validate_artifacts.py`` checks. Before
    this the record was rebuilt from scratch every run, which was harmless only
    while every run built everything.

    The record is carried forward **only when it describes the same corpus**. A
    different row count means every stage's numbers refer to something else, and
    a half-stale record that still looks complete is worse than one that is
    obviously fresh.
    """
    if not paths.PROJECTION_STATS_JSON.exists():
        return dict(shape)
    try:
        prior = json.loads(paths.PROJECTION_STATS_JSON.read_text())
    except (json.JSONDecodeError, OSError):
        log("previous projection_stats.json is unreadable; starting a fresh record")
        return dict(shape)
    if not all(prior.get(k) == v for k, v in shape.items()):
        log("previous projection_stats.json describes a different corpus; "
            "starting a fresh record")
        return dict(shape)
    prior.update(shape)
    return prior


def write_coords(coords: np.ndarray, path: Path) -> None:
    cols = ["x", "y", "z"][: coords.shape[1]]
    df = pd.DataFrame({c: coords[:, i].astype(np.float32) for i, c in enumerate(cols)})
    df.to_parquet(path, index=False)
    log(f"wrote {path.name} {coords.shape}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build joint PCA/UMAP projection artifacts.")
    ap.add_argument("--umap-neighbors", type=int, default=30,
                    help="UMAP n_neighbors. 30 since 2026-07-23: 15 wins on the "
                         "two local metrics but buys them with the large-scale "
                         "arrangement neither metric scores; see run_umap.")
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
    ap.add_argument("--skip-tsne", action="store_true",
                    help="Skip the t-SNE stage. It is the most expensive one in "
                         "the build, because 3-d t-SNE cannot use the "
                         "interpolation accelerator and falls back to "
                         "Barnes-Hut; see run_tsne.")
    ap.add_argument("--tsne-perplexity", type=int, default=TSNE_PERPLEXITY,
                    help="t-SNE perplexity. The neighbour graph is built at "
                         "3x this, which is openTSNE's own rule.")
    ap.add_argument("--tsne-jobs", type=int, default=-1,
                    help="Threads for the t-SNE optimization. NOTE: this is a "
                         "no-op on a stock macOS openTSNE wheel, which is "
                         "built without OpenMP, so the fit is single-threaded "
                         "whatever you pass. See run_tsne.")
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
    stats: dict = load_prior_stats({"n_archs4": int(n_archs4),
                                    "n_osdr": int(len(osdr_norm)),
                                    "total": int(total)})

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

    # --- The neighbour-based reductions: UMAP, then t-SNE -------------------
    # Both need random access to every row while they fit, so the 1.93 GB
    # normalized corpus is materialized once here and shared, rather than each
    # stage paying for its own copy. It is dropped as soon as the last graph
    # that needs it has been built, which is what keeps peak RSS during t-SNE's
    # optimization down to the affinity matrix plus the embedding.
    if not (args.skip_umap and args.skip_tsne):
        x = load_normalized_corpus(mm, n_archs4, osdr_norm, args.batch)

        if not args.skip_umap:
            t_umap = time.time()
            stats["umap_fit"] = "full corpus, no landmark subsample"
            stats["umap_method"] = "densmap" if args.densmap else "umap"
            stats["umap_neighbors"] = int(args.umap_neighbors)
            stats["umap_min_dist"] = 0.1
            stats["umap_metric"] = "cosine"
            stats["umap_input"] = "raw 512-d L2-normalized"
            stats["umap_init"] = "exact full-corpus PCA, scaled (not spectral)"
            if args.densmap:
                stats["dens_lambda"] = float(args.dens_lambda)

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
            del umap3, knn
            stats["umap_seconds"] = round(time.time() - t_umap, 1)
            save_stats()

        if not args.skip_tsne:
            t_tsne = time.time()
            perp = int(args.tsne_perplexity)
            stats["tsne_fit"] = "full corpus, no landmark subsample"
            stats["tsne_perplexity"] = perp
            stats["tsne_knn"] = 3 * perp
            stats["tsne_metric"] = "cosine"
            stats["tsne_input"] = "raw 512-d L2-normalized"
            stats["tsne_init"] = "exact full-corpus PCA, rescaled to std 1e-4"
            stats["tsne_early_exaggeration"] = TSNE_EARLY_EXAGGERATION
            stats["tsne_early_iter"] = TSNE_EARLY_ITER
            stats["tsne_iter"] = TSNE_ITER
            stats["tsne_momentum"] = TSNE_MOMENTUM
            stats["tsne_learning_rate"] = "auto (n/exaggeration)"
            # Recorded per dimensionality because they genuinely differ: the
            # interpolation accelerator refuses more than two output
            # dimensions, so the 3-d map is a Barnes-Hut layout and the 2-d one
            # is not. A reader comparing the two should be told that.
            stats["tsne2_negative_gradient"] = "FIt-SNE"
            stats["tsne3_negative_gradient"] = "Barnes-Hut"
            stats["tsne_jobs"] = int(args.tsne_jobs)

            t = time.time()
            aff = build_tsne_affinities(x, perp, args.seed, args.knn_jobs,
                                        args.tsne_jobs)
            # Includes the neighbour graph, which is the bulk of it.
            stats["tsne_affinity_seconds"] = round(time.time() - t, 1)
            save_stats()

            # From here t-SNE needs only P and the PCA init, and P is the
            # largest allocation in the whole build, so the corpus goes now
            # rather than at the end of the block.
            del x

            t = time.time()
            tsne2 = run_tsne(aff, tsne_init_from_pca(pca3, 2, args.seed), 2,
                             args.seed, args.tsne_jobs)
            stats["tsne2_seconds"] = round(time.time() - t, 1)
            write_coords(tsne2, paths.COORDS_TSNE2)
            save_stats()
            del tsne2

            t = time.time()
            tsne3 = run_tsne(aff, tsne_init_from_pca(pca3, 3, args.seed), 3,
                             args.seed, args.tsne_jobs)
            stats["tsne3_seconds"] = round(time.time() - t, 1)
            write_coords(tsne3, paths.COORDS_TSNE3)
            save_stats()
            del tsne3, aff
            stats["tsne_seconds"] = round(time.time() - t_tsne, 1)
        else:
            del x

    save_stats()
    log(f"ALL DONE. stats -> {paths.PROJECTION_STATS_JSON.name}")


if __name__ == "__main__":
    main()
