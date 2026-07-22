"""The offline projection build, where "full corpus" has to mean what it says.

Two claims in `precompute/build_projections.py` are load-bearing and invisible
in the output: that the PCA is the *exact* decomposition of every point rather
than a good approximation of one, and that the corpus is streamed through both
passes in the fixed global order every other artifact is indexed by. A drifted
component or a transposed block would still produce a plausible-looking map, so
neither can be checked by eye.

`fit_exact_pca` is scored against `sklearn.decomposition.PCA` fit on the
materialized matrix, which is the thing it claims to be equal to.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from precompute import build_projections as bp  # noqa: E402
from precompute import validate_artifacts as va  # noqa: E402


def _corpus(n: int = 4000, d: int = bp.EMB_DIM, seed: int = 0) -> np.ndarray:
    """L2-normalized float32 vectors with planted low-rank structure.

    Isotropic noise would give a nearly flat spectrum, where every ordering of
    the components is equally defensible and a sign or ordering bug would not
    show up. Planting a handful of strong directions gives the leading
    components something unambiguous to find.
    """
    rng = np.random.default_rng(seed)
    basis = rng.normal(size=(8, d)).astype(np.float32)
    loadings = rng.normal(size=(n, 8)).astype(np.float32) * np.array(
        [6.0, 4.5, 3.0, 2.2, 1.6, 1.2, 0.9, 0.7], dtype=np.float32)
    x = loadings @ basis + rng.normal(scale=0.7, size=(n, d)).astype(np.float32)
    return bp.l2_normalize(x)


def _stream(x: np.ndarray, batch: int = 512):
    for s in range(0, len(x), batch):
        e = min(s + batch, len(x))
        yield s, e, x[s:e]


# --- The exactness claim ----------------------------------------------------

def test_exact_pca_matches_a_full_sklearn_fit():
    """Not "close to" a full-corpus fit. The same fit, to float64 round-off."""
    from sklearn.decomposition import PCA

    x = _corpus()
    components, evr, mean = bp.fit_exact_pca(_stream(x), len(x))

    ref = PCA(n_components=50, svd_solver="full").fit(x.astype(np.float64))

    assert np.allclose(mean, ref.mean_, atol=1e-9)
    assert np.abs(evr[:50] - ref.explained_variance_ratio_).max() < 1e-9

    # Components are only defined up to sign, and both sides apply the same
    # largest-absolute-entry rule, so they should agree outright rather than
    # only up to sign.
    for k in range(10):
        assert np.abs(np.dot(components[k], ref.components_[k])) > 1 - 1e-8
        assert np.allclose(components[k], ref.components_[k], atol=1e-6), (
            f"component {k} disagrees with sklearn beyond a sign flip")


def test_the_whole_spectrum_is_returned_and_sums_to_one():
    """All 512 eigenvalues, so the recorded profile is the real one.

    The previous build fit 50 components and reported a cumulative figure over
    those 50. An exact decomposition has no reason to truncate, and validate
    _artifacts.py now checks that the spectrum sums to 1 as evidence the fit
    was not silently swapped back for a truncated one.
    """
    x = _corpus()
    components, evr, _ = bp.fit_exact_pca(_stream(x), len(x))
    assert components.shape == (bp.EMB_DIM, bp.EMB_DIM)
    assert evr.shape == (bp.EMB_DIM,)
    assert evr.sum() == pytest.approx(1.0, abs=1e-12)
    assert (np.diff(evr) <= 1e-12).all(), "the spectrum is not sorted descending"
    assert (evr >= 0).all(), "a variance ratio came out negative"


def test_components_are_orthonormal():
    x = _corpus()
    components, _, _ = bp.fit_exact_pca(_stream(x), len(x))
    top = components[:20]
    assert np.abs(top @ top.T - np.eye(20)).max() < 1e-9


def test_signs_are_deterministic_across_block_sizes():
    """A rebuild must not mirror the map for reasons invisible in the data.

    Eigensolvers are free to return -v for v, and the streaming accumulation
    order changes with the batch size, so the sign rule is what keeps
    coordinates stable run to run.
    """
    x = _corpus()
    a, evr_a, _ = bp.fit_exact_pca(_stream(x, batch=512), len(x))
    b, evr_b, _ = bp.fit_exact_pca(_stream(x, batch=997), len(x))
    assert np.abs(evr_a - evr_b).max() < 1e-12
    assert np.abs(a[:20] - b[:20]).max() < 1e-9


def test_a_subsample_fit_is_measurably_different():
    """Guard the point of the change: sampling really does move the answer.

    If a 10% subsample reproduced the exact fit, "full corpus" would be a
    distinction without a difference and this test would be the place that says
    so. It does not, so the test pins the gap instead.
    """
    x = _corpus(n=6000, seed=3)
    exact, _, _ = bp.fit_exact_pca(_stream(x), len(x))
    sub = x[np.random.default_rng(1).choice(len(x), 600, replace=False)]
    approx, _, _ = bp.fit_exact_pca(_stream(sub), len(sub))
    worst = min(abs(float(np.dot(exact[k], approx[k]))) for k in range(10))
    assert worst < 0.9999, "a 10% subsample reproduced the exact components"


# --- The transform, and the order everything else is indexed by -------------

def test_transform_reproduces_sklearn_coordinates():
    from sklearn.decomposition import PCA

    x = _corpus()
    components, _, mean = bp.fit_exact_pca(_stream(x), len(x))
    got = bp.transform_pca(_stream(x), len(x), components[:3], mean)

    ref = PCA(n_components=3, svd_solver="full").fit_transform(x.astype(np.float64))
    assert got.shape == (len(x), 3)
    assert np.abs(got - ref).max() < 1e-4


def test_transform_is_centred_on_the_corpus():
    x = _corpus()
    components, _, mean = bp.fit_exact_pca(_stream(x), len(x))
    coords = bp.transform_pca(_stream(x), len(x), components[:3], mean)
    assert np.abs(coords.mean(axis=0)).max() < 1e-5
    # Variance along the axes must fall away, or the components are misordered.
    v = coords.var(axis=0)
    assert v[0] > v[1] > v[2]


def test_stream_covers_every_row_once_in_global_order():
    """ARCHS4 first in memmap order, then OSDR, which is the order every
    artifact is positionally joined on."""
    rng = np.random.default_rng(5)
    n_archs4, n_osdr = 250, 30
    mm = (rng.normal(size=(n_archs4, bp.EMB_DIM)) * 12).astype(np.float16)
    osdr = bp.l2_normalize(rng.normal(size=(n_osdr, bp.EMB_DIM)).astype(np.float32))

    seen = np.zeros(n_archs4 + n_osdr, dtype=int)
    rebuilt = np.empty((n_archs4 + n_osdr, bp.EMB_DIM), dtype=np.float32)
    for s, e, block in bp.stream_corpus(mm, n_archs4, osdr, batch=64):
        seen[s:e] += 1
        rebuilt[s:e] = block

    assert (seen == 1).all(), "a row was skipped or yielded twice"
    assert np.allclose(np.linalg.norm(rebuilt, axis=1), 1.0, atol=1e-5), (
        "the stream must L2-normalize every block")
    assert np.allclose(rebuilt[n_archs4:], osdr, atol=1e-6), (
        "the OSDR block is not at the tail of the global order")
    expect0 = bp.l2_normalize(np.asarray(mm[:1], dtype=np.float32))
    assert np.allclose(rebuilt[0], expect0[0], atol=1e-6)


def test_normalization_removes_the_magnitude_axis():
    """Invariant 2, at fixture scale: an unnormalized fit is magnitude-dominated.

    The real corpus measures PC1 at 57.8% before normalization and 40.9% after,
    and the same collapse has to be visible here or the fixture is not
    exercising the thing the invariant protects.

    The directions must share a strong mean, which is the part that is easy to
    get wrong: isotropic directions scaled by a random magnitude do *not*
    produce a large PC1, because that magnitude variance is spread evenly over
    all 512 axes instead of landing on one. Real encoder output is concentrated
    around a common direction, so scaling it varies the corpus along that one
    direction, and that is the axis PC1 finds.

    The concentration below is measured, not assumed: over a 40,000-sample read
    of the real ARCHS4 memmap, normalized vectors sit at mean cosine 0.929 to
    the corpus mean direction (10th-90th percentile 0.875 to 0.975), which is
    the anchor-to-noise ratio ANCHOR_SCALE reproduces in 512 dimensions.
    """
    rng = np.random.default_rng(11)
    ANCHOR_SCALE = 57.0
    anchor = rng.normal(size=(1, bp.EMB_DIM)).astype(np.float32)
    anchor /= np.linalg.norm(anchor)
    direction = bp.l2_normalize(
        anchor * ANCHOR_SCALE + rng.normal(size=(3000, bp.EMB_DIM)).astype(np.float32))
    assert (direction @ anchor.T).mean() == pytest.approx(0.929, abs=0.02), (
        "the fixture no longer reproduces the measured corpus concentration")
    scale = rng.uniform(6.7, 25.5, size=(3000, 1)).astype(np.float32)
    raw = direction * scale

    _, evr_raw, _ = bp.fit_exact_pca(_stream(raw), len(raw))
    _, evr_norm, _ = bp.fit_exact_pca(_stream(bp.l2_normalize(raw)), len(raw))

    # One axis, not a spectrum. The exact share depends on how much real
    # structure sits underneath the magnitude, which the fixture does not try to
    # reproduce, so what is asserted is the shape of the failure: unnormalized,
    # PC1 stands an order of magnitude clear of PC2 and is pure magnitude.
    assert evr_raw[0] > 10 * evr_raw[1], (
        f"no single axis dominated the unnormalized fit "
        f"(PC1 {evr_raw[0]:.3f}, PC2 {evr_raw[1]:.3f})")
    assert evr_norm[0] < evr_raw[0] / 10, (
        f"normalization did not remove the magnitude axis "
        f"(raw PC1 {evr_raw[0]:.3f} vs normalized {evr_norm[0]:.3f})")


# --- The quality gate's own arithmetic --------------------------------------
# `validate_artifacts.py --quality` is what stands between a scrambled
# projection and a green build, so the functions that define "truth" for it are
# checked against a reference rather than trusted.

def test_exact_knn_agrees_with_sklearn_on_cosine():
    """`_exact_knn` is the ground truth every recall number is measured against.

    It is hand-rolled block matrix multiplication, chosen so the measurement is
    exact rather than approximate, which means a bug in it would silently move
    every score it produces.
    """
    from sklearn.neighbors import NearestNeighbors

    x = _corpus(n=800, d=64, seed=9)
    got = va._exact_knn(x, k=10, block=137)  # block size deliberately not a divisor

    nn = NearestNeighbors(n_neighbors=11, metric="cosine").fit(x)
    _, ref = nn.kneighbors(x, n_neighbors=11)
    ref = ref[:, 1:]  # drop self

    assert got.shape == ref.shape
    agree = np.mean([len(set(got[i]) & set(ref[i])) / 10 for i in range(len(x))])
    assert agree > 0.999, f"only {agree:.4f} agreement with an exact cosine kNN"
    assert (got != np.arange(len(x))[:, None]).all(), "a point is its own neighbour"


def test_recall_is_the_fraction_of_true_neighbours_recovered():
    truth = np.array([[1, 2, 3], [0, 2, 3]])
    assert va._recall(truth, truth.copy()) == pytest.approx(1.0)
    assert va._recall(truth, np.array([[9, 8, 7], [9, 8, 7]])) == pytest.approx(0.0)
    # one of three recovered in each row
    assert va._recall(truth, np.array([[1, 8, 7], [0, 8, 7]])) == pytest.approx(1 / 3)


def test_purity_counts_only_points_with_a_known_label():
    labels = np.array(["Liver", "Liver", "Heart", "Unknown"], dtype=object)
    neighbours = np.array([[1, 2], [0, 2], [0, 1], [0, 1]])
    known = np.array([True, True, True, False])
    # point 0: neighbours Liver, Heart -> 0.5; point 1: Liver, Heart -> 0.5;
    # point 2: Liver, Liver -> 0.0. The Unknown row must not be counted at all.
    assert va._purity(neighbours, labels, known) == pytest.approx(1 / 3)
    assert np.isnan(va._purity(neighbours, labels, np.zeros(4, bool)))


def test_purity_of_a_perfectly_clustered_labelling_is_one():
    labels = np.array(["A", "A", "B", "B"], dtype=object)
    neighbours = np.array([[1], [0], [3], [2]])
    assert va._purity(neighbours, labels, np.ones(4, bool)) == pytest.approx(1.0)
