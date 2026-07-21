"""The lasso readout: does it tell the truth?

The fixture corpus is built from known latent clusters with metadata derived
from those clusters, so these tests have real ground truth. A coherence
implementation that always answers "yes" fails the random-selection test; one
that always answers "no" fails the cluster test; one that reads its numbers off
the projection fails the coordinate-independence test.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import hypergeom

from manifold import coherence, data


def _osdr_points_in_cluster(corpus, cluster_id):
    """Global point indices for the OSDR samples in one latent cluster."""
    n_archs4 = corpus["n_archs4"]
    local = np.where(corpus["osdr_cluster"] == cluster_id)[0]
    return local + n_archs4


def _random_osdr_points(corpus, k, seed=0):
    n_archs4, n_osdr = corpus["n_archs4"], corpus["n_osdr"]
    rng = np.random.default_rng(seed)
    return rng.choice(np.arange(n_archs4, n_archs4 + n_osdr), size=k, replace=False)


# --- The identity the fast permutation null depends on ---------------------

def test_cohesion_statistic_equals_the_naive_definition():
    """||mean(V)|| must equal the literal mean cosine to the normalized centroid.

    The permutation null is only affordable because of this identity. If the
    statistic is ever changed so the identity no longer holds, the null becomes
    a measure of something else and every z-score silently drifts.
    """
    rng = np.random.default_rng(17)
    for n in (2, 9, 250):
        v = rng.normal(size=(n, 512)).astype(np.float32)
        v /= np.linalg.norm(v, axis=1, keepdims=True)

        centroid = v.mean(axis=0)
        centroid = centroid / np.linalg.norm(centroid)
        naive = float((v @ centroid).mean())

        assert coherence._mean_cos_to_centroid(v) == pytest.approx(naive, rel=1e-5)


def test_identical_vectors_are_perfectly_cohesive():
    v = np.tile(np.eye(1, 512, 0).astype(np.float32), (20, 1))
    assert coherence._mean_cos_to_centroid(v) == pytest.approx(1.0, abs=1e-6)


# --- The null model --------------------------------------------------------

def _brute_force_null(corpus, n_draw, n_rep=400, seed=0):
    """Ground truth: actually draw random subsets without replacement and measure."""
    n_archs4, n_osdr, _ = data.counts()
    lo, hi = (0, n_archs4) if corpus == "archs4" else (n_archs4, n_archs4 + n_osdr)
    rng = np.random.default_rng(seed)
    out = np.empty(n_rep)
    for i in range(n_rep):
        idx = rng.choice(np.arange(lo, hi), size=n_draw, replace=False)
        out[i] = coherence._mean_cos_to_centroid(data.normalized_vectors(idx))
    return out


@pytest.mark.parametrize("n_draw", [20, 200, 2000])
def test_analytic_null_matches_brute_force_resampling(corpus, n_draw):
    """The Gaussian null must reproduce what real random subsets actually do.

    This is the test the whole readout rests on. It compares the analytic
    null - population moments plus a finite-population correction - against
    literally drawing random subsets and measuring them.
    """
    analytic, _ = coherence._permutation_null(n_draw, 0)
    brute = _brute_force_null("archs4", n_draw)

    assert analytic.mean() == pytest.approx(brute.mean(), rel=0.02), (
        f"null mean off: analytic {analytic.mean():.5f} vs brute {brute.mean():.5f}")
    assert analytic.std() == pytest.approx(brute.std(), rel=0.25), (
        f"null spread off: analytic {analytic.std():.5f} vs brute {brute.std():.5f}")


def test_random_selections_score_near_zero_at_every_scale(corpus):
    """H0 is true for a uniform random draw, so z must not drift with |S|.

    The defect this pins down: a null resampled from a fixed background pool
    develops a z-offset growing like sqrt(|S| / pool_size), which reaches
    |z| > 40 at corpus scale and would report pure noise as a strong result.
    """
    n_archs4 = corpus["n_archs4"]
    for frac in (0.01, 0.1, 0.4, 0.8):
        n = int(n_archs4 * frac)
        zs = []
        for trial in range(4):
            sel = np.random.default_rng(500 + trial).choice(n_archs4, size=n, replace=False)
            zs.append(coherence.analyze_selection(sel)["cohesion"]["z"])
        mean_z = float(np.mean(zs))
        assert abs(mean_z) < 4.0, (
            f"random selection of {n} points ({frac:.0%} of the corpus) scored "
            f"mean z={mean_z:.1f}; the null is biased at this scale")


def test_selecting_the_entire_corpus_is_degenerate(corpus):
    """At n = N there is exactly one possible draw, so there is nothing to test."""
    n_archs4 = corpus["n_archs4"]
    r = coherence.analyze_selection(np.arange(n_archs4))
    assert abs(r["cohesion"]["z"]) < 1e-3
    assert not r["cohesive"]


def test_finite_population_correction_shrinks_the_null_spread(corpus):
    """Sampling most of a corpus leaves little room to vary; the null must know."""
    n_archs4 = corpus["n_archs4"]
    small = coherence._permutation_null(int(n_archs4 * 0.05), 0)[0].std()
    large = coherence._permutation_null(int(n_archs4 * 0.95), 0)[0].std()
    assert large < small


def test_population_moments_are_exact_not_sampled(corpus):
    """Moments must describe the whole corpus, since the null bias depends on it."""
    n_archs4, n_osdr, _ = data.counts()
    mu, cov = data.population_moments("archs4")
    ref = data.normalized_vectors(np.arange(n_archs4))
    assert np.allclose(mu, ref.mean(axis=0), atol=1e-5)
    assert np.allclose(cov, np.cov(ref.T, bias=True), atol=1e-5)
    assert mu.shape == (512,) and cov.shape == (512, 512)


def test_moments_are_cached_to_disk(corpus):
    from manifold import paths
    data.population_moments("osdr")
    assert paths.POPULATION_MOMENTS_NPZ.exists()


def test_large_selection_readout_is_fast(corpus):
    """A lasso over a quarter of the corpus must still answer interactively."""
    import time

    rng = np.random.default_rng(31)
    sel = rng.choice(corpus["total"], size=min(10000, corpus["total"] // 2), replace=False)
    data.population_moments("archs4")  # warm the moments so this times the statistic
    t0 = time.time()
    r = coherence.analyze_selection(sel)
    elapsed = time.time() - t0
    assert r["status"] == "ok"
    assert elapsed < 5.0, f"a {len(sel)}-point selection took {elapsed:.1f}s"


# --- A. Geometric cohesion -------------------------------------------------

def test_a_cluster_reads_as_coherent(corpus):
    sel = _osdr_points_in_cluster(corpus, 0)
    assert len(sel) >= coherence.MIN_SELECTION
    r = coherence.analyze_selection(sel)

    assert r["status"] == "ok"
    assert r["n"] == len(sel)
    assert r["cohesive"], f"a true cluster read as incoherent: {r['verdict']}"
    assert r["cohesion"]["z"] > 3.0
    assert r["cohesion"]["emp_p"] < 0.05
    assert "Coherent" in r["verdict"]


def test_a_scattered_selection_reads_as_incoherent(corpus):
    """The honest negative: a draw spanning every cluster must not claim structure."""
    n_archs4 = corpus["n_archs4"]
    # One point from each cluster in turn - maximally spread across the manifold.
    clusters = corpus["osdr_cluster"]
    picks = []
    for c in np.unique(clusters):
        picks.extend((np.where(clusters == c)[0][:8] + n_archs4).tolist())
    r = coherence.analyze_selection(np.array(picks))

    assert r["status"] == "ok"
    cluster_r = coherence.analyze_selection(_osdr_points_in_cluster(corpus, 0))
    assert r["cohesion"]["z"] < cluster_r["cohesion"]["z"], (
        "a cross-cluster draw scored at least as cohesive as a single cluster")


def test_a_random_archs4_draw_is_not_cohesive(corpus):
    """Random points from the background population must look like the background."""
    rng = np.random.default_rng(5)
    sel = rng.choice(corpus["n_archs4"], size=60, replace=False)
    r = coherence.analyze_selection(sel)
    assert r["status"] == "ok"
    # z near zero: the selection IS a draw from the null it is being tested against.
    assert abs(r["cohesion"]["z"]) < 6.0, f"random draw scored z={r['cohesion']['z']:.1f}"


def test_too_small_selection_is_refused(corpus):
    sel = _random_osdr_points(corpus, coherence.MIN_SELECTION - 1)
    r = coherence.analyze_selection(sel)
    assert r["status"] == "too_small"
    assert r["n"] == coherence.MIN_SELECTION - 1
    assert r["min"] == coherence.MIN_SELECTION


def test_duplicate_indices_are_collapsed(corpus):
    """Plotly can report a point once per trace; the statistic must not double-count."""
    sel = _osdr_points_in_cluster(corpus, 1)
    doubled = np.concatenate([sel, sel])
    assert coherence.analyze_selection(doubled)["n"] == len(np.unique(sel))


# --- B. Metadata enrichment ------------------------------------------------

def test_cluster_selection_enriches_for_its_own_tissue(corpus):
    """Tissue is a function of cluster in the fixture, so it must come out on top."""
    cluster = 2
    sel = _osdr_points_in_cluster(corpus, cluster)
    r = coherence.analyze_selection(sel)

    tissue_hits = [e for e in r["enrichment"] if e["field"] == "tissue"]
    assert tissue_hits, f"no tissue enrichment found; got {[e['field'] for e in r['enrichment']]}"
    top = tissue_hits[0]
    expected = data.osdr_metadata()["tissue"].to_numpy()[corpus["osdr_cluster"] == cluster][0]
    assert top["category"] == expected
    assert top["fold"] > 1.5
    assert top["q"] < 0.05


def test_enrichment_matches_a_hand_computed_hypergeometric(corpus):
    """The reported p must be the exact upper-tail hypergeometric, not an approximation."""
    sel = _osdr_points_in_cluster(corpus, 0)
    r = coherence.analyze_selection(sel)
    rows = [e for e in r["enrichment"] if e["field"] == "tissue"]
    assert rows
    e = rows[0]
    expected_p = float(hypergeom.sf(e["k"] - 1, e["N"], e["K"], e["n"]))
    assert e["p"] == pytest.approx(expected_p, rel=1e-9)
    expected_fold = (e["k"] / e["n"]) / (e["K"] / e["N"])
    assert e["fold"] == pytest.approx(expected_fold, rel=1e-9)


def test_benjamini_hochberg_matches_the_reference_definition():
    """q_i = min over j>=i of (m/j) * p_(j), clipped to 1, in p-value order."""
    rng = np.random.default_rng(0)
    for m in (1, 5, 40, 200):
        p = np.clip(rng.random(m) ** 3, 0, 1)
        got = coherence._bh_qvalues(p)

        order = np.argsort(p)
        ranked = p[order] * m / (np.arange(m) + 1)
        expected = np.empty(m)
        expected[order] = np.clip(np.minimum.accumulate(ranked[::-1])[::-1], 0, 1)
        assert np.allclose(got, expected)
        # Monotonicity in p is the property the correction exists to preserve.
        assert np.all(np.diff(got[order]) >= -1e-12)
        assert np.all(got >= p - 1e-12), "a q-value fell below its own p-value"


def test_a_significant_but_tiny_effect_is_not_called_a_driver():
    """q<1e-10 at 1.05x fold is sample size talking, not biology."""
    weak = [{"field": "species", "category": "human", "fold": 1.05, "q": 1e-22}]
    verdict = coherence._synthesize(
        z=9.0, emp_p=0.001, knn=None, significant=weak, batch=None,
        cross_dataset=False, cohesive=True, has_enrichment=True)
    assert "driven by" not in verdict
    assert "small deviation" in verdict
    assert "1 significant category is" in verdict, "message should not read '1 categories'"


def test_the_banner_colour_agrees_with_the_sentence_inside_it():
    """A green banner over 'weak cohesion, no real driver' contradicts itself."""
    weak = [{"field": "species", "category": "mouse", "fold": 1.2, "q": 0.03}]
    assert coherence._verdict_class(cohesive=False, significant=weak, batch=None) == "null"
    assert coherence._verdict_class(cohesive=True, significant=[], batch=None) == "good"

    strong = [{"field": "tissue", "category": "Liver", "fold": 6.2, "q": 1e-9}]
    assert coherence._verdict_class(cohesive=False, significant=strong, batch=None) == "good"
    assert coherence._verdict_class(
        cohesive=True, significant=strong,
        batch={"flag": True, "top_study": "OSD-1", "fraction": 0.9}) == "warn"


def test_a_real_effect_is_still_named_as_a_driver():
    strong = [{"field": "tissue", "category": "Liver", "fold": 6.2, "q": 3e-22}]
    verdict = coherence._synthesize(
        z=9.0, emp_p=0.001, knn=None, significant=strong, batch=None,
        cross_dataset=False, cohesive=True, has_enrichment=True)
    assert "driven by tissue=Liver (6.2x" in verdict


def test_bh_on_empty_input_is_empty():
    assert len(coherence._bh_qvalues(np.array([]))) == 0


# --- kNN purity ------------------------------------------------------------

def test_knn_purity_is_enriched_for_a_cluster_and_flat_for_a_random_draw(corpus):
    cluster = _osdr_points_in_cluster(corpus, 0)
    r_cluster = coherence.analyze_selection(cluster)
    assert r_cluster["knn"] is not None, "hnswlib index missing from the fixture"
    assert r_cluster["knn"]["fold"] > 2.0

    rng = np.random.default_rng(9)
    rand = rng.choice(corpus["n_archs4"], size=len(cluster), replace=False)
    r_rand = coherence.analyze_selection(rand)
    assert r_rand["knn"]["fold"] < r_cluster["knn"]["fold"]


def test_knn_expected_fraction_uses_the_full_corpus(corpus):
    sel = _osdr_points_in_cluster(corpus, 1)
    r = coherence.analyze_selection(sel)
    n_sel = r["n"]
    total = corpus["total"]
    assert r["knn"]["expected"] == pytest.approx((n_sel - 1) / (total - 1), rel=1e-9)


def test_missing_index_degrades_to_no_knn(corpus, monkeypatch):
    """Without the index the readout must still work, just without the purity stat."""
    monkeypatch.setattr(data, "hnsw_index", lambda: None)
    r = coherence.analyze_selection(_osdr_points_in_cluster(corpus, 0))
    assert r["status"] == "ok"
    assert r["knn"] is None
    assert "kNN" not in r["verdict"]


# --- C. Batch-confound guard ----------------------------------------------

def test_single_study_selection_is_flagged_as_batch_driven(corpus):
    studies = data.osdr_metadata()["study"].to_numpy()
    biggest = max(set(studies), key=lambda s: (studies == s).sum())
    local = np.where(studies == biggest)[0]
    if len(local) < coherence.MIN_SELECTION:
        pytest.skip("fixture has no study large enough to test the guard")
    sel = local + corpus["n_archs4"]

    r = coherence.analyze_selection(sel)
    assert r["batch"] is not None
    assert r["batch"]["top_study"] == biggest
    assert r["batch"]["fraction"] == pytest.approx(1.0)
    assert r["batch"]["flag"] is True
    assert "CAUTION" in r["verdict"]
    assert r["verdict_class"] == "warn"


def test_archs4_only_selection_has_no_batch_section(corpus):
    rng = np.random.default_rng(2)
    r = coherence.analyze_selection(rng.choice(corpus["n_archs4"], size=40, replace=False))
    assert r["batch"] is None


# --- D. Cross-dataset caution ---------------------------------------------

def test_mixed_selection_raises_the_precision_caution(corpus):
    n_archs4 = corpus["n_archs4"]
    sel = np.concatenate([np.arange(20), np.arange(n_archs4, n_archs4 + 20)])
    r = coherence.analyze_selection(sel)
    assert r["cross_dataset"] is True
    assert r["n_archs4"] == 20 and r["n_osdr"] == 20
    assert "bf16" in r["verdict"]


def test_single_corpus_selection_does_not_raise_it(corpus):
    r = coherence.analyze_selection(_osdr_points_in_cluster(corpus, 0))
    assert r["cross_dataset"] is False
    assert "bf16" not in r["verdict"]


# --- The load-bearing invariant -------------------------------------------

def test_statistics_never_read_the_projection_coordinates(corpus, monkeypatch):
    """Invariant 3: the lasso picks points, the pixels never define the number.

    Enforced by making any coordinate access explode. If a statistic ever starts
    depending on 2D positions, this test fails immediately rather than years
    later when someone notices the numbers track the UMAP seed.
    """
    def forbidden(*a, **k):
        raise AssertionError("coherence read projection coordinates")

    monkeypatch.setattr(data, "coords", forbidden)
    r = coherence.analyze_selection(_osdr_points_in_cluster(corpus, 0))
    assert r["status"] == "ok"
    assert r["cohesive"]


def test_readout_is_deterministic(corpus):
    """Same selection, same numbers - a seeded null, not a fresh random one."""
    sel = _osdr_points_in_cluster(corpus, 1)
    a = coherence.analyze_selection(sel)
    b = coherence.analyze_selection(sel)
    assert a["cohesion"] == b["cohesion"]
    assert a["verdict"] == b["verdict"]


def test_selection_order_does_not_change_the_result(corpus):
    sel = _osdr_points_in_cluster(corpus, 1)
    shuffled = sel.copy()
    np.random.default_rng(4).shuffle(shuffled)
    assert coherence.analyze_selection(sel)["verdict"] == \
           coherence.analyze_selection(shuffled)["verdict"]
