"""The data layer: global point order, vector gathers, and color-by lookups.

These are the tests that matter most, because every defect they catch is silent.
A point index that resolves to the wrong row does not raise; it just labels a
liver sample as kidney and reports a confident, wrong statistic.
"""

from __future__ import annotations

import numpy as np

import pytest

from manifold import data, paths


def test_global_order_is_archs4_then_osdr(corpus):
    """Row i < n_archs4 is ARCHS4; the rest are OSDR, in that fixed order."""
    n_archs4, n_osdr, total = data.counts()
    assert (n_archs4, n_osdr, total) == (corpus["n_archs4"], corpus["n_osdr"], corpus["total"])

    pm = data.points_meta()
    assert len(pm) == total
    assert (pm["dataset"].to_numpy()[:n_archs4] == 0).all()
    assert (pm["dataset"].to_numpy()[n_archs4:] == 1).all()
    # src_index must restart at 0 for the OSDR block, since it indexes the npy.
    assert pm["src_index"].to_numpy()[n_archs4] == 0
    assert pm["src_index"].to_numpy()[-1] == n_osdr - 1


def test_every_artifact_shares_the_global_order(corpus):
    """Coordinates, identity table, and the OSDR metadata all agree on length."""
    n_archs4, n_osdr, total = data.counts()
    for method in ("pca", "umap"):
        for dims, width in (("2d", 2), ("3d", 3)):
            c = data.coords(method, dims)
            assert c.shape == (total, width), f"{method}/{dims} has shape {c.shape}"
    assert len(data.osdr_metadata()) == n_osdr
    assert len(data.archs4_geo()) == n_archs4


def test_normalized_vectors_resolve_the_right_rows(corpus):
    """A gather over mixed corpora must place each vector at its request position.

    The ARCHS4 branch sorts indices for memmap locality and then scatters the
    results back. An inverted permutation there would silently return every
    selected point's vector under a different point's identity.
    """
    n_archs4, n_osdr, total = data.counts()
    rng = np.random.default_rng(3)
    # Deliberately unsorted, with duplicates removed, spanning both corpora.
    idx = np.concatenate([
        rng.choice(n_archs4, size=25, replace=False),
        rng.choice(np.arange(n_archs4, total), size=10, replace=False),
    ])
    rng.shuffle(idx)

    got = data.normalized_vectors(idx)
    assert got.shape == (len(idx), 512)
    assert np.allclose(np.linalg.norm(got, axis=1), 1.0, atol=1e-5)

    # Reference: fetch each index one at a time, which needs no permutation.
    for pos, i in enumerate(idx):
        one = data.normalized_vectors(np.array([i]))[0]
        assert np.allclose(got[pos], one, atol=1e-6), f"row {pos} (index {i}) misplaced"


def test_normalized_vectors_match_the_source_arrays(corpus):
    """The gather reads the real memmap/npy, not a reordered copy of them."""
    n_archs4, _, total = data.counts()
    mm = data._archs4_memmap()
    osdr = data._osdr_embeddings()

    def unit(v):
        return v / np.linalg.norm(v)

    for i in (0, 1, n_archs4 - 1):
        assert np.allclose(data.normalized_vectors(np.array([i]))[0],
                           unit(np.asarray(mm[i], dtype=np.float32)), atol=1e-5)
    for j in (0, 5, len(osdr) - 1):
        assert np.allclose(data.normalized_vectors(np.array([n_archs4 + j]))[0],
                           unit(osdr[j].astype(np.float32)), atol=1e-6)


def test_species_labels_cover_the_whole_corpus(corpus):
    labels = data.species_labels()
    assert len(labels) == corpus["total"]
    assert set(np.unique(labels)) <= {"human", "mouse"}
    # OSDR is the mouse spaceflight corpus; every OSDR point must be mouse.
    assert (labels[corpus["n_archs4"]:] == "mouse").all()


def test_osdr_field_values_align_with_metadata_rows(corpus):
    n_osdr = corpus["n_osdr"]
    for field in data.OSDR_FIELDS:
        vals = data.osdr_field_values(field)
        assert len(vals) == n_osdr, f"{field} has {len(vals)} values for {n_osdr} points"
        assert vals.index[0] == 0 and vals.index[-1] == n_osdr - 1
    # Values must line up positionally with the metadata frame itself.
    meta = data.osdr_metadata()
    assert list(data.osdr_field_values("tissue")) == list(meta["tissue"].astype(str))


def test_unknown_field_degrades_instead_of_raising(corpus):
    vals = data.osdr_field_values("no_such_field")
    assert len(vals) == corpus["n_osdr"]
    assert set(vals) == {"Unknown"}


@pytest.mark.parametrize("raw,expected", [
    ("Space Flight", "Space Flight"),
    ("spaceflight", "Space Flight"),
    ("Ground Control", "Ground"),
    ("Ground control", "Ground"),
    ("Ground Control Rerun", "Ground"),
    ("Basal Control", "Ground"),
    ("Vivarium Control", "Ground"),
    ("Cohort Control #1", "Ground"),
    ("", "Unknown"),
    ("nan", "Unknown"),
])
def test_flight_status_is_the_binary_contrast(raw, expected):
    assert data._flight_status(raw) == expected


def test_control_arms_stay_distinct_under_the_raw_field(corpus):
    """The coarse Flight/Ground field must not be the only view of the arm.

    Basal, Vivarium, and Ground controls are different experiments. Collapsing
    them is fine for the headline contrast but must not be the only option, or
    real structure becomes invisible.
    """
    assert "spaceflight" in data.OSDR_FIELDS and "flight_status" in data.OSDR_FIELDS
    arms = set(data.osdr_field_values("spaceflight"))
    statuses = set(data.osdr_field_values("flight_status"))
    assert len(arms) >= len(statuses)
    assert statuses <= {"Space Flight", "Ground", "Unknown"}


def test_method_availability_reflects_disk(corpus):
    assert data.method_available("pca")
    assert data.method_available("umap")


def test_missing_method_returns_empty_not_error(corpus, monkeypatch, tmp_path):
    """A projection that was never built must yield an empty array, not a crash."""
    missing = tmp_path / "nope.parquet"
    monkeypatch.setitem(data.METHODS, "pca", {"2d": missing, "3d": missing, "density": "pca2"})
    data.coords.cache_clear()
    try:
        assert data.coords("pca", "2d").shape == (0, 2)
        assert not data.method_available("pca")
    finally:
        data.coords.cache_clear()


def test_cache_dir_is_the_fixture_not_the_repo():
    """Guard the guard: a leaked env override would make the suite test prod data."""
    assert "bridge-manifold-fixture-" in str(paths.CACHE_DIR)
    assert "bridge-manifold-fixture-" in str(paths.BRIDGE_RNA_ROOT)
