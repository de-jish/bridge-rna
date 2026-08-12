"""The data layer: global point order and color-by lookups.

These are the tests that matter most, because every defect they catch is silent.
A row that resolves to the wrong point does not raise; it just paints a liver
sample with the kidney colour, and nothing on screen says so.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

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
    for method in data.METHODS:
        for dims, width in (("2d", 2), ("3d", 3)):
            c = data.coords(method, dims)
            assert c.shape == (total, width), f"{method}/{dims} has shape {c.shape}"
    assert len(data.osdr_metadata()) == n_osdr


def test_species_labels_cover_the_whole_corpus(corpus):
    labels = data.species_labels()
    assert len(labels) == corpus["total"]
    assert set(np.unique(labels)) <= {"human", "mouse"}
    # OSDR is the mouse spaceflight corpus; every OSDR point must be mouse.
    assert (labels[corpus["n_archs4"]:] == "mouse").all()


def test_osdr_field_values_align_with_metadata_rows(corpus):
    n_osdr = corpus["n_osdr"]
    vals = data.osdr_field_values("tissue")
    assert len(vals) == n_osdr, f"{len(vals)} values for {n_osdr} points"
    assert vals.index[0] == 0 and vals.index[-1] == n_osdr - 1
    # Values must line up positionally with the metadata frame itself.
    meta = data.osdr_metadata()
    assert list(vals) == list(meta["tissue"].astype(str))


def test_unknown_field_degrades_instead_of_raising(corpus):
    vals = data.osdr_field_values("no_such_field")
    assert len(vals) == corpus["n_osdr"]
    assert set(vals) == {"Unknown"}


def test_the_metadata_frame_carries_no_derived_columns(corpus):
    """`osdr_metadata` returns the parquet, unmodified.

    It used to add a derived `flight_status`, collapsing OSDR's seven raw
    control arms onto Flight vs Ground for a color-by that no longer exists. A
    derived column with no reader is a column that goes stale silently, and the
    collapse is one the retrieval half deliberately refuses to make - a basal
    animal and a vivarium animal are different experiments - so it must not
    reappear here as an apparently-authoritative field of the label table.
    """
    meta = data.osdr_metadata()
    on_disk = pd.read_parquet(paths.OSDR_METADATA_PARQUET)
    assert list(meta.columns) == list(on_disk.columns)
    assert "flight_status" not in meta.columns
    # The raw arm is untouched: find.py prints it, and cohorts group by it.
    assert "spaceflight" in meta.columns


def test_method_availability_reflects_disk(corpus):
    for method in data.METHODS:
        assert data.method_available(method), f"{method} coordinates are missing"


def test_missing_method_returns_empty_not_error(corpus, monkeypatch, tmp_path):
    """A projection that was never built must yield an empty array, not a crash."""
    missing = tmp_path / "nope.parquet"
    monkeypatch.setitem(data.METHODS, "pca", {"2d": missing, "3d": missing})
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


# --- Tissue loaders ---------------------------------------------------------

def test_archs4_tissue_spans_the_archs4_block(corpus):
    labels = data.archs4_tissue()
    assert labels is not None, "the fixture writes a metadata join; it should load"
    assert len(labels) == corpus["n_archs4"]


def test_archs4_tissue_is_none_without_the_join(corpus, without_archs4_metadata):
    assert data.archs4_tissue() is None


def test_osdr_tissue_is_canonicalized_not_raw(corpus):
    """OSDR's raw values are hyper-specific; the loader must fold them.

    If this returned raw values, the "Tissue" color-by would put ARCHS4 and
    OSDR in disjoint category sets and the shared legend would be a fiction.
    """
    from manifold import tissue

    labels = data.osdr_tissue()
    assert len(labels) == corpus["n_osdr"]
    assert set(labels) <= set(tissue.BUCKETS)
    raw = set(data.osdr_field_values("tissue"))
    assert set(labels) != raw, "loader returned raw OSDR values, not buckets"
