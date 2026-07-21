"""Figure construction and the sampling that feeds it.

The invariant these protect is the one that connects the picture to the science:
every drawn glyph must carry the *global* corpus index of the sample it depicts,
because that index is what a lasso hands to the 512-d readout. A glyph drawn at
the right pixel with the wrong customdata produces a confident statistic about
samples the user never selected.
"""

from __future__ import annotations

import numpy as np
import pytest

from manifold import data, render, sampling, theme


# --- Stratified sampling ---------------------------------------------------

def test_sample_respects_the_budget_and_the_species_mix():
    species = np.concatenate([np.zeros(8000, np.int8), np.ones(2000, np.int8)])
    idx = sampling.stratified_archs4_sample(species, n_target=1000, seed=0)

    assert len(idx) == pytest.approx(1000, abs=2)
    assert len(np.unique(idx)) == len(idx), "sampled the same point twice"
    assert idx.max() < len(species)
    assert (np.diff(idx) > 0).all(), "indices are not sorted"

    human_frac = (species[idx] == 0).mean()
    assert human_frac == pytest.approx(0.8, abs=0.02), "species proportions were not preserved"


def test_sample_returns_everything_when_the_pool_is_small():
    species = np.zeros(50, np.int8)
    idx = sampling.stratified_archs4_sample(species, n_target=1000)
    assert len(idx) == 50


def test_sample_is_deterministic_for_a_seed():
    species = np.concatenate([np.zeros(500, np.int8), np.ones(500, np.int8)])
    a = sampling.stratified_archs4_sample(species, 200, seed=3)
    b = sampling.stratified_archs4_sample(species, 200, seed=3)
    c = sampling.stratified_archs4_sample(species, 200, seed=4)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_a_rare_class_is_never_dropped():
    """One mouse among 10,000 humans must still be eligible to appear."""
    species = np.zeros(10000, np.int8)
    species[123] = 1
    idx = sampling.stratified_archs4_sample(species, n_target=100, seed=0)
    assert 123 in idx


def test_viewport_mask_selects_the_window():
    coords = np.array([[0.0, 0.0], [5.0, 5.0], [-3.0, 2.0], [1.0, 1.0]])
    mask = sampling.viewport_mask(coords, (-1.0, 2.0, -1.0, 2.0))
    assert list(mask) == [True, False, False, True]


def test_mask_restricts_the_sample_pool():
    species = np.zeros(1000, np.int8)
    mask = np.zeros(1000, bool)
    mask[200:260] = True
    idx = sampling.stratified_archs4_sample(species, n_target=1000, seed=0, mask=mask)
    assert set(idx) == set(range(200, 260))


# --- Figure construction ---------------------------------------------------

ALL_LAYERS = ["archs4", "osdr", "density"]


def _customdata_indices(fig):
    out = []
    for tr in fig.data:
        if tr.customdata is not None:
            out.extend(int(c[0]) if isinstance(c, (list, tuple)) else int(c) for c in tr.customdata)
    return np.array(out, dtype=np.int64)


def test_figure_builds_for_every_control_combination(corpus):
    for method in ("pca", "umap"):
        for dims in ("2d", "3d"):
            for color_by in ("species", "flight_status", "tissue", "study"):
                fig, legend, badges = render.build_figure(
                    method, dims, color_by, ALL_LAYERS, 2000, None)
                assert len(fig.data) > 0, f"{method}/{dims}/{color_by} drew nothing"
                assert fig.layout.paper_bgcolor == theme.PLOT_BG


def test_every_glyph_carries_its_global_index(corpus):
    """customdata must be the global corpus index, and land on the right coords."""
    coords = data.coords("pca", "2d")
    fig, _, _ = render.build_figure("pca", "2d", "tissue", ALL_LAYERS, 1500, None)

    seen = 0
    for tr in fig.data:
        if tr.customdata is None:
            continue
        idx = np.array([int(c[0]) if isinstance(c, (list, tuple)) else int(c) for c in tr.customdata], dtype=np.int64)
        assert idx.min() >= 0 and idx.max() < corpus["total"]
        assert np.allclose(np.asarray(tr.x, dtype=np.float32), coords[idx, 0], atol=1e-5)
        assert np.allclose(np.asarray(tr.y, dtype=np.float32), coords[idx, 1], atol=1e-5)
        seen += len(idx)
    assert seen > 0


def test_osdr_indices_land_in_the_osdr_block(corpus):
    """OSDR glyphs must carry indices >= n_archs4, never raw 0-based OSDR rows."""
    fig, _, _ = render.build_figure("pca", "2d", "tissue", ["osdr"], 1000, None)
    idx = _customdata_indices(fig)
    assert len(idx) == corpus["n_osdr"]
    assert idx.min() >= corpus["n_archs4"]
    assert sorted(idx) == list(range(corpus["n_archs4"], corpus["total"]))


def test_archs4_indices_land_in_the_archs4_block(corpus):
    fig, _, _ = render.build_figure("pca", "2d", "species", ["archs4"], 1000, None)
    idx = _customdata_indices(fig)
    assert idx.max() < corpus["n_archs4"]


def test_layer_toggles_actually_remove_layers(corpus):
    n_osdr = corpus["n_osdr"]
    only_osdr = _customdata_indices(render.build_figure(
        "pca", "2d", "tissue", ["osdr"], 1000, None)[0])
    assert len(only_osdr) == n_osdr

    fig_none, _, badges = render.build_figure("pca", "2d", "tissue", [], 1000, None)
    assert len(_customdata_indices(fig_none)) == 0
    assert badges == []


def test_density_underlay_is_placed_at_the_recorded_extent(corpus):
    fig, _, _ = render.build_figure("pca", "2d", "tissue", ALL_LAYERS, 1000, None)
    images = fig.layout.images
    assert len(images) == 1
    ext = data.stats()["density_pca2"]
    img = images[0]
    assert img.x == pytest.approx(ext["x0"])
    assert img.y == pytest.approx(ext["y1"])
    assert img.sizex == pytest.approx(ext["x1"] - ext["x0"])
    assert img.sizey == pytest.approx(ext["y1"] - ext["y0"])
    assert img.layer == "below"


def test_density_is_not_drawn_in_3d(corpus):
    fig, _, _ = render.build_figure("pca", "3d", "tissue", ALL_LAYERS, 1000, None)
    assert not fig.layout.images


def test_budget_caps_the_live_archs4_glyphs(corpus):
    for budget in (200, 800):
        fig, _, _ = render.build_figure("pca", "2d", "species", ["archs4"], budget, None)
        drawn = len(_customdata_indices(fig))
        assert drawn <= budget + 4, f"budget {budget} drew {drawn}"


def test_viewport_restricts_the_sample_to_the_window(corpus):
    coords = data.coords("pca", "2d")[: corpus["n_archs4"]]
    x0, x1 = np.percentile(coords[:, 0], [40, 60])
    y0, y1 = np.percentile(coords[:, 1], [40, 60])
    fig, _, _ = render.build_figure("pca", "2d", "species", ["archs4"], 2000, (x0, x1, y0, y1))
    idx = _customdata_indices(fig)
    assert len(idx) > 0
    pts = coords[idx]
    assert (pts[:, 0] >= x0 - 1e-6).all() and (pts[:, 0] <= x1 + 1e-6).all()
    assert (pts[:, 1] >= y0 - 1e-6).all() and (pts[:, 1] <= y1 + 1e-6).all()


def test_legend_reports_categories_with_counts(corpus):
    _, legend, _ = render.build_figure("pca", "2d", "tissue", ["osdr"], 1000, None)
    items = legend["items"]
    assert items, "no legend produced for a categorical color-by"
    assert sum(i["count"] for i in items) == corpus["n_osdr"]
    assert len({i["color"] for i in items}) == len(items), "two categories share a color"
    labels = [i["label"] for i in items]
    assert len(labels) == len(set(labels))


def test_high_cardinality_color_by_collapses_into_other(corpus):
    """Past the palette size, the tail must fold into one neutral Other bucket."""
    _, legend, _ = render.build_figure("pca", "2d", "study", ["osdr"], 1000, None)
    labels = [i["label"] for i in legend["items"]]
    n_studies = data.osdr_field_values("study").nunique()
    if n_studies > render.TOP_N:
        assert "Other" in labels
        assert len(labels) <= render.TOP_N + 1
        other = next(i for i in legend["items"] if i["label"] == "Other")
        assert other["color"] == theme.OTHER_COLOR


def test_missing_projection_renders_a_message_not_a_crash(corpus, monkeypatch, tmp_path):
    missing = tmp_path / "absent.parquet"
    monkeypatch.setitem(data.METHODS, "umap", {"2d": missing, "3d": missing, "density": "umap2"})
    data.coords.cache_clear()
    try:
        fig, legend, badges = render.build_figure("umap", "2d", "tissue", ALL_LAYERS, 1000, None)
        assert len(fig.data) == 0
        assert any("not available" in b for b in badges)
        assert fig.layout.annotations
    finally:
        data.coords.cache_clear()


def test_osdr_markers_are_visually_distinct_from_the_cloud(corpus):
    """The spaceflight overlay has to read above a 100k-point background."""
    fig, _, _ = render.build_figure("pca", "2d", "tissue", ALL_LAYERS, 1000, None)
    archs4 = [t for t in fig.data if t.name == "ARCHS4"]
    osdr = [t for t in fig.data if t.name != "ARCHS4"]
    assert archs4 and osdr
    assert osdr[0].marker.size > archs4[0].marker.size
    assert osdr[0].marker.symbol == theme.OSDR_SYMBOL
    assert osdr[0].marker.line.color == theme.OSDR_OUTLINE
    assert osdr[0].marker.line.width > 0
