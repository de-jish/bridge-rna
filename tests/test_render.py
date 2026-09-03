"""Figure construction and the sampling that feeds it.

The invariant these protect is the one that connects the picture to the data:
every drawn glyph must sit at the coordinates of the sample it depicts and carry
that sample's colour. A glyph drawn at the right pixel under the wrong label is
a silent lie, and on a 942,563-point scatter nobody would ever see it by eye.

Identity is checked by recomputing the sample indices and comparing coordinates,
rather than by reading them back out of ``customdata``. The ARCHS4 traces no
longer carry customdata at all: it existed to hand global indices to the lasso
readout, and with that gone it was roughly 600 KB of dead payload per figure.
"""

from __future__ import annotations

import numpy as np
import pytest


# --- The retrieval overlay ---------------------------------------------------

_RETRIEVAL = {
    "hit_points": [1, 2, 3],
    "hit_labels": ["GSM1", "GSM2", "GSM3"],
    "hit_scores": [0.997, 0.995, 0.993],
    "query_point": 10,
    "query_label": "OSD-100|Mmus_EYE",
}

_NEIGHBORHOOD = {
    "label": "OSD-100",
    "points": [1, 2, 3],
    "rows": [
        {"rank": 1, "gsm": "GSM1", "gse": "GSE1", "score": .99,
         "tissue": "Eye", "species": "mouse"},
        {"rank": 2, "gsm": "GSM2", "gse": "GSE1", "score": .98,
         "tissue": "Eye", "species": "mouse"},
        {"rank": 3, "gsm": "GSM3", "gse": "GSE2", "score": .97,
         "tissue": "Brain", "species": "human"},
    ],
    "focus_points": [2],
    "returned": 3,
    "locatable": 3,
}


@pytest.mark.parametrize("is_3d", [False, True])
def test_evidence_neighborhood_is_one_uniform_open_trace_with_optional_focus(
        is_3d):
    """Evidence is rank metadata, not a rank ramp or projected boundary.

    A scalar marker style and markers-only mode prevent color/size gradients,
    connecting lines, and hull-like geometry.  The seven-field customdata row
    is also the click/hover contract consumed by the explorer.
    """
    from manifold import render, theme

    coords = np.random.default_rng(11).normal(
        size=(50, 3 if is_3d else 2)).astype(np.float32)
    base, focus = render._neighborhood_traces(coords, is_3d, _NEIGHBORHOOD)

    expected_type = "scatter3d" if is_3d else "scattergl"
    assert base.type == expected_type
    assert base.name == "512-D evidence neighbor"
    assert base.marker.symbol == "circle-open"
    assert base.marker.size == theme.NEIGHBORHOOD_SIZE * (0.5 if is_3d else 1)
    assert isinstance(base.marker.color, str)
    assert "lines" not in (base.mode or "")
    assert list(base.customdata[0]) == [
        "neighborhood", 1, "GSM1", "GSE1", .99, "Eye", "mouse"]
    assert len(base.customdata) == 3

    assert focus is not None
    assert focus.name == "focused evidence neighbor"
    assert len(focus.x) == 1
    assert focus.customdata is None, "only the capped evidence trace carries metadata"
    hit_rings = [
        trace for trace in render._retrieval_traces(coords, is_3d, _COMPARISON)
        if trace.name == "retrieved hit"
    ]
    required_diameter = max(
        trace.marker.size * (np.sqrt(2) if "square" in trace.marker.symbol else 1)
        for trace in hit_rings
    )
    assert focus.marker.size > required_diameter, (
        "the focus circle must enclose every requested-hit ring, including "
        "the comparison square's diagonal")


def test_evidence_neighborhood_omits_stale_indices_without_changing_counts():
    """A stale map index cannot take down the figure or rewrite evidence totals."""
    from manifold import render

    overlay = {
        **_NEIGHBORHOOD,
        "points": [1, 999999],
        "rows": [_NEIGHBORHOOD["rows"][0], _NEIGHBORHOOD["rows"][1]],
        "focus_points": [999999],
        "returned": 3,
        "locatable": 2,
    }
    base, focus = render._neighborhood_traces(
        np.zeros((5, 2), dtype=np.float32), False, overlay)

    assert len(base.x) == 1
    assert list(base.customdata[0])[2] == "GSM1"
    assert focus is None
    assert overlay["returned"] == 3 and overlay["locatable"] == 2


def test_evidence_sits_behind_requested_hits_and_focus_sits_above_them(corpus):
    """The wider context must not replace or cover the requested-hit rings."""
    from manifold import render

    neighborhood = {
        **_NEIGHBORHOOD,
        "points": [1, 999999],
        "rows": _NEIGHBORHOOD["rows"][:2],
        "focus_points": [1],
        "returned": 3,
        "locatable": 2,
    }
    fig, _, badges = render.build_figure(
        "pca", "2d", "species", ["archs4"], 1000, None,
        retrieval=_RETRIEVAL, neighborhood=neighborhood)
    names = [trace.name for trace in fig.data]
    evidence_at = names.index("512-D evidence neighbor")
    focus_at = names.index("focused evidence neighbor")
    hit_positions = [i for i, name in enumerate(names) if name == "retrieved hit"]

    assert hit_positions
    assert evidence_at < min(hit_positions)
    assert focus_at > max(hit_positions)
    assert "Evidence neighborhood: <b>1</b> of 3 locatable" in badges


@pytest.mark.parametrize("dims,is_3d", [("2d", False), ("3d", True)])
def test_the_overlay_builds_in_both_dimensionalities(dims, is_3d):
    """3-D took the whole figure callback down with a 500.

    `Scatter3d` accepts a much smaller symbol set than `Scatter` and rejects
    the rest outright rather than degrading: `star` raises, and so does the
    2-D-only `cliponaxis`. Nothing caught it because the overlay had no test
    and the browser check does not open 3-D with a retrieval showing.
    """
    from manifold import render

    coords = np.random.default_rng(0).normal(size=(50, 3 if is_3d else 2))
    traces = render._retrieval_traces(coords.astype(np.float32), is_3d, _RETRIEVAL)
    assert traces, "the overlay drew nothing"
    expected = "scatter3d" if is_3d else "scatter"
    assert all(t.type == expected for t in traces)


@pytest.mark.parametrize("is_3d", [False, True])
def test_the_overlay_never_emits_a_line_trace(is_3d):
    """The one prohibition that must never be relaxed.

    A line between the query and a hit has a length; that length is a UMAP
    distance; a UMAP distance is not quantitative. Reading it as similarity is
    exactly the misreading the whole overlay is designed to prevent.
    """
    from manifold import render

    coords = np.random.default_rng(1).normal(size=(50, 3 if is_3d else 2))
    traces = render._retrieval_traces(coords.astype(np.float32), is_3d, _RETRIEVAL)
    for t in traces:
        assert "lines" not in (t.mode or ""), f"{t.name} draws lines"


def test_every_hit_ring_is_identical():
    """No size, opacity, or colour ramp across rank: the top hits differ by
    about 0.0016 cosine (the top-5 span for the OSD-100 query; 0.004 is the
    top-20 span), and any ramp asserts a difference that is not there."""
    from manifold import render

    coords = np.random.default_rng(2).normal(size=(50, 2)).astype(np.float32)
    hits = next(t for t in render._retrieval_traces(coords, False, _RETRIEVAL)
                if t.name == "retrieved hit")
    assert np.isscalar(hits.marker.size) or isinstance(hits.marker.size, (int, float))
    assert isinstance(hits.marker.color, str), "a per-point colour array is a ramp"


def test_the_overlay_ignores_out_of_range_points():
    """A stale retrieval must not index past the end of a rebuilt corpus."""
    from manifold import render

    coords = np.random.default_rng(3).normal(size=(5, 2)).astype(np.float32)
    traces = render._retrieval_traces(
        coords, False,
        {**_RETRIEVAL, "hit_points": [1, 999999], "query_point": 42})
    hits = next((t for t in traces if t.name == "retrieved hit"), None)
    assert hits is not None and len(hits.x) == 1
    assert not any(t.name == "query" for t in traces), "an out-of-range query was drawn"


from manifold import colorby, data, render, sampling, theme


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

ALL_LAYERS = ["archs4", "osdr"]


def _drawn_xy(fig):
    """Every drawn point as an (x, y) array, across all traces."""
    xs, ys = [], []
    for tr in fig.data:
        xs.extend(np.asarray(tr.x, dtype=np.float64).tolist())
        ys.extend(np.asarray(tr.y, dtype=np.float64).tolist())
    return np.column_stack([xs, ys]) if xs else np.empty((0, 2))


def _osdr_traces(fig):
    """Traces carrying the OSDR glyph, identified by symbol rather than by name."""
    return [t for t in fig.data if getattr(t.marker, "symbol", None) == theme.OSDR_SYMBOL]


def test_figure_builds_for_every_control_combination(corpus):
    # Driven off the registry rather than a literal list, so a projection added
    # to data.METHODS cannot ship without ever having been drawn.
    for method in data.METHODS:
        for dims in ("2d", "3d"):
            for color_by in (c.key for c in colorby.REGISTRY):
                fig, legend, badges = render.build_figure(
                    method, dims, color_by, ALL_LAYERS, 2000, None)
                assert len(fig.data) > 0, f"{method}/{dims}/{color_by} drew nothing"
                assert fig.layout.paper_bgcolor == theme.PLOT_BG


def test_every_drawn_point_sits_on_a_real_corpus_coordinate(corpus):
    """No glyph may be invented, duplicated, or displaced from its sample."""
    coords = data.coords("pca", "2d")
    fig, _, _ = render.build_figure("pca", "2d", "tissue", ALL_LAYERS, 1500, None)
    drawn = _drawn_xy(fig)
    assert len(drawn) > 0

    known = {(round(float(x), 5), round(float(y), 5)) for x, y in coords}
    missing = [p for p in drawn if (round(p[0], 5), round(p[1], 5)) not in known]
    assert not missing, f"{len(missing)} glyphs are not at any corpus coordinate"


def test_osdr_overlay_draws_every_osdr_point_exactly_once(corpus):
    coords = data.coords("pca", "2d")
    n_archs4, n_osdr = corpus["n_archs4"], corpus["n_osdr"]
    fig, _, _ = render.build_figure("pca", "2d", "tissue", ["osdr"], 1000, None)
    drawn = _drawn_xy(fig)
    assert len(drawn) == n_osdr

    expected = coords[n_archs4:][:, :2]
    assert np.allclose(np.sort(drawn, axis=0), np.sort(expected, axis=0), atol=1e-4)


def test_archs4_layer_draws_only_archs4_points(corpus):
    coords = data.coords("pca", "2d")
    n_archs4 = corpus["n_archs4"]
    fig, _, _ = render.build_figure("pca", "2d", "species", ["archs4"], 1000, None)
    drawn = _drawn_xy(fig)
    osdr_only = {(round(float(x), 5), round(float(y), 5))
                 for x, y in coords[n_archs4:][:, :2]}
    leaked = [p for p in drawn if (round(p[0], 5), round(p[1], 5)) in osdr_only]
    assert not leaked, "the ARCHS4 layer drew OSDR points"


def test_a_category_keeps_one_colour_across_both_corpora(corpus):
    """The whole point of the shared palette: one tissue, one colour, everywhere.

    Ranking categories per layer - the previous behaviour - gave the same tissue
    two different colours whenever the corpora ranked their categories
    differently, which makes the legend actively wrong.
    """
    fig, legend, _ = render.build_figure("pca", "2d", "tissue", ALL_LAYERS, 2000, None)
    expected = {row["label"]: row["color"] for row in legend["items"]}
    for trace in fig.data:
        assert trace.name in expected, f"trace {trace.name!r} is not in the legend"
        assert trace.marker.color == expected[trace.name], (
            f"{trace.name} drawn in {trace.marker.color}, legend says "
            f"{expected[trace.name]}")


def test_legend_counts_track_the_drawn_sample(corpus):
    """Legend counts are the points actually plotted, so they move with the
    point budget instead of reporting the whole corpus.

    Species covers both corpora with no residual bucket, so every drawn glyph
    lands in a legend row and the counts must total exactly the glyphs drawn.
    """
    fig_s, small, _ = render.build_figure("pca", "2d", "species", ALL_LAYERS, 200, None)
    fig_l, large, _ = render.build_figure("pca", "2d", "species", ALL_LAYERS, 3000, None)
    small_total = sum(i["count"] for i in small["items"])
    large_total = sum(i["count"] for i in large["items"])
    assert small_total < large_total, "a smaller budget must show smaller counts"
    assert small_total == len(_drawn_xy(fig_s)), "counts must total the drawn glyphs"
    assert large_total == len(_drawn_xy(fig_l))


def test_a_category_with_nothing_drawn_drops_from_the_legend(corpus):
    """The legend shows what is plotted, so a category with no drawn points is
    absent rather than shown as a zero.

    In the fixture OSDR is all one species while ARCHS4 carries both, so drawing
    only the OSDR layer leaves the ARCHS4-only species with nothing on screen -
    and it must leave the legend too, with no zero-count row.
    """
    _, both, _ = render.build_figure("pca", "2d", "species", ALL_LAYERS, 4000, None)
    _, osdr_only, _ = render.build_figure("pca", "2d", "species", ["osdr"], 4000, None)
    assert len(both["items"]) == 2, "both species should show when both layers draw"
    assert len(osdr_only["items"]) == 1, "the ARCHS4-only species must drop out"
    assert all(i["count"] > 0 for i in osdr_only["items"]), "no zero-count rows"


def test_3d_caps_the_cloud_and_the_legend_counts_the_cap(corpus, monkeypatch):
    """3-D never draws more than the rotation cap, and the legend counts the
    capped glyphs rather than the budget that was asked for."""
    monkeypatch.setattr(render, "SCATTER3D_ARCHS4_CAP", 500)
    fig, legend, _ = render.build_figure(
        "pca", "3d", "species", ["archs4"], corpus["n_archs4"], None)
    drawn = len(_drawn_xy(fig))
    assert drawn <= 500, f"3-D drew {drawn}, above the cap"
    assert sum(i["count"] for i in legend["items"]) == drawn, (
        "the legend must count the capped glyphs, not the budget")


# --- The no-grey-cloud contract --------------------------------------------

def test_a_field_that_cannot_describe_archs4_draws_it_as_faint_context(
        corpus, without_archs4_metadata):
    """The core fix: ARCHS4 keeps its shape on screen without impersonating data.

    It is drawn in one context colour that is not in the palette, faintly, and
    labelled as context, rather than taking a grey palette slot and reading as
    "measured, and empty here".

    Reached through Tissue on a machine with no GEO join, which is how this
    branch is reached at all now that the nine OSDR-only fields are gone - and
    it is the state a fresh clone starts in, so it is the case most worth
    pinning rather than a contrivance standing in for the deleted ones.
    """
    fig, _, badges = render.build_figure(
        "pca", "2d", "tissue", ALL_LAYERS, 2000, None)
    assert len(_drawn_xy(fig)) > corpus["n_osdr"], "no context cloud was drawn"
    context = [t for t in fig.data if t.marker.color == theme.ARCHS4_CONTEXT]
    assert context, "the context cloud is not using the context colour"
    assert context[0].marker.opacity < 0.5, "context must be faint, not data-like"
    assert theme.ARCHS4_CONTEXT not in theme.CATEGORICAL
    assert any("context only" in b for b in badges), badges


def test_the_context_cloud_is_drawn_in_3d_too(corpus, without_archs4_metadata):
    """Context is not a 2-D fallback; it is what an uncovered corpus always gets."""
    fig, _, badges = render.build_figure(
        "pca", "3d", "tissue", ALL_LAYERS, 1000, None)
    context = [t for t in fig.data if t.marker.color == theme.ARCHS4_CONTEXT]
    assert context, "3-D lost the ARCHS4 context cloud"
    assert any("context only" in b for b in badges), badges


def test_context_points_never_enter_the_legend(corpus, without_archs4_metadata):
    """Uncovered points have no value under this field, so they get no swatch."""
    _, legend, _ = render.build_figure(
        "pca", "2d", "tissue", ALL_LAYERS, 2000, None)
    labels = [i["label"] for i in legend["items"]]
    assert colorby.NOT_COVERED not in labels
    assert sum(i["count"] for i in legend["items"]) == corpus["n_osdr"]


# --- Layers, budget, viewport ----------------------------------------------

def test_layer_toggles_actually_remove_layers(corpus):
    only_osdr = _drawn_xy(render.build_figure(
        "pca", "2d", "tissue", ["osdr"], 1000, None)[0])
    assert len(only_osdr) == corpus["n_osdr"]

    fig_none, _, badges = render.build_figure("pca", "2d", "tissue", [], 1000, None)
    assert len(_drawn_xy(fig_none)) == 0
    assert badges == []


# --- The memoized colour plan -----------------------------------------------

def test_the_color_plan_is_compact_integer_codes(corpus):
    """The plan is memoized for every colour-by, so its size is load-bearing.

    An array of display-label strings looks equivalent and is not: under pandas
    3.0 a string Series materializes a fresh Python str per element on
    ``.to_numpy()``, so it held 942,563 distinct objects to express 13 distinct
    values, measured at 127 MB per colour-by. Across the registry that is more
    than a gigabyte of cache for a map that otherwise opens 81.5 MB.
    """
    codes, legend = render._color_plan("tissue")
    _, _, total = data.counts()
    assert codes.dtype == np.int16, f"codes widened to {codes.dtype}"
    assert len(codes) == total
    assert codes.nbytes == total * 2
    assert codes.max() < len(legend), "a code points past the end of the legend"
    assert codes.min() >= render.NOT_COVERED_CODE


def test_every_covered_point_lands_in_exactly_one_legend_row(corpus):
    """Codes are the only link between a glyph and its swatch, so the mapping
    from points to legend rows must be total and must agree with the counts."""
    for key in (c.key for c in colorby.REGISTRY):
        codes, legend = render._color_plan(key)
        counted = sum(row["count"] for row in legend)
        assert int((codes >= 0).sum()) == counted, (
            f"{key}: {int((codes >= 0).sum())} points carry a slot but the "
            f"legend totals {counted}")
        for slot, row in enumerate(legend):
            assert int((codes == slot).sum()) == row["count"], (
                f"{key}: legend row {row['label']!r} claims {row['count']} "
                f"points but {int((codes == slot).sum())} carry its slot")


def test_the_color_plan_is_cached_and_returns_the_same_object(corpus):
    a, _ = render._color_plan("tissue")
    b, _ = render._color_plan("tissue")
    assert a is b, "the colour plan is being recomputed on every figure build"


def test_the_figure_carries_no_layout_images(corpus):
    """Nothing is painted underneath the glyphs any more.

    The density raster used to sit here as a `layout.images` underlay. Every
    point on screen is now a real glyph at a real sample's coordinates, and a
    picture that reappeared under them would be a second, unlabelled encoding
    of the same data.
    """
    for dims in ("2d", "3d"):
        for color_by in (c.key for c in colorby.REGISTRY):
            fig, _, _ = render.build_figure("pca", dims, color_by, ALL_LAYERS,
                                            1000, None)
            assert not fig.layout.images, f"{dims}/{color_by} drew an underlay"


def test_the_whole_corpus_can_be_drawn_when_the_budget_allows(corpus):
    """The budget tops out at every point, not at a sample of them."""
    fig, _, badges = render.build_figure(
        "pca", "2d", "species", ALL_LAYERS, corpus["total"], None)
    assert len(_drawn_xy(fig)) == corpus["total"]
    assert any(f"{corpus['n_archs4']:,}" in b for b in badges), badges


def test_budget_caps_the_live_archs4_glyphs(corpus):
    for budget in (200, 800):
        fig, _, _ = render.build_figure("pca", "2d", "species", ["archs4"], budget, None)
        drawn = len(_drawn_xy(fig))
        assert drawn <= budget + 4, f"budget {budget} drew {drawn}"


def test_viewport_restricts_the_sample_to_the_window(corpus):
    coords = data.coords("pca", "2d")[: corpus["n_archs4"]]
    x0, x1 = np.percentile(coords[:, 0], [40, 60])
    y0, y1 = np.percentile(coords[:, 1], [40, 60])
    fig, _, _ = render.build_figure("pca", "2d", "species", ["archs4"], 2000, (x0, x1, y0, y1))
    pts = _drawn_xy(fig)
    assert len(pts) > 0
    assert (pts[:, 0] >= x0 - 1e-4).all() and (pts[:, 0] <= x1 + 1e-4).all()
    assert (pts[:, 1] >= y0 - 1e-4).all() and (pts[:, 1] <= y1 + 1e-4).all()


# --- Legend -----------------------------------------------------------------

def test_legend_reports_categories_with_counts(corpus):
    """Counts total the points actually plotted, so toggling the ARCHS4 layer
    off leaves even a whole-map field showing only its OSDR points.

    With only the OSDR layer drawn, every field totals n_osdr - a whole-map
    field's ARCHS4 categories simply have nothing on screen to count.
    """
    for key in (c.key for c in colorby.REGISTRY):
        _, legend, _ = render.build_figure("pca", "2d", key, ["osdr"], 1000, None)
        items = legend["items"]
        assert items, f"no legend produced for {key}"
        assert sum(i["count"] for i in items) == corpus["n_osdr"]
        labels = [i["label"] for i in items]
        assert len(labels) == len(set(labels)), f"{key} repeats a legend category"


def test_high_cardinality_color_by_collapses_into_other(corpus):
    """Past the palette size, the tail must fold into one neutral Other bucket.

    Tissue is what exercises this now. It used to be Study, whose 70 OSD
    accessions overflowed the eleven slots comfortably; with the OSDR-only
    fields gone, Tissue is both the only high-cardinality field left and the one
    where the overflow is real rather than arranged - the shared vocabulary has
    39 buckets against a palette of eleven, so the tail is not a test fixture.
    """
    _, legend, _ = render.build_figure("pca", "2d", "tissue", ALL_LAYERS, 4000, None)
    labels = [i["label"] for i in legend["items"]]
    n_tissues = len({t for t in colorby.labels("tissue")
                     if not colorby.is_residual(t)})
    assert n_tissues > render.TOP_N, (
        f"only {n_tissues} tissue buckets drawn; nothing overflows to test")
    assert "Other" in labels
    assert len(labels) <= render.TOP_N + 2  # + Other, + Unknown
    other = next(i for i in legend["items"] if i["label"] == "Other")
    assert other["color"] == theme.OTHER_COLOR


def test_residual_categories_never_take_a_bright_slot(corpus):
    """"Unknown" must not outrank a real category for a palette colour."""
    _, legend, _ = render.build_figure("pca", "2d", "tissue", ALL_LAYERS, 2000, None)
    for row in legend["items"]:
        if colorby.is_residual(row["label"]):
            assert row["color"] not in theme.CATEGORICAL, (
                f"{row['label']} took a categorical colour")


def test_unknown_and_other_stay_distinguishable(corpus):
    """They are different facts and must not share a swatch."""
    assert theme.residual_color("Unknown") != theme.residual_color("Other")


def test_missing_projection_renders_a_message_not_a_crash(corpus, monkeypatch, tmp_path):
    missing = tmp_path / "absent.parquet"
    monkeypatch.setitem(data.METHODS, "umap", {"2d": missing, "3d": missing})
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
    osdr = _osdr_traces(fig)
    archs4 = [t for t in fig.data if t not in osdr]
    assert osdr and archs4
    assert osdr[0].marker.size > archs4[0].marker.size
    assert osdr[0].marker.line.color == theme.OSDR_OUTLINE
    assert osdr[0].marker.line.width > 0


def test_osdr_hover_names_the_sample_and_its_category(corpus):
    fig, _, _ = render.build_figure("pca", "2d", "tissue", ["osdr"], 1000, None)
    trace = _osdr_traces(fig)[0]
    assert trace.hovertemplate, "the OSDR overlay lost its hover"
    assert trace.customdata is not None
    assert len(trace.customdata[0]) == 2, "hover payload should be [sample_key, category]"


def test_archs4_cloud_carries_no_hover_or_customdata(corpus):
    """Hover hit-testing dominates the frame cost, and the per-point payload is
    dead weight now that nothing consumes indices.

    This is also the guard on a cut feature, which is the reason it now has a
    number on it. Inspecting an arbitrary cloud point by clicking it is not
    buildable the obvious way: `hoverinfo="skip"` suppresses click *picking* as
    well as hover, so the cloud emits no `plotly_click` at all, and the tempting
    fix is to turn hover back on. Measured in a real browser against the full
    942,563-point corpus, restyling these 13 traces to `hoverinfo="x+y"` makes
    them clickable and takes synchronous mousemove handling to a **median of
    239.8 ms, max 243.7 ms**, sustained rather than a one-time index build -
    which is a map that feels frozen under the cursor. `hoverinfo="none"` does
    not restore picking either.

    So if this test fails because someone enabled hover to make the cloud
    clickable, the fix is not to update the test. See
    `docs/design-notes.md#finding-a-sample-on-the-map` section 7.
    """
    fig, _, _ = render.build_figure("pca", "2d", "species", ["archs4"], 1000, None)
    for trace in fig.data:
        assert trace.hovertemplate is None
        assert trace.hoverinfo == "skip"
        assert trace.customdata is None


def test_residual_glyphs_are_drawn_underneath_the_categories(corpus):
    """Plotly paints in insertion order, so the grey bucket must go first.

    With ~308,000 residual glyphs added last they painted over every category
    that carried information, and the map read as grey where it was not.
    """
    fig, legend, _ = render.build_figure("pca", "2d", "tissue", ["archs4"], 2000, None)
    names = [t.name for t in fig.data]
    residual_positions = [i for i, n in enumerate(names) if colorby.is_residual(n)]
    category_positions = [i for i, n in enumerate(names) if not colorby.is_residual(n)]
    if residual_positions and category_positions:
        assert max(residual_positions) < min(category_positions), (
            f"residual buckets are not underneath: {names}")


def test_residual_glyphs_recede_in_the_archs4_cloud(corpus):
    """Points with no usable label must not compete with points that have one."""
    fig, _, _ = render.build_figure("pca", "2d", "tissue", ["archs4"], 2000, None)
    residual = [t for t in fig.data if colorby.is_residual(t.name)]
    named = [t for t in fig.data if not colorby.is_residual(t.name)]
    if residual and named:
        assert residual[0].marker.opacity < named[0].marker.opacity
        assert residual[0].marker.size < named[0].marker.size


def test_the_osdr_overlay_never_recedes(corpus):
    """Only 2,108 OSDR points exist; dimming any of them loses the overlay."""
    fig, _, _ = render.build_figure("pca", "2d", "tissue", ["osdr"], 2000, None)
    opacities = {t.marker.opacity for t in _osdr_traces(fig)}
    assert len(opacities) == 1, f"OSDR glyphs drawn at mixed weights: {opacities}"


# --- Two cohorts drawn at once (the compare-against case) --------------------

_COMPARISON = {
    "cohorts": [
        {"role": "a", "label": "Liver · Space Flight",
         "hit_points": [1, 2, 3], "hit_labels": ["GSM1", "GSM2", "GSM3"],
         "hit_scores": [0.997, 0.995, 0.993], "query_points": [10, 11, 12]},
        {"role": "b", "label": "Liver · Ground Control",
         "hit_points": [2, 3, 7], "hit_labels": ["GSM2", "GSM3", "GSM7"],
         "hit_scores": [0.996, 0.994, 0.992], "query_points": [13, 14]},
    ],
    "hit_points": [1, 2, 3, 2, 3, 7],
    "hit_labels": ["GSM1", "GSM2", "GSM3", "GSM2", "GSM3", "GSM7"],
    "hit_scores": [0.997, 0.995, 0.993, 0.996, 0.994, 0.992],
    "shared_points": [2, 3],
    "query_point": 10,
    "query_points": [10, 11, 12, 13, 14],
    "query_label": "Liver · Space Flight vs Liver · Ground Control",
}


def _coords(n=50, is_3d=False, seed=7):
    return np.random.default_rng(seed).normal(
        size=(n, 3 if is_3d else 2)).astype(np.float32)


@pytest.mark.parametrize("is_3d", [False, True])
def test_a_comparison_draws_both_cohorts_in_both_dimensionalities(is_3d):
    """3-D is the case that broke the single-cohort overlay with a 500, because
    Scatter3d rejects an unknown symbol outright. Both hit symbols used here are
    deliberately in its vocabulary, so the encoding is identical in 2-D and 3-D
    rather than needing a fallback that would mean something different."""
    from manifold import render

    traces = render._retrieval_traces(_coords(is_3d=is_3d), is_3d, _COMPARISON)
    expected = "scatter3d" if is_3d else "scatter"
    assert all(t.type == expected for t in traces)
    assert len([t for t in traces if t.name == "query"]) == 2
    assert len([t for t in traces if t.name == "retrieved hit"]) == 2


def test_the_two_cohorts_members_differ_by_hue_and_not_by_symbol():
    """Hue is safe on a member because a member is a filled mark with a white
    outline: its findability comes from the outline, so the fill only has to
    separate A from B. Symbol is already spent telling a member from a hit."""
    from manifold import render, theme

    traces = render._retrieval_traces(_coords(), False, _COMPARISON)
    members = [t for t in traces if t.name == "query"]
    assert [t.marker.color for t in members] == [theme.RETRIEVAL_QUERY,
                                                 theme.RETRIEVAL_QUERY_B]
    assert len({t.marker.symbol for t in members}) == 1, "symbol must not differ"


def test_the_two_cohorts_hits_differ_by_symbol_and_are_both_white():
    """No hue clears 3:1 against the worst categorical tissue bucket - the
    comparison network's own three measure 1.00 to 1.07 - which is the same
    measurement that made the ring white in the first place. A two-cohort map
    can afford a lost hit even less, because a lost hit is also a lost group."""
    from manifold import render, theme

    traces = render._retrieval_traces(_coords(), False, _COMPARISON)
    hits = [t for t in traces if t.name == "retrieved hit"]
    assert {t.marker.color for t in hits} == {theme.RETRIEVAL_HIT_RING}
    assert [t.marker.symbol for t in hits] == [theme.RETRIEVAL_HIT_SYMBOL,
                                               theme.RETRIEVAL_HIT_SYMBOL_B]


def test_a_hit_both_cohorts_retrieved_carries_both_marks():
    """The shared set is emergent, never computed: a shared hit is in both hit
    lists, so it receives both traces and draws as a ring inside a square. That
    is why the shared count on the map cannot drift from the one the status
    banner quotes - it is not calculated twice."""
    from manifold import render

    coords = _coords()
    traces = render._retrieval_traces(coords, False, _COMPARISON)
    a, b = [t for t in traces if t.name == "retrieved hit"]
    drawn_a = {(round(float(x), 6), round(float(y), 6)) for x, y in zip(a.x, a.y)}
    drawn_b = {(round(float(x), 6), round(float(y), 6)) for x, y in zip(b.x, b.y)}
    shared = {(round(float(coords[i][0]), 6), round(float(coords[i][1]), 6))
              for i in _COMPARISON["shared_points"]}
    assert shared <= drawn_a and shared <= drawn_b, (
        "a shared hit must receive both cohorts' marks")
    assert b.marker.size > a.marker.size, (
        "B's square must circumscribe A's ring, or one hides the other")


def test_the_square_is_drawn_after_the_ring_it_surrounds():
    """Trace order is back to front. A 27 px square emitted first would sit
    under the 20 px ring; emitted last it frames it."""
    from manifold import render, theme

    traces = render._retrieval_traces(_coords(), False, _COMPARISON)
    order = [t.marker.symbol for t in traces if t.name == "retrieved hit"]
    assert order.index(theme.RETRIEVAL_HIT_SYMBOL_B) > \
        order.index(theme.RETRIEVAL_HIT_SYMBOL)
    names = [t.name for t in traces]
    assert names.index("query") > names.index("retrieved hit"), (
        "members are the thing being asked about and go on top")
    assert names.index("query halo") < names.index("retrieved hit")


def test_rank_numerals_are_dropped_only_when_two_cohorts_are_drawn():
    """Two competing numeral sets over the same few hundred pixels is
    illegible, and the hover carries strictly more: a shared hit names both
    cohorts, both 512-d ranks and both cosines. A single cohort keeps them."""
    from manifold import render

    coords = _coords()
    both = [t for t in render._retrieval_traces(coords, False, _COMPARISON)
            if t.name == "retrieved hit"]
    assert all("text" not in (t.mode or "") for t in both)

    alone = [t for t in render._retrieval_traces(coords, False, _RETRIEVAL)
             if t.name == "retrieved hit"]
    assert all("text" in (t.mode or "") for t in alone)
    assert list(alone[0].text) == ["1", "2", "3"]


def test_a_hits_hover_names_the_cohort_that_retrieved_it_only_when_ambiguous():
    from manifold import render

    coords = _coords()
    a, b = [t for t in render._retrieval_traces(coords, False, _COMPARISON)
            if t.name == "retrieved hit"]
    assert all("Space Flight" in row[1] for row in a.customdata)
    assert all("Ground Control" in row[1] for row in b.customdata)

    alone = next(t for t in render._retrieval_traces(coords, False, _RETRIEVAL)
                 if t.name == "retrieved hit")
    assert all(row[1].startswith("512-d rank") for row in alone.customdata), (
        "one cohort has nothing to disambiguate against")


def test_the_halo_fades_as_the_cohort_grows():
    """The halo was the one part of the overlay that got louder as the cohort
    got larger: every member of a 38-animal cohort carried a full-width ring at
    0.50 alpha and the overlap composited into a disc. Total ink is now roughly
    constant, and the ring narrows once there is more than one member."""
    from manifold import theme

    alphas = [float(theme.halo_rgba(theme.RETRIEVAL_QUERY_RGB, k).rsplit(",", 1)[1]
                    .strip(" )")) for k in (1, 2, 4, 10, 38)]
    assert alphas == sorted(alphas, reverse=True), alphas
    assert alphas[0] == theme.RETRIEVAL_QUERY_HALO_ALPHA, "k=1 must be unchanged"
    assert min(alphas) >= 0.14, "a large cohort's halo must not vanish"
    assert (theme.RETRIEVAL_QUERY_HALO_SIZE_POOLED
            < theme.RETRIEVAL_QUERY_HALO_SIZE)


def test_map_rank_is_measured_from_the_nearest_member_of_its_own_cohort():
    """It used to be measured from query_points[0], whichever animal came first
    in metadata order. That was merely unprincipled for one cohort; for the
    second arm of a comparison it would have ranked B's hits from an A animal."""
    from manifold import render

    # Two well-separated islands. Points 3 and 4 are cohort B's members and
    # point 5 is its hit, all on the right; points 0-2 are cohort A on the left.
    coords = np.array([[0.0, 0.0], [0.1, 0.0], [0.2, 0.0],      # 0-2 left
                       [10.0, 0.0], [10.1, 0.0], [10.4, 0.0]],  # 3-5 right
                      dtype=np.float32)
    ranks = render._map_ranks(coords, hits=[5], members=[3, 4])
    assert ranks == [2], (
        "measured from member 4, only points 3 and 4 lie nearer than the hit")

    # The old behaviour: every hit ranked from members[0]. For cohort B that is
    # an animal in the other cohort entirely, and the whole far island then
    # counts as nearer.
    far = render._map_ranks(coords, hits=[5], members=[0])
    assert far == [5], "measured from the far island, everything is nearer"


def test_map_ranks_sweeps_each_winning_member_once():
    """A full-corpus pass per member would be 38 passes over 942,563 points for
    the largest cohort. Only the members that actually win a hit are swept."""
    from manifold import render

    coords = _coords(n=400, seed=3)
    calls = {"n": 0}
    real = np.einsum

    def counting(*args, **kwargs):
        if args and args[0] == "ij,ij->i":
            calls["n"] += 1
        return real(*args, **kwargs)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(np, "einsum", counting)
        ranks = render._map_ranks(coords, hits=[1, 2, 3], members=list(range(20)))
    assert all(r is not None for r in ranks)
    assert calls["n"] <= 3, f"swept {calls['n']} times for 3 hits"


def test_a_flat_payload_still_renders_as_one_cohort():
    """`hits-store` is session-persisted, so the map can be handed a payload
    written before cohorts existed in the overlay. It must draw, not raise."""
    from manifold import render, theme

    traces = render._retrieval_traces(_coords(), False, _RETRIEVAL)
    members = [t for t in traces if t.name == "query"]
    assert len(members) == 1
    assert members[0].marker.color == theme.RETRIEVAL_QUERY


def test_the_hover_says_pooled_member_only_when_something_was_pooled():
    """For a single-sample or uploaded search the one drawn member *is* the
    query, so calling it a "pooled member" describes a cohort that does not
    exist. The phrase is earned by k >= 2, not by the code path."""
    from manifold import render

    coords = _coords()
    alone = next(t for t in render._retrieval_traces(coords, False, _RETRIEVAL)
                 if t.name == "retrieved hit")
    assert all("map rank" in row[1] for row in alone.customdata)
    assert not any("pooled member" in row[1] for row in alone.customdata)

    pooled = [t for t in render._retrieval_traces(coords, False, _COMPARISON)
              if t.name == "retrieved hit"]
    assert all("map rank" in row[1] for t in pooled for row in t.customdata)


def test_the_comparison_network_names_its_hits_until_they_would_collide():
    """The single-query network labels every hit; this one labelled none.

    A comparison is the figure most likely to end up in a slide or a paper, and
    without labels its accessions lived only in a tooltip - so on paper it
    carried no identities at all. They are drawn up to the point where they
    would stop being readable: the two arms share one vertical rhythm, so the
    node count is 2*k in the worst case, and at k=30 sixty labels on one column
    would overlap. Same rule as the map's RETRIEVAL_MAX_NUMERALS.
    """
    import pandas as pd

    from bridge_rna.figures import COMPARISON_MAX_LABELS, build_comparison_figure

    def arm(prefix: str, n: int) -> pd.DataFrame:
        return pd.DataFrame({
            "gsm": [f"{prefix}{i:07d}" for i in range(n)],
            "gse": ["GSE1"] * n,
            "score": [0.99 - i * 0.001 for i in range(n)],
            "source_name": [""] * n,
            "tissue": [""] * n,
        })

    def query(label: str) -> pd.Series:
        return pd.Series({"cohort_label": label, "sample_name": label,
                          "sample_id": label, "members": "a\nb"})

    small = build_comparison_figure(query("A"), arm("GSM", 4),
                                    query("B"), arm("GSN", 4))
    labelled = [t for t in small.data
                if t.mode and "text" in t.mode and t.text and len(t.text) > 1]
    assert labelled, "a small comparison writes no accessions on the figure"
    written = {s for t in labelled for s in t.text}
    assert "GSM0000000" in written and "GSN0000000" in written

    n = COMPARISON_MAX_LABELS  # 2n nodes, comfortably over the threshold
    big = build_comparison_figure(query("A"), arm("GSM", n),
                                  query("B"), arm("GSN", n))
    hit_traces = [t for t in big.data if t.showlegend]
    assert hit_traces, "the hit bands are gone"
    assert all(t.mode == "markers" for t in hit_traces), \
        "a crowded comparison still writes overlapping accessions"
