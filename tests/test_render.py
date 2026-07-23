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
    for method in ("pca", "umap"):
        for dims in ("2d", "3d"):
            for color_by in ("species", "flight_status", "tissue", "study"):
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

def test_an_osdr_only_field_draws_archs4_as_faint_context_not_as_a_category(corpus):
    """The core fix: ARCHS4 keeps its shape on screen without impersonating data.

    It is drawn in one context colour that is not in the palette, faintly, and
    labelled as context, rather than taking a grey palette slot and reading as
    "measured, and empty here".
    """
    fig, _, badges = render.build_figure(
        "pca", "2d", "flight_status", ALL_LAYERS, 2000, None)
    assert len(_drawn_xy(fig)) > corpus["n_osdr"], "no context cloud was drawn"
    context = [t for t in fig.data if t.marker.color == theme.ARCHS4_CONTEXT]
    assert context, "the context cloud is not using the context colour"
    assert context[0].marker.opacity < 0.5, "context must be faint, not data-like"
    assert theme.ARCHS4_CONTEXT not in theme.CATEGORICAL
    assert any("context only" in b for b in badges), badges


def test_the_context_cloud_is_drawn_in_3d_too(corpus):
    """Context is not a 2-D fallback; it is what an uncovered corpus always gets."""
    fig, _, badges = render.build_figure(
        "pca", "3d", "flight_status", ALL_LAYERS, 1000, None)
    context = [t for t in fig.data if t.marker.color == theme.ARCHS4_CONTEXT]
    assert context, "3-D lost the ARCHS4 context cloud"
    assert any("context only" in b for b in badges), badges


def test_context_points_never_enter_the_legend(corpus):
    """Uncovered points have no value under this field, so they get no swatch."""
    _, legend, _ = render.build_figure(
        "pca", "2d", "flight_status", ALL_LAYERS, 2000, None)
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

def test_the_colour_plan_is_compact_integer_codes(corpus):
    """The plan is memoized for every colour-by, so its size is load-bearing.

    An array of display-label strings looks equivalent and is not: under pandas
    3.0 a string Series materializes a fresh Python str per element on
    ``.to_numpy()``, so it held 942,563 distinct objects to express 13 distinct
    values, measured at 127 MB per colour-by. Across the registry that is more
    than a gigabyte of cache for a map that otherwise opens 81.5 MB.
    """
    codes, legend = render._colour_plan("tissue")
    _, _, total = data.counts()
    assert codes.dtype == np.int16, f"codes widened to {codes.dtype}"
    assert len(codes) == total
    assert codes.nbytes == total * 2
    assert codes.max() < len(legend), "a code points past the end of the legend"
    assert codes.min() >= render.NOT_COVERED_CODE


def test_every_covered_point_lands_in_exactly_one_legend_row(corpus):
    """Codes are the only link between a glyph and its swatch, so the mapping
    from points to legend rows must be total and must agree with the counts."""
    for key in ("tissue", "species", "flight_status"):
        codes, legend = render._colour_plan(key)
        counted = sum(row["count"] for row in legend)
        assert int((codes >= 0).sum()) == counted, (
            f"{key}: {int((codes >= 0).sum())} points carry a slot but the "
            f"legend totals {counted}")
        for slot, row in enumerate(legend):
            assert int((codes == slot).sum()) == row["count"], (
                f"{key}: legend row {row['label']!r} claims {row['count']} "
                f"points but {int((codes == slot).sum())} carry its slot")


def test_the_colour_plan_is_cached_and_returns_the_same_object(corpus):
    a, _ = render._colour_plan("tissue")
    b, _ = render._colour_plan("tissue")
    assert a is b, "the colour plan is being recomputed on every figure build"


def test_the_figure_carries_no_layout_images(corpus):
    """Nothing is painted underneath the glyphs any more.

    The density raster used to sit here as a `layout.images` underlay. Every
    point on screen is now a real glyph at a real sample's coordinates, and a
    picture that reappeared under them would be a second, unlabelled encoding
    of the same data.
    """
    for dims in ("2d", "3d"):
        for color_by in ("tissue", "flight_status"):
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

    With only the OSDR layer drawn, an OSDR-only field and a whole-map field
    both total n_osdr - the whole-map field's ARCHS4 categories simply have
    nothing on screen to count.
    """
    for key in ("flight_status", "tissue"):
        _, legend, _ = render.build_figure("pca", "2d", key, ["osdr"], 1000, None)
        items = legend["items"]
        assert items, f"no legend produced for {key}"
        assert sum(i["count"] for i in items) == corpus["n_osdr"]
        labels = [i["label"] for i in items]
        assert len(labels) == len(set(labels)), f"{key} repeats a legend category"


def test_high_cardinality_color_by_collapses_into_other(corpus):
    """Past the palette size, the tail must fold into one neutral Other bucket."""
    _, legend, _ = render.build_figure("pca", "2d", "study", ["osdr"], 1000, None)
    labels = [i["label"] for i in legend["items"]]
    n_studies = data.osdr_field_values("study").nunique()
    if n_studies > render.TOP_N:
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
    """Hover hit-testing dominates the frame cost at 100k glyphs, and the
    per-point payload is dead weight now that nothing consumes indices."""
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
