"""The serving app: layout wiring, callback plumbing, and the coverage readout.

A Dash app fails at runtime, in the browser, when a callback names a component
that does not exist - there is no import-time check. These tests do that check
statically: every callback Input/Output/State id must be present in the layout
tree, and every className the Python emits must exist in the stylesheet.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from dash import html

import app as shell
from manifold import callbacks, colorby, data, layout, paths, preflight, render, theme


# --- Layout / callback wiring ---------------------------------------------

def _walk(component):
    """Yield every component in a Dash layout tree."""
    yield component
    children = getattr(component, "children", None)
    if children is None:
        return
    if not isinstance(children, (list, tuple)):
        children = [children]
    for child in children:
        if hasattr(child, "children") or hasattr(child, "id"):
            yield from _walk(child)


@pytest.fixture(scope="module")
def app():
    return shell.build_app()


@pytest.fixture(scope="module")
def map_view():
    """The map view alone.

    Everything below asserting on the map's own controls takes this rather than
    `app.layout`, which is now the shell and mounts the *retrieval* view on the
    default route.
    """
    return layout.build_view()


@pytest.fixture(scope="module")
def mounted_ids(app):
    """Every component id that any route can put on the page.

    The router mounts one view at a time, so no single layout contains them
    all - and `suppress_callback_exceptions` is on, which means a callback
    pointing at an id that exists in *neither* view fails silently in the
    browser instead of loudly at startup. That is precisely the failure this
    file exists to catch, so the union is what a callback is checked against.
    """
    ids: set = set()
    trees = [shell.map_unavailable_view()]
    from bridge_rna import layout as rna_layout

    trees.append(rna_layout.build_view())
    trees.append(layout.build_view())
    # The shell itself, built outside a request so it takes the default route.
    trees.append(app.layout() if callable(app.layout) else app.layout)
    for tree in trees:
        ids |= {_id_key(getattr(c, "id", None)) for c in _walk(tree)}
    ids.discard(None)
    return ids


def _id_key(component_id):
    """A hashable, comparable form of a component id.

    Dash ids are either a string or a dict, and the dict form - pattern-matching
    ids like ``{"type": "facet-chip", "key": "tissue"}`` - is what the cohort
    facet chips use. A dict is unhashable, so both of the checks in this file
    used to raise ``TypeError`` the moment a pattern-matching id entered a view
    rather than reporting on it. They are exactly the ids most worth checking,
    since a duplicated or misspelled one fails silently in the browser.
    """
    if isinstance(component_id, dict):
        return tuple(sorted((str(k), str(v)) for k, v in component_id.items()))
    return component_id


# The wildcard sentinels Dash serializes into a pattern-matching id.
_WILDCARDS = ("ALL", "MATCH", "ALLSMALLER")


def _resolve_wildcard(ref, mounted_ids):
    """Return a mounted id that satisfies a pattern-matching reference, or None.

    `None` also means "this was not a pattern id at all", which lets the caller
    fall back to a literal comparison.
    """
    if not (isinstance(ref, str) and ref.startswith("{")):
        return None
    try:
        pattern = json.loads(ref)
    except ValueError:
        return None
    fixed = {str(k): str(v) for k, v in pattern.items()
             if not (isinstance(v, list) and v and v[0] in _WILDCARDS)}
    for mounted in mounted_ids:
        if not isinstance(mounted, tuple):
            continue
        as_dict = dict(mounted)
        if all(as_dict.get(k) == v for k, v in fixed.items()):
            return mounted
    return None


def test_app_builds(app):
    assert app.layout is not None
    assert app.title == "Bridge RNA"


def test_every_callback_target_exists_in_some_view(app, mounted_ids):
    referenced = set()
    for cb in app.callback_map.values():
        for item in list(cb["inputs"]) + list(cb.get("state", [])):
            ref = item["id"]
            # A pattern-matching input arrives as a JSON *string* naming a whole
            # family, e.g. '{"key":["ALL"],"type":"facet-chip"}'. It is satisfied
            # when some mounted component matches on every non-wildcard key, so
            # it is resolved here rather than compared literally - a literal
            # comparison can never match and would make the check unusable for
            # exactly the ids it is most valuable on.
            resolved = _resolve_wildcard(ref, mounted_ids)
            if resolved is not None:
                referenced.add(resolved)
            else:
                referenced.add(_id_key(ref))
    for key in app.callback_map:
        # A pattern-matching callback's key is a JSON blob, not `id.prop`, so
        # the regex below would shred it into fragments that match no component.
        # Those callbacks are already covered by the `inputs` walk above.
        if key.lstrip().startswith("{") or '{"' in key:
            continue
        for part in re.findall(r"([A-Za-z0-9_-]+)\.[A-Za-z]", key):
            referenced.add(part)

    missing = {r for r in referenced if r not in mounted_ids}
    assert not missing, f"callbacks reference components no view mounts: {sorted(missing)}"


@pytest.mark.parametrize("name", ["retrieve", "map"])
def test_no_view_contains_a_duplicate_component_id(name):
    """Dash raises DuplicateIdError, but only for ids in the *initial* layout.

    A view mounted by a callback is never validated, so a duplicate can sit in
    one for as long as nobody loads that route first. This repository shipped
    exactly that: the retrieval view carried two `hits-store` components, and it
    stayed invisible until the shell began serving views in the initial layout.
    """
    import collections

    from bridge_rna import layout as rna_layout

    tree = rna_layout.build_view() if name == "retrieve" else layout.build_view()
    ids = [_id_key(i) for i in (getattr(c, "id", None) for c in _walk(tree)) if i]
    duplicated = [i for i, n in collections.Counter(ids).items() if n > 1]
    assert not duplicated, f"{name} view has duplicate ids: {duplicated}"


def test_a_cold_loaded_deep_link_opens_on_the_right_study():
    """A pasted `/?q=` link must land on the sample's study on the initial load.

    Not via the live callback - that carries `prevent_initial_call` because
    firing it on load left both dropdowns empty - but via `_initial_study`,
    which reads the request at layout-build time. This test drives that helper
    inside a real request context, since that is the only place it does its job.
    """
    from urllib.parse import quote

    from bridge_rna import layout
    from bridge_rna.layout import default_study

    server = shell.build_app().server

    # Any sample in a study other than the default: a plain default would be
    # the wrong study, so only the ?q handling can return the right one.
    # `_initial_study` maps sample -> study and does not consult retrievability,
    # so this does not depend on the cache the test fixture stands in for.
    target = next(r for r in layout.samples_df.itertuples()
                  if r.study_id != default_study)

    with server.test_request_context(f"/?q={quote(target.sample_id)}"):
        assert layout._initial_study() == target.study_id
    with server.test_request_context("/"):
        assert layout._initial_study() == default_study
    # A `?q` naming an unknown sample falls back to the default rather than
    # opening on nothing.
    with server.test_request_context("/?q=OSD-000%7Cnot-a-sample"):
        assert layout._initial_study() == default_study


def test_both_halves_register_their_callbacks(app):
    """A router that mounts a view but forgets its callbacks looks fine until
    a control is touched. Assert one signature callback from each half."""
    keys = " ".join(app.callback_map)
    assert "manifold-graph.figure" in keys, "the map's render callback is missing"
    assert "network-graph.figure" in keys, "the retrieval's search callback is missing"
    assert "page-content.children" in keys, "the router callback is missing"


def test_the_required_controls_exist(map_view):
    ids = {getattr(c, "id", None) for c in _walk(map_view)}
    for required in ("manifold-graph", "color-by", "coverage", "color-by-hint",
                     "method", "method-params", "dims", "layers", "budget",
                     "plot-badges", "legend"):
        assert required in ids, f"layout is missing #{required}"


def test_the_selection_readout_is_gone(map_view):
    """The lasso feature was removed; no part of its panel may survive."""
    ids = {getattr(c, "id", None) for c in _walk(map_view)}
    assert "readout-body" not in ids and "readout" not in ids
    classes = {getattr(c, "className", "") for c in _walk(map_view)}
    assert not any("bm-readout" in str(c) for c in classes)


def test_legend_parts_are_static_so_dash_can_validate_them(map_view):
    ids = {getattr(c, "id", None) for c in _walk(map_view)}
    for required in ("legend-title", "legend-search", "legend-list", "legend-store"):
        assert required in ids, f"{required} is only created at runtime"


def test_budget_tiers_follow_dimensionality(corpus):
    """3-D cannot draw the whole corpus, so it must not offer to.

    In 2-D the tiers climb to "All"; in 3-D they stop at the rotation cap, with
    no "All" pill that would silently redraw as 40,000.
    """
    n_archs4 = str(data.counts()[0])
    two_d = layout.budget_options("2d")
    assert any(o["label"] == "All" for o in two_d)
    assert two_d[-1]["value"] == n_archs4
    assert layout.default_budget("2d") == n_archs4

    three_d = layout.budget_options("3d")
    assert not any(o["label"] == "All" for o in three_d)
    assert all(int(o["value"]) <= render.SCATTER3D_ARCHS4_CAP for o in three_d)
    assert max(int(o["value"]) for o in three_d) == render.SCATTER3D_ARCHS4_CAP
    assert layout.default_budget("3d") == str(render.SCATTER3D_ARCHS4_CAP)


def test_switching_dimensionality_clamps_an_out_of_range_budget(corpus):
    """A 2-D "All" cannot survive into 3-D, and a 3-D tier cannot survive back
    into 2-D; a value valid for the new dimensionality is kept."""
    n_archs4 = str(data.counts()[0])
    cap = str(render.SCATTER3D_ARCHS4_CAP)
    # "All" is not a 3-D tier, so it resets to the 3-D default.
    assert layout.resolve_budget("3d", n_archs4) == cap
    # A value already valid in 3-D is preserved.
    assert layout.resolve_budget("3d", cap) == cap
    # Coming back to 2-D, a 3-D-only value resets to the 2-D default.
    assert layout.resolve_budget("2d", cap) == n_archs4


# --- The Projection control and its parameter readout ------------------------

def test_every_registered_projection_is_offered(corpus):
    """The pills are derived from the registry, not hand-listed beside it."""
    options = layout.method_options()
    assert [o["value"] for o in options] == [k for k, _ in layout.METHOD_LABELS]
    assert set(o["value"] for o in options) == set(data.METHODS)
    assert not any(o["disabled"] for o in options), "the fixture builds all three"


def test_an_unbuilt_projection_is_disabled_not_hidden(corpus, monkeypatch,
                                                      tmp_path):
    """Hiding it would make the app look like it never had the feature."""
    missing = tmp_path / "nope.parquet"
    monkeypatch.setitem(data.METHODS, "tsne", {"2d": missing, "3d": missing})
    options = layout.method_options()
    tsne = next(o for o in options if o["value"] == "tsne")
    assert tsne["disabled"] is True
    assert "tsne" in {o["value"] for o in options}, "it must still be listed"


def test_the_default_projection_skips_past_unbuilt_ones(corpus, monkeypatch,
                                                        tmp_path):
    missing = tmp_path / "nope.parquet"
    assert layout.default_method() == "umap"
    monkeypatch.setitem(data.METHODS, "umap", {"2d": missing, "3d": missing})
    assert layout.default_method() == "tsne"
    monkeypatch.setitem(data.METHODS, "tsne", {"2d": missing, "3d": missing})
    assert layout.default_method() == "pca"


def test_projection_params_report_the_build_record(corpus):
    """The rail states what the build actually did, read off its record."""
    stats = data.projection_stats()
    flat = {p + v + s for p, v, s in layout.projection_params("umap", "2d")}
    assert any(f"n_neighbors={stats['umap_neighbors']}" == t for t in flat)
    assert any("cosine" == t for t in flat)
    assert any(f"{stats['total']:,}" in t for t in flat), "the corpus count"

    pca = {p + v + s for p, v, s in layout.projection_params("pca", "2d")}
    assert any("eigendecomposition" in t for t in pca)
    assert any("%" in t for t in pca), "PC1's share"


def test_tsne_params_name_the_gradient_method_that_actually_ran(corpus):
    """openTSNE's interpolation accelerator refuses 3 dimensions, so the two
    maps are built by different algorithms and the rail must not claim one."""
    two = {p + v + s for p, v, s in layout.projection_params("tsne", "2d")}
    three = {p + v + s for p, v, s in layout.projection_params("tsne", "3d")}
    assert "FIt-SNE" in two and "Barnes-Hut" not in two
    assert "Barnes-Hut" in three and "FIt-SNE" not in three
    # Everything else about the fit is shared, so only that one chip differs.
    assert two - {"FIt-SNE"} == three - {"Barnes-Hut"}


def test_the_init_chip_follows_the_record_not_a_constant(corpus, monkeypatch):
    """The starting layout is the difference between the corpora separating or
    not, so the rail must name the one the build actually used.

    It was hardcoded "PCA init" for a while, which would have kept saying PCA
    after the build switched to a spectral start - the exact kind of stale
    confidence the readout exists to avoid.
    """
    def with_init(value):
        base = dict(data.projection_stats())
        base["umap_init"] = value
        monkeypatch.setattr(data, "projection_stats", lambda: base)
        return {p + v + s for p, v, s in layout.projection_params("umap", "2d")}

    assert "spectral init" in with_init(
        "spectral (normalized Laplacian, shifted Lanczos, ncv=32)")
    assert "PCA init" in with_init("exact full-corpus PCA, scaled (not spectral)")


def test_projection_params_drop_what_the_record_does_not_carry(corpus,
                                                               monkeypatch):
    """An older cache shows fewer parameters, never a blank slot."""
    monkeypatch.setattr(data, "projection_stats", lambda: {"total": 10})
    params = layout.projection_params("umap", "2d")
    assert params == [("fit on all ", "10", " points")]
    assert all(v for _, v, _ in params), "no chip may render an empty value"


def test_projection_params_say_nothing_about_coordinates_that_are_missing(
        corpus, monkeypatch, tmp_path):
    """An interrupted build leaves a complete record beside a missing parquet.

    Every stage saves its stats before the next one runs, and the 3-D t-SNE fit
    is 81% of the build's wall clock, so a run interrupted there leaves
    coords_tsne2.parquet plus a full tsne_* record and no coords_tsne3.parquet.
    The 2-D pill should stay enabled - that map is real - but the rail must not
    describe a 3-D fit that never finished, or it asserts a Barnes-Hut layout
    over 942,563 points beside a plot reading "coordinates not built yet".
    """
    missing = tmp_path / "nope.parquet"
    built = data.METHODS["tsne"]["2d"]
    monkeypatch.setitem(data.METHODS, "tsne", {"2d": built, "3d": missing})

    assert data.method_available("tsne"), "the 2-D map is built and still offered"
    assert data.coords_available("tsne", "2d")
    assert not data.coords_available("tsne", "3d")

    assert layout.projection_params("tsne", "2d"), "2-D still describes itself"
    assert layout.projection_params("tsne", "3d") == []
    assert layout.projection_params_children("tsne", "3d") == []


def test_projection_params_survive_a_cache_with_no_record(corpus, monkeypatch):
    """It is a label, not a correctness gate: no record means no chips."""
    monkeypatch.setattr(data, "projection_stats", lambda: {})
    for method in data.METHODS:
        for dims in ("2d", "3d"):
            assert layout.projection_params(method, dims) == []
            assert layout.projection_params_children(method, dims) == []


def test_projection_stats_returns_empty_for_a_missing_file(tmp_path,
                                                           monkeypatch):
    monkeypatch.setattr(paths, "PROJECTION_STATS_JSON", tmp_path / "gone.json")
    data.projection_stats.cache_clear()
    try:
        assert data.projection_stats() == {}
    finally:
        data.projection_stats.cache_clear()


def test_projection_stats_returns_empty_for_a_corrupt_file(tmp_path,
                                                           monkeypatch):
    """A truncated write must not take the map down for the sake of a label."""
    bad = tmp_path / "projection_stats.json"
    bad.write_text('{"total": 942563')  # truncated mid-object
    monkeypatch.setattr(paths, "PROJECTION_STATS_JSON", bad)
    data.projection_stats.cache_clear()
    try:
        assert data.projection_stats() == {}
    finally:
        data.projection_stats.cache_clear()


def test_every_output_has_exactly_one_writer(app):
    """Two callbacks writing one output race; Dash only rejects some cases."""
    from collections import Counter

    written = Counter()
    for key in app.callback_map:
        for target in key.strip(".").split("..."):
            if target:
                written[target] += 1
    duplicates = [t for t, n in written.items() if n > 1]
    assert not duplicates, f"outputs with multiple writers: {duplicates}"


def test_legend_search_filters_the_rendered_rows():
    """The filter box has to actually filter - it was inert."""
    store = {"title": "Study", "items": [
        {"label": "OSD-100", "color": "#111", "count": 5},
        {"label": "OSD-200", "color": "#222", "count": 3},
        {"label": "Other", "color": "#333", "count": 1},
    ]}
    assert len(callbacks.filtered_legend_rows(store, None)) == 3
    assert len(callbacks.filtered_legend_rows(store, "OSD-1")) == 1
    assert len(callbacks.filtered_legend_rows(store, "other")) == 1, "should ignore case"
    empty = callbacks.filtered_legend_rows(store, "zzz")
    assert getattr(empty, "className", "") == "bm-legend-empty"


def test_legend_filter_survives_an_empty_store():
    assert callbacks.filtered_legend_rows(None, None) == []


def test_graph_offers_no_selection_tool(map_view):
    """Both selection tools must be gone from the modebar and the drag mode.

    Leaving lasso2d enabled would let a user draw a marquee that silently does
    nothing - a promise the app no longer keeps. Note the old config removed
    box-select but not the lasso, so this needs asserting, not assuming.
    """
    graph = next(c for c in _walk(map_view) if getattr(c, "id", None) == "manifold-graph")
    assert graph.config["displaylogo"] is False
    assert graph.config["scrollZoom"] is True
    removed = set(graph.config["modeBarButtonsToRemove"])
    assert {"lasso2d", "select2d"} <= removed
    for is_3d in (False, True):
        assert theme.base_figure_layout(is_3d)["dragmode"] == "pan"


# --- Viewport interpretation ----------------------------------------------

def test_zoom_event_becomes_a_viewport():
    vp = callbacks._viewport_from_relayout({
        "xaxis.range[0]": 3.0, "xaxis.range[1]": 1.0,
        "yaxis.range[0]": 8.0, "yaxis.range[1]": 2.0,
    })
    assert vp == (1.0, 3.0, 2.0, 8.0), "axis ranges were not normalized to min/max"


def test_autorange_resets_the_viewport():
    assert callbacks._viewport_from_relayout({"xaxis.autorange": True}) is None
    assert callbacks._viewport_from_relayout({"autosize": True}) is None
    assert callbacks._viewport_from_relayout(None) is None
    assert callbacks._viewport_from_relayout({}) is None


def test_non_zoom_events_leave_the_sample_alone():
    """A hover or dragmode change must not trigger a resample."""
    assert callbacks._viewport_from_relayout({"dragmode": "pan"}) == "unchanged"
    assert callbacks._viewport_from_relayout({"hovermode": "closest"}) == "unchanged"


# --- Plot badge markup ------------------------------------------------------

def test_bold_markup_is_parsed_not_injected():
    parts = callbacks._html_with_bold("ARCHS4 live: <b>100,000</b> pts")
    assert any(isinstance(p, html.B) and p.children == "100,000" for p in parts)
    assert "".join(p if isinstance(p, str) else p.children for p in parts) == \
        "ARCHS4 live: 100,000 pts"


def test_bold_markup_survives_text_without_tags():
    assert callbacks._html_with_bold("plain") == ["plain"]


# --- Styling --------------------------------------------------------------

def test_the_serving_app_does_not_import_the_scientific_stack():
    """Starting the app must not pull in torch, umap, sklearn, or pynndescent.

    The map draws precomputed coordinates and the retrieval's fast path is a
    memmap scan, so neither needs a model at import time. Keeping it that way is
    what lets the app start on a machine with no checkpoint, and what keeps
    startup off the multi-second torch import. A stray module-scope `import
    torch` would not fail anything - it would just quietly make that untrue.
    """
    import ast

    heavy = {"torch", "umap", "sklearn", "pynndescent", "openTSNE"}
    offenders = []
    files = (list((paths.REPO_ROOT / "bridge_rna").glob("*.py"))
             + list((paths.REPO_ROOT / "manifold").glob("*.py"))
             + [paths.REPO_ROOT / "app.py"])
    for py in sorted(files):
        for node in ast.parse(py.read_text()).body:  # module scope only
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            for name in names:
                if name.split(".")[0] in heavy:
                    offenders.append(f"{py.name}: {name}")
    assert not offenders, f"module-scope scientific imports: {offenders}"


def _all_css() -> str:
    """Every stylesheet Dash serves, concatenated in the order it serves them."""
    return "\n".join(p.read_text()
                     for p in sorted(paths.ASSETS_DIR.glob("*.css")))


def _python_classnames(*dirs: Path) -> set[str]:
    # Dash exposes several className props; a class hook applied through any of
    # them is just as broken if no stylesheet defines it.
    prop = r'(?:className|inputClassName|labelClassName)="([^"]+)"'
    used: set[str] = set()
    for d in dirs:
        files = sorted(d.glob("*.py")) if d.is_dir() else [d]
        for py in files:
            for match in re.findall(prop, py.read_text()):
                used.update(match.split())
    return used


def test_every_classname_used_in_python_exists_in_some_stylesheet():
    """Now that both views share one page, this has to span both.

    A class the map emits could be defined only in the retrieval stylesheet and
    still work, so checking each half against its own file would be checking
    something that is no longer true.
    """
    defined = set(re.findall(r"\.([a-zA-Z][a-zA-Z0-9_-]*)", _all_css()))
    used = _python_classnames(
        paths.REPO_ROOT / "manifold",
        paths.REPO_ROOT / "bridge_rna",
        paths.REPO_ROOT / "app.py",
    )
    # Dash supplies its own component classes; only ours are our problem.
    ours = {c for c in used if c.startswith(("bm-", "app-"))}
    missing = sorted(ours - defined)
    assert not missing, f"classNames with no CSS rule: {missing}"


def test_the_stylesheets_define_each_token_exactly_once():
    """One token layer, so a value cannot depend on which file sorts later.

    The two views arrived with their own :root blocks, and five Dash component
    tokens disagreed - meaning the alphabetically-later stylesheet silently
    decided how the other view's controls rendered on hover.
    """
    import collections

    counts: collections.Counter = collections.Counter()
    for path in sorted(paths.ASSETS_DIR.glob("*.css")):
        for block in re.findall(r":root\s*\{([^}]*)\}", path.read_text(), re.S):
            for name, _ in re.findall(r"(--[\w-]+)\s*:\s*([^;]+);", block):
                counts[name] += 1
    duplicated = sorted(t for t, n in counts.items() if n > 1)
    assert not duplicated, f"tokens defined in more than one place: {duplicated}"
    assert counts, "no design tokens found at all"


def test_theme_matches_the_bridge_rna_tokens():
    """The chrome must stay pixel-identical to Bridge RNA; only the plot is dark."""
    css = _all_css()
    for token, value in [
        ("--bg-canvas", theme.BG_CANVAS), ("--bg-panel", theme.BG_PANEL),
        ("--accent", theme.ACCENT), ("--header-bg", theme.HEADER_BG),
        ("--header-line", theme.HEADER_LINE), ("--plot-bg", theme.PLOT_BG),
    ]:
        assert f"{token}: {value}" in css, f"{token} drifted from {value}"


def test_categorical_palette_has_no_duplicate_hues():
    assert len(set(theme.CATEGORICAL)) == len(theme.CATEGORICAL)
    assert theme.OTHER_COLOR not in theme.CATEGORICAL


def test_preflight_reports_missing_artifacts(tmp_path):
    problems = preflight.check_artifacts([("nothing", tmp_path / "absent.bin")])
    assert len(problems) == 1 and "missing nothing" in problems[0]


def test_preflight_detects_an_unresolved_lfs_pointer(tmp_path):
    stub = tmp_path / "model.pt"
    stub.write_bytes(b"version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 1\n")
    assert preflight.is_lfs_pointer(stub)
    problems = preflight.check_artifacts([("checkpoint", stub)])
    assert "Git LFS pointer" in problems[0]


def test_preflight_passes_for_a_real_file(tmp_path):
    real = tmp_path / "real.bin"
    real.write_bytes(b"\x00" * 8192)
    assert not preflight.is_lfs_pointer(real)
    assert preflight.check_artifacts([("real", real)]) == []


# --- The coverage readout ---------------------------------------------------

def _text(component) -> str:
    """Flatten a Dash component tree to its visible text."""
    if isinstance(component, str):
        return component
    if isinstance(component, (list, tuple)):
        return " ".join(_text(c) for c in component)
    children = getattr(component, "children", None)
    return _text(children) if children is not None else ""


def test_coverage_states_the_exact_point_count(corpus):
    """The answer to "why is most of my map not coloured?", given up front."""
    text = _text(callbacks.coverage_children("flight_status"))
    assert f"{corpus['n_osdr']:,}" in text
    assert f"{corpus['total']:,}" in text
    assert "context" in text.lower(), (
        "the readout must say what happens to the points it does not colour")


def test_coverage_says_so_when_a_field_paints_everything(corpus):
    text = _text(callbacks.coverage_children("species"))
    assert f"{corpus['total']:,}" in text
    assert "all" in text.lower()


def test_coverage_bar_is_amber_only_for_a_partial_field(corpus):
    def fill_class(key):
        bar = callbacks.coverage_children(key)[0]
        return bar.children.className

    assert "partial" not in fill_class("species")
    assert "partial" in fill_class("flight_status")


def test_coverage_offers_the_fix_when_the_join_is_missing(
        corpus, without_archs4_metadata):
    text = _text(callbacks.coverage_children("tissue"))
    assert "fetch_archs4_meta" in text


def test_coverage_shows_no_fix_when_nothing_is_missing(corpus):
    assert "fetch_archs4_meta" not in _text(callbacks.coverage_children("tissue"))


def test_every_field_has_a_hint_or_deliberately_none(corpus):
    """A hint is optional, but it must be a string the layout can render."""
    for spec in colorby.REGISTRY:
        assert isinstance(spec.hint, str)


def test_the_cross_corpus_caution_is_not_shown_on_the_rail(map_view):
    """The standing "Reading across corpora" panel was removed from the UI.

    The batch effect it described is still documented (README, CLAUDE.md); it
    just no longer sits on the control rail as a paragraph of microcopy. This
    pins the removal so the verbose panel does not creep back in.
    """
    classes = {getattr(c, "className", "") for c in _walk(map_view)}
    assert not any("bm-caution" in str(c) for c in classes)
    assert "54x" not in _text(map_view)


# --- A comparison, translated onto the map -----------------------------------


def _comparison_payload(members_a, members_b):
    """A hits-store payload shaped exactly as `run_cohort_search` writes one."""
    def hit(i, gsm):
        return {"gsm": gsm, "score": 0.99, "archs4_index": i}
    return {
        "sample_id": "COHORT|study=OSD-1|tissue=Liver|spaceflight=Space Flight",
        "mode": "cohort",
        "hits": [hit(0, "GSM1"), hit(1, "GSM2"), hit(2, "GSM3")],
        "query": {"cohort_label": "Liver · Space Flight", "is_cohort": "1"},
        "member_ids": list(members_a),
        "comparison": {
            "query_b": {"cohort_label": "Liver · Ground Control", "is_cohort": "1"},
            "hits_b": [hit(1, "GSM2"), hit(2, "GSM3"), hit(5, "GSM6")],
            "shared": ["GSM2", "GSM3"],
            "overlap": 0.5,
            "facet": "spaceflight arm",
            "member_ids": list(members_b),
        },
    }


@pytest.fixture
def two_cohorts(corpus):
    """Two disjoint groups of real OSDR sample keys from the fixture corpus."""
    keys = data.osdr_metadata()["sample_key"].astype(str).tolist()
    assert len(keys) >= 5
    return keys[:3], keys[3:5]


def test_a_comparison_becomes_two_drawable_cohorts(two_cohorts):
    """The payload already carried everything needed - cohort B's hits have
    always had `archs4_index` - and the overlay simply never looked at it. A
    user who ran flight against ground saw one arm and no sign of the other."""
    a, b = two_cohorts
    overlay = callbacks._retrieval_overlay(_comparison_payload(a, b))
    assert overlay is not None
    assert [c["role"] for c in overlay["cohorts"]] == ["a", "b"]
    assert [c["label"] for c in overlay["cohorts"]] == ["Liver · Space Flight",
                                                        "Liver · Ground Control"]
    assert [len(c["query_points"]) for c in overlay["cohorts"]] == [3, 2]


def test_the_flat_keys_are_the_union_of_both_cohorts(two_cohorts):
    """`_frame_for` and the retrieval summary read the flat keys. Keeping them a
    union rather than cohort A's own values is what lets both of them frame and
    describe a comparison without either learning what a comparison is."""
    a, b = two_cohorts
    overlay = callbacks._retrieval_overlay(_comparison_payload(a, b))
    assert overlay["hit_points"] == [p for c in overlay["cohorts"]
                                     for p in c["hit_points"]]
    assert overlay["query_points"] == [p for c in overlay["cohorts"]
                                       for p in c["query_points"]]
    assert len(overlay["query_points"]) == 5


def test_shared_points_are_the_intersection_of_the_two_hit_sets(two_cohorts):
    a, b = two_cohorts
    overlay = callbacks._retrieval_overlay(_comparison_payload(a, b))
    first, second = (set(c["hit_points"]) for c in overlay["cohorts"])
    assert set(overlay["shared_points"]) == first & second
    assert overlay["shared_points"], "this fixture is built to share two hits"


def test_unticking_an_arm_removes_it_from_everything_downstream(two_cohorts):
    """The escape hatch when two large cohorts crowd one region. It has to
    reach the frame as well as the figure, or framing zooms out past what is
    actually drawn."""
    a, b = two_cohorts
    payload = _comparison_payload(a, b)
    only_a = callbacks._retrieval_overlay(payload, ("a",))
    assert [c["role"] for c in only_a["cohorts"]] == ["a"]
    assert len(only_a["query_points"]) == 3
    assert only_a["shared_points"] == [], "nothing to share with one cohort"

    only_b = callbacks._retrieval_overlay(payload, ("b",))
    assert [c["role"] for c in only_b["cohorts"]] == ["b"]
    assert len(only_b["query_points"]) == 2


def test_the_checklist_value_maps_to_the_cohorts_it_asks_for():
    """Cohort A stays "on" rather than becoming "a" so a single-query retrieval
    keeps the exact value this control has always held, and a session store
    written before comparisons could be drawn still resolves."""
    assert callbacks._roles_from_checklist(["on"]) == ("a",)
    assert callbacks._roles_from_checklist(["on", "b"]) == ("a", "b")
    assert callbacks._roles_from_checklist(["b"]) == ("b",)
    assert callbacks._roles_from_checklist([]) == ()
    assert callbacks._roles_from_checklist(None) == ()


def test_a_single_cohort_payload_is_unchanged_by_the_comparison_machinery(
        two_cohorts):
    """A comparison must be the single-cohort case plus a second thing. If the
    one-cohort overlay drifts, every existing behaviour drifts with it."""
    a, _b = two_cohorts
    payload = _comparison_payload(a, [])
    payload.pop("comparison")
    overlay = callbacks._retrieval_overlay(payload)
    assert len(overlay["cohorts"]) == 1
    assert overlay["query_label"] == "3 pooled cohort samples"
    assert overlay["shared_points"] == []
    assert overlay["query_point"] == overlay["query_points"][0]


def test_a_single_sample_retrieval_is_still_named_by_its_sample_key(corpus):
    """The map rail prints this, and it must not become "1 pooled cohort
    samples" for a plain Sample-mode search."""
    key = str(data.osdr_metadata()["sample_key"].iloc[0])
    overlay = callbacks._retrieval_overlay(
        {"sample_id": key, "hits": [{"gsm": "GSM1", "score": 0.9,
                                     "archs4_index": 0}]})
    assert overlay["query_label"] == key
    assert len(overlay["cohorts"]) == 1


def test_the_map_swatches_match_the_colours_plotly_draws():
    """Plotly cannot read a CSS variable, so the rail's two-cohort key mirrors
    the plot's hexes by hand. A drifted swatch would label the wrong cohort."""
    css = _all_css()
    for selector, value in [(".bm-retrieval-swatch.is-a", theme.RETRIEVAL_QUERY),
                            (".bm-retrieval-swatch.is-b", theme.RETRIEVAL_QUERY_B)]:
        assert f"{selector} {{ background: {value}; }}" in css, (
            f"{selector} drifted from {value}")


def test_the_second_cohorts_hit_symbol_is_valid_in_three_dimensions():
    """Scatter3d rejects an unknown symbol outright rather than degrading - the
    failure that took the whole figure callback down with a 500 the first time
    3-D was opened with a retrieval showing."""
    import plotly.graph_objects as go

    # Constructing the trace is the check: Scatter3d validates the symbol at
    # construction and raises ValueError on one it does not know.
    for symbol in (theme.RETRIEVAL_HIT_SYMBOL, theme.RETRIEVAL_HIT_SYMBOL_B):
        go.Scatter3d(x=[0], y=[0], z=[0], mode="markers",
                     marker=dict(symbol=symbol))

    with pytest.raises(ValueError):
        go.Scatter3d(x=[0], y=[0], z=[0], mode="markers",
                     marker=dict(symbol="star"))  # the reason 3-D uses diamond


# --- The router must not repaint a view that is already on screen ------------


def test_the_router_declines_to_repaint_the_route_already_painted():
    """`prevent_initial_call` is not enough on its own, and the gap it left was a
    live bug. `dcc.Location` publishes `pathname` once it mounts, which Dash
    reads as a change, so the router rebuilt the view the server had already
    painted. Rebuilding the retrieval view is not free - it reads the OSDR
    catalog and renders the sample preview - so its response landed a few
    hundred milliseconds after load, on top of whatever the user had done in the
    meantime. Clicking Cohort on arrival opened the cohort panel and then closed
    it again, 6 times out of 6, and the click was innocent: its own callback ran
    and returned the right answer, then the router overwrote the subtree it had
    just updated."""
    assert shell.navigation_for("/", "retrieve") is None
    assert shell.navigation_for("/map", "map") is None


def test_the_router_still_swaps_when_the_route_actually_changes():
    assert shell.navigation_for("/map", "retrieve")["key"] == "map"
    assert shell.navigation_for("/", "map")["key"] == "retrieve"


def test_an_unknown_path_resolves_to_the_default_route():
    assert shell.navigation_for("/nope", "map")["key"] == shell.DEFAULT_ROUTE["key"]
    assert shell.navigation_for("/nope", shell.DEFAULT_ROUTE["key"]) is None, (
        "and does not repaint the default view when that is already showing")


def test_a_query_string_link_is_not_treated_as_a_navigation():
    """`/?q=<sample_id>` is the deep link the served layout already handles at
    build time. Repainting it would throw that away and restart the view."""
    assert shell.navigation_for("/", "retrieve") is None


def test_the_router_writes_back_the_route_it_painted(app):
    """The store has to be an Output as well as a State, or the first real
    navigation is the only one that ever happens."""
    writers = [cb for cb in app.callback_map.values()
               if any(o.component_id == "route-store"
                      for o in (cb["output"] if isinstance(cb["output"], list)
                                else [cb["output"]]))]
    assert len(writers) == 1, "route-store must have exactly one writer"


def test_an_uploaded_retrieval_is_not_labelled_as_a_pooled_cohort(corpus):
    """An uploaded sample has no `sample_key`, so it has no coordinate on this
    map. Reporting that as "0 pooled cohort samples" would be false twice over,
    and it is what the rail said once the label moved out of its guard: the
    branch tested how many members were *locatable*, not whether anything was
    pooled."""
    overlay = callbacks._retrieval_overlay({
        "sample_id": "UPLOAD|my_counts.csv::SampleA",
        "mode": "uploaded",
        "hits": [{"gsm": "GSM1", "score": 0.9, "archs4_index": 0}],
        "query": {"sample_name": "SampleA"},
    })
    assert overlay is not None, "an uploaded search still draws its hits"
    assert overlay["query_points"] == []
    assert overlay["query_label"] == "", (
        "an empty label is what lets the rail fall back to 'an OSDR sample'")


def test_the_second_query_star_opens_the_second_cohort(corpus):
    """The comparison figure draws a `query2` node and every node it draws is
    clickable. This branch was added because clicking it reported "No metadata
    found"; it then raised TypeError instead, which is worse - the callback
    errors and the inspector silently keeps showing the previous node."""
    import pandas as pd
    from bridge_rna import panels

    a = pd.Series({"sample_id": "COHORT|a", "cohort_label": "Liver · Flight",
                   "is_cohort": "1", "study_id": "OSD-137", "grouped_by": "Study",
                   "stability": "0.72 at k = 6", "members": "OSD-137|x\nOSD-137|y",
                   "excluded": "", "outliers": ""})
    b = pd.Series({**a.to_dict(), "sample_id": "COHORT|b",
                   "cohort_label": "Liver · Ground"})
    hits = pd.DataFrame([{"gsm": "GSM1", "score": 0.9, "gse": "GSE1"}])

    panel = panels.build_details_panel(
        query=a, selected_payload={"kind": "query2", "node_id": "COHORT|b"},
        hits_df=hits, query_b=b)
    # ensure_ascii would escape the middot in a cohort label, so the
    # assertions below could never match and would pass vacuously.
    text = json.dumps(panel, default=str, ensure_ascii=False)
    assert "Liver · Ground" in text, "the second star must open the second cohort"
    assert "Liver · Flight" not in text, "and not the first"


def test_a_comparison_without_a_second_query_says_so_rather_than_raising(corpus):
    import pandas as pd
    from bridge_rna import panels

    a = pd.Series({"sample_id": "S1", "sample_name": "S1"})
    panel = panels.build_details_panel(
        query=a, selected_payload={"kind": "query2", "node_id": "nope"},
        hits_df=pd.DataFrame([{"gsm": "GSM1", "score": 0.9}]), query_b=None)
    assert "no second cohort" in json.dumps(panel, default=str,
                                            ensure_ascii=False)
