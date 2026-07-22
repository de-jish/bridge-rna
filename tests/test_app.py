"""The serving app: layout wiring, callback plumbing, and the coverage readout.

A Dash app fails at runtime, in the browser, when a callback names a component
that does not exist - there is no import-time check. These tests do that check
statically: every callback Input/Output/State id must be present in the layout
tree, and every className the Python emits must exist in the stylesheet.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from dash import html

import app as shell
from manifold import callbacks, colorby, layout, paths, preflight, theme


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
        ids |= {getattr(c, "id", None) for c in _walk(tree)}
    ids.discard(None)
    return ids


def test_app_builds(app):
    assert app.layout is not None
    assert app.title == "Bridge RNA"


def test_every_callback_target_exists_in_some_view(app, mounted_ids):
    referenced = set()
    for cb in app.callback_map.values():
        for item in list(cb["inputs"]) + list(cb.get("state", [])):
            referenced.add(item["id"])
    for key in app.callback_map:
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
    ids = [i for i in (getattr(c, "id", None) for c in _walk(tree)) if i]
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
                     "method", "dims", "layers", "budget", "plot-badges", "legend"):
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

    heavy = {"torch", "umap", "sklearn", "pynndescent"}
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


def test_the_batch_effect_caution_is_still_disclosed_somewhere(map_view):
    """Removing the readout deleted the only place this was ever said.

    The measured 54x cross-corpus effect is a property of the map, not of any
    selection, so losing the panel must not lose the warning.
    """
    text = _text(map_view)
    assert "54x" in text
    assert "corpora" in text.lower()
