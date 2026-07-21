"""The serving app: layout wiring, callback plumbing, and readout rendering.

A Dash app fails at runtime, in the browser, when a callback names a component
that does not exist - there is no import-time check. These tests do that check
statically: every callback Input/Output/State id must be present in the layout
tree, and every className the Python emits must exist in the stylesheet.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest
from dash import html

import app_manifold
from manifold import callbacks, coherence, data, layout, paths, preflight, theme


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
    return app_manifold.build_app()


def test_app_builds(app):
    assert app.layout is not None
    assert app.title == "Bridge Manifold"


def test_every_callback_target_exists_in_the_layout(app):
    ids = {getattr(c, "id", None) for c in _walk(app.layout)}
    ids.discard(None)

    referenced = set()
    for cb in app.callback_map.values():
        for item in list(cb["inputs"]) + list(cb.get("state", [])):
            referenced.add(item["id"])
    for key in app.callback_map:
        for part in re.findall(r"([A-Za-z0-9_-]+)\.[A-Za-z]", key):
            referenced.add(part)

    missing = {r for r in referenced if r not in ids}
    assert not missing, f"callbacks reference components not in the layout: {sorted(missing)}"


def test_the_graph_and_readout_exist(app):
    ids = {getattr(c, "id", None) for c in _walk(app.layout)}
    for required in ("manifold-graph", "readout-body", "color-by", "method",
                     "dims", "layers", "budget", "plot-badges", "legend"):
        assert required in ids, f"layout is missing #{required}"


def test_color_by_options_are_all_renderable(corpus):
    """Every option the menu offers must actually produce a figure."""
    for opt in layout.color_by_options():
        fig, legend, badges = __import__("manifold.render", fromlist=["render"]).build_figure(
            "pca", "2d", opt["value"], ["archs4", "osdr", "density"], 1000, None)
        assert len(fig.data) > 0, f"color-by {opt['value']} drew nothing"


def test_archs4_tissue_option_hidden_until_the_join_exists(corpus):
    assert not data.archs4_tissue_available()
    values = {o["value"] for o in layout.color_by_options()}
    assert "archs4_tissue" not in values


def test_legend_parts_are_static_so_dash_can_validate_them(app):
    ids = {getattr(c, "id", None) for c in _walk(app.layout)}
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
    store = {"title": "Study (OSDR)", "items": [
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


def test_readout_clears_when_the_view_changes(corpus):
    """A statistic for a lasso that is no longer on screen is worse than none."""
    sel = {"points": [{"customdata": [corpus["n_archs4"] + i]} for i in range(30)]}

    kept = callbacks.readout_for(sel, triggered_id="manifold-graph")
    assert "No points selected" not in _readout_text(kept)

    for control in ("color-by", "method", "dims", "budget", "layers"):
        cleared = callbacks.readout_for(sel, triggered_id=control)
        assert "No points selected" in _readout_text(cleared), (
            f"readout survived a change to {control}")


def test_graph_config_disables_the_plotly_logo_and_keeps_lasso(app):
    graph = next(c for c in _walk(app.layout) if getattr(c, "id", None) == "manifold-graph")
    assert graph.config["displaylogo"] is False
    assert graph.config["scrollZoom"] is True
    assert theme.base_figure_layout()["dragmode"] == "lasso"


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
    assert callbacks._viewport_from_relayout({"dragmode": "lasso"}) == "unchanged"
    assert callbacks._viewport_from_relayout({"hovermode": "closest"}) == "unchanged"


# --- Selection extraction -------------------------------------------------

def test_selection_indices_reads_customdata():
    sel = {"points": [{"customdata": [5]}, {"customdata": [9]}, {"customdata": [5]}]}
    assert list(callbacks.selection_indices(sel)) == [5, 9, 5]


def test_selection_indices_skips_points_without_customdata():
    sel = {"points": [{"customdata": [1]}, {"pointIndex": 3}, {"customdata": None}]}
    assert list(callbacks.selection_indices(sel)) == [1]


def test_selection_indices_handles_empty_and_missing():
    for empty in (None, {}, {"points": []}):
        assert len(callbacks.selection_indices(empty)) == 0


def test_selection_indices_accepts_a_scalar_customdata():
    assert list(callbacks.selection_indices({"points": [{"customdata": 7}]})) == [7]


# --- Readout rendering ----------------------------------------------------

def _readout_text(component) -> str:
    """Flatten a Dash component tree to its visible text."""
    if isinstance(component, str):
        return component
    if isinstance(component, (list, tuple)):
        return " ".join(_readout_text(c) for c in component)
    children = getattr(component, "children", None)
    return _readout_text(children) if children is not None else ""


def test_readout_renders_a_coherent_selection(corpus):
    n_archs4 = corpus["n_archs4"]
    sel = np.where(corpus["osdr_cluster"] == 0)[0] + n_archs4
    result = coherence.analyze_selection(sel)
    text = _readout_text(callbacks._render_readout(result))

    assert "Coherent" in text
    assert "Cohesion z" in text and "kNN-purity" in text
    assert "Enriched features" in text


def test_readout_renders_the_honest_negative(corpus):
    result = {
        "status": "ok", "n": 40, "n_archs4": 40, "n_osdr": 0,
        "cohesion": {"obs": 0.1, "null_mean": 0.1, "null_std": 0.01, "z": 0.2, "emp_p": 0.5},
        "knn": None, "enrichment": [], "batch": None, "cross_dataset": False,
        "cohesive": False, "has_enrichment": False,
        "verdict": "This selection resembles a random draw.", "verdict_class": "null",
    }
    text = _readout_text(callbacks._render_readout(result))
    assert "random draw" in text
    assert "No categorical feature enriched" in text


def test_readout_refuses_a_tiny_selection(corpus):
    result = coherence.analyze_selection(np.arange(corpus["n_archs4"],
                                                   corpus["n_archs4"] + 3))
    text = _readout_text(callbacks._render_readout(result))
    assert "3 point" in text
    assert str(coherence.MIN_SELECTION) in text


def test_bold_markup_is_parsed_not_injected():
    parts = callbacks._html_with_bold("ARCHS4 live: <b>100,000</b> pts")
    assert any(isinstance(p, html.B) and p.children == "100,000" for p in parts)
    assert "".join(p if isinstance(p, str) else p.children for p in parts) == \
        "ARCHS4 live: 100,000 pts"


def test_bold_markup_survives_text_without_tags():
    assert callbacks._html_with_bold("plain") == ["plain"]


# --- Styling --------------------------------------------------------------

def test_every_classname_used_in_python_exists_in_the_stylesheet():
    css = (paths.ASSETS_DIR / "manifold.css").read_text()
    defined = set(re.findall(r"\.([a-zA-Z][a-zA-Z0-9_-]*)", css))

    # Dash exposes several className props; a class hook applied through any of
    # them is just as broken if the stylesheet never defines it.
    prop = r'(?:className|inputClassName|labelClassName)="([^"]+)"'
    used: set[str] = set()
    for py in sorted(Path(paths.MANIFOLD_ROOT / "manifold").glob("*.py")):
        for match in re.findall(prop, py.read_text()):
            used.update(match.split())

    # Dash supplies its own component classes; only ours are our problem.
    ours = {c for c in used if c.startswith("bm-")}
    missing = sorted(ours - defined)
    assert not missing, f"classNames with no CSS rule: {missing}"


def test_theme_matches_the_bridge_rna_tokens():
    """The chrome must stay pixel-identical to Bridge RNA; only the plot is dark."""
    css = (paths.ASSETS_DIR / "manifold.css").read_text()
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
