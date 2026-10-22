"""Loading skeletons, and the results table's fit and header styling (2026-10-22).

The skeletons replace two things a reader saw on every visit to the map:
Dash's bare "Loading..." text while the bundles arrived, and Plotly's default
white axes from -1 to 6 while the 942,563-point figure was built. The table
rules fix a comparison that ran past its frame at laptop widths and headers
that wore their columns' cell fonts.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"


def _css(name: str) -> str:
    return (ASSETS / name).read_text()


@pytest.fixture(scope="module")
def app():
    import app as shell

    return shell.build_app()


# --- The map's first figure -----------------------------------------------------

def test_the_graph_starts_on_a_navy_placeholder_not_plotlys_white_default():
    from manifold import layout, theme

    figure = layout.placeholder_figure()
    assert figure["data"] == []
    assert figure["layout"]["paper_bgcolor"] == theme.PLOT_BG
    assert figure["layout"]["plot_bgcolor"] == theme.PLOT_BG
    assert figure["layout"]["xaxis"]["visible"] is False
    assert figure["layout"]["yaxis"]["visible"] is False
    assert figure["layout"]["meta"] == {"placeholder": True}

    graph = _find(layout.build_view(), "manifold-graph")
    assert graph.figure == figure


def test_the_skeleton_names_what_is_being_drawn_and_hides_when_the_figure_lands(app):
    from manifold import data, layout

    skeleton = layout.map_skeleton()
    assert skeleton.id == "map-skeleton" and skeleton.role == "status"
    _n_archs4, _n_osdr, total = data.counts()
    assert f"Drawing {total:,} samples" in str(skeleton)

    # One clientside callback flips it, from the figure alone: a 942,563-point
    # figure is never posted to the server to change a class.
    assert "map-skeleton.className" in app.callback_map
    entry = app.callback_map["map-skeleton.className"]
    assert [i["id"] for i in entry["inputs"]] == ["manifold-graph"]
    inline = "\n".join(app._inline_scripts)
    assert "bm-map-skeleton is-done" in inline and "placeholder" in inline


def test_the_skeleton_never_takes_pointer_events_and_fades_when_done():
    css = _css("map.css")
    block = _rule(css, ".bm-map-skeleton")
    assert "pointer-events: none" in block
    done = _rule(css, ".bm-map-skeleton.is-done")
    assert "visibility: hidden" in done and "opacity: 0" in done


# --- The boot shell -------------------------------------------------------------

def test_the_boot_shell_is_shown_only_while_dash_is_still_loading():
    import app as app_module

    page = app_module.INDEX_STRING
    assert "{%app_entry%}" in page
    assert page.index('class="boot-skeleton"') < page.index("{%app_entry%}")
    assert 'aria-hidden="true"' in page.split('class="boot-skeleton"')[1][:40]

    css = _css("01-shell.css")
    assert re.search(r"\.boot-skeleton\s*\{\s*display:\s*none;", css)
    assert "body:has(._dash-loading) .boot-skeleton" in css
    # Dash's own text leaves the screen but stays for assistive technology.
    loading = _rule(css, "._dash-loading")
    assert "clip-path: inset(50%)" in loading and "display: none" not in loading


def test_the_sweep_stops_under_reduced_motion():
    css = _css("01-shell.css")
    reduced = css[css.index("@media (prefers-reduced-motion: reduce)"):]
    assert "animation: none" in reduced.split("}")[0] + reduced.split("}")[1]


# --- The results table ----------------------------------------------------------

def _widths(css: str, selector: str) -> int:
    m = re.search(re.escape(selector) + r"[^{]*\{\s*width:\s*(\d+)px", css)
    assert m, selector
    return int(m.group(1))


@pytest.mark.parametrize("comparison", [False, True])
def test_column_thresholds_are_the_sum_of_the_visible_fixed_widths(comparison):
    """Each container-query threshold must equal the fixed widths still shown
    plus 140 px for Title, or a width change silently brings back the sideways
    scroll a comparison had at 1,440 px."""
    css = _css("retrieve.css")
    if comparison:
        ranks = 2 * (_widths(css, ".results-table--comparison .col-rank_a")
                     + _widths(css, ".results-table--comparison .col-score_a"))
    else:
        ranks = (_widths(css, ".results-table .col-rank_a")
                 + _widths(css, ".results-table .col-score_a"))
    gsm = _widths(css, ".results-table .col-gsm")
    gse = _widths(css, ".results-table .col-gse")
    tissue = _widths(css, ".results-table .col-tissue")
    species = _widths(css, ".results-table .col-species")
    title = 140
    expected = {
        "species": ranks + gsm + gse + tissue + species + title - 1,
        "gse": ranks + gsm + gse + tissue + title - 1,
        "title": ranks + gsm + tissue + title - 1,
        "tissue": ranks + gsm + tissue - 1,
    }
    scope = (r"\.results-table--comparison" if comparison
             else r"\.results-table:not\(\.results-table--comparison\)")
    found = {}
    for width, column in re.findall(
            r"@container results-table \(max-width: (\d+)px\) \{\s*" + scope
            + r" \.col-(\w+) \{ display: none; \}", css):
        found[column] = int(width)
    assert found == expected


def test_headers_are_labels_not_values():
    css = _css("retrieve.css")
    block = _rule(css, ".results-table thead th.col-num")
    assert "font-family: var(--font-heading)" in block
    assert "font-style: normal" in block


def test_a_dropdown_search_box_draws_one_box_around_icon_and_field():
    css = _css("02-controls.css")
    container = _rule(css, ".dash-dropdown-content .dash-dropdown-search-container")
    assert "border: 1px solid" in container
    assert ":focus-within" in css
    assert re.search(r"\.dash-dropdown-search:focus-visible\s*\{\s*outline:\s*none",
                     css)


# --- helpers --------------------------------------------------------------------

def _rule(css: str, selector: str) -> str:
    """The declarations of the rule whose selector list starts a line with it."""
    m = re.search(r"(?m)^" + re.escape(selector) + r"[^{]*\{([^}]*)\}", css)
    assert m, selector
    return m.group(1)


def _find(node, wanted_id):
    if getattr(node, "id", None) == wanted_id:
        return node
    children = getattr(node, "children", None)
    if children is None:
        return None
    for child in children if isinstance(children, (list, tuple)) else [children]:
        found = _find(child, wanted_id)
        if found is not None:
            return found
    return None
