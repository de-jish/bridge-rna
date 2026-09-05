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
    # The find box's suggestion rows are the one component family that cannot be
    # mounted up front: there are between zero and SUGGESTION_LIMIT of them and
    # each carries its own identifier in its id, which is what lets a click
    # commit the row that was clicked rather than an index into a list that the
    # next keystroke has already replaced. So the check mounts what the view can
    # *render*, through the same builder the callback uses - a typo in the id
    # then fails here exactly as it would for a static component.
    trees.append(html.Div(layout.suggestion_children(
        {"items": [{"value": "GSM1", "label": "GSM1", "detail": "", "kind": "gsm"}],
         "reason": ""})))
    trees.append(html.Div(layout.neighborhood_studies_children([
        {"gse": "GSE1", "display_gse": "GSE1", "sample_count": 1,
         "best_rank": 1, "dominant_tissue": "Eye"},
    ], None)))
    trees.append(html.Div(layout.neighborhood_samples_children([
        {"rank": 1, "gsm": "GSM1", "gse": "GSE1", "title": "Eye",
         "tissue": "Eye", "species": "mouse", "score": 0.9,
         "archs4_index": 1},
    ], None)))
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


def test_new_results_clear_ai_text_without_requesting_a_summary(app, monkeypatch):
    from types import SimpleNamespace
    from bridge_rna import callbacks as retrieval_callbacks

    monkeypatch.setattr(retrieval_callbacks, "ctx",
                        SimpleNamespace(triggered_id="hits-store"))
    def unexpected_request():
        pytest.fail("Changing results must not request an AI service")
    monkeypatch.setattr(retrieval_callbacks, "unavailable_reason", unexpected_request)
    callback = _callback_for(app, "ai-summary-button")
    assert callback(1, {"sample_id": "new query", "hits": []}) == ("", "")
    assert callback(1, None) == ("", "")


def test_source_identifiers_link_to_the_record_without_changing_metadata(monkeypatch):
    import pandas as pd
    from bridge_rna import panels

    monkeypatch.setattr(panels, "_fetch_osdr_study_summary", lambda _: {})
    query = pd.Series({"study_id": "OSD-100", "sample_name": "eye sample",
                       "sample_id": "OSD-100|eye sample"})
    hits = pd.DataFrame([{"gsm": "GSM4256053", "gse": "GSE143281",
                          "title": "Source title — unchanged", "score": 0.99}])
    expected_urls = {
        "OSD-100": "https://osdr.nasa.gov/bio/repo/data/studies/OSD-100",
        "GSM4256053": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM4256053",
        "GSE143281": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE143281",
    }
    for selected, accession in [(None, "OSD-100"),
                                ({"kind": "gsm", "node_id": "GSM4256053"}, "GSM4256053"),
                                ({"kind": "gse", "node_id": "GSE143281"}, "GSE143281")]:
        tree = html.Div(panels.build_details_panel(query, selected, hits))
        links = [c for c in _walk(tree) if isinstance(c, html.A)]
        assert any(c.children == accession and c.href == expected_urls[accession]
                   for c in links)
        assert all(c.href.startswith("https://") for c in links)
        if accession == "GSM4256053":
            assert "Source title — unchanged" in _text(tree)


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


def test_evidence_payload_is_json_safe_and_uses_the_fixed_depth():
    import numpy as np
    import pandas as pd

    from bridge_rna import callbacks as retrieval_callbacks

    frame = pd.DataFrame([
        {"gsm": "GSM1", "score": np.float32(0.9),
         "archs4_index": np.int64(7)},
        {"gsm": "GSM2", "score": np.float32(0.8),
         "archs4_index": np.int64(8)},
    ])
    payload = retrieval_callbacks._evidence_payload(frame, label="Cohort A")

    assert payload["depth_requested"] == 250
    assert payload["depth_returned"] == 2
    assert [row["rank"] for row in payload["hits"]] == [1, 2]
    assert all(type(row["rank"]) is int for row in payload["hits"])
    assert json.loads(json.dumps(payload)) == payload


def test_missing_supported_evidence_payload_requires_rerunning_the_retrieval():
    from bridge_rna import callbacks as retrieval_callbacks

    payload = retrieval_callbacks._evidence_payload(None, label="ignored")

    assert payload["available"] is False
    assert payload["depth_requested"] == 250
    assert payload["depth_returned"] == 0
    assert "saved retrieval predates evidence neighborhoods" in payload["reason"]
    assert "Run this supported retrieval again" in payload["reason"]


@pytest.mark.parametrize("mode", ["demo", "precomputed"])
def test_fresh_unsupported_evidence_payload_does_not_invite_the_same_rerun(mode):
    from bridge_rna import callbacks as retrieval_callbacks

    payload = retrieval_callbacks._evidence_payload(
        None, label="ignored", mode=mode)

    assert payload["available"] is False
    assert f"{mode} mode cannot provide" in payload["reason"]
    assert "Rerunning that same mode will not add it" in payload["reason"]
    assert "cached OSDR sample, uploaded counts, or cohort retrieval" \
        in payload["reason"]


def _callback_for(app, trigger_id):
    matches = [
        entry["callback"]
        for entry in app.callback_map.values()
        if any(item["id"] == trigger_id for item in entry["inputs"])
    ]
    assert len(matches) == 1
    return getattr(matches[0], "__wrapped__", matches[0])


def _retrieval_frame(prefix, count, start=0):
    import pandas as pd

    return pd.DataFrame([
        {
            "gsm": f"{prefix}{i}",
            "gse": f"GSE{i}",
            "title": f"{prefix} title {i}",
            "source_name": f"{prefix} source {i}",
            "characteristics": "",
            "tissue": "Liver",
            "species": "Mus musculus",
            "score": 0.99 - i / 100,
            "archs4_index": start + i,
        }
        for i in range(count)
    ])


def test_cached_sample_callback_carries_its_evidence_neighborhood(
    app, monkeypatch
):
    from bridge_rna import callbacks as retrieval_callbacks

    hits = _retrieval_frame("GSM-A-", 2)
    neighborhood = _retrieval_frame("GSM-A-", 4)
    monkeypatch.setattr(
        retrieval_callbacks,
        "search_hits_with_neighborhood",
        lambda **_kwargs: (hits, neighborhood, "cached"),
    )
    monkeypatch.setattr(
        retrieval_callbacks,
        "build_network_figure",
        lambda *, query, hits_df: {"requested_count": len(hits_df)},
    )
    query = retrieval_callbacks.samples_df.iloc[0]

    network, payload, _status = _callback_for(app, "search-button")(
        1, query["sample_id"], 2, "", []
    )

    assert network["requested_count"] == 2
    assert len(payload["hits"]) == 2
    assert payload["neighborhood"]["available"] is True
    assert payload["neighborhood"]["label"] == query["sample_name"]
    assert [row["gsm"] for row in payload["neighborhood"]["hits"]] == [
        "GSM-A-0", "GSM-A-1", "GSM-A-2", "GSM-A-3"
    ]


def test_demo_sample_callback_keeps_hits_and_marks_evidence_unavailable(
    app, monkeypatch
):
    from bridge_rna import callbacks as retrieval_callbacks

    hits = _retrieval_frame("GSM-D-", 2)
    monkeypatch.setattr(
        retrieval_callbacks,
        "search_hits_with_neighborhood",
        lambda **_kwargs: (hits, None, "demo"),
    )
    monkeypatch.setattr(
        retrieval_callbacks,
        "build_network_figure",
        lambda *, query, hits_df: {"requested_count": len(hits_df)},
    )
    query = retrieval_callbacks.samples_df.iloc[0]

    network, payload, _status = _callback_for(app, "search-button")(
        1, query["sample_id"], 2, "", []
    )

    assert network["requested_count"] == 2
    assert [row["gsm"] for row in payload["hits"]] == ["GSM-D-0", "GSM-D-1"]
    assert payload["neighborhood"]["available"] is False
    assert "demo mode cannot provide" in payload["neighborhood"]["reason"]
    assert "Rerunning that same mode will not add it" \
        in payload["neighborhood"]["reason"]


def test_upload_callback_carries_evidence_without_expanding_the_network(
    app, monkeypatch
):
    from bridge_rna import callbacks as retrieval_callbacks

    hits = _retrieval_frame("GSM-U-", 2)
    neighborhood = _retrieval_frame("GSM-U-", 4)
    monkeypatch.setattr(
        retrieval_callbacks,
        "run_uploaded_retrieval_with_neighborhood",
        lambda **_kwargs: (hits, neighborhood),
    )
    monkeypatch.setattr(
        retrieval_callbacks,
        "build_network_figure",
        lambda *, query, hits_df: {"requested_count": len(hits_df)},
    )

    network, payload, _status = _callback_for(app, "upload-search-button")(
        1,
        {"path": "counts.csv", "filename": "counts.csv", "columns": ["Sample A"]},
        "Sample A",
        2,
        "",
        [],
    )

    assert network["requested_count"] == 2
    assert len(payload["hits"]) == 2
    assert payload["neighborhood"]["label"] == "Sample A"
    assert payload["neighborhood"]["depth_returned"] == 4


def test_comparison_callback_keeps_independent_evidence_for_both_arms(
    app, monkeypatch
):
    import numpy as np

    from bridge_rna import callbacks as retrieval_callbacks
    from bridge_rna import cohorts as C

    facets = ("study", "tissue", "spaceflight")
    cohort_a = C.Cohort(
        "COHORT|a", "OSD-1", facets,
        {"study": "OSD-1", "tissue": "Liver", "spaceflight": "Flight"},
        ("A1", "A2"),
    )
    cohort_b = C.Cohort(
        "COHORT|b", "OSD-1", facets,
        {"study": "OSD-1", "tissue": "Liver", "spaceflight": "Ground"},
        ("B1", "B2"),
    )
    hits_a = _retrieval_frame("GSM-A-", 2, start=10)
    hits_b = _retrieval_frame("GSM-B-", 2, start=20)
    neighborhood_a = _retrieval_frame("GSM-A-", 4, start=10)
    neighborhood_b = _retrieval_frame("GSM-B-", 5, start=20)
    rows = np.asarray([[1.0, 0.0], [0.9, 0.1]], dtype=np.float32)
    by_id = {cohort_a.cohort_id: cohort_a, cohort_b.cohort_id: cohort_b}
    monkeypatch.setattr(
        C, "find_cohort", lambda cohort_id, facets=None: by_id.get(cohort_id)
    )
    results = {
        cohort_a.members: (hits_a, neighborhood_a, rows, None),
        cohort_b.members: (hits_b, neighborhood_b, rows, None),
    }
    monkeypatch.setattr(
        retrieval_callbacks,
        "run_cohort_retrieval_with_neighborhood",
        lambda members, topk: results[tuple(members)],
    )
    seen = {}

    def comparison_figure(**kwargs):
        seen.update(kwargs)
        return {"comparison": True}

    monkeypatch.setattr(
        retrieval_callbacks, "build_comparison_figure", comparison_figure
    )

    network, payload, _status = _callback_for(app, "cohort-search-button")(
        1,
        cohort_a.cohort_id,
        list(cohort_a.members),
        cohort_b.cohort_id,
        list(facets),
        2,
        "",
        [],
    )

    assert network == {"comparison": True}
    assert len(seen["hits_a"]) == len(seen["hits_b"]) == 2
    assert len(payload["hits"]) == len(payload["comparison"]["hits_b"]) == 2
    assert payload["neighborhood"]["label"] == cohort_a.label
    assert payload["comparison"]["neighborhood_b"]["label"] == cohort_b.label
    assert payload["neighborhood"]["depth_returned"] == 4
    assert payload["comparison"]["neighborhood_b"]["depth_returned"] == 5
    assert payload["neighborhood"]["hits"][0]["gsm"] == "GSM-A-0"
    assert payload["comparison"]["neighborhood_b"]["hits"][0]["gsm"] == "GSM-B-0"


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
    for required in ("legend-title", "legend-search", "legend-search-group",
                     "legend-list", "legend-store"):
        assert required in ids, f"{required} is only created at runtime"


def test_neighborhood_explorer_parts_are_static_so_dash_can_validate_them(map_view):
    ids = {getattr(c, "id", None) for c in _walk(map_view)}
    required = {
        "explore-neighborhood", "neighborhood-drawer", "neighborhood-close",
        "neighborhood-heading", "neighborhood-arm", "neighborhood-tab",
        "neighborhood-search", "neighborhood-body",
        "neighborhood-open-store", "neighborhood-focus-store",
        "neighborhood-focus-sink",
    }
    assert required <= ids, f"layout is missing {sorted(required - ids)}"


def test_one_callback_owns_neighborhood_open_state(app):
    writers = [k for k in app.callback_map
               if "neighborhood-open-store.data" in k]
    assert len(writers) == 1
    inputs = {i["id"] for i in app.callback_map[writers[0]]["inputs"]}
    assert {"frame-retrieval", "explore-neighborhood", "neighborhood-close",
            "hits-store"} <= inputs


def test_explorer_does_not_make_the_viewport_multi_writer(app):
    writers = [k for k in app.callback_map if "viewport-store.data" in k]
    assert len(writers) == 1


def test_tab_and_search_changes_do_not_redraw_the_map(app):
    figure_key = next(k for k in app.callback_map
                      if "manifold-graph.figure" in k)
    figure_inputs = {(i["id"], i["property"])
                     for i in app.callback_map[figure_key]["inputs"]}
    assert {("neighborhood-open-store", "data"),
            ("neighborhood-arm", "value"),
            ("neighborhood-focus-store", "data")} <= figure_inputs
    assert ("neighborhood-tab", "value") not in figure_inputs
    assert ("neighborhood-search", "value") not in figure_inputs


@pytest.mark.parametrize(
    "trigger,previous,has_retrieval,expected",
    [
        (None, None, False,
         {"open": False, "opener": "explore-neighborhood",
          "focus_target": None}),
        ("frame-retrieval", None, True,
         {"open": True, "opener": "frame-retrieval",
          "focus_target": "neighborhood-heading"}),
        ("explore-neighborhood", {"open": False, "opener": "frame-retrieval"},
         True, {"open": True, "opener": "explore-neighborhood",
                "focus_target": "neighborhood-heading"}),
        ("neighborhood-close", {"open": True, "opener": "frame-retrieval"},
         True, {"open": False, "opener": "frame-retrieval",
                "focus_target": "frame-retrieval"}),
        ("neighborhood-close", {"open": False,
                                "opener": "frame-retrieval"},
         True, {"open": False, "opener": "frame-retrieval",
                "focus_target": None}),
        ("hits-store", {"open": True, "opener": "explore-neighborhood"},
         True, {"open": False, "opener": "explore-neighborhood",
                "focus_target": None}),
        ("hits-store", {"open": False, "opener": "frame-retrieval"},
         True, {"open": False, "opener": "frame-retrieval",
                "focus_target": None}),
        ("frame-retrieval", {"open": False, "opener": "explore-neighborhood"},
         False, {"open": False, "opener": "explore-neighborhood",
                 "focus_target": None}),
    ],
)
def test_neighborhood_open_state_has_one_deterministic_transition(
        trigger, previous, has_retrieval, expected):
    assert callbacks.next_neighborhood_open_state(
        trigger, previous, has_retrieval) == expected


def test_neighborhood_overview_renders_metrics_coverage_and_ranked_groups():
    summary = {
        "available": True,
        "depth": 4,
        "sentence": "Eye is the leading tissue: 2 of 3 samples with tissue metadata.",
        "study_count": 2,
        "top_three_study_samples": 3,
        "score": {"count": 4, "median": 0.975, "minimum": 0.95,
                  "maximum": 0.99},
        "tissue": {"covered": 3, "items": [
            {"label": "Eye", "count": 2, "share": 2 / 3},
            {"label": "Brain", "count": 1, "share": 1 / 3},
        ]},
        "species": {"covered": 3, "items": [
            {"label": "Mus musculus", "count": 2, "share": 2 / 3},
            {"label": "Homo sapiens", "count": 1, "share": 1 / 3},
        ]},
        "leading_studies": [
            {"display_gse": "GSE10", "sample_count": 2,
             "dominant_tissue": "Eye"},
            {"display_gse": "GSE20", "sample_count": 1,
             "dominant_tissue": "Brain"},
        ],
    }

    children = layout.neighborhood_overview_children(summary)
    text = _text(children)
    classes = _classes(children)

    assert summary["sentence"] in text
    assert "Tissue categories 3 of 4" in text
    assert "Species metadata 3 of 4" in text
    assert all(value in text for value in ("4", "2", "0.975"))
    assert "GSE10" in text and "GSE20" in text
    assert "bm-neighborhood-metrics" in classes
    assert classes.count("bm-neighborhood-bar-fill") == 4


def test_neighborhood_studies_use_separate_buttons_and_geo_links():
    groups = [
        {"gse": "GSE10", "display_gse": "GSE10", "sample_count": 2,
         "best_rank": 1, "title": "Retina study", "dominant_tissue": "Eye",
         "dominant_tissue_count": 2, "tissue_covered": 2},
        {"gse": "", "display_gse": "No GSE recorded", "sample_count": 1,
         "best_rank": 4, "title": "", "dominant_tissue": "",
         "dominant_tissue_count": 0, "tissue_covered": 0},
    ]
    children = layout.neighborhood_studies_children(
        groups, {"kind": "study", "value": "GSE10"})
    tree = html.Div(children)
    buttons = [c for c in _walk(tree) if type(c).__name__ == "Button"]
    links = [c for c in _walk(tree) if type(c).__name__ == "A"]

    assert [button.id for button in buttons] == [
        {"type": "neighborhood-study", "value": "GSE10"},
        {"type": "neighborhood-study",
         "value": layout.NO_GSE_FOCUS_VALUE},
    ]
    assert "is-selected" in buttons[0].className
    assert getattr(buttons[0], "aria-pressed") == "true"
    assert getattr(buttons[1], "aria-pressed") == "false"
    assert buttons[1].disabled is False
    assert "Retina study · Eye · 2 of 2 with tissue metadata" in _text(buttons[0])
    assert "Sample title not recorded · Tissue not recorded · 0 of 0 with tissue metadata" \
        in _text(buttons[1])
    assert len(links) == 1
    assert links[0].href.endswith("?acc=GSE10")
    assert not any(type(c).__name__ == "A" for c in _walk(buttons[0]))


def test_unassigned_study_button_focuses_rows_with_blank_gse():
    groups = [
        {"gse": "", "display_gse": "No GSE recorded", "sample_count": 1,
         "best_rank": 1, "dominant_tissue": "Eye"},
    ]
    button = next(c for c in _walk(html.Div(
        layout.neighborhood_studies_children(groups, None)
    )) if type(c).__name__ == "Button")

    focus = callbacks.next_neighborhood_focus(button.id, 1)
    payload = {"neighborhood": {
        "available": True, "label": "A", "depth_returned": 1,
        "hits": [{"rank": 1, "gsm": "GSM1", "gse": "",
                  "score": 0.99, "archs4_index": 4}],
    }}
    overlay = callbacks._neighborhood_overlay(payload, "a", focus)

    assert "No GSE recorded" in _text(button)
    assert overlay["focus_points"] == [4]


def test_neighborhood_samples_keep_unlocatable_rows_selectable_and_show_detail():
    rows = [
        {"rank": 1, "gsm": "GSM1", "gse": "GSE10", "title": "Retina A",
         "tissue": "Eye", "species": "Mus musculus", "score": 0.99,
         "archs4_index": 4},
        {"rank": 2, "gsm": "GSM2", "gse": "", "title": "Brain B",
         "tissue": "Brain", "species": "Homo sapiens", "score": None,
         "archs4_index": None},
    ]
    children = layout.neighborhood_samples_children(
        rows, {"kind": "sample", "value": "GSM2"})
    tree = html.Div(children)
    buttons = [c for c in _walk(tree) if type(c).__name__ == "Button"]
    links = [c for c in _walk(tree) if type(c).__name__ == "A"]

    assert [button.id["value"] for button in buttons] == ["GSM1", "GSM2"]
    assert buttons[1].disabled is False
    assert "is-selected" in buttons[1].className
    assert getattr(buttons[1], "aria-pressed") == "true"
    assert "Brain B" in _text(children)
    assert any(link.href.endswith("?acc=GSM2") for link in links)


def test_neighborhood_builders_have_explicit_empty_and_unavailable_states():
    assert "No studies" in _text(layout.neighborhood_studies_children([], None))
    assert "No samples" in _text(layout.neighborhood_samples_children([], None))
    unavailable = layout.neighborhood_unavailable_children(
        "Run this retrieval again to build its evidence neighborhood.")
    assert "Run this retrieval again" in _text(unavailable)
    assert "bm-neighborhood-empty" in _classes(unavailable)


def test_comparison_arm_options_name_both_cohorts_without_merging_them():
    payload = {
        "query": {"cohort_label": "Flight"},
        "neighborhood": {"available": True, "label": "Flight", "hits": []},
        "comparison": {
            "query_b": {"cohort_label": "Ground"},
            "neighborhood_b": {"available": True, "label": "Ground", "hits": []},
        },
    }
    options, style = callbacks.neighborhood_arm_options(payload)
    assert options == [
        {"label": "A · Flight", "value": "a"},
        {"label": "B · Ground", "value": "b"},
    ]
    assert style == {}

    single, single_style = callbacks.neighborhood_arm_options({
        "query": {"sample_name": "Sample 1"},
        "neighborhood": {"available": True, "label": "Sample 1", "hits": []},
    })
    assert single == [{"label": "Sample 1", "value": "a"}]
    assert single_style == {"display": "none"}


def test_drawer_state_reads_only_the_selected_comparison_arm():
    arm_a = _neighborhood_payload("Flight")
    arm_b = _neighborhood_payload("Ground", 10)
    arm_b["hits"] = arm_b["hits"][:1]
    arm_b["depth_returned"] = 1
    payload = {
        "hits": list(arm_a["hits"]),
        "neighborhood": arm_a,
        "comparison": {"hits_b": list(arm_b["hits"]),
                       "neighborhood_b": arm_b},
    }

    state = callbacks.neighborhood_drawer_state(
        payload, "b", "samples", "", None)

    assert state["heading"] == "Ground"
    assert state["meta"] == (
        "1 requested hit · 1 returned · "
        "exact top-250 cosine neighborhood in 512-D"
    )
    assert state["tabs"][1]["label"] == "Studies 1"
    assert state["tabs"][2]["label"] == "Samples 1"
    buttons = [c for c in _walk(html.Div(state["body"]))
               if type(c).__name__ == "Button"]
    assert [button.id["value"] for button in buttons] == ["GSM1"]
    assert state["search_style"] == {}


def test_single_drawer_meta_names_requested_hits_and_exact_cosine_context():
    payload = {
        "hits": [{"gsm": "GSM1"}, {"gsm": "GSM2"}],
        "neighborhood": _neighborhood_payload("Sample 1"),
    }

    state = callbacks.neighborhood_drawer_state(
        payload, "a", "overview", "", None)

    assert state["meta"] == (
        "2 requested hits · 3 returned · "
        "exact top-250 cosine neighborhood in 512-D"
    )


def test_comparison_drawer_meta_reads_requested_count_from_each_active_arm():
    payload = {
        "hits": [{"gsm": "A1"}, {"gsm": "A2"}],
        "neighborhood": _neighborhood_payload("Flight"),
        "comparison": {
            "hits_b": [{"gsm": "B1"}],
            "neighborhood_b": _neighborhood_payload("Ground", 10),
        },
    }

    meta_a = callbacks.neighborhood_drawer_state(
        payload, "a", "overview", "", None)["meta"]
    meta_b = callbacks.neighborhood_drawer_state(
        payload, "b", "overview", "", None)["meta"]

    assert meta_a.startswith("2 requested hits · 3 returned")
    assert meta_b.startswith("1 requested hit · 3 returned")
    assert meta_a.endswith("exact top-250 cosine neighborhood in 512-D")
    assert meta_b.endswith("exact top-250 cosine neighborhood in 512-D")


def test_drawer_state_explains_an_unavailable_neighborhood():
    reason = "Run this retrieval again to build its evidence neighborhood."
    state = callbacks.neighborhood_drawer_state(
        {"neighborhood": {"available": False, "reason": reason}},
        "a", "overview", "", None)
    assert reason in _text(state["body"])
    assert state["search_style"] == {"display": "none"}


def test_drawer_distinguishes_stale_supported_from_fresh_unsupported_results():
    stale = callbacks.neighborhood_drawer_state(
        {"mode": "cached", "hits": [{"gsm": "GSM1"}]},
        "a", "overview", "", None)
    unsupported_reason = (
        "The demo mode cannot provide an exact top-250 cosine neighborhood in "
        "512-D. Rerunning that same mode will not add it. Use a cached OSDR "
        "sample, uploaded counts, or cohort retrieval instead."
    )
    unsupported = callbacks.neighborhood_drawer_state(
        {"mode": "demo", "hits": [{"gsm": "GSM1"}],
         "neighborhood": {"available": False, "reason": unsupported_reason}},
        "a", "overview", "", None)

    stale_text = _text(stale["body"])
    unsupported_text = _text(unsupported["body"])
    assert "saved retrieval predates evidence neighborhoods" in stale_text
    assert "Run this supported retrieval again" in stale_text
    assert "demo mode cannot provide" in unsupported_text
    assert "Rerunning that same mode will not add it" in unsupported_text
    assert stale_text != unsupported_text


def test_unavailable_comparison_arm_keeps_its_cohort_name():
    reason = "Run this retrieval again."
    state = callbacks.neighborhood_drawer_state({
        "query": {"cohort_label": "Flight"},
        "neighborhood": {"available": True, "label": "Flight", "hits": []},
        "comparison": {
            "query_b": {"cohort_label": "Ground"},
            "neighborhood_b": {"available": False, "reason": reason,
                               "label": "", "hits": []},
        },
    }, "b", "overview", "", None)

    assert state["heading"] == "Ground"
    assert reason in _text(state["body"])


@pytest.mark.parametrize(
    "trigger,value,expected",
    [
        ("hits-store", {"anything": True}, None),
        ({"type": "neighborhood-study", "value": "GSE10"}, 1,
         {"kind": "study", "value": "GSE10"}),
        ({"type": "neighborhood-sample", "value": "GSM2"}, 1,
         {"kind": "sample", "value": "GSM2"}),
        ("manifold-graph", {"points": [{"customdata": [
            "neighborhood", 2, "GSM2", "GSE10", 0.98, "Eye", "mouse"]}]},
         {"kind": "sample", "value": "GSM2"}),
    ],
)
def test_neighborhood_focus_accepts_only_explicit_evidence_actions(
        trigger, value, expected):
    assert callbacks.next_neighborhood_focus(trigger, value) == expected


@pytest.mark.parametrize(
    "trigger,current",
    [
        ({"type": "neighborhood-study", "value": "GSE10"},
         {"kind": "study", "value": "GSE10"}),
        ({"type": "neighborhood-sample", "value": "GSM2"},
         {"kind": "sample", "value": "GSM2"}),
    ],
)
def test_activating_the_focused_row_again_clears_focus(trigger, current):
    assert callbacks.next_neighborhood_focus(trigger, 2, current) is None


def test_cleared_row_focus_renders_aria_pressed_false():
    trigger = {"type": "neighborhood-study", "value": "GSE10"}
    focus = callbacks.next_neighborhood_focus(
        trigger, 2, {"kind": "study", "value": "GSE10"})
    group = {
        "gse": "GSE10", "display_gse": "GSE10", "sample_count": 1,
        "best_rank": 1, "title": "Study", "dominant_tissue": "Eye",
        "dominant_tissue_count": 1, "tissue_covered": 1,
    }

    button = next(c for c in _walk(html.Div(
        layout.neighborhood_studies_children([group], focus)
    )) if type(c).__name__ == "Button")

    assert getattr(button, "aria-pressed") == "false"
    assert "is-selected" not in button.className


def test_neighborhood_focus_ignores_mounts_and_non_evidence_map_clicks():
    assert callbacks.next_neighborhood_focus(
        {"type": "neighborhood-study", "value": "GSE10"}, 0
    ) is callbacks.no_update
    assert callbacks.next_neighborhood_focus(
        "manifold-graph", {"points": [{"customdata": ["OSD-1|sample"]}]}
    ) is callbacks.no_update
    assert callbacks.next_neighborhood_focus("manifold-graph", None) \
        is callbacks.no_update


def test_one_callback_owns_neighborhood_focus_state(app):
    writers = [k for k in app.callback_map
               if "neighborhood-focus-store.data" in k]
    assert len(writers) == 1
    inputs = {(str(i["id"]), i["property"])
              for i in app.callback_map[writers[0]]["inputs"]}
    assert ("manifold-graph", "clickData") in inputs
    assert ("hits-store", "data") in inputs
    assert any("neighborhood-study" in component_id
               for component_id, _property in inputs)
    assert any("neighborhood-sample" in component_id
               for component_id, _property in inputs)
    states = {(str(item["id"]), item["property"])
              for item in app.callback_map[writers[0]]["state"]}
    assert ("neighborhood-focus-store", "data") in states


def test_one_callback_owns_focus_and_map_selected_tab_transition(app):
    focus_writer = next(k for k in app.callback_map
                        if "neighborhood-focus-store.data" in k)
    tab_writer = next(k for k in app.callback_map
                      if "neighborhood-tab.value" in k)

    assert focus_writer == tab_writer


def test_map_neighborhood_click_reveals_samples_without_clearing_controls():
    click = {"points": [{"customdata": [
        "neighborhood", 2, "GSM2", "GSE10", 0.98, "Eye", "mouse",
    ]}]}

    focus, tab, arm, search = callbacks.next_neighborhood_interaction(
        "manifold-graph", click)

    assert focus == {"kind": "sample", "value": "GSM2"}
    assert tab == "samples"
    assert arm is callbacks.no_update
    assert search is callbacks.no_update


def test_selected_sample_detail_ignores_filter_but_list_keeps_filter():
    payload = {"neighborhood": {
        "available": True, "label": "A", "depth_returned": 2,
        "hits": [
            {"rank": 1, "gsm": "GSM1", "gse": "GSE1",
             "title": "Retina sample", "tissue": "Eye",
             "species": "mouse", "score": 0.99, "archs4_index": 1},
            {"rank": 2, "gsm": "GSM2", "gse": "GSE2",
             "title": "Brain sample", "tissue": "Brain",
             "species": "mouse", "score": 0.98, "archs4_index": 2},
        ],
    }}

    state = callbacks.neighborhood_drawer_state(
        payload, "a", "samples", "GSM1",
        {"kind": "sample", "value": "GSM2"})
    tree = html.Div(state["body"])
    buttons = [c for c in _walk(tree) if type(c).__name__ == "Button"]
    detail_links = [c for c in _walk(tree)
                    if getattr(c, "className", None) == "bm-neighborhood-geo"]

    assert [button.id["value"] for button in buttons] == ["GSM1"]
    assert "Brain sample" in _text(tree)
    assert any(link.href.endswith("?acc=GSM2") for link in detail_links)


def test_selected_sample_detail_survives_an_empty_filtered_list():
    payload = {"neighborhood": {
        "available": True, "label": "A", "depth_returned": 1,
        "hits": [
            {"rank": 1, "gsm": "GSM2", "gse": "GSE2",
             "title": "Brain sample", "tissue": "Brain",
             "species": "mouse", "score": 0.98, "archs4_index": 2},
        ],
    }}

    state = callbacks.neighborhood_drawer_state(
        payload, "a", "samples", "no matches",
        {"kind": "sample", "value": "GSM2"})

    assert "Brain sample" in _text(state["body"])
    assert "No samples match this search" in _text(state["body"])


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


# --- What counts as committing a search --------------------------------------

class _Ctx:
    """A stand-in for `dash.ctx`, which only exists inside a live callback."""

    def __init__(self, triggered_id=None, triggered=()):
        self.triggered_id = triggered_id
        self.triggered = list(triggered)


def test_a_re_rendered_suggestion_family_is_not_a_click(monkeypatch):
    """A pattern-matching `ALL` input fires when its family is re-rendered.

    This family is re-rendered on every keystroke, with every `n_clicks` back at
    zero, so "some suggestion input fired" is not "a suggestion was chosen".
    """
    ids = [{"type": "find-suggestion", "value": v}
           for v in ("GSM1", "GSM9000005", "GSM3")]

    # A keystroke: the whole family re-rendered, nothing clicked.
    monkeypatch.setattr(callbacks, "ctx", _Ctx(ids[0], []))
    assert callbacks.chosen_suggestion([0, 0, 0], ids) is None

    # A real click on the second row.
    monkeypatch.setattr(callbacks, "ctx", _Ctx(ids[1], []))
    assert callbacks.chosen_suggestion([0, 1, 0], ids) == {
        "value": "GSM9000005", "seq": 1}

    # Choosing the same row twice must be two distinct payloads, or the second
    # write would not change the store and would commit nothing.
    assert callbacks.chosen_suggestion([0, 2, 0], ids) == {
        "value": "GSM9000005", "seq": 2}

    # Enter or blur is not a suggestion at all.
    monkeypatch.setattr(callbacks, "ctx", _Ctx("find-input", []))
    assert callbacks.chosen_suggestion([0, 1, 0], ids) is None


def test_a_component_merely_appearing_is_not_a_commit(monkeypatch):
    """Dash wakes a callback when one of its inputs newly appears, and reports
    that with a `None` value. Read as an event, it committed the empty box and
    wiped the marks from a search the reader had not touched."""
    monkeypatch.setattr(callbacks, "ctx", _Ctx(
        "find-input", [{"prop_id": "find-input.n_blur", "value": None}]))
    assert not callbacks._fired("find-input.n_blur")
    monkeypatch.setattr(callbacks, "ctx", _Ctx(
        "find-input", [{"prop_id": "find-input.n_blur", "value": 1}]))
    assert callbacks._fired("find-input.n_blur")
    monkeypatch.setattr(callbacks, "ctx", _Ctx(None, []))
    assert not callbacks._fired("find-input.n_submit")


def test_nothing_that_changes_per_keystroke_can_reach_the_search(app):
    """The search callback must not share a wake-up with the suggestion list.

    Two separate defects made this a rule rather than a preference, and the
    second is the reason the chosen row arrives as a store instead of as its own
    click. `find-input.value` on this callback ran a real search - and rebuilt a
    942,563-point figure - on every letter typed. The suggestion family's `ALL`
    input did something worse: it woke this callback on every keystroke with
    nothing to do, and **Dash discards the response to a request a newer request
    for the same callback supersedes**, so a keystroke's no-op overtook an Enter
    that the server had already answered correctly. Measured on the real app,
    the first find of every session was silently lost.

    Every input here must therefore change only when someone has finished.
    """
    key = next(k for k in app.callback_map if "find-store.data" in k)
    inputs = {(str(i["id"]), i["property"])
              for i in app.callback_map[key]["inputs"]}
    assert ("find-input", "value") not in inputs
    assert not [i for i in inputs if "find-suggestion" in i[0]], (
        f"the suggestion family can wake the search callback: {inputs}")
    assert {("find-input", "n_submit"), ("find-input", "n_blur"),
            ("find-chosen", "data")} <= inputs


def test_the_suggestion_list_is_the_only_thing_a_keystroke_redraws(app):
    """And the corollary: `find-input.value` reaches exactly one callback."""
    readers = [k for k, cb in app.callback_map.items()
               if any(i["id"] == "find-input" and i["property"] == "value"
                      for i in cb["inputs"])]
    assert readers == ["find-suggestions.children"], readers


# --- Framing and its one reverse --------------------------------------------

def test_the_reset_is_offered_only_when_the_map_is_framed():
    """A visible control that does nothing is what the lasso was removed for.

    "Framed" is a property of `viewport-store` and of nothing else, which is the
    finding that made one reset enough: framing a find, framing a retrieval and
    the user's own scroll-zoom are three ways to write one value, and by the
    time it is written they are indistinguishable.
    """
    window = [0.0, 1.0, 0.0, 1.0]
    assert callbacks.is_framed(window, "2d")
    assert not callbacks.is_framed(None, "2d")
    assert not callbacks.is_framed([], "2d")
    # 3-D never frames: frame_points declines and the camera ignores axis
    # ranges, so a reset there would be a click with no visible effect.
    assert not callbacks.is_framed(window, "3d")


def test_2d_aspect_constraint_preserves_ranges_when_the_canvas_resizes():
    """Docking a drawer must not turn an Explore click into a hidden zoom.

    The x/y scales remain equal, but Plotly must absorb a new canvas aspect
    ratio in the axis domains.  Its default ``range`` constraint instead
    changed the visible x range when the neighborhood drawer narrowed the map.
    """
    layout_2d = theme.base_figure_layout(False)
    assert layout_2d["xaxis"]["constrain"] == "domain"
    assert layout_2d["yaxis"]["constrain"] == "domain"


def test_neighborhood_only_update_leaves_unframed_axes_untouched():
    """Explorer marks may change without asking Plotly to autorange again."""
    assert callbacks.viewport_axes(None, "2d", preserve=True) == {}


def test_an_unframed_figure_releases_the_axes_rather_than_omitting_them():
    """`uirevision="keep"` preserves the user's zoom unless the figure changes
    the attribute, so "no range key" is not the same instruction as "autorange".
    Omitting it left a reset that redrew the whole corpus into the window the
    user was already looking at."""
    assert theme.base_figure_layout(False)["uirevision"], (
        "this test is about uirevision; the layout no longer sets one")

    framed = callbacks.viewport_axes([1.0, 2.0, 3.0, 4.0], "2d")
    assert framed["xaxis"]["range"] == [1.0, 2.0]
    assert framed["yaxis"]["range"] == [3.0, 4.0]

    freed = callbacks.viewport_axes(None, "2d")
    assert freed["xaxis"]["autorange"] is True
    assert freed["yaxis"]["autorange"] is True
    assert "range" not in freed["xaxis"]

    assert callbacks.viewport_axes([1.0, 2.0, 3.0, 4.0], "3d") == {}


def test_one_callback_owns_the_viewport_and_every_framing_action_feeds_it(app):
    """Three actions narrow the view and one undoes them, through one writer.

    Two callbacks writing one Output is a race Dash only sometimes rejects, and
    a reset handler bolted onto each frame button would have been exactly that -
    plus a third state, a plain scroll-zoom, that neither of them could undo.
    """
    writers = [key for key in app.callback_map if "viewport-store.data" in key]
    assert len(writers) == 1, f"viewport-store has {len(writers)} writers"
    inputs = {i["id"] for i in app.callback_map[writers[0]]["inputs"]}
    assert {"frame-find", "frame-retrieval", "reset-view"} <= inputs


def test_the_reset_clears_the_viewport_and_nothing_else(app):
    """Unframing must keep the query, the results and the selection.

    It writes one Output, so there is nothing else it *could* clear - which is
    the argument, and is worth pinning rather than re-reading the callback.
    """
    key = next(k for k in app.callback_map if "viewport-store.data" in k)
    assert key.strip(".").split("...") == ["viewport-store.data"]


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


def test_the_dash_tokens_that_carry_text_use_the_accessible_blue():
    """Dash spends its "interactive strong" token on text, not only on fills.

    It paints an option's label with it on `:hover` and `:focus-within`, at a
    specificity of (0,3,1) - `:not(:has(input[disabled]))` is worth more than
    it looks - so it outranks what either view writes for its own controls.
    Pointing it at --accent put 3.76:1 text on screen from a rule neither
    stylesheet owns; pointing it at --accent-text fixes every Dash control at
    once. The `:focus-within` half only became reachable when the radio inside
    a segmented pill stopped being `display: none`, which is why this was
    invisible until the keyboard fix landed.
    """
    css = _all_css()
    match = re.search(r"--Dash-Fill-Interactive-Strong:\s*var\((--[\w-]+)\)", css)
    assert match, "Dash's interactive-strong token is no longer mapped"
    assert match.group(1) == "--accent-text", (
        "Dash paints text with this token, so it must be the AA-safe blue")


def test_the_retrieval_input_rules_cannot_reach_the_map():
    """`dash-input` is Dash's class, not ours, and it is on every dcc.Input.

    Styling `.dash-input .dash-input-element` from retrieve.css therefore
    reached the map's legend filter as well, which sits on the navy canvas -
    and turned it into a white box with dark text. Caught in a screenshot, not
    by any assertion, which is why there is one now. The retrieval view's own
    rules stay inside the retrieval view's own container.
    """
    retrieve = (paths.ASSETS_DIR / "retrieve.css").read_text()
    unscoped = re.findall(r"^\s*\.dash-input\s", retrieve, re.M)
    assert not unscoped, (
        "retrieve.css styles .dash-input without scoping it; Dash puts that "
        "class on the map's legend filter too")
    assert ".sidebar .dash-input .dash-input-element" in retrieve


def test_the_map_view_has_a_landmark_and_a_heading(map_view):
    """The map had a page title and no structure under it.

    The retrieval view already offered a nav, a controls aside, a main and an
    inspector aside; the map was a `div` inside a `div`, so the only heading
    below the shell's H1 was nothing at all and there was no way to jump to
    either region.
    """
    tags = [type(c).__name__ for c in _walk(map_view)]
    assert "Main" in tags, "the plot is not a main landmark"
    assert "Aside" in tags, "the control rail is not a complementary landmark"
    headings = [c for c in _walk(map_view) if type(c).__name__ == "H2"]
    assert headings, "the map offers no heading below the shell's H1"


def test_the_map_graph_relayouts_when_its_container_resizes(map_view):
    """Plotly keeps the geometry it was first laid out with unless told not to.

    The map's container changes size for reasons no callback fires on - the
    stacked breakpoint below 900 px, a phone rotating, a window being dragged -
    and without this the scatter drew a quadrant of the corpus into the whole
    canvas. The retrieval view's graph has always set it.
    """
    graphs = [c for c in _walk(map_view) if getattr(c, "id", None) == "manifold-graph"]
    assert graphs, "the map graph is gone"
    assert graphs[0].config.get("responsive") is True


def test_the_map_stylesheet_stacks_below_a_breakpoint():
    """A fixed 268 px rail beside a fixed 228 px key is 496 px of furniture.

    On a 393 px phone that left the plot a 125 px strip with the key sitting on
    top of the rail. The map carried no media query at all until 2026-08-11, so
    this pins the two things that fix it: the columns stack, and the plot stops
    taking its height from a column it no longer has.
    """
    css = _all_css()
    assert "@media (max-width: 900px)" in css
    stacked = css.split("@media (max-width: 900px)", 1)[1]
    assert ".bm-body { flex-direction: column; }" in stacked
    assert "height: 68vh" in stacked


def test_key_and_legend_rows_overflow_rather_than_compress():
    """A column flex item shrinks below its content height by default.

    Both panels sit inside a height-capped box, so without `flex: none` the
    rows compressed into each other and lost their descenders instead of the
    list scrolling - there was never any overflow for `overflow-y: auto` to
    act on. Reproduced at 393 px, where the panel is capped at 46% of the
    canvas.
    """
    css = _all_css()
    assert ".bm-legend-list > .bm-legend-item { flex: none; }" in css
    key_rule = css.split(".bm-key-row {", 1)[1].split("}", 1)[0]
    assert "flex: none" in key_rule


def test_every_dropdown_and_slider_is_named_by_something(mounted_ids, app):
    """A control announced only by its own value is an unnamed control.

    Dash 4 renders a Dropdown as a button whose `aria-labelledby` points at the
    span holding its *value*, and a Slider as a Radix thumb with no name at
    all - so "OSDR study: OSD-100" reached a screen reader as "OSD-100,
    button", and the top-k slider as "5, slider". Neither can be fixed with a
    `for` attribute, because neither renders a labelable element.

    Both views therefore wrap each control in `role="group"` with
    `aria-labelledby` on the heading that already names it. This asserts every
    one of those groups exists, and that each points at an id that is really on
    the page - a dangling `aria-labelledby` is worse than none, because it
    silently resolves to the empty string.
    """
    from bridge_rna import layout as rna_layout

    trees = [rna_layout.build_view(), layout.build_view()]
    ids: set = set()
    named: dict = {}
    for tree in trees:
        for component in _walk(tree):
            cid = getattr(component, "id", None)
            if isinstance(cid, str):
                ids.add(cid)
            labelled_by = getattr(component, "aria-labelledby", None)
            if labelled_by:
                named[labelled_by] = component

    # Every control that Dash renders without a usable name of its own.
    for control in ("study-dropdown", "sample-dropdown", "cohort-dropdown",
                    "cohort-compare-dropdown", "upload-sample-column",
                    "topk-slider", "color-by", "layers", "method", "dims",
                    "budget"):
        assert control in ids, f"{control} is no longer on any view"

    for label_id in ("study-dropdown-label", "sample-dropdown-label",
                     "cohort-dropdown-label", "cohort-compare-label",
                     "upload-column-label", "topk-slider-label",
                     "cohort-facets-label", "color-by-label", "layers-label",
                     "projection-label", "dims-label", "budget-label"):
        assert label_id in named, f"no group is labelled by {label_id}"
        assert label_id in ids, f"{label_id} is referenced but never rendered"


def test_theme_matches_the_bridge_rna_tokens():
    """The chrome must stay pixel-identical to Bridge RNA; only the plot is dark.

    `theme.py` is a mirror of the stylesheet, not a second source of truth, and
    a mirror that is never checked is just a stale copy: TEXT_MUTED sat at
    #8a99ac for a while after the token was darkened for contrast. Every
    constant this module claims to reuse verbatim is asserted here.
    """
    css = _all_css()
    for token, value in [
        ("--bg-canvas", theme.BG_CANVAS), ("--bg-panel", theme.BG_PANEL),
        ("--bg-panel-raised", theme.BG_PANEL_RAISED), ("--bg-inset", theme.BG_INSET),
        ("--text-primary", theme.TEXT_PRIMARY),
        ("--text-secondary", theme.TEXT_SECONDARY),
        ("--text-muted", theme.TEXT_MUTED),
        ("--accent", theme.ACCENT), ("--accent-text", theme.ACCENT_TEXT),
        ("--accent-hover", theme.ACCENT_HOVER),
        ("--accent-teal", theme.ACCENT_TEAL), ("--accent-warm", theme.ACCENT_WARM),
        ("--header-bg", theme.HEADER_BG), ("--header-fg", theme.HEADER_FG),
        ("--header-line", theme.HEADER_LINE), ("--plot-bg", theme.PLOT_BG),
        ("--status-good", theme.STATUS_GOOD), ("--status-error", theme.STATUS_ERROR),
        ("--status-warn", theme.STATUS_WARN),
    ]:
        assert f"{token}: {value}" in css, f"{token} drifted from {value}"


def test_every_text_token_clears_wcag_aa_on_every_surface():
    """Text contrast is a correctness gate here, not a preference.

    Both text tiers below --text-primary failed AA before 2026-08-10 -
    --text-muted at 2.90:1 carried every rail label, kicker and hint, and
    --accent at 3.76:1 carried the primary button's white label and every blue
    link. This pins the fix against every ground the stylesheet defines, so a
    future palette edit that reintroduces the failure fails here instead of in
    a browser.
    """
    css = _all_css()

    def token(name: str) -> str:
        match = re.search(rf"{re.escape(name)}:\s*(#[0-9a-fA-F]{{6}})\s*;", css)
        assert match, f"{name} is not defined as a hex literal"
        return match.group(1)

    def relative_luminance(hex_color: str) -> float:
        raw = hex_color.lstrip("#")
        channels = [int(raw[i:i + 2], 16) / 255 for i in (0, 2, 4)]
        linear = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
                  for c in channels]
        return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]

    def ratio(a: str, b: str) -> float:
        la, lb = relative_luminance(a), relative_luminance(b)
        return (max(la, lb) + 0.05) / (min(la, lb) + 0.05)

    # Every light surface text can land on, including the tinted ones: a chip
    # in --accent-soft and a banner body in --status-error-soft are exactly
    # where a grey chosen against plain white quietly stops clearing the bar.
    grounds = ["--bg-panel", "--bg-canvas", "--bg-panel-raised", "--bg-inset",
               "--accent-soft", "--accent-teal-soft", "--accent-warm-soft",
               "--status-good-soft", "--status-error-soft", "--status-warn-soft",
               "--status-info-soft"]
    foregrounds = ["--text-primary", "--text-secondary", "--text-muted",
                   "--accent-text"]

    failures = []
    for fg in foregrounds:
        for bg in grounds:
            r = ratio(token(fg), token(bg))
            if r < 4.5:
                failures.append(f"{fg} on {bg}: {r:.2f}:1")
    # White type on the primary button's ground is the other direction.
    for bg in ("--accent-text", "--accent-hover"):
        r = ratio("#ffffff", token(bg))
        if r < 4.5:
            failures.append(f"white on {bg}: {r:.2f}:1")
    assert not failures, "text below WCAG AA 4.5:1 -> " + "; ".join(failures)


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


def test_coverage_states_the_exact_point_count(corpus, without_archs4_metadata):
    """The answer to "why is most of my map not coloured?", given up front.

    Asserted on degraded Tissue, which is the only partial field left after the
    nine OSDR-only ones were removed - and the one a fresh clone actually meets.
    """
    text = _text(callbacks.coverage_children("tissue"))
    assert f"{corpus['n_osdr']:,}" in text
    assert f"{corpus['total']:,}" in text
    assert "context" in text.lower(), (
        "the readout must say what happens to the points it does not colour")


def test_coverage_says_nothing_when_a_field_paints_everything(corpus):
    """The full bar is the whole answer.

    "Colours all 942,563 points." sat under it and only restated the bar in
    words. The readout exists to answer "why is most of my map not colored?",
    which is a question a whole-map field never raises, so it now says nothing
    at all rather than reassuring at length.
    """
    children = callbacks.coverage_children("species")
    assert len(children) == 1, "a whole-map field is the bar and nothing else"
    assert _text(children).strip() == ""


def _coverage_fill_class(key: str) -> str:
    return callbacks.coverage_children(key)[0].children.className


def test_coverage_bar_is_full_for_a_whole_map_field(corpus):
    assert "partial" not in _coverage_fill_class("species")
    assert "partial" not in _coverage_fill_class("tissue")


def test_coverage_bar_is_amber_for_a_partial_field(corpus, without_archs4_metadata):
    assert "partial" in _coverage_fill_class("tissue")
    assert "partial" not in _coverage_fill_class("species"), (
        "species reads the identity table and is whole-map with any cache")


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


def _neighborhood_payload(label="A", offset=0):
    return {
        "available": True, "label": label, "depth_requested": 250,
        "depth_returned": 3, "metric": "cosine", "space": "embedding-512d",
        "hits": [
            {"rank": 1, "gsm": "GSM1", "gse": "GSE1", "tissue": "Eye",
             "species": "mouse", "score": .99, "archs4_index": offset + 1},
            {"rank": 2, "gsm": "GSM2", "gse": "GSE1", "tissue": "Eye",
             "species": "mouse", "score": .98, "archs4_index": offset + 2},
            {"rank": 3, "gsm": "GSM3", "gse": "GSE2", "tissue": "Brain",
             "species": "human", "score": .97, "archs4_index": None},
        ],
    }


def test_neighborhood_overlay_selects_an_arm_and_focus_points():
    payload = {"neighborhood": _neighborhood_payload("A"),
               "comparison": {"neighborhood_b": _neighborhood_payload("B", 10)}}
    overlay = callbacks._neighborhood_overlay(
        payload, "a", {"kind": "study", "value": "GSE1"})
    assert overlay["label"] == "A"
    assert overlay["points"] == [1, 2]
    assert overlay["focus_points"] == [1, 2]
    assert overlay["returned"] == 3 and overlay["locatable"] == 2
    assert callbacks._neighborhood_overlay(payload, "b", None)["points"] == [11, 12]

    sample = callbacks._neighborhood_overlay(
        payload, "a", {"kind": "sample", "value": "GSM2"})
    assert sample["focus_points"] == [2]


def test_neighborhood_overlay_omits_non_integer_map_indices():
    neighborhood = _neighborhood_payload()
    neighborhood["hits"][0]["archs4_index"] = "1"
    neighborhood["hits"][1]["archs4_index"] = 2.0

    overlay = callbacks._neighborhood_overlay(
        {"neighborhood": neighborhood}, "a", None)

    assert overlay["points"] == []
    assert overlay["rows"] == []
    assert overlay["returned"] == 3 and overlay["locatable"] == 0


def test_unavailable_neighborhood_draws_nothing():
    assert callbacks._neighborhood_overlay(
        {"neighborhood": {"available": False, "hits": []}}, "a", None) is None


# --- The key on the plot -----------------------------------------------------
#
# The map draws four encodings at once - corpus tissue hue, member fill hue,
# hit ring shape, and corpus glyph shape - and until this key existed it
# explained exactly one of them. These pin the structure, not the wording.


def _classes(component) -> list[str]:
    """Every className in a Dash component tree, in document order."""
    out: list[str] = []
    if isinstance(component, (list, tuple)):
        for c in component:
            out += _classes(c)
        return out
    cls = getattr(component, "className", None)
    if cls:
        out.append(str(cls))
    children = getattr(component, "children", None)
    if children is not None:
        out += _classes(children)
    return out


def _key_shapes(children) -> list[str]:
    """The glyph shape of each key row, in order."""
    return [c.split("is-", 1)[1] for c in _classes(children)
            if c.startswith("bm-key-glyph is-")]


def test_a_plain_search_gets_a_two_row_key_and_no_headings(corpus):
    """The single-query state must not pay for the comparison. With one query
    on screen there is nothing for a name or a group heading to distinguish it
    from, so the key is the query mark, the hit mark, and nothing else."""
    key = str(data.osdr_metadata()["sample_key"].iloc[0])
    overlay = callbacks._retrieval_overlay(
        {"sample_id": key, "hits": [{"gsm": "GSM1", "score": 0.9,
                                     "archs4_index": 0}]})
    children = layout.retrieval_key_children(overlay, ("a", "b"), "2d")
    assert _key_shapes(children) == ["star", "hit-a"]
    assert "bm-key bm-key--retrieval" in _classes(children)
    assert "bm-key-head" not in _classes(children)
    assert "the query sample" in _text(children)
    assert "pooled member" not in _text(children), (
        "one sample is not a pool; the phrase is earned by k >= 2")


def test_drawn_evidence_neighborhood_adds_its_open_dot_to_the_key(corpus):
    key = str(data.osdr_metadata()["sample_key"].iloc[0])
    retrieval = callbacks._retrieval_overlay(
        {"sample_id": key, "hits": [{"gsm": "GSM1", "score": 0.9,
                                      "archs4_index": 0}]})
    neighborhood = callbacks._neighborhood_overlay(
        {"neighborhood": _neighborhood_payload()}, "a", None)

    children = layout.retrieval_key_children(
        retrieval, ("a",), "2d", neighborhood=neighborhood)

    assert _key_shapes(children) == ["star", "neighborhood", "hit-a"]
    assert "512-D evidence neighbor" in _text(children)
    neighborhood_glyph = next(
        c for child in children for c in _walk(child)
        if getattr(c, "className", "") == "bm-key-glyph is-neighborhood")
    assert theme.NEIGHBORHOOD_COLOR in str(neighborhood_glyph.style)

    without_marks = layout.retrieval_key_children(
        retrieval, ("a",), "2d",
        neighborhood={**neighborhood, "points": [], "locatable": 0})
    assert "neighborhood" not in _key_shapes(without_marks)


def test_evidence_key_counts_only_indices_the_current_map_can_draw(corpus):
    payload = _neighborhood_payload()
    payload["hits"][1]["archs4_index"] = corpus["n_archs4"] + 10
    neighborhood = callbacks._neighborhood_overlay(
        {"neighborhood": payload}, "a", None)

    children = layout.retrieval_key_children(
        None, ("a",), "2d", neighborhood=neighborhood)
    row = next(
        c for child in children for c in _walk(child)
        if getattr(c, "className", "") == "bm-key-row")

    assert neighborhood["locatable"] == 1
    assert _text(row).strip() == "512-D evidence neighbor 1"


def test_all_stale_evidence_indices_draw_no_key(corpus):
    payload = _neighborhood_payload()
    payload["hits"][0]["archs4_index"] = corpus["n_archs4"] + 10
    payload["hits"][1]["archs4_index"] = corpus["n_archs4"] + 11
    neighborhood = callbacks._neighborhood_overlay(
        {"neighborhood": payload}, "a", None)

    children = layout.retrieval_key_children(
        None, ("a",), "2d", neighborhood=neighborhood)

    assert neighborhood["locatable"] == 0
    assert children == []


def test_a_comparison_key_is_grouped_by_role_not_by_cohort(two_cohorts):
    """The encoding has two factors and they do not share a channel: hue names
    the cohort on a member, ring shape names it on a hit. Listing each cohort's
    marks together buries that. Listing the two member rows adjacent and the two
    hit rows adjacent shows it - one pair differs only in hue, the next only in
    shape - so the layout is the explanation and no sentence has to assert it.
    """
    a, b = two_cohorts
    overlay = callbacks._retrieval_overlay(_comparison_payload(a, b))
    children = layout.retrieval_key_children(overlay, ("a", "b"), "2d")

    assert _key_shapes(children) == ["star", "star", "hit-a", "hit-b",
                                     "hit-both"]
    # The rule that separates this key from the colour list below it hangs off
    # this modifier. It was missing here once - the comparison branch returns
    # from a different indent level than the single-query branch, so an edit
    # that added the class to both matched only one - and the key sat flush
    # against "Color · Tissue" with nothing between them.
    assert "bm-key bm-key--retrieval" in _classes(children)
    text = _text(children)
    assert "Pooled members" in text and "Retrieved hits" in text
    # Each cohort is named in both of its rows, so neither channel has to be
    # carried across a heading by memory.
    for label in ("Liver · Space Flight", "Liver · Ground Control"):
        assert text.count(label) == 2, f"{label} is not named in both its rows"


def test_the_doubled_mark_is_rowed_only_when_both_arms_are_drawn(two_cohorts):
    """A hit retrieved by both is one point wearing both cohorts' rings, so it
    cannot exist with an arm hidden. It is also not a third cohort, which is why
    it is the last row of the hits group rather than a group of its own."""
    a, b = two_cohorts
    overlay = callbacks._retrieval_overlay(_comparison_payload(a, b))

    both = layout.retrieval_key_children(overlay, ("a", "b"), "2d")
    assert "retrieved by both" in _text(both)
    assert _key_shapes(both)[-1] == "hit-both"

    one = layout.retrieval_key_children(overlay, ("a",), "2d")
    assert "retrieved by both" not in _text(one)
    assert "hit-both" not in _key_shapes(one)


def test_the_single_query_key_obeys_the_tick_too(corpus):
    """The one tick hides the whole overlay, so the key has to recede with it.

    This branch ignored `roles` at first, which meant unticking "Show it on the
    map" left the plot with no star and no ring while the key went on counting
    one member and five hits - the exact failure the count rule exists to
    prevent, in the commonest state the map has.
    """
    key = str(data.osdr_metadata()["sample_key"].iloc[0])
    overlay = callbacks._retrieval_overlay(
        {"sample_id": key, "hits": [{"gsm": "GSM1", "score": 0.9,
                                     "archs4_index": 0}]})
    shown = layout.retrieval_key_children(overlay, ("a",), "2d")
    assert "1" in _text(shown) and "hidden" not in _text(shown)

    hidden = layout.retrieval_key_children(overlay, (), "2d")
    assert _key_shapes(hidden) == _key_shapes(shown), "the marks stay named"
    assert _classes(hidden).count("bm-key-row is-hidden") == 2
    assert _text(hidden).count("hidden") == 2


def test_a_hidden_arm_keeps_both_its_rows_receded(two_cohorts):
    """Dropping them would leave a gold glyph on screen a moment later with
    nothing to look it up in, and would make the key disagree with the ticks
    that produced it. The hue and the shape it owns stay named; only the counts
    go, because there is nothing on screen to count."""
    a, b = two_cohorts
    overlay = callbacks._retrieval_overlay(_comparison_payload(a, b))
    children = layout.retrieval_key_children(overlay, ("a",), "2d")

    assert _key_shapes(children) == ["star", "star", "hit-a", "hit-b"]
    assert _classes(children).count("bm-key-row is-hidden") == 2
    assert _text(children).count("hidden") == 2


def test_the_key_follows_the_three_dimensional_symbol_substitution(two_cohorts):
    """`Scatter3d` rejects `star` outright, so a member draws as a diamond
    there. A key still showing a star would assert a mark 3-D does not draw,
    which is worse than the silence it replaced. The hit shapes are deliberately
    unchanged - that is why shape, not hue, carries cohort identity."""
    a, b = two_cohorts
    overlay = callbacks._retrieval_overlay(_comparison_payload(a, b))
    assert _key_shapes(layout.retrieval_key_children(overlay, ("a", "b"), "3d")) \
        == ["diamond", "diamond", "hit-a", "hit-b", "hit-both"]


def test_an_uploaded_query_gets_no_member_row(corpus):
    """An uploaded sample was never embedded into this map, so it has no
    coordinate and draws no member mark. A row for it would key a glyph that is
    not there."""
    overlay = callbacks._retrieval_overlay(
        {"sample_id": "UPLOAD|counts.csv::S1",
         "hits": [{"gsm": "GSM1", "score": 0.9, "archs4_index": 0}]})
    children = layout.retrieval_key_children(overlay, ("a", "b"), "2d")
    assert _key_shapes(children) == ["hit-a"]


def test_a_shared_hit_is_one_sample_wherever_it_is_counted(two_cohorts):
    """`hit_points` is a concatenation across the arms, so a hit both cohorts
    retrieved is in it twice. Quoting its length made the map say "10 hits, 2 of
    them retrieved by both" for the same comparison whose banner on the other
    view said "share 2 of 8 retrieved samples" - two surfaces, one search, two
    numbers, and a subset relation that cannot hold either way. Both surfaces
    count distinct samples now."""
    a, b = two_cohorts
    overlay = callbacks._retrieval_overlay(_comparison_payload(a, b))
    per_arm = sum(len(c["hit_points"]) for c in overlay["cohorts"])
    distinct = len(set(overlay["hit_points"]))
    assert overlay["shared_points"], "this fixture is built to share two hits"
    assert distinct == per_arm - len(overlay["shared_points"])
    assert distinct < len(overlay["hit_points"]), "the concatenation double-counts"

    _fig, _legend, badges = render.build_figure(
        "pca", "2d", "species", ["archs4"], 5000, None, retrieval=overlay)
    badge = next(b for b in badges if "cohorts" in b)
    assert f"<b>{distinct}</b> samples" in badge, badge
    assert f"<b>{len(overlay['shared_points'])}</b> retrieved by both" in badge


def test_the_retrieval_group_carries_no_standing_paragraph():
    """The rail's retrieval group is a control, a summary line and a button.

    It used to end with a permanent caveat about the two rank orderings
    disagreeing. That fact is still true and is still shown, in the hover
    `render._map_ranks` builds, which is where the reader meets it at the moment
    it applies; the rail was telling them to go and read a tooltip. The caveat
    is pinned in REMOVED_COPY below - this asserts the shape that replaced it,
    so a different three-line paragraph cannot quietly take its place.
    """
    group = next(c for c in _walk(layout.control_rail())
                 if getattr(c, "id", None) == "retrieval-group")
    hints = [c for c in _walk(group)
             if "bm-hint" in str(getattr(c, "className", ""))]
    assert len(hints) == 1, (
        "the retrieval group has more than the summary line under its control")
    assert hints[0].id == "retrieval-summary"


def test_the_corpus_key_rows_only_the_layers_that_are_drawn(corpus):
    """A key is what you are looking at. That a diamond means OSDR had been
    true since the map was built and was stated nowhere a viewer could read."""
    both = layout.corpus_key_children(["archs4", "osdr"])
    assert _key_shapes(both) == ["corpus-archs4", "corpus-osdr"]
    assert "ARCHS4" in _text(both) and "OSDR" in _text(both)

    assert _key_shapes(layout.corpus_key_children(["osdr"])) == ["corpus-osdr"]
    assert layout.corpus_key_children([]) == []
    assert layout.corpus_key_children(None) == []


def test_the_key_glyph_shapes_all_have_a_stylesheet_rule():
    """A shape with no rule renders as a 14px empty box, which is a key row
    pointing at nothing. `test_every_classname_used_in_python_exists_in_some
    _stylesheet` cannot catch these: the class is built by string interpolation
    from the shape name, so it never appears in a className literal."""
    css = _all_css()
    shapes = ["star", "diamond", "hit-a", "hit-b", "hit-both",
              "found", "corpus-archs4", "corpus-osdr"]
    missing = [s for s in shapes if f".bm-key-glyph.is-{s}" not in css]
    assert not missing, f"key glyph shapes with no CSS rule: {missing}"


#: Copy that was deliberately removed from the interface. Each of these was a
#: sentence the rail or the panel used to carry, and each was cut for a reason
#: recorded beside the code that used to emit it. They are pinned here because
#: prose regresses silently: nothing else in the suite would notice a paragraph
#: coming back, and several of these had already outlived the thing they
#: described.
REMOVED_COPY = (
    "Not a difference vector",
    "Nothing is dropped for you",
    "Adds roughly two seconds per hit",
    "One glyph per sample",
    "Each glyph is a pooled member",
    "no line is drawn",
    "anatomical vocabulary",
    # Removed 2026-08-11. Both said something true; neither said it where the
    # reader needed it. The rank disagreement is in every hit's hover, and the
    # trailing clause narrated a picture already on screen.
    "Hover a hit for its rank",
    "drawn where they sit in the space",
    # The three one-line mode hints under Sample / Cohort / Upload. Each
    # restated its own tab, above a panel that answers the same question with
    # real controls a moment later.
    "One OSDR sample against every ARCHS4 sample",
    "A whole experimental group, pooled into one query",
    "A counts file the corpus has never seen",
    # The rail names the quantity, not the mechanism, and this was its only
    # piece of implementation vocabulary. The component id is still `budget`.
    "ARCHS4 point budget",
)


def test_the_removed_copy_stays_removed():
    """Neither view may say any of these again."""
    source = "\n".join(
        p.read_text()
        for d in ("manifold", "bridge_rna")
        for p in sorted((paths.REPO_ROOT / d).glob("*.py"))) + \
        (paths.REPO_ROOT / "app.py").read_text()
    back = [s for s in REMOVED_COPY if s in source]
    assert not back, f"copy that was deliberately removed is back: {back}"


def test_the_app_spells_color_the_american_way():
    """One application, one spelling.

    The map's own identifiers were already American - `colorby`, `color_by`,
    `color_for_index`, `#color-by` - while its prose and two of its user-facing
    strings were not, and `render._colour_plan` sat two lines from
    `theme.color_for_index`. The retrieval view had one user-visible instance
    left, in the comparison legend strip, which would have put two spellings on
    screen for the same two-cohort search.
    """
    offenders = []
    for path in (sorted((paths.REPO_ROOT / "manifold").glob("*.py"))
                 + sorted((paths.REPO_ROOT / "bridge_rna").glob("*.py"))
                 + sorted(paths.ASSETS_DIR.glob("*.css"))
                 + [paths.REPO_ROOT / "app.py"]):
        for i, line in enumerate(path.read_text().splitlines(), 1):
            if "colour" in line.lower():
                offenders.append(f"{path.name}:{i}")
    assert not offenders, f"British spelling in the app: {offenders}"


def _inline_backgrounds(component) -> set[str]:
    """Every `style={"background": ...}` value in a Dash component tree."""
    out: set[str] = set()
    if isinstance(component, (list, tuple)):
        for c in component:
            out |= _inline_backgrounds(c)
        return out
    style = getattr(component, "style", None)
    if isinstance(style, dict) and style.get("background"):
        out.add(style["background"])
    children = getattr(component, "children", None)
    if children is not None:
        out |= _inline_backgrounds(children)
    return out


def test_the_map_key_reads_its_hues_from_the_theme(two_cohorts):
    """There is no second copy of a cohort hue left to drift.

    The rail's old two-cohort key mirrored RETRIEVAL_QUERY and
    RETRIEVAL_QUERY_B into map.css by hand, because Plotly cannot read a CSS
    variable - so the swatch and the glyph were two spellings of one hex, and a
    drifted swatch would have labelled the wrong cohort. The key on the plot
    sets them inline from `theme` instead. The stylesheet keeps the shapes,
    which have no Python constant to disagree with.
    """
    a, b = two_cohorts
    overlay = callbacks._retrieval_overlay(_comparison_payload(a, b))
    fills = _inline_backgrounds(
        layout.retrieval_key_children(overlay, ("a", "b"), "2d"))
    assert theme.RETRIEVAL_QUERY in fills
    assert theme.RETRIEVAL_QUERY_B in fills

    map_css = (paths.ASSETS_DIR / "map.css").read_text()
    for value in (theme.RETRIEVAL_QUERY, theme.RETRIEVAL_QUERY_B):
        assert value not in map_css, (
            f"{value} is mirrored into map.css again; the key reads theme")


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
