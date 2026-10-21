"""The map's cuts of hits-store must draw exactly what the full payload drew.

`manifold.callbacks.derive_map_stores` reads hits-store once per visit and
writes three smaller stores, so that nothing on the map that fires per
interaction posts the whole payload (472 KB per sample at k=1,000; see
docs/map-overlay-stores.md). That is only a performance change if every
consumer returns the same value from its cut as it did from the payload, and
this file is where that is proved: for a single sample, an upload, a cohort, a
comparison and a mode with no evidence, for every tick combination, with the
drawer open and closed and focused, in 2-D and 3-D.

Every store travels through the browser as JSON, so every comparison here is
made after a JSON round trip of both sides, which is what the callbacks
actually receive.
"""

from __future__ import annotations

import itertools
import json

import dash
import pandas as pd
import plotly.utils
import pytest
from dash import html

from bridge_rna import neighborhoods
from manifold import callbacks, data, layout, render

ROLES = ("a", "b")
ROLE_SETS = [(), ("a",), ("b",), ("a", "b")]
HIT_FIELDS = ("gsm", "score", "gse", "title", "source_name", "characteristics",
              "tissue", "species", "geo_summary", "geo_design", "pubmed_ids",
              "archs4_index")


def _rt(value):
    """What a callback receives after the value has been through the browser."""
    return json.loads(json.dumps(value))


def _js(value) -> str:
    """Components, figures and plain values, compared as the JSON Dash sends."""
    return json.dumps(value, cls=plotly.utils.PlotlyJSONEncoder, sort_keys=True)


def _hits(indices, start_score: float, series: str) -> list[dict]:
    """Hits with every field the real scan writes, long text included."""
    rows = []
    for rank, idx in enumerate(indices):
        rows.append({
            "gsm": f"GSM{100000 + idx}" if idx is not None else "",
            # float32 values, like the real cosine scan's.
            "score": float(pd.Series([start_score - rank * 1e-4],
                                     dtype="float32").iloc[0]),
            # Every seventh series is blank, which the drawer files under its
            # no-GSE study and the map focuses through NO_GSE_FOCUS_VALUE.
            "gse": "" if rank % 7 == 3 else f"{series}{rank % 5}",
            "title": f"Mouse liver replicate {rank} " * 3,
            "source_name": "liver tissue",
            "characteristics": "strain: C57BL/6J; age: 12 weeks; " * 6,
            "tissue": ["Liver", "Kidney", ""][rank % 3],
            "species": ["mouse", "human"][rank % 2],
            "geo_summary": "A long GEO summary. " * 20,
            "geo_design": "A long GEO design. " * 10,
            "pubmed_ids": "12345678",
            "archs4_index": idx,
        })
    return rows


def _neighborhood(hits: list[dict], label: str) -> dict:
    """An evidence arm built by the real builder, from the real hit shape."""
    return neighborhoods.build_payload(pd.DataFrame(hits), label=label)


@pytest.fixture(scope="module")
def osdr_keys(corpus):
    return data.osdr_metadata()["sample_key"].astype(str).tolist()


def _single(osdr_keys, k=60, mode="cached"):
    n_archs4 = data.counts()[0]
    # One hit that predates `archs4_index`, one off the end of the corpus.
    indices = list(range(0, 3 * k, 3))[:k - 2] + [None, n_archs4 + 5]
    hits = _hits(indices, 0.999, "GSE1")
    return {
        "sample_id": osdr_keys[0],
        "entrez_email": "someone@example.org",
        "biopython_enabled": False,
        "mode": mode,
        "hits": hits,
        "score_profile": [h["score"] for h in hits],
        "query": {"sample_id": osdr_keys[0],
                  "sample_name": osdr_keys[0].split("|")[-1],
                  "tissue": "Liver", "accession": osdr_keys[0].split("|")[0]},
        "neighborhood": _neighborhood(hits, osdr_keys[0].split("|")[-1]),
    }


def _uploaded(osdr_keys):
    payload = _single(osdr_keys, mode="uploaded")
    # An uploaded sample has no sample_key, so it is not on the map.
    payload["sample_id"] = "upload:counts.csv|Sample_1"
    payload["query"] = {"sample_id": payload["sample_id"],
                        "sample_name": "Sample_1", "is_upload": "1"}
    return payload


def _demo(osdr_keys):
    payload = _single(osdr_keys, mode="demo")
    payload["neighborhood"] = neighborhoods.unavailable_payload(
        neighborhoods.evidence_unavailable_reason("demo"))
    return payload


def _cohort(osdr_keys, k=60):
    payload = _single(osdr_keys, k=k)
    payload.update({
        "sample_id": "COHORT|study=OSD-1|tissue=Liver|spaceflight=Space Flight",
        "mode": "cohort",
        "query": {"cohort_label": "Liver · Space Flight", "is_cohort": "1",
                  "members": "\n".join(osdr_keys[:4])},
        "member_ids": osdr_keys[:4],
    })
    payload["neighborhood"]["label"] = "Liver · Space Flight"
    return payload


def _comparison(osdr_keys, k=60):
    payload = _cohort(osdr_keys, k=k)
    # Half of B's hits are A's, so shared hits exist and draw twice.
    indices_b = list(range(0, 3 * k, 6))[:k // 2] + list(range(1, 3 * k, 6))[:k - k // 2]
    hits_b = _hits(indices_b, 0.998, "GSE2")
    shared = sorted({h["gsm"] for h in payload["hits"]} & {h["gsm"] for h in hits_b})
    payload["comparison"] = {
        "query_b": {"cohort_label": "Liver · Ground Control", "is_cohort": "1"},
        "hits_b": hits_b,
        "score_profile_b": [h["score"] for h in hits_b],
        "shared": shared,
        "overlap": 0.4,
        "facet": "spaceflight arm",
        "member_ids": osdr_keys[4:7],
        "neighborhood_b": _neighborhood(hits_b, "Liver · Ground Control"),
    }
    return payload


KINDS = ["single", "uploaded", "demo", "cohort", "comparison"]


@pytest.fixture(scope="module")
def payloads(osdr_keys):
    built = {"single": _single(osdr_keys), "uploaded": _uploaded(osdr_keys),
             "demo": _demo(osdr_keys), "cohort": _cohort(osdr_keys),
             "comparison": _comparison(osdr_keys)}
    return {kind: _rt(p) for kind, p in built.items()}


def _cuts(payload):
    overlay, evidence, summary = callbacks.map_stores(payload)
    return _rt(overlay), _rt(evidence), _rt(summary)


def _focuses(payload):
    """No focus, a sample, a study, and the no-GSE study, each from the payload."""
    rows = (payload.get("neighborhood") or {}).get("hits") or []
    sample = next((r["gsm"] for r in rows if r.get("gsm")), "GSM0")
    study = next((r["gse"] for r in rows if r.get("gse")), "GSE0")
    return [None, {"kind": "sample", "value": sample},
            {"kind": "study", "value": study},
            {"kind": "study", "value": layout.NO_GSE_FOCUS_VALUE}]


# --- The overlay the figure draws --------------------------------------------

@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("roles", ROLE_SETS + [None])
def test_the_retrieval_overlay_is_identical_from_the_cut(payloads, kind, roles):
    payload = payloads[kind]
    overlay, _evidence, _summary = _cuts(payload)
    args = () if roles is None else (roles,)
    assert (callbacks._retrieval_overlay(overlay, *args)
            == callbacks._retrieval_overlay(payload, *args))


@pytest.mark.parametrize("kind", KINDS)
def test_the_evidence_overlay_draws_the_same_rows_from_the_cut(payloads, kind):
    """`rows` keeps only the fields the figure reads; everything else is equal."""
    payload = payloads[kind]
    overlay, _evidence, _summary = _cuts(payload)
    for arm, focus, roles in itertools.product(ROLES, _focuses(payload), ROLE_SETS):
        cut = callbacks._neighborhood_overlays(overlay, arm, focus, roles)
        full = callbacks._neighborhood_overlays(payload, arm, focus, roles)
        if full is None:
            assert cut is None
            continue
        for cohort in full["cohorts"]:
            cohort["rows"] = [{f: row[f] for f in callbacks.MAP_NEIGHBOR_FIELDS
                               if f in row} for row in cohort["rows"]]
        assert cut == full, (arm, focus, roles)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("dims", ["2d", "3d"])
def test_the_figure_and_its_key_are_byte_identical_from_the_cut(payloads, kind, dims):
    """The end-to-end form of the two tests above, and the one that fails first
    if the map starts reading a field the cut leaves behind."""
    payload = payloads[kind]
    overlay, _evidence, _summary = _cuts(payload)
    focus = _focuses(payload)[1]
    for roles, drawer_open in itertools.product(ROLE_SETS, (False, True)):
        built = []
        for source in (payload, overlay):
            retrieval = (callbacks._retrieval_overlay(source, roles)
                         if roles else None)
            evidence = (callbacks._neighborhood_overlays(source, "a", focus, roles)
                        if drawer_open else None)
            fig, legend, badges = render.build_figure(
                "pca", dims, "species", ["archs4", "osdr"], 1000, None,
                retrieval=retrieval, neighborhood=evidence)
            key = layout.retrieval_key_children(
                callbacks._retrieval_overlay(source), roles, dims,
                neighborhood=evidence)
            built.append((fig.to_json(), _js(legend), badges, _js(key)))
        assert built[0] == built[1], (roles, drawer_open)


# --- The drawer ---------------------------------------------------------------

@pytest.mark.parametrize("kind", KINDS)
def test_the_drawer_is_identical_from_the_evidence_cut(payloads, kind):
    payload = payloads[kind]
    _overlay, evidence, _summary = _cuts(payload)
    for arm, tab, search, focus in itertools.product(
            ROLES, ("overview", "studies", "samples", "nonsense"),
            ("", "liver", "GSE1"), _focuses(payload)):
        assert (_js(callbacks.neighborhood_drawer_state(evidence, arm, tab, search, focus))
                == _js(callbacks.neighborhood_drawer_state(payload, arm, tab, search, focus))), \
            (arm, tab, search, focus)


# --- The rail and the viewport ------------------------------------------------

@pytest.mark.parametrize("kind", KINDS)
def test_fit_results_frames_the_same_window_from_the_summary(payloads, kind):
    payload = payloads[kind]
    _overlay, _evidence, summary = _cuts(payload)
    for method, dims, roles in itertools.product(
            data.METHODS, ("2d", "3d"), callbacks.FRAME_ROLE_SETS):
        assert (callbacks.summary_frame(summary, method, dims, roles)
                == callbacks._frame_for(payload, method, dims, roles)), \
            (method, dims, roles)


def test_fit_results_actually_frames_something(payloads):
    """The equality above would hold trivially if every window were None."""
    _overlay, _evidence, summary = _cuts(payloads["comparison"])
    windows = [callbacks.summary_frame(summary, "pca", "2d", roles)
               for roles in callbacks.FRAME_ROLE_SETS]
    assert all(w and len(w) == 4 for w in windows)
    assert all(x0 < x1 and y0 < y1 for x0, x1, y0, y1 in windows)


def test_there_is_nothing_to_summarize_without_a_retrieval():
    assert callbacks.map_stores(None) == (None, None, None)
    assert callbacks.retrieval_summary({}) is None
    assert callbacks.summary_frame(None, "pca", "2d", ("a",)) is None


def _old_summary_children(hits_payload, show_retrieval):
    """`describe_the_retrieval` as it read the full payload, before the cut."""
    full = callbacks._retrieval_overlay(hits_payload)
    if full is None:
        return ""
    roles = callbacks._roles_from_checklist(show_retrieval)
    shown = [c for c in full["cohorts"] if c["role"] in roles]
    if not shown:
        return ("Not drawn. Tick it above to put it back on the map."
                if len(full["cohorts"]) < 2 else
                "Neither arm is drawn. Tick one to put it back on the map.")
    if len(full["cohorts"]) < 2:
        n = len(set(full["hit_points"]))
        query = full["query_label"] or "an OSDR sample"
        return [html.B(query.split("|")[-1]),
                f" and its {n} nearest ARCHS4 neighbor{'s' if n != 1 else ''}."]
    if len(shown) < 2:
        return [html.B(shown[0]["label"] or "One arm"),
                " only. Tick both to see which samples they share."]
    return ["Both arms drawn, ", html.B(f"{len(set(full['hit_points']))}"),
            " samples retrieved, ", html.B(f"{len(full['shared_points'])}"),
            " of them by both."]


def _old_group_outputs(hits_payload, dims, freeze):
    """`show_retrieval_group` as it read the full payload, before the cut."""
    single = [{"label": " Show it on the map", "value": "on"}]

    def ticks(options, value):
        return ((dash.no_update, dash.no_update) if freeze else (options, value))

    overlay = callbacks._retrieval_overlay(hits_payload)
    if overlay is None:
        return ({"display": "none"}, "Fit results", *ticks(single, ["on"]))
    action_label = "Fit results" if dims == "2d" else "View results"
    if len(overlay["cohorts"]) < 2:
        return {}, action_label, *ticks(single, ["on"])
    options = [{"label": f"  {c['label'] or 'cohort'}", "value": key}
               for c, key in zip(overlay["cohorts"], ("on", "b"))]
    return {}, action_label, *ticks(options, ["on", "b"])


@pytest.mark.parametrize("kind", KINDS + ["none"])
def test_the_rail_says_the_same_thing_from_the_summary(payloads, kind):
    payload = payloads.get(kind)
    _overlay, _evidence, summary = _cuts(payload)
    for ticks in ([], ["on"], ["b"], ["on", "b"], None):
        assert (_js(callbacks.retrieval_summary_children(summary, ticks))
                == _js(_old_summary_children(payload, ticks))), ticks
    for dims, freeze in itertools.product(("2d", "3d"), (False, True)):
        assert (callbacks.retrieval_group_outputs(summary, dims, freeze)
                == _old_group_outputs(payload, dims, freeze)), (dims, freeze)


def test_a_comparison_summary_counts_distinct_and_shared_hits(payloads):
    """Guards the two numbers against being restated from the concatenation."""
    payload = payloads["comparison"]
    _overlay, _evidence, summary = _cuts(payload)
    overlay = callbacks._retrieval_overlay(payload)
    a, b = (set(c["hit_points"]) for c in overlay["cohorts"])
    assert summary["hit_count"] == len(a | b) < len(overlay["hit_points"])
    assert summary["shared_count"] == len(a & b) > 0


# --- What the cuts weigh ------------------------------------------------------

def test_the_cuts_carry_only_what_the_map_reads(payloads):
    overlay, evidence, _summary = _cuts(payloads["comparison"])
    for hits in (overlay["hits"], overlay["comparison"]["hits_b"]):
        assert {k for h in hits for k in h} == set(callbacks.MAP_HIT_FIELDS)
    for arm in (overlay["neighborhood"], overlay["comparison"]["neighborhood_b"]):
        assert {k for h in arm["hits"] for k in h} <= set(callbacks.MAP_NEIGHBOR_FIELDS)
    assert "hits" not in evidence
    assert "hits_b" not in evidence["comparison"]


def test_at_k_1000_the_rail_store_does_not_grow_with_k(osdr_keys):
    """The store `update_viewport` carries on every zoom stays a few hundred
    bytes however many hits there are, and the figure's cut is a small share of
    the payload."""
    small = _rt(_comparison(osdr_keys, k=60))
    large = _rt(_comparison(osdr_keys, k=1000))
    summaries = [json.dumps(callbacks.retrieval_summary(p)) for p in (small, large)]
    assert len(summaries[1]) < 2_000
    assert abs(len(summaries[1]) - len(summaries[0])) < 32
    overlay = json.dumps(callbacks.overlay_payload(large))
    assert len(overlay) < 0.2 * len(json.dumps(large))


# --- The wiring ---------------------------------------------------------------

@pytest.fixture(scope="module")
def map_callbacks():
    """The map's own callbacks, registered alone on a bare app."""
    app = dash.Dash(__name__)
    callbacks.register(app)
    return app.callback_map


def _deps(entry, kind):
    return {(str(d["id"]), d["property"]) for d in entry.get(kind) or []}


def test_only_the_derivation_reads_the_whole_payload(map_callbacks):
    """The property, not the symptom: no map callback that fires per
    interaction may carry hits-store, as Input or as State."""
    readers = [key for key, entry in map_callbacks.items()
               if ("hits-store", "data") in _deps(entry, "inputs") | _deps(entry, "state")]
    assert len(readers) == 1, readers
    assert "map-overlay-store.data" in readers[0]
    assert "map-evidence-store.data" in readers[0]
    assert "map-retrieval-store.data" in readers[0]


def test_the_derivation_reruns_when_the_map_mounts(map_callbacks):
    """hits-store lives on the shell and never remounts, so without a map input
    a visit after a client-side route change would draw no retrieval."""
    entry = next(e for k, e in map_callbacks.items() if "map-overlay-store.data" in k)
    assert ("map-workspace", "id") in _deps(entry, "inputs")
    assert not entry.get("prevent_initial_call")


@pytest.mark.parametrize("output,store", [
    ("manifold-graph.figure", "map-overlay-store"),
    ("neighborhood-body.children", "map-evidence-store"),
    ("retrieval-summary.children", "map-retrieval-store"),
    ("retrieval-group.style", "map-retrieval-store"),
])
def test_each_consumer_reads_its_own_cut(map_callbacks, output, store):
    entry = next(e for k, e in map_callbacks.items() if output in k)
    assert (store, "data") in _deps(entry, "inputs")


def test_the_viewport_carries_only_the_summary(map_callbacks):
    entry = next(e for k, e in map_callbacks.items() if "viewport-store.data" in k)
    assert ("map-retrieval-store", "data") in _deps(entry, "state")
    assert not {d for d in _deps(entry, "inputs") | _deps(entry, "state")
                if d[0] in ("map-overlay-store", "map-evidence-store")}


def test_the_stores_are_in_the_map_view():
    ids = set()
    stack = [layout.build_view()]
    while stack:
        node = stack.pop()
        ids.add(getattr(node, "id", None))
        children = getattr(node, "children", None)
        if isinstance(children, (list, tuple)):
            stack.extend(children)
        elif children is not None and hasattr(children, "children"):
            stack.append(children)
    assert {"map-overlay-store", "map-evidence-store", "map-retrieval-store",
            "map-workspace"} <= ids
