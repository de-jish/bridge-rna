"""Both independently ranked match sets must survive the map handoff.

Since 2026-10-22 the neighborhood explorer is the retrieved matches: each arm's
stored `neighborhood` is a marker, its rows are that arm's hits, and the map
draws one teal match mark per hit rather than a separate evidence layer.
"""

import numpy as np
import pandas as pd
import pytest

from bridge_rna import figures
from manifold import callbacks, layout, render, theme

K = 40
B_START = 20  # B's matches overlap A's last 20, so 20 matches are shared.


def payload():
    def hits(start):
        return [{"archs4_index": i, "gsm": f"GSM{i}", "gse": f"GSE{i % 3}",
                 "score": 1 - rank / 1000}
                for rank, i in enumerate(range(start, start + K), 1)]

    def marker(label):
        return {"available": True, "reason": "", "label": label,
                "from_hits": True}
    return {"mode": "cohort",
            "hits": hits(0), "neighborhood": marker("Flight"),
            "query": {"cohort_label": "Flight", "is_cohort": "1"},
            "comparison": {"hits_b": hits(B_START),
                           "neighborhood_b": marker("Ground"),
                           "query_b": {"cohort_label": "Ground",
                                       "is_cohort": "1"}}}


@pytest.mark.parametrize("dims", ["2d", "3d"])
def test_both_match_sets_keep_coordinates_ranks_and_shared_hover(corpus, dims):
    roles = ("a", "b")
    retrieval = callbacks._retrieval_overlay(payload(), roles)
    evidence = callbacks._neighborhood_overlays(payload(), "a", None, roles)
    fig, _, _ = render.build_figure(
        "pca", dims, "species", [], 1000, None,
        retrieval=retrieval, neighborhood=evidence)

    assert not any(t.name == "512-D evidence neighbor" for t in fig.data)
    traces = [t for t in fig.data if t.name == "retrieved match"]
    assert len(traces) == 2, "exactly one match trace per cohort"
    assert [len(t.x) for t in traces] == [K, K]
    assert [t.marker.symbol for t in traces] == [theme.MATCH_SYMBOL,
                                                 theme.MATCH_SYMBOL_B]
    assert traces[1].marker.size > traces[0].marker.size
    assert all(t.marker.color == theme.MATCH_COLOR for t in traces)
    coords = render.data.coords("pca", dims)
    for trace, start in zip(traces, (0, B_START)):
        np.testing.assert_array_equal(trace.x, coords[start:start + K, 0])
        assert [r[1] for r in trace.customdata] == list(range(1, K + 1))

    # The explorer lists exactly the matches drawn, in rank order.
    for cohort, trace in zip(evidence["cohorts"], traces):
        assert cohort["returned"] == cohort["locatable"] == K
        assert [row["gsm"] for row in cohort["rows"]] == \
            [r[2] for r in trace.customdata]

    # GSM20 is A's 21st match and B's first: either trace's hover names both.
    for row in (traces[0].customdata[B_START], traces[1].customdata[0]):
        assert row[2] == f"GSM{B_START}"
        assert "Flight" in row[8] and "Ground" in row[8]
        assert f"rank {B_START + 1} of {K}" in row[8] and "0.9790" in row[8]
        assert "0.9990" in row[8]

    key = str(layout.retrieval_key_children(retrieval, roles, dims))
    assert "is-match-b" in key and "Flight" in key and "Ground" in key
    assert "is-neighborhood" not in key


def test_arm_selection_changes_focus_without_removing_other_matches(corpus):
    focused = f"GSM{B_START + 10}"
    evidence = callbacks._neighborhood_overlays(
        payload(), "b", {"kind": "sample", "value": focused}, ("a", "b"))
    assert [c["focus_points"] for c in evidence["cohorts"]] == \
        [[], [B_START + 10]]
    hidden = callbacks._neighborhood_overlays(payload(), "a", None, ("b",))
    assert [c["role"] for c in hidden["cohorts"]] == ["b"]
    assert callbacks._neighborhood_overlays(payload(), "a", None, ()) is None


def test_clicking_b_evidence_opens_its_own_sample_details():
    click = {"points": [{"customdata": [
        "neighborhood", 1, "GSM300", "GSE1", .99, "", "", "b", ""]}]}
    focus, tab, arm, _ = callbacks.next_neighborhood_interaction("manifold-graph", click)
    assert (focus, tab, arm) == ({"kind": "sample", "value": "GSM300"}, "samples", "b")


def test_every_network_edge_has_one_width_independent_of_cosine():
    query = pd.Series({"sample_id": "A", "sample_name": "A"})
    hits = pd.DataFrame({"gsm": ["GSM1", "GSM2", "GSM3"],
                         "gse": ["GSE1"] * 3, "score": [.999, .9, .5]})
    single = figures.build_network_figure(query, hits)
    comparison = figures.build_comparison_figure(query, hits, query, hits.iloc[::-1])
    widths = {t.line.width for fig in (single, comparison)
              for t in fig.data if t.mode == "lines"}
    assert len(widths) == 1

