"""Both independently ranked evidence sets must survive the map handoff."""

import numpy as np
import pandas as pd
import pytest

from bridge_rna import figures
from manifold import callbacks, layout, render, theme


def payload():
    def arm(label, start):
        return {"available": True, "label": label, "depth_returned": 250,
                "hits": [{"archs4_index": i, "gsm": f"GSM{i}",
                          "rank": rank, "score": 1 - rank / 1000}
                         for rank, i in enumerate(range(start, start + 250), 1)]}
    return {"neighborhood": arm("Flight", 0),
            "comparison": {"neighborhood_b": arm("Ground", 125)}}


@pytest.mark.parametrize("dims", ["2d", "3d"])
def test_both_top_250_sets_keep_coordinates_ranks_and_shared_hover(corpus, dims):
    evidence = callbacks._neighborhood_overlays(payload(), "a", None, ("a", "b"))
    fig, _, _ = render.build_figure(
        "pca", dims, "species", [], 1000, None, neighborhood=evidence)
    traces = [t for t in fig.data if t.name == "512-D evidence neighbor"]
    assert len(traces) == 2
    assert [len(t.x) for t in traces] == [250, 250]
    assert [t.marker.symbol for t in traces] == ["circle-open", "square-open"]
    assert traces[1].marker.size > traces[0].marker.size
    assert all(t.marker.color == theme.NEIGHBORHOOD_COLOR for t in traces)
    coords = render.data.coords("pca", dims)
    for trace, start in zip(traces, (0, 125)):
        np.testing.assert_array_equal(trace.x, coords[start:start + 250, 0])
        assert [r[1] for r in trace.customdata] == list(range(1, 251))
    for row in (traces[0].customdata[125], traces[1].customdata[0]):
        assert "Flight" in row[8] and "Ground" in row[8]
        assert "126" in row[8] and "0.8740" in row[8]
        assert "0.9990" in row[8]
    key = str(layout.retrieval_key_children(None, ("a", "b"), dims, evidence))
    assert "is-neighborhood-b" in key and "Flight" in key and "Ground" in key


def test_arm_selection_changes_focus_without_removing_other_neighborhood(corpus):
    evidence = callbacks._neighborhood_overlays(
        payload(), "b", {"kind": "sample", "value": "GSM130"}, ("a", "b"))
    assert [c["focus_points"] for c in evidence["cohorts"]] == [[], [130]]
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

