"""The results panel's figures: score by rank and series.

Every assertion is on the figure JSON, which is what Dash ships to the browser,
so a test here fails on the same thing a reader would see go wrong.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from bridge_rna import result_plots as RP
from bridge_rna.figures import GRAPH_THEME

REFERENCE = 940_455


# ------------------------------------------------------------------- fixtures

def _profile(n: int = 1000, top: float = 0.9970, bottom: float = 0.9634) -> list[float]:
    """A monotone profile shaped like the measured one: flat at the top."""
    r = np.arange(1, n + 1)
    return list(top - (top - bottom) * np.log10(r) / np.log10(n))


def _row(i: int, rank_a=None, rank_b=None, gse: str | None = None,
         profile_a=None, profile_b=None) -> dict:
    pa, pb = profile_a or _profile(), profile_b or _profile(top=0.9950, bottom=0.9610)
    return {
        "gsm": f"GSM{1000 + i}", "gse": f"GSE{i % 4}" if gse is None else gse,
        "title": f"Sample {i} <title> & more", "source_name": "liver",
        "tissue": "Liver", "species": "mouse", "archs4_index": 50 + i,
        "rank_a": rank_a, "score_a": pa[rank_a - 1] if rank_a else None,
        "rank_b": rank_b, "score_b": pb[rank_b - 1] if rank_b else None,
        "identical": [],
    }


def _single(k: int = 30, profile=None, mode: str = "cached", query: str = "OSD-100|s1") -> dict:
    profile = _profile() if profile is None else profile
    return {"mode": mode, "k": k, "comparison": False, "label_a": "OSD-100 eye",
            "label_b": "", "query_id": query, "reference_count": REFERENCE,
            "profile_a": profile, "profile_b": [],
            "rows": [_row(i, rank_a=i + 1, profile_a=profile or None) for i in range(k)]}


def _comparison(k: int = 10) -> dict:
    """A in ranks 1..k for rows 0..k-1, B in ranks 1..k for rows 4..k+3."""
    rows = []
    for i in range(k + 4):
        ra = i + 1 if i < k else None
        rb = i - 3 if 4 <= i < k + 4 else None
        rows.append(_row(i, rank_a=ra, rank_b=rb))
    return {"mode": "cohort", "k": k, "comparison": True,
            "label_a": "OSD-48 liver flight", "label_b": "OSD-48 liver ground",
            "query_id": "OSD-48|liver", "reference_count": REFERENCE,
            "profile_a": _profile(), "profile_b": _profile(top=0.9950, bottom=0.9610),
            "rows": rows}


def _json(fig) -> dict:
    return json.loads(fig.to_json())


def _traces(fig, role: str) -> list[dict]:
    return [t for t in _json(fig)["data"] if (t.get("meta") or {}).get("role") == role]


def _is_empty_state(fig) -> bool:
    data = _json(fig)
    return not data["data"] and len(data["layout"].get("annotations", [])) == 1


def _all_figures():
    single = _single()
    comp = _comparison()
    return [
        RP.score_rank_figure(single, single["rows"][:20], selected_gsm="GSM1003"),
        RP.score_rank_figure(comp, comp["rows"][2:], selected_gsm="GSM1005"),
        RP.score_rank_figure(None, None),
        RP.series_figure(single, single["rows"], selected_gsm="GSM1001"),
        RP.series_figure(comp, comp["rows"], selected_gse="GSE2"),
        RP.series_figure(single, []),
    ]


# --------------------------------------------------------------- constants

def test_the_view_caps_are_the_documented_ones():
    assert RP.SERIES_MAX_BARS == 20
    assert RP.NETWORK_MAX_ROWS == 30


# ------------------------------------------------------------- score by rank

def test_score_figure_draws_the_profile_on_a_log_rank_axis():
    results = _single()
    fig = RP.score_rank_figure(results, results["rows"])
    data = _json(fig)
    xaxis = data["layout"]["xaxis"]
    assert xaxis["type"] == "log"
    lo, hi = xaxis["range"]
    assert 10 ** lo < 1 and 10 ** hi > 1000, "the axis must show rank 1 to rank 1,000"
    assert "940,455" in xaxis["title"]["text"]
    assert data["layout"]["yaxis"]["title"]["text"] == "Cosine similarity"

    (line,) = _traces(fig, "profile")
    assert line["mode"] == "lines"
    assert line["x"][0] == 1 and line["x"][-1] == 1000 and len(line["y"]) == 1000
    assert line["hoverinfo"] == "skip"
    assert "customdata" not in line


def test_score_axis_is_fitted_to_flat_scores_not_anchored_at_zero():
    results = _single()
    fig = RP.score_rank_figure(results, results["rows"])
    lo, hi = _json(fig)["layout"]["yaxis"]["range"]
    assert 0.95 < lo < min(results["profile_a"])
    assert max(results["profile_a"]) < hi < 1.0
    # The span of the axis is a few hundredths, not the 0..1 of a cosine.
    assert hi - lo < 0.05


def test_rows_in_the_table_are_emphasized_and_filtered_rows_muted():
    results = _single(k=30)
    hidden = {"GSM1003", "GSM1007", "GSM1020"}
    visible = [r for r in results["rows"] if r["gsm"] not in hidden]
    fig = RP.score_rank_figure(results, visible)
    (shown,) = _traces(fig, "in_table")
    (muted,) = _traces(fig, "filtered_out")
    assert len(shown["x"]) == len(visible) == 27
    assert len(muted["x"]) == 3
    assert {c[1] for c in muted["customdata"]} == hidden
    assert shown["marker"]["color"] == GRAPH_THEME["gsm"]
    assert muted["marker"]["color"] != shown["marker"]["color"]
    # Ranks are the true ranks; filtering never renumbers.
    assert sorted(muted["x"]) == [4, 8, 21]
    assert all("Filtered out" in h for h in muted["hovertext"])
    # Two marker states need a key; the profile line does not.
    assert _json(fig)["layout"]["showlegend"] is True


def test_every_marker_carries_gsm_customdata_and_a_six_decimal_hover():
    results = _single(k=12)
    fig = RP.score_rank_figure(results, results["rows"][:9])
    for role in ("in_table", "filtered_out"):
        for t in _traces(fig, role):
            assert len(t["customdata"]) == len(t["x"])
            for (kind, gsm), rank in zip(t["customdata"], t["x"]):
                assert kind == "gsm"
                row = results["rows"][rank - 1]
                assert gsm == row["gsm"]
            for h, rank in zip(t["hovertext"], t["x"]):
                row = results["rows"][rank - 1]
                assert f"{row['score_a']:.6f}" in h
                assert f"Rank {rank:,}" in h
                assert row["gse"] in h
                # Source metadata is escaped, never interpreted as markup.
                assert "&lt;title&gt;" in h and "<title>" not in h


def test_a_single_unfiltered_query_needs_no_legend():
    results = _single()
    fig = RP.score_rank_figure(results, results["rows"])
    assert not _traces(fig, "filtered_out")
    assert _json(fig)["layout"]["showlegend"] is False


def test_the_k_rule_is_drawn_at_k_and_labelled_with_k():
    results = _single(k=30)
    layout = _json(RP.score_rank_figure(results, results["rows"]))["layout"]
    (rule,) = layout["shapes"]
    # A shape on a log axis is placed by data value in Plotly 6 (verified in a
    # rendered figure); the annotation by log10 of it.
    assert rule["x0"] == rule["x1"] == 30
    (label,) = layout["annotations"]
    assert label["text"] == "Top 30"
    assert label["x"] == pytest.approx(math.log10(30))


def test_the_selected_row_is_ringed():
    results = _single(k=30)
    fig = RP.score_rank_figure(results, results["rows"], selected_gsm="GSM1004")
    (ring,) = _traces(fig, "selection")
    assert ring["x"] == [5]
    assert ring["marker"]["symbol"] == "circle-open"
    assert ring["hoverinfo"] == "skip" and "customdata" not in ring
    assert not _traces(RP.score_rank_figure(results, results["rows"]), "selection")
    assert not _traces(RP.score_rank_figure(results, results["rows"],
                                            selected_gsm="GSM9999"), "selection")


def test_comparison_uses_both_cohort_hues_and_both_shapes():
    results = _comparison(k=10)
    fig = RP.score_rank_figure(results, results["rows"], selected_gsm="GSM1005")
    shown = {t["meta"]["arm"]: t for t in _traces(fig, "in_table")}
    assert shown["a"]["marker"]["color"] == GRAPH_THEME["cohort_a"]
    assert shown["b"]["marker"]["color"] == GRAPH_THEME["cohort_b"]
    assert shown["a"]["marker"]["symbol"] == "circle"
    assert shown["b"]["marker"]["symbol"] == "square"
    assert shown["a"]["name"] == "OSD-48 liver flight"
    assert shown["b"]["name"] == "OSD-48 liver ground"
    assert shown["a"]["showlegend"] and shown["b"]["showlegend"]
    assert len(_traces(fig, "profile")) == 2
    # GSM1005 is in both top-k lists, so it is ringed on both curves.
    rings = {t["meta"]["arm"]: t for t in _traces(fig, "selection")}
    assert rings["a"]["x"] == [6] and rings["b"]["x"] == [2]
    assert rings["b"]["marker"]["symbol"] == "square-open"
    # The hover reads across both arms.
    hover = shown["a"]["hovertext"][shown["a"]["x"].index(6)]
    assert "OSD-48 liver flight: rank 6" in hover
    assert "OSD-48 liver ground: rank 2" in hover
    only_a = shown["a"]["hovertext"][shown["a"]["x"].index(1)]
    assert "OSD-48 liver ground: not in its top 10" in only_a


def test_comparison_mutes_filtered_rows_in_their_cohort_shape():
    results = _comparison(k=10)
    visible = [r for r in results["rows"] if r["gsm"] not in {"GSM1000", "GSM1012"}]
    fig = RP.score_rank_figure(results, visible)
    muted = {t["meta"]["arm"]: t for t in _traces(fig, "filtered_out")}
    assert muted["a"]["marker"]["symbol"] == "circle"
    assert muted["b"]["marker"]["symbol"] == "square"
    assert [c[1] for c in muted["a"]["customdata"]] == ["GSM1000"]
    assert [c[1] for c in muted["b"]["customdata"]] == ["GSM1012"]
    assert sum(bool(t["showlegend"]) for t in muted.values()) == 1


def test_score_figure_without_a_profile_still_draws_the_matches():
    """The demo path returns no score profile."""
    results = _single(k=5, profile=[], mode="demo")
    fig = RP.score_rank_figure(results, results["rows"])
    assert not _traces(fig, "profile")
    (shown,) = _traces(fig, "in_table")
    assert shown["x"] == [1, 2, 3, 4, 5]


@pytest.mark.parametrize("results", [None, {}, {"rows": []}])
def test_score_figure_empty_state(results):
    assert _is_empty_state(RP.score_rank_figure(results, []))


def test_a_filter_change_keeps_the_zoom_and_a_new_search_resets_it():
    results = _single()
    a = _json(RP.score_rank_figure(results, results["rows"]))["layout"]["uirevision"]
    b = _json(RP.score_rank_figure(results, results["rows"][:3]))["layout"]["uirevision"]
    other = _single(query="OSD-100|s2")
    c = _json(RP.score_rank_figure(other, other["rows"]))["layout"]["uirevision"]
    assert a == b != c
    assert _json(RP.score_rank_figure(results, [], revision="mine"))["layout"]["uirevision"] == "mine"


# -------------------------------------------------------------------- series

def test_series_counts_orders_by_count_then_best_rank():
    rows = [_row(0, 1, gse="GSE9"), _row(1, 2, gse="GSE5"), _row(2, 3, gse="GSE9"),
            _row(3, 4, gse="GSE5"), _row(4, 5, gse="GSE7"), _row(5, 6, gse=""),
            _row(6, 7, gse="GSE7")]
    results = {"comparison": False, "rows": rows}
    counts = RP.series_counts(results, rows)
    assert [c["gse"] for c in counts] == ["GSE9", "GSE5", "GSE7", ""]
    assert [c["count"] for c in counts] == [2, 2, 2, 1]
    assert [c["best_rank"] for c in counts] == [1, 2, 5, 6]
    assert counts[-1]["label"] == "No series recorded"
    assert "a_only" not in counts[0]


def test_series_counts_split_a_comparison_by_retrieved_by():
    results = _comparison(k=10)
    counts = {c["gse"]: c for c in RP.series_counts(results, results["rows"])}
    for c in counts.values():
        assert c["a_only"] + c["both"] + c["b_only"] == c["count"]
    # Rows 0..3 are A only, 4..9 both, 10..13 B only; gse is i % 4.
    assert counts["GSE0"] == {"gse": "GSE0", "label": "GSE0", "count": 4,
                              "best_rank": 1, "a_only": 1, "both": 2, "b_only": 1}


def test_series_counts_return_every_series_and_the_figure_draws_twenty():
    rows = [_row(i, rank_a=i + 1, gse=f"GSE{100 + i // 2}") for i in range(60)]
    results = {"comparison": False, "k": 60, "rows": rows}
    counts = RP.series_counts(results, rows)
    assert len(counts) == 30
    fig = RP.series_figure(results, rows)
    data = _json(fig)
    cats = data["layout"]["yaxis"]["categoryarray"]
    assert cats == [c["gse"] for c in counts[:RP.SERIES_MAX_BARS]]
    assert data["layout"]["yaxis"]["autorange"] == "reversed", "largest at the top"
    (bars,) = _traces(fig, "series")
    assert bars["orientation"] == "h"
    assert bars["text"] == [str(c["count"]) for c in counts[:20]]
    assert bars["customdata"] == [["gse", c["gse"]] for c in counts[:20]]


def test_the_no_series_bar_is_counted_but_not_clickable():
    rows = [_row(0, 1, gse=""), _row(1, 2, gse=""), _row(2, 3, gse="GSE4")]
    results = {"comparison": False, "k": 3, "rows": rows}
    fig = RP.series_figure(results, rows)
    by_click = {t["meta"]["clickable"]: t for t in _traces(fig, "series")}
    assert by_click[False]["y"] == ["No series recorded"]
    assert by_click[False]["x"] == [2]
    assert "customdata" not in by_click[False]
    assert by_click[True]["customdata"] == [["gse", "GSE4"]]


def test_comparison_series_bars_stack_only_both_only():
    results = _comparison(k=10)
    fig = RP.series_figure(results, results["rows"])
    data = _json(fig)
    assert data["layout"]["barmode"] == "stack"
    parts = {t["meta"]["part"]: t for t in _traces(fig, "series") if t["meta"]["clickable"]}
    assert parts["a_only"]["name"] == "Only OSD-48 liver flight"
    assert parts["both"]["name"] == "Both"
    assert parts["b_only"]["name"] == "Only OSD-48 liver ground"
    assert parts["a_only"]["marker"]["color"] == GRAPH_THEME["cohort_a"]
    assert parts["both"]["marker"]["color"] == GRAPH_THEME["cohort_shared"]
    assert parts["b_only"]["marker"]["color"] == GRAPH_THEME["cohort_b"]
    counts = {c["gse"]: c for c in RP.series_counts(results, results["rows"])}
    for key, t in parts.items():
        for (kind, gse), x in zip(t["customdata"], t["x"]):
            assert kind == "gse" and counts[gse][key] == x
    (totals,) = _traces(fig, "totals")
    assert totals["hoverinfo"] == "skip" and "customdata" not in totals


def test_the_selected_series_is_outlined_whether_named_or_reached_by_a_sample():
    results = _single(k=8)
    for kwargs in ({"selected_gse": "GSE2"}, {"selected_gsm": "GSM1006"}):
        fig = RP.series_figure(results, results["rows"], **kwargs)
        (bars,) = _traces(fig, "series")
        widths = dict(zip(bars["y"], bars["marker"]["line"]["width"]))
        assert widths["GSE2"] == 2 and all(w == 1 for g, w in widths.items() if g != "GSE2")
        ticks = _json(fig)["layout"]["yaxis"]["ticktext"]
        assert "<b>GSE2</b>" in ticks


def test_series_empty_states():
    assert _is_empty_state(RP.series_figure(None, None))
    results = _single(k=4)
    fig = RP.series_figure(results, [])
    assert _is_empty_state(fig)
    assert "filters" in _json(fig)["layout"]["annotations"][0]["text"]


# --------------------------------------------------------------- the system

def test_every_figure_shares_the_light_surface_and_never_selects():
    for fig in _all_figures():
        layout = _json(fig)["layout"]
        assert layout["paper_bgcolor"] == GRAPH_THEME["paper_bg"]
        if not fig.data:
            continue
        assert layout["plot_bgcolor"] == GRAPH_THEME["plot_bg"]
        assert layout["clickmode"] == "event"
        assert layout["dragmode"] in ("zoom", "pan", False)
        assert layout["hoverlabel"]["font"]["color"] == GRAPH_THEME["text_primary"]
        assert layout["hoverlabel"]["bgcolor"] == "#ffffff"
        assert {"select2d", "lasso2d"} <= set(layout["modebar"]["remove"])
        assert "title" not in layout or not layout["title"].get("text")


def test_every_clickable_trace_names_a_gsm_or_gse_and_context_traces_are_inert():
    for fig in _all_figures():
        for t in _json(fig)["data"]:
            if "customdata" in t:
                assert t.get("hoverinfo") != "skip"
                cells = t["customdata"]
                for kind, node in cells:
                    assert kind in ("gsm", "gse") and node
            else:
                role = (t.get("meta") or {}).get("role")
                assert t.get("hoverinfo") == "skip" or (
                    role == "series" and not t["meta"]["clickable"]), role


def test_no_dash_characters_in_figures_or_source():
    for fig in _all_figures():
        text = fig.to_json()
        assert "—" not in text and "–" not in text
        assert "\\u2014" not in text and "\\u2013" not in text
    source = Path(RP.__file__).read_text()
    assert "—" not in source and "–" not in source
