"""The results half of the retrieval view: controls, table, plot routing, export.

`results.py` and `result_plots.py` have their own tests; these pin how the view
wires them together, which is where the table, the plot, the export and the AI
summary could start to disagree about which rows are in view.
"""

from __future__ import annotations

import io
import math

import pandas as pd
import pytest

from bridge_rna import results as R
from bridge_rna import results_view as V


# --- Number of matches -----------------------------------------------------

@pytest.mark.parametrize("k", [3, 5, 10, 30, 100, 300, 1000])
def test_every_mark_on_the_slider_lands_on_its_own_k(k):
    assert V.k_from_slider(V._log_k(k)) == k


@pytest.mark.parametrize("position,expected", [
    (math.log10(7), 7),          # small k stays exact
    (2.531, 340),                # 339.6 -> two significant figures
    (2.9999, 1000),
    (0.0, 3),                    # below the floor clamps up
    (None, V.MATCH_COUNT_DEFAULT),
])
def test_a_dragged_slider_offers_a_readable_k(position, expected):
    assert V.k_from_slider(position) == expected


@pytest.mark.parametrize("typed,expected", [
    (47, 47), ("47", 47), (2, 3), (5000, 1000), ("", 5), (None, 5), (12.6, 13),
])
def test_the_exact_box_is_clamped_to_the_supported_range(typed, expected):
    assert V.clamp_match_count(typed) == expected


@pytest.mark.parametrize("typed,position,expected", [
    (47, None, 47),
    (5000, None, 1000),          # an out-of-range k is clamped, never dropped
    (None, V._log_k(300), 300),  # an empty box falls back to the slider
    ("", V._log_k(30), 30),
    ("abc", V._log_k(10), 10),
])
def test_a_search_uses_the_box_and_falls_back_to_the_slider(typed, position, expected):
    assert V.requested_match_count(typed, position) == expected


def test_the_box_carries_no_range_that_would_turn_5000_into_none():
    """Dash 4 reports a number outside min/max as None; the callback clamps instead."""
    box = next(c for c in [V.build_topk_control().children[0].children[1]])
    assert box.id == "topk-input"
    assert getattr(box, "max", None) is None and getattr(box, "min", None) is None


def test_the_control_reaches_one_thousand_and_starts_where_it_did():
    slider = V.build_topk_control().children[1].children[0]
    assert slider.id == "topk-slider"
    assert V.k_from_slider(slider.min) == 3 and V.k_from_slider(slider.max) == 1000
    assert V.k_from_slider(slider.value) == V.MATCH_COUNT_DEFAULT == 5


# --- Rows and the table ------------------------------------------------------

def _row(gsm, rank_a, score_a, rank_b=None, score_b=None, **extra):
    row = {"gsm": gsm, "gse": extra.pop("gse", "GSE1"), "title": f"{gsm} title",
           "source_name": "", "tissue": "Eye", "species": "Mus musculus",
           "archs4_index": extra.pop("archs4_index", None),
           "rank_a": rank_a, "score_a": score_a, "rank_b": rank_b,
           "score_b": score_b, "identical": extra.pop("identical", [])}
    row.update(extra)
    return row


def _results(rows, comparison=False):
    return {"mode": "cohort" if comparison else "cached", "k": 3,
            "comparison": comparison, "label_a": "Query A",
            "label_b": "Query B" if comparison else "", "query_id": "Q",
            "reference_count": 940455, "profile_a": [0.9, 0.8, 0.7],
            "profile_b": [0.85, 0.75] if comparison else [], "rows": rows}


def test_single_table_has_one_rank_and_one_score_column():
    results = _results([_row("GSM1", 1, 0.99), _row("GSM2", 2, 0.98)])
    html = str(V.render_table(results["rows"], results, V.DEFAULT_SORT, "GSM2"))
    for header in ("Rank", "GSM", "Score", "Series", "Title", "Tissue category", "Species"):
        assert f"'{header}'" in html
    assert "Rank A" not in html
    assert "aria-sort" in html and "ascending" in html
    assert "is-selected" in html


def test_comparison_table_names_both_cohorts_ranks_and_hides_missing_ones():
    rows = [_row("GSM1", 1, 0.99, 2, 0.97), _row("GSM2", None, None, 1, 0.98)]
    html = str(V.render_table(rows, _results(rows, comparison=True), V.DEFAULT_SORT))
    for header in ("Rank A", "Score A", "Rank B", "Score B"):
        assert header in html
    assert "Not in top 3" in html


def test_identical_profiles_are_named_in_the_gsm_cell():
    rows = [_row("GSM1", 1, 0.99, identical=["GSM9"]), _row("GSM9", 2, 0.99,
                                                             identical=["GSM1"])]
    html = str(V.render_table(rows, _results(rows), V.DEFAULT_SORT))
    assert "Same profile: GSM9" in html and "Same profile: GSM1" in html


def test_ranks_are_never_renumbered_by_a_filter():
    rows = [_row(f"GSM{i}", i, 1 - i / 100, gse="GSE1" if i % 2 else "GSE2")
            for i in range(1, 7)]
    results = _results(rows)
    filters = R.normalize_filters({"series": ["GSE2"], "series_mode": "only"})
    shown, _hidden = V.visible_rows(results, filters, V.DEFAULT_SORT)
    assert [r["rank_a"] for r in shown] == [2, 4, 6]


def test_filters_from_the_controls_include_the_merge_checkbox():
    values = [["GSE1"], "hide", [], [], None, "", ["on"], "either"]
    filters = V.filters_from_inputs(values)
    assert filters["series"] == ["GSE1"] and filters["series_mode"] == "hide"
    assert filters["merge_identical"] is True
    assert V.filters_from_inputs([None] * len(V.FILTER_INPUTS)) == R.default_filters()


def test_summary_states_the_rank_rule_only_when_ranks_stop_being_contiguous():
    results = _results([_row("GSM1", 1, 0.99), _row("GSM2", 2, 0.98)])
    plain = V._summary(results, 2, 0, filtered=False)
    assert "not biological equivalence" in plain and "Query A" in plain
    assert "positions among all" not in plain
    filtered = V._summary(results, 1, 0, filtered=True)
    assert filtered.startswith("1 of 2 matches shown.")
    assert "positions among all 940,455 reference samples" in filtered
    merged = V._summary(results, 1, 1, filtered=False)
    assert "1 identical profile merged" in merged


# --- Plot routing -------------------------------------------------------------

def test_empty_results_invite_the_active_mode():
    fig, caption, legend = V.build_plot("rank", None, [], None, mode="cohort")
    assert "cohort" in fig.to_json() and caption == "" and legend == ""


def test_a_filter_that_keeps_nothing_says_so_in_the_plot():
    results = _results([_row("GSM1", 1, 0.99)])
    fig, _caption, _legend = V.build_plot("rank", results, [], None)
    assert "No matches pass these filters" in fig.to_json()


def test_network_view_is_capped_and_says_so():
    from bridge_rna import result_plots as P

    n = P.NETWORK_MAX_ROWS + 5
    rows = [_row(f"GSM{i}", i, 1 - i / 1000, gse=f"GSE{i % 3}") for i in range(1, n + 1)]
    results = _results(rows)
    results["query_a"] = {"sample_id": "OSD-1|x", "sample_name": "x"}
    fig, caption, legend = V.build_plot("network", results, rows, None)
    assert legend == "sample"
    assert f"first {P.NETWORK_MAX_ROWS} of {n}" in caption
    gsm_nodes = [c for t in fig.data if getattr(t, "customdata", None) is not None
                 for c in t.customdata if c[0] == "gsm"]
    assert len(gsm_nodes) == P.NETWORK_MAX_ROWS


def test_similarity_needs_locatable_rows():
    results = _results([_row("GSM1", 1, 0.99), _row("GSM2", 2, 0.98)])
    fig, _caption, _legend = V.build_plot("similarity", results, results["rows"], None)
    assert "at least two matches" in fig.to_json()


def test_hits_frame_takes_one_cohorts_rows_in_its_own_rank_order():
    rows = [_row("GSM1", 2, 0.97, 1, 0.99), _row("GSM2", 1, 0.98), _row("GSM3", None, None, 2, 0.9)]
    frame_a = V._hits_frame(rows, "a")
    frame_b = V._hits_frame(rows, "b")
    assert frame_a["gsm"].tolist() == ["GSM2", "GSM1"]
    assert frame_b["gsm"].tolist() == ["GSM1", "GSM3"]
    assert frame_b["score"].tolist() == [0.99, 0.9]


# --- Export --------------------------------------------------------------------

def test_export_is_the_visible_rows_in_table_order():
    rows = [_row(f"GSM{i}", i, 1 - i / 100, gse="GSE1" if i < 3 else "GSE2")
            for i in range(1, 5)]
    results = _results(rows)
    filters = R.normalize_filters({"series": ["GSE2"], "series_mode": "hide"})
    sort = {"column": "rank_a", "descending": True}
    shown, _ = V.visible_rows(results, filters, sort)
    text = R.to_csv(shown, results, filters, sort, {}, "2026-10-21 00:00:00 UTC")
    frame = pd.read_csv(io.StringIO(text), comment="#")
    assert frame["gsm"].tolist() == ["GSM2", "GSM1"]
