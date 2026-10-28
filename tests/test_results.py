"""Tests for `bridge_rna.results`: rows, filters, sort, page and CSV.

Everything here runs on hand-built hit lists shaped like the hits-store payload
the search callbacks write, so none of it needs the memmap or the cache.
"""

from __future__ import annotations

import ast
import io
import math
from pathlib import Path

import pandas as pd
import pytest

from bridge_rna import results as R


# --- builders ---------------------------------------------------------------


def hit(gsm, score, gse="GSE1", title="", source_name="", tissue="", species="",
        archs4_index=None, characteristics=""):
    return {"gsm": gsm, "score": score, "gse": gse, "title": title,
            "source_name": source_name, "tissue": tissue, "species": species,
            "archs4_index": archs4_index, "characteristics": characteristics,
            "geo_summary": "long free text the table never shows"}


def single_payload(hits, profile=None, sample_id="OSD-100|Rep1", name="Rep1"):
    return {"sample_id": sample_id, "mode": "cached", "hits": hits,
            "score_profile": profile if profile is not None else [h["score"] for h in hits],
            "query": {"sample_id": sample_id, "sample_name": name}}


def comparison_payload(hits_a, hits_b):
    return {
        "sample_id": "COHORT|study=OSD-1|tissue=Retina|spaceflight=Flight",
        "mode": "cohort",
        "hits": hits_a,
        "score_profile": [0.9, 0.8],
        "query": {"sample_name": "OSD-1 · Retina · Flight", "cohort_label": "Retina · Flight",
                  "is_cohort": "1"},
        "comparison": {
            "query_b": {"sample_name": "OSD-1 · Retina · Ground",
                        "cohort_label": "Retina · Ground", "is_cohort": "1"},
            "hits_b": hits_b,
            "score_profile_b": [0.95, 0.7, float("nan")],
        },
    }


SINGLE_HITS = [
    hit("GSM10", 0.99, gse="GSE7", title="Retina wt", tissue="Eye",
        species="Mus musculus", archs4_index=10),
    hit("GSM11", 0.98, gse="GSE7", title="Retina ko", tissue="Eye",
        species="Mus musculus", archs4_index=11),
    hit("GSM12", 0.97, gse="GSE8", title="Liver", tissue="Liver",
        species="Homo sapiens", archs4_index=12, source_name="hepatocyte"),
    hit("GSM13", 0.96, gse="GSE9", title="Eye re-deposit", tissue="Eye",
        species="Mus musculus", archs4_index=13),
    hit("GSM14", 0.95, gse="GSE8", title="Kidney", tissue="Kidney",
        species="Homo sapiens", archs4_index=14),
]


def single_results(identical_groups=()):
    return R.build_results(single_payload(SINGLE_HITS), reference_count=940_455,
                           identical_groups=identical_groups)


def comparison_results():
    a = [hit("GSM1", 0.9, archs4_index=1, title="shared"),
         hit("GSM2", 0.8, archs4_index=2, title="a only")]
    b = [hit("GSM3", 0.95, archs4_index=3, title="b only"),
         hit("GSM1", 0.7, archs4_index=1, title="shared")]
    return R.build_results(comparison_payload(a, b), reference_count=100)


def gsms(rows):
    return [r["gsm"] for r in rows]


# --- the module stays pure --------------------------------------------------


def test_the_module_imports_no_dash_memmap_or_retrieval():
    tree = ast.parse(Path(R.__file__).read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
    # Standard library only, so the module is testable on plain lists anywhere.
    assert imported <= {"__future__", "csv", "io", "math", "re", "collections",
                        "collections.abc", "typing"}


def test_no_em_dash_in_the_module():
    assert chr(0x2014) not in Path(R.__file__).read_text()


# --- build_results ----------------------------------------------------------


def test_empty_or_missing_payload_builds_nothing():
    assert R.build_results(None, reference_count=1) is None
    assert R.build_results({}, reference_count=1) is None
    assert R.build_results({"hits": []}, reference_count=1) is None
    assert R.build_results("not a dict", reference_count=1) is None


def test_single_query_contract_shape():
    res = single_results()
    assert set(res) == {"mode", "scoring", "k", "comparison", "label_a", "label_b",
                        "query_id", "reference_count", "profile_a", "profile_b", "rows"}
    # A payload with no scoring recorded predates whitening and was plain cosine.
    assert res["scoring"] == "cosine"
    assert res["mode"] == "cached"
    assert res["k"] == 5
    assert res["comparison"] is False
    assert res["label_a"] == "Rep1"
    assert res["label_b"] == ""
    assert res["query_id"] == "OSD-100|Rep1"
    assert res["reference_count"] == 940_455
    assert res["profile_a"] == [0.99, 0.98, 0.97, 0.96, 0.95]
    assert res["profile_b"] == []
    row = res["rows"][0]
    assert set(row) == {"gsm", "gse", "title", "source_name", "tissue", "species",
                        "archs4_index", "rank_a", "score_a", "cosine_a",
                        "rank_b", "score_b", "cosine_b", "identical"}
    # The long free-text fields stay in hits-store.
    assert "characteristics" not in row and "geo_summary" not in row


def test_single_query_ranks_are_contiguous_from_hit_order():
    res = single_results()
    assert [r["rank_a"] for r in res["rows"]] == [1, 2, 3, 4, 5]
    assert gsms(res["rows"]) == ["GSM10", "GSM11", "GSM12", "GSM13", "GSM14"]
    assert all(r["rank_b"] is None and r["score_b"] is None for r in res["rows"])
    assert [r["score_a"] for r in res["rows"]] == [0.99, 0.98, 0.97, 0.96, 0.95]


def test_rank_comes_from_hit_order_not_from_score():
    # The hits are already best first; a later equal score is still a later rank.
    hits = [hit("GSM2", 0.5, archs4_index=9), hit("GSM1", 0.5, archs4_index=1)]
    res = R.build_results(single_payload(hits), reference_count=10)
    assert [(r["gsm"], r["rank_a"]) for r in res["rows"]] == [("GSM2", 1), ("GSM1", 2)]


def test_missing_text_and_index_become_blank_and_none():
    hits = [{"gsm": "GSM1", "score": 0.9, "gse": float("nan"), "title": "nan",
             "source_name": None, "tissue": "None", "species": "<NA>",
             "archs4_index": float("nan")},
            {"gsm": "GSM2", "score": "0.8", "title": "  padded  ", "archs4_index": "7"}]
    res = R.build_results(single_payload(hits, profile=[]), reference_count=10)
    a, b = res["rows"]
    assert (a["gse"], a["title"], a["source_name"], a["tissue"], a["species"]) == \
        ("", "", "", "", "")
    assert a["archs4_index"] is None
    assert b["title"] == "padded"
    assert b["archs4_index"] == 7 and isinstance(b["archs4_index"], int)
    assert b["score_a"] == 0.8


def test_archs4_index_is_a_plain_int():
    import numpy as np
    hits = [hit("GSM1", np.float32(0.9), archs4_index=np.int64(42))]
    row = R.build_results(single_payload(hits, profile=[]), reference_count=1)["rows"][0]
    assert type(row["archs4_index"]) is int and row["archs4_index"] == 42
    assert type(row["score_a"]) is float


def test_label_falls_back_through_name_to_id():
    payload = single_payload(SINGLE_HITS[:1])
    payload["query"] = {"sample_id": "UPLOAD|f.csv::col"}
    assert R.build_results(payload, reference_count=1)["label_a"] == "UPLOAD|f.csv::col"
    payload["query"] = None
    assert R.build_results(payload, reference_count=1)["label_a"] == "OSD-100|Rep1"


def test_profile_drops_non_finite_values():
    res = comparison_results()
    assert res["profile_a"] == [0.9, 0.8]
    assert res["profile_b"] == [0.95, 0.7]


def test_comparison_joins_both_lists_on_gsm():
    res = comparison_results()
    assert res["comparison"] is True
    assert res["k"] == 2
    assert res["label_a"] == "Retina · Flight"
    assert res["label_b"] == "Retina · Ground"
    by = {r["gsm"]: r for r in res["rows"]}
    assert set(by) == {"GSM1", "GSM2", "GSM3"}
    assert (by["GSM1"]["rank_a"], by["GSM1"]["score_a"]) == (1, 0.9)
    assert (by["GSM1"]["rank_b"], by["GSM1"]["score_b"]) == (2, 0.7)
    assert (by["GSM2"]["rank_b"], by["GSM2"]["score_b"]) == (None, None)
    assert (by["GSM3"]["rank_a"], by["GSM3"]["score_a"]) == (None, None)
    assert by["GSM3"]["rank_b"] == 1


def test_comparison_rows_order_best_rank_then_index():
    # GSM1 (A1) and GSM3 (B1) share best rank 1; archs4_index breaks the tie.
    res = comparison_results()
    assert gsms(res["rows"]) == ["GSM1", "GSM3", "GSM2"]


def test_comparison_k_is_the_longer_list():
    a = [hit("GSM1", 0.9)]
    b = [hit("GSM2", 0.9), hit("GSM3", 0.8), hit("GSM4", 0.7)]
    assert R.build_results(comparison_payload(a, b), reference_count=1)["k"] == 3


def test_order_ties_put_missing_index_last_then_gsm():
    a = [hit("GSMb", 0.9, archs4_index=None)]
    b = [hit("GSMa", 0.9, archs4_index=None), hit("GSMc", 0.8, archs4_index=5)]
    rows = R.build_results(comparison_payload(a, b), reference_count=1)["rows"]
    assert gsms(rows) == ["GSMa", "GSMb", "GSMc"]


def test_metadata_fills_from_either_side():
    a = [hit("GSM1", 0.9, title="", tissue="", archs4_index=None)]
    b = [hit("GSM1", 0.8, title="From B", tissue="Eye", archs4_index=4)]
    row = R.build_results(comparison_payload(a, b), reference_count=1)["rows"][0]
    assert (row["title"], row["tissue"], row["archs4_index"]) == ("From B", "Eye", 4)


def test_a_repeated_gsm_keeps_its_first_rank():
    hits = [hit("GSM1", 0.9), hit("GSM2", 0.8), hit("GSM1", 0.7)]
    rows = R.build_results(single_payload(hits), reference_count=1)["rows"]
    assert [(r["gsm"], r["rank_a"]) for r in rows] == [("GSM1", 1), ("GSM2", 2)]


def test_a_hit_without_accession_keeps_its_rank():
    hits = [hit("", 0.9), hit("GSM2", 0.8), hit(float("nan"), 0.7)]
    rows = R.build_results(single_payload(hits), reference_count=1)["rows"]
    assert [r["rank_a"] for r in rows] == [1, 2, 3]
    assert gsms(rows) == ["", "GSM2", ""]


def test_identical_groups_list_the_other_present_members():
    res = single_results(identical_groups=[["GSM13", "GSM10"], ["GSM12", "GSM999"]])
    by = {r["gsm"]: r for r in res["rows"]}
    assert by["GSM10"]["identical"] == ["GSM13"]
    assert by["GSM13"]["identical"] == ["GSM10"]
    # GSM999 is not in the rows, so GSM12 has no twin to name.
    assert by["GSM12"]["identical"] == []
    assert by["GSM11"]["identical"] == []


def test_identical_groups_larger_than_pairs_list_in_rank_order():
    res = single_results(identical_groups=[("GSM14", "GSM11", "GSM12")])
    by = {r["gsm"]: r for r in res["rows"]}
    assert by["GSM14"]["identical"] == ["GSM11", "GSM12"]
    assert by["GSM11"]["identical"] == ["GSM12", "GSM14"]


# --- normalize_filters ------------------------------------------------------


def test_default_filters_are_inactive_and_valid():
    f = R.default_filters()
    assert f == R.normalize_filters(None)
    assert not R.filters_active(f)
    assert set(f) == {"series", "tissue", "species", "min_score", "text",
                      "retrieved_by"}


@pytest.mark.parametrize("raw", [None, {}, [], "junk", 3, {"unrelated": 1}])
def test_normalize_filters_tolerates_garbage(raw):
    assert R.normalize_filters(raw) == R.default_filters()


def test_normalize_filters_coerces_widget_values():
    f = R.normalize_filters({
        "series": "GSE1", "tissue": ("Eye", None, "", "Eye"),
        "species": None, "min_score": " 0.95 ", "text": "  retina ",
        "retrieved_by": "Both",
        # Retired controls a stale session could still send: ignored.
        "series_mode": "hide", "merge_identical": ["on"],
    })
    assert f == {"series": ["GSE1"], "tissue": ["Eye"],
                 "species": [], "min_score": 0.95, "text": "retina",
                 "retrieved_by": "both"}


@pytest.mark.parametrize("value, expected", [
    ("abc", None), ("", None), (float("nan"), None), (float("inf"), None),
    (True, None), (0, 0.0), ("0.5", 0.5), (1, 1.0),
])
def test_normalize_min_score(value, expected):
    assert R.normalize_filters({"min_score": value})["min_score"] == expected


def test_normalize_rejects_unknown_modes():
    f = R.normalize_filters({"retrieved_by": "neither", "text": 5})
    assert (f["retrieved_by"], f["text"]) == ("either", "")


# --- apply_filters ----------------------------------------------------------


def run(filters, res=None):
    res = res or single_results()
    return gsms(R.apply_filters(res["rows"], filters))


def test_no_filters_keep_every_row():
    assert run(None) == ["GSM10", "GSM11", "GSM12", "GSM13", "GSM14"]


def test_a_series_filter_shows_only_the_chosen_series():
    """Show only is the one series mode since 2026-10-22; Hide was removed."""
    assert run({"series": ["GSE8"]}) == ["GSM12", "GSM14"]
    assert run({"series": ["GSE7", "GSE9"]}) == ["GSM10", "GSM11", "GSM13"]
    assert run({"series": ["GSE8"], "series_mode": "hide"}) == ["GSM12", "GSM14"]


def test_tissue_and_species():
    assert run({"tissue": ["Eye"]}) == ["GSM10", "GSM11", "GSM13"]
    assert run({"species": ["Homo sapiens"]}) == ["GSM12", "GSM14"]


def test_min_score_is_inclusive():
    assert run({"min_score": 0.97}) == ["GSM10", "GSM11", "GSM12"]
    assert run({"min_score": 1.5}) == []


def test_min_score_uses_the_best_present_score_in_a_comparison():
    res = comparison_results()
    # GSM1 scores 0.9 in A and 0.7 in B; its best (0.9) passes 0.85.
    assert run({"min_score": 0.85}, res) == ["GSM1", "GSM3"]


def test_text_search_is_case_insensitive_over_four_fields():
    assert run({"text": "RETINA"}) == ["GSM10", "GSM11"]
    assert run({"text": "gsm12"}) == ["GSM12"]
    assert run({"text": "gse9"}) == ["GSM13"]
    assert run({"text": "Hepato"}) == ["GSM12"]
    # Tissue is not a searched field.
    assert run({"text": "kidney"}) == ["GSM14"]
    assert run({"text": "Liver"}) == ["GSM12"]
    assert run({"text": "Mus musculus"}) == []


def test_filters_combine_with_and():
    assert run({"tissue": ["Eye"], "series": ["GSE7"], "min_score": 0.985}) == ["GSM10"]


def test_filtering_preserves_input_order():
    res = single_results()
    reordered = list(reversed(res["rows"]))
    kept = R.apply_filters(reordered, {"tissue": ["Eye"]})
    assert gsms(kept) == ["GSM13", "GSM11", "GSM10"]


def test_filtering_never_renumbers_ranks():
    kept = R.apply_filters(single_results()["rows"], {"series": ["GSE8"]})
    assert [r["rank_a"] for r in kept] == [3, 5]


def test_identical_profiles_stay_as_separate_rows():
    """No merge: each copy of a re-deposited profile keeps its own row and rank,
    and is marked with the other. The merge checkbox was removed on 2026-10-22."""
    res = single_results(identical_groups=[["GSM13", "GSM10"]])
    kept = R.apply_filters(res["rows"], None)
    assert gsms(kept) == ["GSM10", "GSM11", "GSM12", "GSM13", "GSM14"]
    by = {r["gsm"]: r for r in kept}
    assert by["GSM10"]["identical"] == ["GSM13"] and by["GSM13"]["identical"] == ["GSM10"]


@pytest.mark.parametrize("mode, expected", [
    ("either", ["GSM1", "GSM3", "GSM2"]),
    ("both", ["GSM1"]),
    ("a", ["GSM2"]),
    ("b", ["GSM3"]),
])
def test_retrieved_by_on_a_comparison(mode, expected):
    assert run({"retrieved_by": mode}, comparison_results()) == expected


def test_retrieved_by_is_ignored_on_a_single_query():
    for mode in ("both", "a", "b"):
        assert run({"retrieved_by": mode}) == gsms(single_results()["rows"])


def test_apply_filters_on_no_rows():
    assert R.apply_filters([], {"text": "x"}) == []


# --- sort_rows --------------------------------------------------------------


def sortable():
    """A comparison with gaps in every column, built to exercise missing-last."""
    a = [hit("GSM5", 0.90, gse="GSE2", title="beta", tissue="Eye", species="Mus musculus",
             archs4_index=5),
         hit("GSM1", 0.80, gse="", title="Alpha", tissue="", species="Homo sapiens",
             archs4_index=1),
         hit("GSM3", 0.80, gse="GSE1", title="", tissue="liver", species="",
             archs4_index=3)]
    b = [hit("GSM4", 0.99, gse="gse3", title="gamma", tissue="Brain", species="Mus musculus",
             archs4_index=4),
         hit("GSM5", 0.70, archs4_index=5)]
    return R.build_results(comparison_payload(a, b), reference_count=10)["rows"]


def test_default_order_is_best_rank():
    rows = sortable()
    assert gsms(rows) == ["GSM4", "GSM5", "GSM1", "GSM3"]
    assert gsms(R.sort_rows(list(reversed(rows)), None)) == gsms(rows)


@pytest.mark.parametrize("column, ascending, descending", [
    # rank_a: GSM5=1, GSM1=2, GSM3=3, GSM4 missing
    ("rank_a", ["GSM5", "GSM1", "GSM3", "GSM4"], ["GSM3", "GSM1", "GSM5", "GSM4"]),
    # score_a: GSM5 .9, GSM1 .8 and GSM3 .8 tie (GSM1 has the better rank), GSM4 missing
    ("score_a", ["GSM1", "GSM3", "GSM5", "GSM4"], ["GSM5", "GSM1", "GSM3", "GSM4"]),
    # rank_b: GSM4=1, GSM5=2, the rest missing in best-rank order
    ("rank_b", ["GSM4", "GSM5", "GSM1", "GSM3"], ["GSM5", "GSM4", "GSM1", "GSM3"]),
    ("score_b", ["GSM5", "GSM4", "GSM1", "GSM3"], ["GSM4", "GSM5", "GSM1", "GSM3"]),
    ("gsm", ["GSM1", "GSM3", "GSM4", "GSM5"], ["GSM5", "GSM4", "GSM3", "GSM1"]),
    # gse compares case-insensitively; GSM1 has none and goes last both ways
    ("gse", ["GSM3", "GSM5", "GSM4", "GSM1"], ["GSM4", "GSM5", "GSM3", "GSM1"]),
    ("title", ["GSM1", "GSM5", "GSM4", "GSM3"], ["GSM4", "GSM5", "GSM1", "GSM3"]),
    ("tissue", ["GSM4", "GSM5", "GSM3", "GSM1"], ["GSM3", "GSM5", "GSM4", "GSM1"]),
    # species: GSM4 and GSM5 tie on Mus musculus and fall back to best rank
    ("species", ["GSM1", "GSM4", "GSM5", "GSM3"], ["GSM4", "GSM5", "GSM1", "GSM3"]),
])
def test_sort_every_column_both_directions(column, ascending, descending):
    rows = sortable()
    assert gsms(R.sort_rows(rows, {"column": column, "descending": False})) == ascending
    assert gsms(R.sort_rows(rows, {"column": column, "descending": True})) == descending


def test_every_sort_column_is_covered():
    assert set(R.SORT_COLUMNS) == {"rank_a", "score_a", "rank_b", "score_b", "gsm",
                                   "gse", "title", "tissue", "species"}


def test_sort_ties_fall_back_to_best_rank_in_both_directions():
    rows = R.build_results(single_payload([
        hit("GSM1", 0.9, gse="GSE1"), hit("GSM2", 0.8, gse="GSE1"),
        hit("GSM3", 0.7, gse="GSE1")]), reference_count=1)["rows"]
    shuffled = [rows[2], rows[0], rows[1]]
    for descending in (False, True):
        out = R.sort_rows(shuffled, {"column": "gse", "descending": descending})
        assert gsms(out) == ["GSM1", "GSM2", "GSM3"]


def test_sort_never_renumbers_and_does_not_mutate():
    rows = sortable()
    before = [dict(r) for r in rows]
    out = R.sort_rows(rows, {"column": "title", "descending": True})
    assert rows == before
    assert {r["gsm"]: r["rank_a"] for r in out} == {r["gsm"]: r["rank_a"] for r in rows}


@pytest.mark.parametrize("sort", [None, {}, {"column": "nonsense", "descending": True},
                                  {"column": 3}, "rank_a"])
def test_unknown_sort_is_rank_order(sort):
    rows = sortable()
    assert gsms(R.sort_rows(list(reversed(rows)), sort)) == gsms(rows)


# --- page_rows --------------------------------------------------------------


def numbered(n):
    return [{"gsm": f"GSM{i}", "rank_a": i} for i in range(1, n + 1)]


def test_page_size_is_fifty():
    assert R.PAGE_SIZE == 50


def test_first_middle_and_last_page():
    rows = numbered(412)
    p = R.page_rows(rows, 0)
    assert (p["page"], p["pages"], p["start"], p["end"], p["total"]) == (0, 9, 1, 50, 412)
    assert len(p["rows"]) == 50
    p = R.page_rows(rows, 1)
    assert (p["start"], p["end"]) == (51, 100)
    assert p["rows"][0]["gsm"] == "GSM51"
    p = R.page_rows(rows, 8)
    assert (p["page"], p["start"], p["end"], len(p["rows"])) == (8, 401, 412, 12)


@pytest.mark.parametrize("page, expected", [
    (-3, 0), (99, 2), (None, 0), ("1", 1), ("x", 0), (1.7, 1), (float("nan"), 0),
])
def test_page_is_clamped(page, expected):
    assert R.page_rows(numbered(120), page)["page"] == expected


def test_exact_multiple_has_no_empty_trailing_page():
    p = R.page_rows(numbered(100), 5)
    assert (p["pages"], p["page"], p["start"], p["end"]) == (2, 1, 51, 100)


def test_empty_rows_page():
    p = R.page_rows([], 3)
    assert p == {"rows": [], "page": 0, "pages": 1, "start": 0, "end": 0, "total": 0}


# --- filter_options ---------------------------------------------------------


def test_filter_options_count_and_order():
    opts = R.filter_options(single_results()["rows"])
    assert opts["series"] == [("GSE7", 2), ("GSE8", 2), ("GSE9", 1)]
    assert opts["tissue"] == [("Eye", 3), ("Kidney", 1), ("Liver", 1)]
    assert opts["species"] == [("Mus musculus", 3), ("Homo sapiens", 2)]


def test_filter_options_exclude_empty_values():
    opts = R.filter_options(sortable())
    assert ("", 1) not in opts["series"]
    assert all(value for value, _ in opts["tissue"] + opts["series"] + opts["species"])
    assert opts["species"] == [("Mus musculus", 2), ("Homo sapiens", 1)]


def test_filter_options_on_no_rows():
    assert R.filter_options([]) == {"series": [], "tissue": [], "species": []}


# --- describe_filters and filters_active ------------------------------------


def test_describe_each_filter():
    res = comparison_results()
    lines = R.describe_filters({
        "series": ["GSE1", "GSE2"], "tissue": ["Eye"], "species": ["Mus musculus"],
        "min_score": 0.99, "text": "retina",
        "retrieved_by": "both"}, res)
    assert lines == [
        "Only series: GSE1, GSE2",
        "Tissue category: Eye",
        "Species: Mus musculus",
        "Minimum score 0.99",
        'Text contains "retina"',
        "Retrieved by both cohorts",
    ]


def test_describe_series_and_cohort_names():
    res = comparison_results()
    # A stale Hide mode is ignored: a series filter only ever shows.
    assert R.describe_filters({"series": ["GSE3"], "series_mode": "hide"}, res) == \
        ["Only series: GSE3"]
    assert R.describe_filters({"retrieved_by": "a"}, res) == \
        ["Retrieved only by Retina · Flight"]
    assert R.describe_filters({"retrieved_by": "b"}, res) == \
        ["Retrieved only by Retina · Ground"]


def test_describe_falls_back_to_cohort_letters():
    res = dict(comparison_results(), label_a="", label_b="")
    assert R.describe_filters({"retrieved_by": "b"}, res) == ["Retrieved only by cohort B"]


def test_describe_nothing_when_inactive():
    assert R.describe_filters(None, single_results()) == []
    assert R.describe_filters(R.default_filters(), comparison_results()) == []


def test_retrieved_by_is_not_active_on_a_single_query():
    single = single_results()
    assert R.describe_filters({"retrieved_by": "both"}, single) == []
    assert not R.filters_active({"retrieved_by": "both"}, single)
    assert R.filters_active({"retrieved_by": "both"}, comparison_results())


@pytest.mark.parametrize("change", [
    {"series": ["GSE1"]}, {"tissue": ["Eye"]}, {"species": ["Mus musculus"]},
    {"min_score": 0.0}, {"text": "x"},
    {"retrieved_by": "a"},
])
def test_each_filter_alone_is_active(change):
    assert R.filters_active(dict(R.default_filters(), **change))


def test_retired_controls_alone_are_not_active():
    assert not R.filters_active({"series_mode": "hide", "merge_identical": True})


# --- to_csv -----------------------------------------------------------------


AWKWARD = 'Sample #3, "wild type"\nsecond line # with hash'


def export(res, rows=None, filters=None, sort=None):
    rows = res["rows"] if rows is None else rows
    lookup = {"GSM10": {"characteristics": "age: 8 wk, sex: M"},
              "GSM1": {"characteristics": "tissue: retina; #2"}}
    return R.to_csv(rows, res, filters, sort, lookup, "2026-10-21T15:00:00Z")


def read(text):
    return pd.read_csv(io.StringIO(text), comment="#", keep_default_na=False,
                       dtype=str)


def test_csv_single_round_trip_and_columns():
    res = single_results(identical_groups=[["GSM10", "GSM13"]])
    df = read(export(res))
    assert list(df.columns) == ["rank", "gsm", "score", "gse", "title", "source_name",
                                "characteristics", "tissue_category", "species",
                                "identical_profile_gsms", "archs4_index"]
    assert df["gsm"].tolist() == ["GSM10", "GSM11", "GSM12", "GSM13", "GSM14"]
    assert df["rank"].tolist() == ["1", "2", "3", "4", "5"]
    assert df.loc[0, "characteristics"] == "age: 8 wk, sex: M"
    assert df.loc[1, "characteristics"] == ""
    assert df.loc[0, "identical_profile_gsms"] == "GSM13"
    assert df.loc[0, "archs4_index"] == "10"
    assert df.loc[2, "tissue_category"] == "Liver"


def test_csv_keeps_full_score_precision():
    score = 0.99712345678901234
    res = R.build_results(single_payload([hit("GSM1", score, archs4_index=1)]),
                          reference_count=1)
    df = pd.read_csv(io.StringIO(export(res)), comment="#")
    assert df.loc[0, "score"] == score


def test_csv_round_trips_commas_quotes_newlines_and_hashes():
    titles = [AWKWARD, "#leading hash", "trailing #", 'only "quotes"', "a,b,c"]
    hits = [hit(f"GSM{i}", 0.9 - i / 100, title=t, source_name=t, archs4_index=i)
            for i, t in enumerate(titles)]
    res = R.build_results(single_payload(hits), reference_count=1)
    df = read(export(res))
    assert df["title"].tolist() == titles
    assert df["source_name"].tolist() == titles
    assert len(df) == len(titles)


def test_csv_comparison_columns_and_blank_ranks():
    res = comparison_results()
    text = export(res)
    df = pd.read_csv(io.StringIO(text), comment="#")
    assert list(df.columns[:5]) == ["rank_a", "score_a", "rank_b", "score_b", "gsm"]
    assert "rank" not in df.columns and "score" not in df.columns
    by = df.set_index("gsm")
    assert by.loc["GSM1", "rank_a"] == 1 and by.loc["GSM1", "rank_b"] == 2
    assert math.isnan(by.loc["GSM2", "rank_b"]) and math.isnan(by.loc["GSM3", "score_a"])
    assert by.loc["GSM1", "characteristics"] == "tissue: retina; #2"


def test_csv_rows_follow_the_given_order():
    res = single_results()
    rows = R.sort_rows(res["rows"], {"column": "title", "descending": False})
    df = read(export(res, rows=rows))
    assert df["gsm"].tolist() == gsms(rows)
    assert df["rank"].tolist() == [str(r["rank_a"]) for r in rows]


def test_csv_with_no_rows_still_has_a_header():
    res = single_results()
    df = read(export(res, rows=[]))
    assert len(df) == 0 and "gsm" in df.columns


def provenance(text):
    return [line[2:] for line in text.splitlines() if line.startswith("# ")]


def test_csv_provenance_single():
    res = single_results()
    lines = provenance(export(res, sort={"column": "score_a", "descending": True}))
    assert lines[0] == "Bridge RNA retrieval export"
    assert "Query: Rep1" in lines
    assert "Query ID: OSD-100|Rep1" in lines
    assert "Mode: cached" in lines
    assert "Matches requested (k): 5" in lines
    assert "Reference collection: 940,455 ARCHS4 samples" in lines
    assert "Filters: none" in lines
    assert "Rows: 5" in lines
    assert "Sort: Score, descending" in lines
    assert "Exported at: 2026-10-21T15:00:00Z" in lines
    note = lines[-1]
    assert "positions among all 940,455 reference samples" in note
    assert "not biological equivalence" in note


def test_csv_provenance_comparison_and_filters():
    res = comparison_results()
    filters = {"retrieved_by": "both", "text": 'multi\nline "x"', "series": ["GSE1"]}
    text = export(res, filters=filters, sort={"column": "rank_b"})
    lines = provenance(text)
    assert "Query A: Retina · Flight" in lines
    assert "Query B: Retina · Ground" in lines
    assert "Matches requested (k): 2 per cohort" in lines
    assert "Filter: Only series: GSE1" in lines
    assert "Filter: Retrieved by both cohorts" in lines
    # A newline in a filter value cannot escape the comment line.
    assert 'Filter: Text contains "multi line "x""' in lines
    assert "Filters: none" not in lines
    assert "Sort: Rank (Retina · Ground), ascending" in lines
    # Every line before the header is a comment, so pandas skips all of them.
    head = text.split('"rank_a"')[0]
    assert all(line.startswith("# ") for line in head.splitlines())


def test_csv_default_sort_is_named():
    assert "Sort: Best rank first" in provenance(export(single_results()))
