from __future__ import annotations

import math

import pandas as pd

from bridge_rna import neighborhoods as N


def _frame() -> pd.DataFrame:
    return pd.DataFrame([
        {"gsm": "GSM1", "gse": "GSE10", "title": "Retina A",
         "tissue": "Eye / retina", "species": "Mus musculus",
         "score": 0.99, "archs4_index": 4},
        {"gsm": "GSM2", "gse": "GSE10", "title": "Retina A",
         "tissue": "Eye / retina", "species": "Mus musculus",
         "score": 0.98, "archs4_index": 7},
        {"gsm": "GSM3", "gse": "GSE20", "title": "Brain B",
         "tissue": "Brain / CNS", "species": "Homo sapiens",
         "score": 0.96, "archs4_index": 9},
        {"gsm": "GSM4", "gse": "", "title": "",
         "tissue": None, "species": float("nan"),
         "score": 0.95, "archs4_index": None},
    ])


def test_payload_is_ranked_json_safe_and_explicit_about_depth():
    payload = N.build_payload(_frame(), label="OSD-100 · eye")
    assert payload["depth_requested"] == 250
    assert payload["depth_returned"] == 4
    assert payload["metric"] == "cosine"
    assert payload["space"] == "embedding-512d"
    assert [r["rank"] for r in payload["hits"]] == [1, 2, 3, 4]
    assert payload["hits"][3]["species"] == ""
    assert payload["hits"][3]["archs4_index"] is None
    assert not any(isinstance(v, float) and math.isnan(v)
                   for row in payload["hits"] for v in row.values())


def test_payload_treats_pandas_nullable_text_as_missing_metadata():
    payload = N.build_payload(pd.DataFrame([
        {"gsm": pd.NA, "tissue": pd.NA, "species": pd.NA},
    ]))

    assert payload["hits"][0]["gsm"] == ""
    assert N.summarize(payload)["tissue"] == {"covered": 0, "items": []}


def test_summary_uses_metadata_coverage_as_the_denominator():
    summary = N.summarize(N.build_payload(_frame()))
    assert summary["depth"] == 4
    assert summary["tissue"]["covered"] == 3
    assert summary["tissue"]["items"][0] == {
        "label": "Eye / retina", "count": 2, "share": 2 / 3}
    assert summary["species"]["covered"] == 3
    assert summary["study_count"] == 2
    assert summary["top_three_study_samples"] == 3
    assert "2 of 3" in summary["sentence"]


def test_studies_group_unassigned_rows_and_sort_deterministically():
    groups = N.study_groups(N.build_payload(_frame()))
    assert [g["gse"] for g in groups] == ["GSE10", "GSE20", ""]
    assert groups[0]["sample_count"] == 2
    assert groups[0]["best_rank"] == 1
    assert groups[0]["dominant_tissue"] == "Eye / retina"
    assert groups[-1]["display_gse"] == "No GSE recorded"


def test_sample_filter_matches_accession_study_tissue_species_and_title():
    payload = N.build_payload(_frame())
    assert [r["gsm"] for r in N.sample_rows(payload, "gsm3")] == ["GSM3"]
    assert [r["gsm"] for r in N.sample_rows(payload, "retina")] == ["GSM1", "GSM2"]
    assert len(N.sample_rows(payload, "")) == 4


def test_missing_or_unavailable_payload_has_honest_empty_structures():
    unavailable = N.unavailable_payload("Run this retrieval again.")
    assert N.summarize(unavailable)["available"] is False
    assert N.study_groups(None) == []
    assert N.sample_rows({}, "") == []


def test_summary_uses_numeric_scores_and_reports_missing_tissue_honestly():
    frame = pd.DataFrame([
        {"gsm": "GSM2", "gse": "GSE2", "score": 0.2},
        {"gsm": "GSM1", "gse": "GSE1", "score": "not numeric"},
        {"gsm": "GSM3", "gse": "GSE3", "score": 0.8},
    ])
    summary = N.summarize(N.build_payload(frame))

    assert summary["score"] == {
        "count": 2, "median": 0.5, "minimum": 0.2, "maximum": 0.8}
    assert summary["tissue"] == {"covered": 0, "items": []}
    assert summary["sentence"] == (
        "No tissue metadata is available for the returned depth of 3 samples.")


def test_summary_and_study_ties_are_sorted_by_label_and_rank():
    frame = pd.DataFrame([
        {"gsm": "GSM3", "gse": "GSE2", "tissue": "Zebra", "score": 0.6},
        {"gsm": "GSM1", "gse": "GSE1", "tissue": "Ant", "score": 0.9},
        {"gsm": "GSM2", "gse": "GSE3", "tissue": "Ant", "score": 0.8},
    ])
    payload = N.build_payload(frame)

    assert [item["label"] for item in N.summarize(payload)["tissue"]["items"]] == [
        "Ant", "Zebra"]
    assert [group["gse"] for group in N.study_groups(payload)] == [
        "GSE2", "GSE1", "GSE3"]


def test_unassigned_study_group_is_final_even_when_it_has_more_samples():
    frame = pd.DataFrame([
        {"gsm": "GSM1", "gse": "", "score": 0.9},
        {"gsm": "GSM2", "gse": "", "score": 0.8},
        {"gsm": "GSM3", "gse": "GSE1", "score": 0.7},
    ])

    assert [group["gse"] for group in N.study_groups(N.build_payload(frame))] == [
        "GSE1", ""]
