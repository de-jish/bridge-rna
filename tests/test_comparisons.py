"""Contracts for fixed comparison membership and exact paired evidence."""
import numpy as np
import pandas as pd
import pytest


def test_reviewed_comparisons_reject_incomplete_or_changed_metadata():
    from bridge_rna.reviewed import COMPARISONS, resolve_comparison

    preset = COMPARISONS[0]
    rows = []
    for arm, members in (("Space Flight", preset.flight), ("Ground Control", preset.ground)):
        rows.extend(dict(sample_key=k, study=preset.study, tissue=preset.tissue,
                         spaceflight=arm, strain=preset.strain, sex="Female",
                         duration="37 day", habitat="Rodent Habitat",
                         diet="Unknown", genotype="Unknown") for k in members)
    meta = pd.DataFrame(rows)
    a, b = resolve_comparison(preset.key, meta)
    assert a.members == preset.flight and b.members == preset.ground
    with pytest.raises(ValueError, match="missing"):
        resolve_comparison(preset.key, meta.iloc[:-1])
    changed = meta.copy()
    changed.loc[0, "strain"] = "Different strain"
    with pytest.raises(ValueError, match="strain"):
        resolve_comparison(preset.key, changed)
    with pytest.raises(ValueError, match="duplicate"):
        resolve_comparison(preset.key, pd.concat([meta, meta.iloc[:1]]))
    with pytest.raises(ValueError, match="not source-checked"):
        resolve_comparison("OSD-238", meta)


def test_exact_paired_ranks_include_matches_outside_other_prefix():
    from bridge_rna.comparison import paired_rankings

    x = np.array([[1., 0.], [.8, .6], [.6, .8], [0., 1.]], dtype=np.float32)
    result = paired_rankings(x, np.array([1., 0.]), np.array([0., 1.]), depth=1)
    rows = {r["archs4_index"]: r for r in result["rows"]}
    assert set(rows) == {0, 3}
    assert (rows[0]["rank_a"], rows[0]["rank_b"]) == (1, 4)
    assert (rows[3]["rank_a"], rows[3]["rank_b"]) == (4, 1)
    assert rows[0]["score_b"] == pytest.approx(0.)
    assert result["reference_count"] == 4


def test_paired_scores_match_cosine_and_break_ties_by_reference_index():
    from bridge_rna.comparison import paired_rankings

    rng = np.random.default_rng(42)
    x = rng.normal(size=(51, 8)).astype(np.float32)
    x[1] = x[0]
    q = rng.normal(size=(2, 8)).astype(np.float32)
    result = paired_rankings(x, *q, depth=51)
    expected = (x / np.linalg.norm(x, axis=1, keepdims=True)) @ (
        q / np.linalg.norm(q, axis=1, keepdims=True)).T
    ranks = np.argsort(np.argsort(-expected, axis=0, kind="stable"), axis=0) + 1
    for r in result["rows"]:
        i = r["archs4_index"]
        assert r["score_a"] == pytest.approx(float(expected[i, 0]), abs=2e-7)
        assert r["score_b"] == pytest.approx(float(expected[i, 1]), abs=2e-7)
        assert (r["rank_a"], r["rank_b"]) == tuple(ranks[i])
    rows = {r["archs4_index"]: r for r in result["rows"]}
    assert rows[0]["rank_a"] < rows[1]["rank_a"]


def test_invalid_vectors_cannot_become_ranked_evidence():
    from bridge_rna.comparison import paired_rankings

    with pytest.raises(ValueError):
        paired_rankings(np.eye(2), np.zeros(2), np.ones(2))
    with pytest.raises(ValueError):
        paired_rankings(np.array([[np.nan, 0.]]), np.ones(2), np.ones(2))


def test_identical_group_negative_control_and_swapped_labels():
    from bridge_rna.comparison import paired_rankings

    x = np.random.default_rng(8).normal(size=(100, 6)).astype(np.float32)
    original = x.copy()
    a, b = x[:2]
    same = paired_rankings(x, a, a, depth=30)
    np.testing.assert_array_equal(x, original)
    assert all(r["rank_a"] == r["rank_b"] and r["score_a"] == r["score_b"] for r in same["rows"])
    ab = paired_rankings(x, a, b, depth=30)
    ba = {r["archs4_index"]: r for r in paired_rankings(x, b, a, depth=30)["rows"]}
    for r in ab["rows"]:
        swapped = ba[r["archs4_index"]]
        assert (r["rank_a"], r["rank_b"]) == (swapped["rank_b"], swapped["rank_a"])
        assert (r["score_a"], r["score_b"]) == (swapped["score_b"], swapped["score_a"])


def test_evidence_depth_uses_union_and_groups_each_sample_once():
    from bridge_rna.comparison import evidence_groups

    rows = [dict(gsm="GSM1", gse="GSE1", rank_a=1, rank_b=100),
            dict(gsm="GSM2", gse="GSE1", rank_a=100, rank_b=1),
            dict(gsm="GSM3", gse="", rank_a=101, rank_b=101)]
    groups = evidence_groups(rows, 30)
    assert len(groups) == 1
    assert [r["gsm"] for r in groups[0]["rows"]] == ["GSM1", "GSM2"]
    assert len(evidence_groups(rows, 250)) == 2


def test_context_lookup_is_optional_and_refuses_wrong_identity(tmp_path):
    import sqlite3
    from bridge_rna.comparison import study_context

    rows = [dict(gsm="GSM1", archs4_index=0, gse="GSE1")]
    assert study_context(rows, None)["available"] is False
    path = tmp_path / "meta.sqlite"
    db = sqlite3.connect(path)
    db.executescript("CREATE TABLE samples(gsm TEXT, global_index INTEGER);"
                     "CREATE TABLE sample_studies(gsm TEXT,gse TEXT);"
                     "CREATE TABLE studies(gse TEXT,title TEXT,summary TEXT,overall_design TEXT);"
                     "INSERT INTO samples VALUES('GSM1',99);")
    db.close()
    result = study_context(rows, path)
    assert not result["available"] and "identity" in result["reason"].lower()
    assert not (tmp_path / "missing.sqlite").exists()
    assert not study_context(rows, tmp_path / "missing.sqlite")["available"]


def test_context_preserves_multiple_memberships_without_multiplying_samples(tmp_path):
    import sqlite3
    from bridge_rna.comparison import study_context

    p = tmp_path / "meta.sqlite"
    with sqlite3.connect(p) as db:
        db.executescript("CREATE TABLE samples(gsm TEXT, global_index INTEGER);"
                         "CREATE TABLE sample_studies(gsm TEXT,gse TEXT);"
                         "CREATE TABLE studies(gse TEXT,title TEXT,summary TEXT,overall_design TEXT);"
                         "INSERT INTO samples VALUES('GSM1',0);"
                         "INSERT INTO sample_studies VALUES('GSM1','GSE1'),('GSM1','GSE2');"
                         "INSERT INTO studies VALUES('GSE1','A','Text','Design'),('GSE2','B','','');")
    before = p.read_bytes()
    result = study_context([dict(gsm="GSM1", archs4_index=0, gse="GSE1")], p)
    assert result["available"]
    assert result["memberships"] == {"GSM1": ["GSE1", "GSE2"]}
    assert result["studies"]["GSE1"]["design"] == "Design"
    assert p.read_bytes() == before


def test_comparison_view_shows_both_ranks_and_source_labels():
    from bridge_rna.comparison_ui import evidence_children
    from dash.development.base_component import Component

    payload = {"paired_evidence": dict(reference_count=1000, rows=[dict(
        gsm="GSM1", gse="GSE1", title="Rds treated as control", rank_a=1,
        rank_b=900, score_a=.99, score_b=.98, source_name="retina",
        characteristics="genotype: Rds", tissue="Eye", species="Mus musculus")],
        context=dict(available=False, reason="Local study context is not configured.",
                     studies={}, memberships={})),
        "reviewed_comparison": "osd-194-retina"}
    children = evidence_children(payload, 30)
    def text(c):
        if isinstance(c, (list, tuple)):
            return " ".join(text(x) for x in c)
        return text(getattr(c, "children", "")) if isinstance(c, Component) else str(c)
    rendered = text(children)
    assert "900" in rendered and "0.990000" in rendered and "0.980000" in rendered
    assert "Rds treated as control" in rendered
    assert "independent" in rendered and "not configured" in rendered


def test_new_result_clears_stale_comparison_evidence():
    from bridge_rna.comparison_ui import evidence_children

    assert evidence_children(None, 30) == []
    assert evidence_children({"hits": []}, 30) == []


def test_evidence_pages_cover_every_candidate_exactly_once():
    from bridge_rna.comparison import evidence_page

    rows = [dict(gsm=f"GSM{i}", gse=f"GSE{i % 3}", rank_a=i+1, rank_b=1000+i) for i in range(61)]
    shown = []
    for page in range(3):
        groups, total = evidence_page(rows, 100, page)
        assert total == 61
        part = [r["gsm"] for g in groups for r in g["rows"]]
        assert len(part) == (30 if page < 2 else 1)
        shown.extend(part)
    assert len(set(shown)) == 61
    assert evidence_page(rows, 30, 0)[1] == 30


def test_comparison_remount_does_not_search_and_failure_clears_results(monkeypatch):
    import app
    from dash import no_update
    from bridge_rna import comparison_ui

    application = app.build_app()
    registered = next(v for v in application.callback_map.values()
                      if any(i["id"] == "comparison-search-button" for i in v["inputs"]))
    search = registered["callback"].__wrapped__
    def unavailable(*args):
        raise ValueError("Source-checked samples are missing")
    monkeypatch.setattr(comparison_ui, "run_documented_comparison", unavailable)
    assert search(0, "osd-194-retina", 5) == (no_update, no_update, no_update)
    figure, payload, status = search(1, "osd-194-retina", 5)
    assert payload is None
    assert "Comparison unavailable" in str(figure)
    assert "missing" in str(status)
