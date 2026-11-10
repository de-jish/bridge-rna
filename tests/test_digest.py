"""Shared terms: what a set of matches has in common, counted by series.

Pure tests over lists of dicts and a hand-made corpus table: no embedding, no
memmap, no Dash server. Design: docs/match-digest.md.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

from bridge_rna import digest as D
from bridge_rna import results as R
from bridge_rna import results_view as RV

ROOT = Path(__file__).resolve().parents[1]


def hit(gsm, gse, title="", source_name="", characteristics=""):
    return {"gsm": gsm, "gse": gse, "title": title, "source_name": source_name,
            "characteristics": characteristics, "score": 0.9}


def table(total=1000, **counts):
    return D.TermTable(counts, total)


# --- the tokenizer -----------------------------------------------------------


def test_words_lowercase_split_and_drop_labelling_noise():
    found = D.words("Mouse1 RETINA, RPE-choroid; Rep2 day14 P30 CTL1 cd4 IL-6 2019")
    assert {"retina", "rpe-choroid", "cd4", "il-6"} <= found
    # stopwords, a stopword with a counter, short tokens and bare numbers
    assert not found & {"mouse1", "rep2", "day14", "p30", "ctl1", "2019", "mouse"}


def test_words_tolerate_missing_values():
    assert D.words(None) == set()
    assert D.words(float("nan")) == set()
    assert D.words("") == set()


def test_words_come_only_from_the_fields_the_text_filter_searches():
    """Pressing a word must show the rows it was counted in, so the words come
    from the text filter's own free-text fields and nothing else."""
    assert set(D.TEXT_FIELDS) <= set(R._TEXT_SEARCH_FIELDS)
    row = hit("GSM1", "GSE1", characteristics="tissue: liver")
    assert D.sample_words(row) == set()


# --- counting ----------------------------------------------------------------


def test_a_series_counts_once_however_many_samples_it_deposited():
    hits = [hit(f"GSM{i}", "GSE1", title="liver hepatocyte") for i in range(50)]
    hits += [hit("GSM900", "GSE2", title="liver lobe"),
             hit("GSM901", "GSE3", title="kidney cortex")]
    d = D.shared_terms(hits, table(liver=10, hepatocyte=5, lobe=5, kidney=10, cortex=10))
    assert d["status"] == "ok"
    assert d["series_count"] == 3
    liver = next(t for t in d["terms"] if t["term"] == "liver")
    assert (liver["series"], liver["of"]) == (2, 3)
    # hepatocyte is in 50 samples but one series: not shared across series
    assert "hepatocyte" not in {t["term"] for t in d["terms"]}


def test_ratio_is_observed_series_over_expected_series():
    hits = [hit("A", "GSE1", title="spleen"), hit("B", "GSE2", title="spleen"),
            hit("C", "GSE3", title="thymus"), hit("D", "GSE4", title="liver")]
    d = D.shared_terms(hits, table(total=1000, spleen=20))
    spleen = d["terms"][0]
    assert spleen["expected"] == pytest.approx(4 * 20 / 1000)
    assert spleen["ratio"] == pytest.approx(2 / (4 * 20 / 1000))


def test_a_word_common_across_geo_is_not_listed():
    hits = [hit(str(i), f"GSE{i}", title="liver normal") for i in range(4)]
    # "normal" is used by 60% of all series: 4 of 4 is only 1.7x
    d = D.shared_terms(hits, table(total=1000, liver=20, normal=600))
    assert [t["term"] for t in d["terms"]] == ["liver"]


def test_order_is_series_count_then_ratio_and_capped():
    words = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf",
             "hotel", "india", "juliet", "kilo", "lima"]
    hits = []
    for s in range(6):
        # word i is in the first 2 + i % 5 series, so the series counts differ
        title = " ".join(w for i, w in enumerate(words) if s < 2 + i % 5)
        hits.append(hit(str(s), f"GSE{s}", title=title))
    d = D.shared_terms(hits, table(total=10_000, **{w: 2 for w in words}))
    series = [t["series"] for t in d["terms"]]
    assert series == sorted(series, reverse=True)
    assert len(d["terms"]) == D.MAX_TERMS


def test_a_match_without_a_series_is_its_own_series():
    hits = [hit("GSM1", "", title="retina"), hit("GSM2", "", title="retina")]
    d = D.shared_terms(hits, table(retina=3))
    assert d["series_count"] == 2
    assert d["terms"][0]["series"] == 2


@pytest.mark.parametrize("hits,tbl,status", [
    ([], table(), "empty"),
    ([hit("1", "GSE1", title="liver")], None, "unavailable"),
    ([hit("1", "GSE1", title="liver"), hit("2", "GSE1", title="liver")], table(liver=5),
     "one_series"),
    ([hit("1", "GSE1"), hit("2", "GSE2")], table(), "no_text"),
    ([hit("1", "GSE1", title="liver"), hit("2", "GSE2", title="kidney")],
     table(liver=5, kidney=5), "none"),
])
def test_every_empty_case_says_why(hits, tbl, status):
    d = D.shared_terms(hits, tbl)
    assert d["status"] == status
    assert d["terms"] == []
    if status != "empty":
        assert RV._digest_message(d)


def test_a_comparison_is_described_per_cohort():
    payload = {"hits": [hit("1", "GSE1", title="eye"), hit("2", "GSE2", title="eye")],
               "comparison": {"hits_b": [hit("3", "GSE3", title="lung"),
                                         hit("4", "GSE4", title="lung")]}}
    a, b = D.describe_all(payload, table(eye=3, lung=3))
    assert [t["term"] for t in a["terms"]] == ["eye"]
    assert [t["term"] for t in b["terms"]] == ["lung"]


# --- display -----------------------------------------------------------------


def test_ratio_display_has_no_false_precision():
    assert D.format_ratio(7.44) == "7.4x"
    assert D.format_ratio(18.6) == "19x"
    assert D.format_ratio(3475.0) == ">100x"
    assert D.format_ratio(3475.0, exact=True) == "3,475x"


def test_the_hover_text_states_every_number_behind_the_chip():
    t = {"term": "eye", "series": 8, "of": 22, "corpus_series": 1150,
         "expected": 0.49, "ratio": 16.4}
    text = D.term_title(t, 52_122)
    for part in ("8 of their 22 series", "52,122", "1,150", "0.49", "16x"):
        assert part in text


def test_the_button_name_reads_as_words():
    t = {"term": "rpe", "series": 7, "of": 23, "ratio": 222.0}
    assert D.term_label(t) == ("rpe: 7 of 23 series, more than 100 times its GEO "
                               "rate. Filter the table to it.")
    t["ratio"] = 18.6
    assert "19 times its GEO rate" in D.term_label(t)


def _flatten(node):
    out = [node]
    children = getattr(node, "children", None)
    if children is None or isinstance(children, str):
        return out
    for child in children if isinstance(children, (list, tuple)) else [children]:
        if not isinstance(child, (str, int, float)):
            out.extend(_flatten(child))
    return out


def test_the_view_renders_one_button_per_word_and_marks_the_active_one():
    payload = {"hits": [hit("1", "GSE1", title="retina rpe"),
                        hit("2", "GSE2", title="retina rpe"),
                        hit("3", "GSE3", title="kidney")]}
    stored = RV.describe_payload(payload, table(retina=10, rpe=4, kidney=50))
    children, style = RV.render_digest(stored, "RPE")
    assert style == {}
    buttons = [n for c in children for n in _flatten(c)
               if isinstance(getattr(n, "id", None), dict)]
    assert [b.id["term"] for b in buttons] == ["rpe", "retina"]
    pressed = {b.id["term"]: getattr(b, "aria-pressed") for b in buttons}
    assert pressed == {"rpe": "true", "retina": "false"}


def test_the_view_hides_without_matches():
    assert RV.describe_payload(None) is None
    assert RV.render_digest(None) == ([], {"display": "none"})


# --- the corpus table and its build ------------------------------------------


def _metadata():
    return pd.DataFrame({
        "geo_accession": ["GSM1", "GSM2", "GSM3", "GSM4", "GSM5"],
        "series_id": ["GSE1", "GSE1", "GSE2", "GSE3", None],
        "title": ["liver rep1", "liver rep2", "liver", "kidney", "liver"],
        "source_name": ["", "", "hepatocyte", "", ""],
    })


def test_the_corpus_table_counts_series_and_records_its_total():
    frame = D.build_term_table(_metadata())
    tbl = D.table_from_frame(frame)
    # GSE1, GSE2, GSE3 and the series-less GSM5
    assert tbl.total_series == 4
    assert tbl.counts == {"liver": 3}  # kidney and hepatocyte are in one series each


def test_the_build_script_uses_the_apps_tokenizer_and_validates_itself():
    spec = importlib.util.spec_from_file_location(
        "build_term_index", ROOT / "precompute" / "build_term_index.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.digest is D
    meta = _metadata()
    assert mod.check(D.build_term_table(meta), meta) == []
    broken = D.build_term_table(meta)
    broken.loc[broken["term"] == "__series__", "series"] = 99
    assert mod.check(broken, meta)


def test_a_missing_or_foreign_artifact_reads_as_unavailable(tmp_path, monkeypatch):
    from manifold import paths
    monkeypatch.setattr(paths, "ARCHS4_TERMS_PARQUET", tmp_path / "missing.parquet")
    D.load_term_table.cache_clear()
    try:
        assert D.load_term_table() is None
        bad = tmp_path / "bad.parquet"
        pd.DataFrame({"x": [1]}).to_parquet(bad)
        monkeypatch.setattr(paths, "ARCHS4_TERMS_PARQUET", bad)
        D.load_term_table.cache_clear()
        assert D.load_term_table() is None
    finally:
        D.load_term_table.cache_clear()


def test_the_ai_prompt_receives_the_words_labelled_as_metadata():
    template = (ROOT / "prompts" / "ai_summary_prompt.txt").read_text()
    assert "{shared_terms}" in template
    assert "not the samples' biology" in template
    hits = [hit("1", "GSE1", title="spleen"), hit("2", "GSE2", title="spleen"),
            hit("3", "GSE3", title="liver")]
    text = D.prompt_text(D.shared_terms(hits, table(spleen=10, liver=10)))
    assert text.startswith("- spleen: used by 2 of the hits' 3 series")
    assert D.prompt_text(D.shared_terms(hits, None)).startswith(
        "The GEO word counts are not built")
    # every placeholder the template names is one the callback fills
    import re
    named = set(re.findall(r"\{([a-z_]+)\}", template))
    assert named == {"osdr_metadata", "retrieved_hits_table", "geo_summaries",
                     "shared_terms"}
    assert "used by 2" in template.format(**{k: "" for k in named} | {"shared_terms": text})


def test_the_help_entry_exists_for_the_row():
    from bridge_rna import help as H
    assert "shared-terms" in H.ENTRIES
