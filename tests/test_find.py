"""Resolving an identifier to the points it names on the map.

Every test runs against the synthetic corpus in `tests/fixture_corpus.py`, so
none of this needs the 963 MB memmap or the real cache - `manifold/find.py`
opens no embedding and builds no figure, which is the point of it being its own
module.
"""

from __future__ import annotations

import numpy as np
import pytest

from manifold import data, find


@pytest.fixture(scope="module")
def built_app():
    """The whole shell, for the tests that assert on the callback graph."""
    import app as shell
    return shell.build_app()


@pytest.fixture(autouse=True)
def _clear_index():
    """The index is memoized, and the metadata fixture swaps the parquet out."""
    find.clear_caches()
    yield
    find.clear_caches()


# --- The four grammars -------------------------------------------------------

def test_a_gsm_accession_resolves_to_its_own_point(corpus):
    """An ARCHS4 accession's memmap row *is* its point index - the join the
    whole app is built on - so this is an identity check, not a lookup that
    could be off by one."""
    for row in (0, 1, 999, corpus["n_archs4"] - 1):
        got = find.find(f"GSM{9000000 + row}")
        assert got["reason"] == ""
        assert got["points"] == [row]
        assert got["kind"] == "gsm"


def _a_real_series() -> str:
    """A series id that is actually populated.

    Not `.iloc[0]`: the fixture blanks every 53rd series_id to carry the real
    join's 839 unresolved rows, and row 0 is one of them.
    """
    ids = data.archs4_metadata()["series_id"].astype(str)
    return str(ids[ids.str.strip() != ""].iloc[0])


def test_a_series_resolves_to_every_sample_in_it(corpus):
    meta = data.archs4_metadata()
    series = _a_real_series()
    expected = sorted(np.flatnonzero(meta["series_id"].to_numpy() == series).tolist())
    got = find.find(series)
    assert got["reason"] == ""
    assert got["kind"] == "gse"
    assert got["points"] == expected
    assert len(got["points"]) > 1, "fixture should have multi-sample series"


def test_an_osdr_study_resolves_to_every_sample_in_it(corpus):
    osdr = data.osdr_metadata()
    n_archs4 = corpus["n_archs4"]
    study = str(osdr["study"].iloc[0])
    expected = sorted((n_archs4 + np.flatnonzero(
        osdr["study"].to_numpy() == study)).tolist())
    got = find.find(study)
    assert got["reason"] == ""
    assert got["kind"] == "osd"
    assert got["points"] == expected


def test_an_osdr_sample_resolves_by_full_key_and_by_bare_name(corpus):
    """A user copying from the retrieval view has the full `<study>|<name>`
    key; a user reading a figure caption has only the name."""
    osdr = data.osdr_metadata()
    key = str(osdr["sample_key"].iloc[7])
    name = key.split("|", 1)[1]
    expected = [corpus["n_archs4"] + 7]
    for query in (key, name):
        got = find.find(query)
        assert got["reason"] == "", query
        assert got["points"] == expected, query
        assert got["kind"] == "osdr_sample"


# --- Normalization -----------------------------------------------------------

def test_matching_is_case_insensitive_and_trims_whitespace(corpus):
    for query in ("  GSM9000005  ", "gsm9000005", "Gsm9000005", "\tGSM9000005\n"):
        got = find.find(query)
        assert got["points"] == [5], query


def test_a_study_resolves_with_or_without_its_hyphen(corpus):
    """OSD-100 and OSD100 are the same study to everyone except a string
    compare, and a user typing the second should not be told it does not
    exist."""
    study = str(data.osdr_metadata()["study"].iloc[0])
    plain = study.replace("-", "")
    assert find.find(plain)["points"] == find.find(study)["points"]
    assert find.find(plain)["reason"] == ""


def test_the_label_is_the_canonical_identifier_not_what_was_typed(corpus):
    got = find.find("  gsm9000005 ")
    assert got["label"] == "GSM9000005"


# --- Misses, and telling them apart ------------------------------------------

def test_a_well_formed_accession_that_is_absent_says_so(corpus):
    for query in ("GSM123456789", "GSE999999", "OSD-99999"):
        got = find.find(query)
        assert got["points"] == [], query
        assert got["reason"] == "absent", query


def test_free_text_is_refused_as_a_shape_not_reported_as_absent(corpus):
    """"liver" is the Tissue color-by's question, and the two failures need
    different answers: one tells you to zoom out your expectations, the other
    tells you to use a different control."""
    for query in ("liver", "spaceflight", "mouse brain"):
        got = find.find(query)
        assert got["points"] == [], query
        assert got["reason"] == "shape", query


def test_a_title_substring_that_really_exists_still_resolves_to_nothing(corpus):
    """The fixture's titles are literally "sample 12". Matching one would mean
    the box had quietly become a text search over 940,455 GEO records."""
    titles = data.archs4_metadata()["title"].astype(str)
    assert (titles == "sample 12").any(), "fixture changed; pick another title"
    assert find.find("sample 12")["points"] == []


def test_an_empty_query_is_neither_a_hit_nor_an_error(corpus):
    for query in ("", "   ", None):
        got = find.find(query)
        assert got["points"] == []
        assert got["reason"] == "empty"


# --- The malformed rows in the real metadata ---------------------------------

def test_a_blank_series_id_never_becomes_a_findable_series(corpus):
    """839 rows of the real join carry an empty `series_id` - the samples the
    metadata fetch could not resolve. Parsing the digits off one crashes the
    index build, and coercing it to a number would file those samples under a
    series that does not exist."""
    meta = data.archs4_metadata()
    blank = np.flatnonzero(meta["series_id"].astype(str).str.strip() == "")
    assert len(blank) > 0, "fixture should carry at least one unresolved row"

    # The index builds at all, which is the crash this pins.
    assert find.find("GSE5000")["reason"] == ""
    # And no query reaches those rows through the series path.
    for got in (find.find(""), find.find("GSE"), find.find("GSE0")):
        assert not set(got["points"]) & set(blank.tolist())


def test_every_returned_index_is_a_real_point(corpus):
    total = corpus["total"]
    n_archs4 = corpus["n_archs4"]
    meta = data.archs4_metadata()
    queries = ["GSM9000000", _a_real_series(),
               str(data.osdr_metadata()["study"].iloc[0]),
               str(data.osdr_metadata()["sample_key"].iloc[0])]
    for query in queries:
        got = find.find(query)
        assert got["points"], query
        assert all(0 <= p < total for p in got["points"]), query
        if got["kind"] in ("gsm", "gse"):
            assert all(p < n_archs4 for p in got["points"]), query
        else:
            assert all(p >= n_archs4 for p in got["points"]), query


def test_the_points_come_back_sorted(corpus):
    """The renderer indexes coordinates with them and the frame takes their
    bounding box; both are order-independent, but a stable order keeps the
    marks' draw order stable between identical searches."""
    points = find.find(_a_real_series())["points"]
    assert points == sorted(points)


# --- Coverage: the optional metadata join ------------------------------------

def test_without_the_geo_join_a_gsm_says_what_to_run(corpus, without_archs4_metadata):
    """`cache/archs4_metadata.parquet` is optional, and without it 940,455 of
    the 942,563 points cannot be addressed at all. Reporting that as "absent"
    would tell the user their accession does not exist when the truth is that
    this machine cannot look it up."""
    got = find.find("GSM9000005")
    assert got["points"] == []
    assert got["reason"] == "no_geo_metadata"


def test_without_the_geo_join_osdr_is_still_findable(corpus,
                                                     without_archs4_metadata):
    """The OSDR half needs no join, so it must keep working - degrading the
    whole control because half of it is unavailable would be the failure the
    coverage system exists to prevent."""
    osdr = data.osdr_metadata()
    got = find.find(str(osdr["sample_key"].iloc[3]))
    assert got["reason"] == ""
    assert got["points"] == [corpus["n_archs4"] + 3]


def test_coverage_reports_which_corpora_are_searchable(corpus):
    assert find.searchable() == ("osdr", "archs4")


def test_coverage_drops_archs4_when_the_join_is_missing(corpus,
                                                        without_archs4_metadata):
    assert find.searchable() == ("osdr",)


# --- The index itself --------------------------------------------------------

def test_the_index_is_built_once_and_not_at_import(corpus):
    find.clear_caches()
    assert find._archs4_index.cache_info().currsize == 0
    find.find("GSM9000005")
    find.find("GSM9000006")
    assert find._archs4_index.cache_info().currsize == 1
    assert find._archs4_index.cache_info().hits >= 1


def test_the_index_keys_are_narrow_enough_to_be_cheap(corpus):
    """Measured on the real corpus: int32 keys and int32 rows are 15.0 MB,
    against 96.8 MB for the pandas Index this replaced, on an app whose whole
    working set is 80.8 MB."""
    idx = find._archs4_index()
    for arr in (idx.gsm_keys, idx.gsm_rows, idx.gse_keys, idx.gse_rows):
        assert arr.dtype == np.int32, arr.dtype


def test_a_well_formed_sample_key_the_map_lacks_is_absent_not_a_bad_shape(corpus):
    """788 of the 2,896 OSDR samples the retrieval catalog lists were never
    embedded and have no position here. Answering one of them with "that is not
    an identifier, try the Tissue color-by" would be wrong twice over."""
    for query in ("OSD-141|Mmus_C57-6J_SPL_cells_Rep1_SP1",
                  "Mmus_C57-6J_SPL_cells_Rep1_SP1"):
        got = find.find(query)
        assert got["points"] == [], query
        assert got["reason"] == "absent", query


# --- The mark on the plot, and its key row -----------------------------------

def test_the_found_symbol_is_valid_in_three_dimensions():
    """Scatter3d takes eight symbols and rejects the rest outright rather than
    degrading - the failure that took the whole figure callback down with a 500
    the first time 3-D was opened with a retrieval showing. `x-open` is one of
    the ones it refuses, which is exactly why there are two spellings."""
    import plotly.graph_objects as go
    from manifold import theme

    go.Scatter3d(marker=dict(symbol=theme.FOUND_SYMBOL_3D))   # must not raise
    go.Scattergl(marker=dict(symbol=theme.FOUND_SYMBOL))
    with pytest.raises(ValueError):
        go.Scatter3d(marker=dict(symbol=theme.FOUND_SYMBOL))


def _found_traces_in(fig):
    return [t for t in fig.data if getattr(t, "name", None) == "found"]


def test_a_found_set_is_drawn_and_badged(corpus):
    from manifold import render
    found = find.find("GSM9000005")
    fig, _, badges = render.build_figure(
        "pca", "2d", "tissue", ["archs4", "osdr"], 1500, None, found=found)
    traces = _found_traces_in(fig)
    assert len(traces) == 1
    assert len(traces[0].x) == 1
    assert any("GSM9000005" in b for b in badges)


def test_nothing_is_drawn_or_badged_without_a_find(corpus):
    from manifold import render
    for found in (None, {"points": [], "label": "GSE999"}):
        fig, _, badges = render.build_figure(
            "pca", "2d", "tissue", ["archs4", "osdr"], 1500, None, found=found)
        assert _found_traces_in(fig) == []
        assert not any("Found" in b for b in badges)


def test_the_marks_are_capped_and_the_cap_says_what_it_dropped(corpus,
                                                               monkeypatch):
    """A silent cap reads as "this is all of them". GSE228590 is 8,764 samples
    on the real corpus, and 305 series carry more than 200."""
    from manifold import render, theme

    monkeypatch.setattr(theme, "FIND_MAX_MARKS", 10)
    found = {"points": list(range(200)), "label": "GSE5000"}
    fig, _, badges = render.build_figure(
        "pca", "2d", "tissue", ["archs4", "osdr"], 1500, None, found=found)
    assert len(_found_traces_in(fig)[0].x) == 10
    badge = next(b for b in badges if "GSE5000" in b)
    assert "10" in badge and "200" in badge, badge


def test_the_key_row_counts_what_is_drawn_not_what_exists(corpus, monkeypatch):
    """The key's standing rule: a count is read as "how many am I looking at"."""
    from manifold import layout, theme

    monkeypatch.setattr(theme, "FIND_MAX_MARKS", 10)
    rows = layout.found_key_children({"points": list(range(200)),
                                      "label": "GSE5000"})
    text = str(rows)
    assert "GSE5000" in text
    assert "10" in text and "200" not in text


def test_a_find_with_no_points_gets_no_key_row(corpus):
    from manifold import layout
    assert layout.found_key_children(None) == []
    assert layout.found_key_children({"points": [], "label": "x"}) == []


def test_the_found_layer_is_webgl_in_two_dimensions(corpus):
    """`_retrieval_traces` uses the non-gl Scatter deliberately, for at most
    k+2 points that need markers+text centred. A series is up to 8,764 marks
    and draws no text, so it must not inherit that choice."""
    import plotly.graph_objects as go
    from manifold import render

    found = {"points": list(range(300)), "label": "GSE5000"}
    fig, _, _ = render.build_figure(
        "pca", "2d", "tissue", ["archs4", "osdr"], 1500, None, found=found)
    assert isinstance(_found_traces_in(fig)[0], go.Scattergl)

    fig3, _, _ = render.build_figure(
        "pca", "3d", "tissue", ["archs4", "osdr"], 1500, None, found=found)
    assert isinstance(_found_traces_in(fig3)[0], go.Scatter3d)


def test_a_found_point_outside_the_corpus_is_dropped_not_drawn(corpus):
    from manifold import render
    found = {"points": [0, corpus["total"] + 5, -1], "label": "GSE5000"}
    fig, _, _ = render.build_figure(
        "pca", "2d", "tissue", ["archs4", "osdr"], 1500, None, found=found)
    assert len(_found_traces_in(fig)[0].x) == 1


def test_several_marks_are_drawn_smaller_than_a_lone_one(corpus):
    """A study's samples are often nearly coincident - OSD-100's twelve frame
    into 1.08 units of x on the real corpus - so at full size they composite
    into one blot. Same rule and same 0.7 as a pooled cohort's members."""
    from manifold import render, theme

    one, _, _ = render._found_traces(
        data.coords("pca", "2d"), False, {"points": [1], "label": "GSM1"})
    many, _, _ = render._found_traces(
        data.coords("pca", "2d"), False, {"points": [1, 2, 3], "label": "GSE1"})
    assert one[0].marker.size == theme.FOUND_SIZE
    assert many[0].marker.size == pytest.approx(theme.FOUND_SIZE * 0.7)


def test_the_status_says_something_different_for_each_kind_of_miss(corpus):
    """Answering "liver" and "GSM999999999" with one sentence tells the first
    user their search is broken and the second nothing about whether this
    machine could have looked it up."""
    from manifold import callbacks

    def text(query):
        return str(callbacks.find_status_children(find.find(query)))

    shape = text("liver")
    absent = text("GSM123456789")
    assert "Color by" in shape and "not on this map" not in shape
    assert "not on this map" in absent and "Color by" not in absent
    assert shape != absent
    assert callbacks.find_status_children(find.find(""))  == ""
    assert callbacks.find_status_children(None) == ""


def test_the_no_metadata_status_names_the_command_the_color_by_names(
        corpus, without_archs4_metadata):
    """One sentence for the one missing artifact. Two controls depend on that
    optional join and they must not send the user after two commands."""
    from manifold import callbacks, colorby

    text = str(callbacks.find_status_children(find.find("GSM9000005")))
    assert colorby.ARCHS4_META_HINT in text


def test_the_status_states_the_cap_as_well_as_the_badge(corpus, monkeypatch):
    from manifold import callbacks, theme

    monkeypatch.setattr(theme, "FIND_MAX_MARKS", 10)
    text = str(callbacks.find_status_children(
        {"points": list(range(200)), "label": "GSE5000", "reason": ""}))
    assert "200" in text and "10" in text


def test_finding_something_never_moves_the_viewport_by_itself(built_app):
    """Framing is a button and never a consequence of typing, and there are two
    reasons - the second stronger than the first.

    A found set can span the map: an OSDR study's 192 samples framed to 1.22x
    the corpus width before `_clamped_to_corpus` went in, so an automatic frame
    would zoom the user *out* as the result of a search. And a 2-D neighbourhood
    is not a similarity neighbourhood here - the map's 20 nearest points overlap
    the true cosine top-20 by a median of 0 - so dropping someone into a zoomed
    view of their sample's surroundings invites reading those surroundings as
    related when they are not. `_frame_for`'s docstring already made this call
    for the retrieval; this pins that the find did not quietly reverse it.

    Structural rather than behavioural on purpose: `find-store` being an Input
    of the viewport callback is the thing that would make framing automatic, so
    the test looks for that rather than for a symptom.
    """
    app = built_app
    viewport = [k for k in app.callback_map if "viewport-store" in k]
    assert len(viewport) == 1
    inputs = [i["id"] for i in app.callback_map[viewport[0]]["inputs"]]
    assert "frame-find" in inputs, "the frame button must reach the viewport"
    assert "find-store" not in inputs, (
        "find-store is an Input of the viewport callback, so a search now "
        "moves the map on its own")


def test_the_find_store_is_declared_in_the_view_not_only_as_output():
    """A component that exists only as callback output cannot be validated by
    Dash at startup: a typo in its id fails silently at runtime instead."""
    from manifold import layout

    def walk(node):
        yield node
        children = getattr(node, "children", None)
        if isinstance(children, (list, tuple)):
            for c in children:
                yield from walk(c)
        elif children is not None:
            yield from walk(children)

    ids = {getattr(n, "id", None) for n in walk(layout.build_view())}
    assert "find-store" in ids
    assert "find-input" in ids
    assert "frame-find" in ids
