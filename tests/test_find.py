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


# --- The shared record panel -------------------------------------------------

def test_an_archs4_sample_is_described_with_its_geo_record(corpus):
    record = find.describe(find.find("GSM9000005"))
    assert record["kind"] == "archs4"
    assert record["title"] == "GSM9000005"
    assert record["geo"] == "GSM9000005"
    assert not record["sample_key"], "a GEO sample offers no OSDR retrieval"
    assert dict(record["rows"])["Series"].startswith("GSE")


def test_an_osdr_sample_is_described_with_its_retrieval_key(corpus):
    key = str(data.osdr_metadata()["sample_key"].iloc[7])
    record = find.describe(find.find(key))
    assert record["kind"] == "osdr"
    assert record["title"] == key.split("|", 1)[1]
    assert record["sample_key"] == key
    assert not record["geo"], "an OSDR sample has no GEO record to open"
    assert "Study" in dict(record["rows"])


def test_a_set_reports_its_count_and_does_not_repeat_its_own_name(corpus):
    """"GSE5000 / Series: GSE5000" says it twice. The heading is the
    identifier, so the rows carry only what the heading does not."""
    record = find.describe(find.find(_a_real_series()))
    assert record["kind"] == "set"
    assert list(dict(record["rows"])) == ["Samples on the map"]
    assert record["title"] not in dict(record["rows"]).values()


def test_an_osdr_study_offers_no_geo_link(corpus):
    """A study is an OSDR concept and has no NCBI record; a link to one would
    404."""
    record = find.describe(find.find(str(data.osdr_metadata()["study"].iloc[0])))
    assert record["geo"] == ""
    assert record["sample_key"] == ""


def test_nothing_found_describes_nothing(corpus):
    assert find.describe(None) is None
    assert find.describe(find.find("liver")) is None


def test_an_external_record_is_an_anchor_and_a_retrieval_is_a_dash_link(corpus):
    """A dcc.Link pointing off-site hands its href to Dash's client-side
    router, which tries to resolve an NCBI URL as an application route instead
    of leaving the page."""
    from dash import dcc, html
    from manifold import layout

    external = layout.selected_panel_children(find.describe(find.find("GSM9000005")))
    anchors = [c for c in external if isinstance(c, html.A)]
    assert len(anchors) == 1
    assert anchors[0].href.startswith(find.GEO_ACC_URL)
    assert anchors[0].target == "_blank"
    assert not [c for c in external if isinstance(c, dcc.Link)]

    key = str(data.osdr_metadata()["sample_key"].iloc[0])
    internal = layout.selected_panel_children(find.describe(find.find(key)))
    links = [c for c in internal if isinstance(c, dcc.Link)]
    assert len(links) == 1
    assert links[0].href.startswith("/?q=")
    assert not [c for c in internal if isinstance(c, html.A)]


# --- Suggestions -------------------------------------------------------------
#
# The box completes an identifier while it is being typed. Everything below is
# about the one rule that keeps that from becoming the free-text search this
# module refused to build: a suggestion is a *prefix of a grammar*, never a
# guess at what a word might have meant.

def _labels(query, **kw):
    return [item["label"] for item in find.suggest(query, **kw)["items"]]


def test_a_partial_accession_is_completed(corpus):
    """The index holds numbers, not strings, so a prefix is a union of decade
    ranges rather than a string compare - and it has to actually work."""
    labels = _labels("GSM90000")
    assert labels, "a real accession prefix offered nothing"
    assert all(label.startswith("GSM90000") for label in labels), labels
    assert len(labels) == len(set(labels)), "a sample was offered twice"


def test_a_completed_prefix_offers_the_exact_match_first(corpus):
    """Typing a whole accession must put that accession at the top, not bury it
    under the longer numbers that happen to start with it."""
    assert _labels("GSM9000005")[0] == "GSM9000005"


def test_every_suggested_value_resolves_through_find(corpus):
    """The contract that keeps one resolution path: selecting a suggestion puts
    its `value` in the box, so a value `find()` cannot resolve would be a row
    that silently does nothing.
    """
    for query in ("GSM90000", "GSE500", "OSD-1", "SAMPLE_00"):
        items = find.suggest(query)["items"]
        assert items, f"{query} offered nothing to check"
        for item in items:
            got = find.find(item["value"])
            assert got["reason"] == "", f"{item['value']!r} does not resolve"
            assert got["points"], f"{item['value']!r} resolves to no points"


def test_a_series_suggestion_counts_its_samples(corpus):
    """The detail has to distinguish two adjacent series, and the only thing
    that does is how many samples each holds."""
    meta = data.archs4_metadata()
    series = _a_real_series()
    expected = int((meta["series_id"].astype(str) == series).sum())
    item = next(i for i in find.suggest(series)["items"] if i["label"] == series)
    assert f"{expected:,}" in item["detail"]


def test_an_osdr_sample_is_matched_anywhere_in_its_name(corpus):
    """A substring rather than a prefix, and deliberately: a sample name is
    itself an identifier, there are 2,108 of them rather than 940,455, and
    nobody remembers which end of one they are sure of."""
    key = str(data.osdr_metadata()["sample_key"].iloc[3])
    name = key.split("|", 1)[1]
    items = find.suggest(name[-6:])["items"]
    assert key in [i["value"] for i in items]
    chosen = next(i for i in items if i["value"] == key)
    assert chosen["label"] == name, "the row shows the name, not the 39-char key"
    assert chosen["value"] == key, "but commits the key, which is unambiguous"


def test_a_study_prefix_offers_studies_before_their_samples(corpus):
    """"OSD-100" is far more often the whole intent than a step towards one of
    its twelve samples, and the samples are one keystroke away."""
    items = find.suggest("OSD-1")["items"]
    kinds = [i["kind"] for i in items]
    assert "osd" in kinds
    assert kinds.index("osd") == 0
    assert "osdr_sample" not in kinds[:kinds.count("osd")]


def test_free_text_is_refused_here_exactly_as_it_is_in_find(corpus):
    """The decision this module was built around. `archs4_metadata.parquet`
    carries titles and a substring scan of them is affordable; offering it as a
    dropdown would reintroduce, as a feature, the search that was rejected -
    "liver" returning hundreds of rows and reading as a biological query."""
    for query in ("liver", "mouse brain", "spaceflight"):
        got = find.suggest(query)
        assert got["items"] == [], f"{query!r} was answered with suggestions"
        assert got["reason"] == find.SHAPE


def test_a_miss_says_which_kind_of_miss_it_was(corpus):
    """Answering "liver" and "GSM99999999" with the same empty list tells the
    first user their search is broken and the second nothing about whether this
    machine could have looked it up - the same distinction `find()` draws, drawn
    with the same constants."""
    assert find.suggest("")["reason"] == find.EMPTY
    assert find.suggest("   ")["reason"] == find.EMPTY
    assert find.suggest("GSM99999999")["reason"] == find.ABSENT
    assert find.suggest("OSD-99999")["reason"] == find.ABSENT
    assert find.suggest("liver")["reason"] == find.SHAPE


def test_one_letter_suggests_nothing(corpus):
    """A single character matches most of the corpus and completes nothing."""
    assert find.suggest("M")["items"] == []


def test_the_limit_is_honoured_and_a_zero_limit_offers_nothing(corpus):
    assert len(find.suggest("GSM9", limit=3)["items"]) == 3
    assert find.suggest("GSM9", limit=0)["items"] == []
    assert len(find.suggest("GSM9")["items"]) <= find.SUGGESTION_LIMIT


def test_geo_suggestions_say_so_when_the_join_is_missing(corpus,
                                                         without_archs4_metadata):
    """Reporting a GEO accession as absent on a machine that cannot look one up
    would be invariant 5 by another route - and it is the reason `searchable()`
    reads the availability predicate rather than re-deriving a path."""
    assert find.suggest("GSM90000")["reason"] == find.NO_GEO_METADATA
    assert find.suggest("GSE5000")["reason"] == find.NO_GEO_METADATA
    # OSDR needs only the label table, which the app cannot start without.
    assert find.suggest("OSD-1")["items"], "OSDR stopped being suggestible"


def test_the_empty_rows_of_the_real_join_do_not_break_a_prefix(corpus):
    """839 rows of the real join carry an empty series_id, and the fixture
    carries the same shape. `int(s[3:])` raises on one of those and would take
    the whole suggestion path down with it."""
    meta = data.archs4_metadata()
    assert (meta["series_id"].astype(str).str.strip() == "").any(), (
        "the fixture no longer carries the blank series ids this guards")
    assert _labels("GSE5")


# --- The rows the layout builds from them ------------------------------------

def test_a_suggestion_row_carries_its_own_identifier_in_its_id(corpus):
    """A click commits the row that was clicked. The alternative - an index into
    a parallel store - can be invalidated by the keystroke that lands between
    the click and the callback reading it."""
    from manifold import layout

    rows = layout.suggestion_children(find.suggest("GSM90000"))
    values = [r.id["value"] for r in rows]
    assert values == [i["value"] for i in find.suggest("GSM90000")["items"]]
    assert all(r.id["type"] == "find-suggestion" for r in rows)
    assert all(getattr(r, "role", None) == "option" for r in rows)


def test_a_well_formed_miss_gets_a_row_and_free_text_gets_silence(corpus):
    """A dropdown volunteering "no matches" under the word "liver" would
    contradict the sentence below the box, which says free text is not what this
    control takes."""
    from manifold import layout

    absent = layout.suggestion_children(find.suggest("GSM99999999"))
    assert len(absent) == 1
    assert absent[0].className == "bm-suggest-empty"
    assert getattr(absent[0], "role", None) != "option", (
        "the no-match row must not be selectable")

    assert layout.suggestion_children(find.suggest("liver")) == []
    assert layout.suggestion_children(find.suggest("")) == []
    assert layout.suggestion_children(None) == []
