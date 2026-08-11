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
