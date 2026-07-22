"""The retrieval half, and the join that lets it meet the map.

These run against the same synthetic corpus as the manifold tests: the fixture
writes a real float16 memmap, a real `sample_locations.parquet`, and a real
`osdr_sample_embeddings.float32.npy`, so the cached path can be exercised end to
end without the 963 MB artifact or the multi-hour embedding job.

The contract under test is the one the whole merged app rests on: an OSDR
sample has the *same key* on both sides, and an ARCHS4 hit's memmap row is the
*same integer* as its manifold point. Neither is enforced by a schema anywhere,
so both are enforced here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bridge_rna import osdr, retrieval


@pytest.fixture(autouse=True)
def _point_retrieval_at_the_fixture(monkeypatch, corpus):
    """Aim the ARCHS4 loaders at the synthetic stub instead of the real repo.

    `bridge_rna.config` resolves paths from `__file__`, so without this the
    module-level EMBEDDING_DIR points at the real 963 MB memmap. The lru_caches
    downstream have to be cleared on both sides, or a cached real handle would
    leak into a test and a cached fixture handle would leak out of one.
    """
    retrieval._cached_osdr_embeddings.cache_clear()
    retrieval._archs4_annotations.cache_clear()
    retrieval._ARCHS4_CACHE.clear()
    monkeypatch.setattr(
        retrieval, "EMBEDDING_DIR",
        corpus["bridge_rna_root"] / "archs4_sample_embeddings_full")
    yield
    retrieval._cached_osdr_embeddings.cache_clear()
    retrieval._archs4_annotations.cache_clear()
    retrieval._ARCHS4_CACHE.clear()


def _samples_frame(corpus) -> pd.DataFrame:
    """The retrieval-side sample table, built from the fixture's OSDR keys.

    Deliberately reconstructed by splitting `sample_key` rather than copied, so
    the test fails if the two halves ever disagree on how the key is formed.
    """
    keys = corpus["osdr_metadata"]["sample_key"].astype(str)
    study, name = zip(*(k.split("|", 1) for k in keys))
    return pd.DataFrame({
        "sample_id": keys, "study_id": list(study), "sample_name": list(name),
        "tissue": corpus["osdr_metadata"]["tissue"].to_numpy(),
        "condition": corpus["osdr_metadata"]["spaceflight"].to_numpy(),
    })


# --- The key contract -------------------------------------------------------

def test_osdr_sample_id_is_built_the_same_way_on_both_sides(tmp_path):
    """`load_osdr_samples` must produce the key `embed_osdr.py` writes.

    Both build "<accession>|<sample name>". If either side ever changes, a
    retrieval and a point on the map stop referring to the same sample, and
    nothing would raise - the fast path would simply never find a vector and
    every query would silently fall back to the 22-second subprocess.
    """
    tsv = tmp_path / "meta.tsv"
    tsv.write_text(
        "id.accession\tid.sample name\tstudy.characteristics.material type\n"
        "OSD-100\tMmus_C57-6J_EYE_FLT_Rep1_M23\tleft eye\n"
        "OSD-104\tMmus_BAL_LVR_GC_Rep2\tliver\n"
    )
    df = osdr.load_osdr_samples(tsv)
    assert df["sample_id"].tolist() == [
        "OSD-100|Mmus_C57-6J_EYE_FLT_Rep1_M23",
        "OSD-104|Mmus_BAL_LVR_GC_Rep2",
    ]


def test_every_fixture_osdr_key_resolves_to_a_query_vector(corpus):
    n, usable = retrieval.cached_query_coverage()
    assert usable and n == corpus["n_osdr"]
    for key in corpus["osdr_metadata"]["sample_key"].astype(str):
        assert retrieval.cached_query_vector(key) is not None


def test_an_unknown_sample_has_no_cached_vector():
    assert retrieval.cached_query_vector("OSD-999|not-a-real-sample") is None


# --- The cached path --------------------------------------------------------

def test_cached_retrieval_reproduces_a_brute_force_cosine_ranking(corpus):
    """The fast path must return the true top-k, not an approximation of it.

    Scored against a dense cosine over the whole fixture corpus computed here
    in float64, which is the definition the app claims to implement.
    """
    key = str(corpus["osdr_metadata"]["sample_key"].iloc[3])
    hits = retrieval.run_cached_query_retrieval(key, topk=10)

    q = retrieval.cached_query_vector(key).astype(np.float64)
    q /= np.linalg.norm(q)
    vecs, _, _ = retrieval._load_archs4_index()
    x = np.asarray(vecs, dtype=np.float64)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    truth = np.argsort(-(x @ q))[:10]

    assert hits["archs4_index"].tolist() == truth.tolist()
    assert hits["score"].is_monotonic_decreasing


def test_cached_hits_carry_the_geo_annotation_the_slow_path_leaves_empty(corpus):
    key = str(corpus["osdr_metadata"]["sample_key"].iloc[0])
    hits = retrieval.run_cached_query_retrieval(key, topk=5)
    for col in ("gsm", "gse", "title", "source_name", "tissue", "species"):
        assert col in hits.columns
    assert hits["gsm"].str.startswith("GSM").all()
    assert (hits["tissue"].str.len() > 0).all()
    assert hits["species"].isin(["Homo sapiens", "Mus musculus"]).all()


def test_the_hit_index_addresses_the_same_point_on_the_map(corpus):
    """`archs4_index` must be both the memmap row and the manifold point index.

    This is the join the whole integration rests on: the retrieval returns a row
    of the embedding memmap, and the map addresses that same sample at that same
    offset because ARCHS4 occupies rows 0..n_archs4-1 of the global point order.
    """
    from manifold import data as mdata

    key = str(corpus["osdr_metadata"]["sample_key"].iloc[1])
    hits = retrieval.run_cached_query_retrieval(key, topk=5)

    meta = mdata.points_meta()
    geo = pd.read_parquet(corpus["cache_dir"] / "archs4_geo.parquet")
    for _, hit in hits.iterrows():
        point = int(hit["archs4_index"])
        assert meta["dataset"].iloc[point] == 0, "hit must land on an ARCHS4 point"
        assert int(meta["src_index"].iloc[point]) == point
        assert geo["geo_accession"].iloc[point] == hit["gsm"]


def test_search_hits_reports_the_cached_mode(corpus):
    df = _samples_frame(corpus)
    hits, mode = retrieval.search_hits(
        df, str(df["sample_id"].iloc[2]), topk=4,
        enable_biopython_metadata=False)
    assert mode == "cached"
    assert len(hits) == 4


def test_search_hits_rejects_an_unknown_sample(corpus):
    with pytest.raises(ValueError, match="Unknown sample_id"):
        retrieval.search_hits(_samples_frame(corpus), "nope", topk=3)


# --- The guards -------------------------------------------------------------

def test_a_positional_length_mismatch_refuses_the_cached_path(monkeypatch, corpus,
                                                              capsys):
    """Embeddings and keys are joined by position, so a mismatch is silent.

    Truncating the key table would otherwise attribute every query vector to the
    wrong sample and still return a confident, well-formed answer. The path has
    to refuse rather than degrade.
    """
    from manifold import paths as mpaths

    short = corpus["cache_dir"] / "short_osdr_metadata.parquet"
    pd.read_parquet(mpaths.OSDR_METADATA_PARQUET).head(5).to_parquet(short,
                                                                     index=False)
    retrieval._cached_osdr_embeddings.cache_clear()
    monkeypatch.setattr(mpaths, "OSDR_METADATA_PARQUET", short)

    assert retrieval._cached_osdr_embeddings() is None
    assert "joined positionally" in capsys.readouterr().err
    retrieval._cached_osdr_embeddings.cache_clear()


def test_a_missing_cache_falls_through_instead_of_raising(monkeypatch):
    from manifold import paths as mpaths

    retrieval._cached_osdr_embeddings.cache_clear()
    monkeypatch.setattr(mpaths, "OSDR_EMBEDDINGS_NPY",
                        mpaths.CACHE_DIR / "does-not-exist.npy")
    assert retrieval._cached_osdr_embeddings() is None
    assert retrieval.cached_query_coverage() == (0, False)
    retrieval._cached_osdr_embeddings.cache_clear()


def test_missing_geo_metadata_yields_blanks_rather_than_the_string_nan(monkeypatch,
                                                                      corpus):
    """pandas 3.0 leaves NA through `astype(str)`.

    A literal "nan" in a GSE column is read downstream as a real accession and
    linked to on GEO. Blank is the honest rendering of a field GEO never filled.
    """
    from manifold import data as mdata

    df = mdata.archs4_metadata().copy()
    df.loc[: len(df) // 2, "series_id"] = None
    df.loc[: len(df) // 2, "title"] = None
    # Patch the source rather than the memoized reader, so the lru_cache stays a
    # real lru_cache for the autouse teardown to clear.
    retrieval._archs4_annotations.cache_clear()
    monkeypatch.setattr(mdata, "archs4_metadata", lambda: df)

    key = str(corpus["osdr_metadata"]["sample_key"].iloc[0])
    hits = retrieval.run_cached_query_retrieval(key, topk=20)
    for col in ("gse", "title"):
        assert not hits[col].isin(["nan", "None", "<NA>"]).any()
    assert (hits["gse"] == "").any(), "the blanked rows must actually be reached"


def test_a_sample_with_no_counts_column_is_unavailable_not_slow(tmp_path, corpus):
    """The third tier is the one that was missed, and it is the one that matters.

    A sample whose name appears in no column of its own study's counts matrix
    cannot be answered by any path: the cached vector does not exist and
    `demo_osdr_top5.py` raises. Calling that "slow" would send someone to wait
    22 seconds for a guaranteed failure.
    """
    counts = tmp_path / "counts.csv"
    counts.write_text("gene,SAMPLE_PRESENT,SAMPLE_OTHER\nActb,5,7\n")
    flew = "Space Flight"

    assert retrieval.sample_tier(
        "OSD-999|SAMPLE_PRESENT", "SAMPLE_PRESENT", str(counts), flew
    ) == retrieval.TIER_SUBPROCESS
    assert retrieval.sample_tier(
        "OSD-999|SAMPLE_ABSENT", "SAMPLE_ABSENT", str(counts), flew
    ) == retrieval.TIER_UNAVAILABLE
    # No counts file recorded at all is equally unanswerable.
    assert retrieval.sample_tier(
        "OSD-999|SAMPLE_PRESENT", "SAMPLE_PRESENT", "", flew
    ) == retrieval.TIER_UNAVAILABLE


@pytest.mark.parametrize("condition", ["", "   ", "nan", "None", "NA", "n/a"])
def test_no_spaceflight_value_means_unavailable_not_slow(tmp_path, condition):
    """The filter that the first version of `sample_tier` missed.

    `demo_osdr_top5.py` drops rows with no recorded spaceflight value *before*
    it looks for the requested sample name, so such a sample raises "not found
    after filtering" rather than being slow. Classifying it as `subprocess`
    told the user to wait 22 seconds for a guaranteed failure - and 733 of the
    788 unavailable samples fail for exactly this reason.
    """
    counts = tmp_path / "counts.csv"
    counts.write_text("gene,SAMPLE_PRESENT\nActb,5\n")
    assert retrieval.sample_tier(
        "OSD-999|SAMPLE_PRESENT", "SAMPLE_PRESENT", str(counts), condition
    ) == retrieval.TIER_UNAVAILABLE


def test_a_cached_sample_is_cached_whatever_else_is_missing(corpus):
    """The cached vector wins: it exists, so no filter needs re-deriving."""
    key = str(corpus["osdr_metadata"]["sample_key"].iloc[0])
    assert retrieval.sample_tier(key, "irrelevant", "", "") == retrieval.TIER_CACHED


def test_the_cached_path_never_opens_a_checkpoint_or_shells_out(monkeypatch, corpus):
    """The fast path must not reach the subprocess. A regression there would be
    invisible except as a 44x slowdown, which no assertion elsewhere would catch."""
    import subprocess

    def explode(*a, **k):
        raise AssertionError("cached retrieval must not launch a subprocess")

    monkeypatch.setattr(subprocess, "run", explode)
    df = _samples_frame(corpus)
    hits, mode = retrieval.search_hits(df, str(df["sample_id"].iloc[0]), topk=3,
                                       enable_biopython_metadata=False)
    assert mode == "cached" and len(hits) == 3
