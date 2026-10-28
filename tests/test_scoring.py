"""Whitened matching: the scan formula, the artifact guard, and the fallback.

The app ranks by `(x - mu) . v / r_x` rather than by transforming every
reference vector, so the first thing pinned here is that the two are the same
ranking. Built on the synthetic fixture corpus with the build script's own
`fit` and `row_norms`; the real-corpus version of the same check runs inside
`precompute/build_whitening.py` every time the artifact is written.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from bridge_rna import retrieval, scoring

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "precompute"))
import build_whitening  # noqa: E402


@pytest.fixture(autouse=True)
def _fixture_index(monkeypatch, corpus):
    def clear():
        retrieval._cached_osdr_embeddings.cache_clear()
        retrieval._archs4_annotations.cache_clear()
        retrieval._ARCHS4_CACHE.clear()
        scoring.load_whitening.cache_clear()

    clear()
    monkeypatch.setattr(retrieval, "EMBEDDING_DIR",
                        corpus["bridge_rna_root"] / "archs4_sample_embeddings_full")
    yield
    clear()


@pytest.fixture
def whitened(monkeypatch, tmp_path):
    """Fit the transform on the fixture index and install it as the artifact."""
    from manifold import paths

    vecs, _, d = retrieval._load_archs4_index()
    mu, lam, V = build_whitening.fit(vecs)
    W = V * (1.0 / np.sqrt(np.maximum(lam, scoring.FLOOR * lam[0])))
    mu32 = mu.astype(np.float32)
    norms = build_whitening.row_norms(vecs, mu32, W)
    path = tmp_path / "whitening.npz"
    np.savez(path, mu=mu32, W=W, row_norms=norms, eigenvalues=lam,
             meta=np.array(json.dumps({
                 "index_bytes": int(vecs.nbytes), "floor": scoring.FLOOR,
                 "index_fingerprint": scoring.index_fingerprint(vecs),
                 "transform_id": "fixture"})))
    monkeypatch.setattr(paths, "WHITENING_NPZ", path)
    scoring.load_whitening.cache_clear()
    w = scoring.load_whitening()
    assert w is not None
    yield w
    scoring.load_whitening.cache_clear()


def _explicit_scores(vecs, w, q):
    x = np.asarray(vecs, dtype=np.float32)
    x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)
    zx = (x.astype(np.float64) - w.mu) @ w.W
    zx /= np.linalg.norm(zx, axis=1, keepdims=True)
    zq = (q / np.linalg.norm(q) - w.mu) @ w.W
    return zx @ (zq / np.linalg.norm(zq))


def test_the_scan_ranks_exactly_as_the_explicit_transform(whitened):
    vecs, _, _ = retrieval._load_archs4_index()
    osdr, _ = retrieval._cached_osdr_embeddings()
    for q in osdr[:5].astype(np.float64):
        idx, score = retrieval._topk_cosine_from_memmap(vecs, q, 50,
                                                        scoring=scoring.WHITENED)
        exact = _explicit_scores(vecs, whitened, q)
        expected = np.lexsort((np.arange(len(exact)), -exact))[:50]
        np.testing.assert_array_equal(idx, expected)
        np.testing.assert_allclose(score, exact[idx], atol=2e-5)


def test_whitening_changes_the_ranking(whitened):
    """If it never reordered anything it would be pointless to ship."""
    vecs, _, _ = retrieval._load_archs4_index()
    osdr, _ = retrieval._cached_osdr_embeddings()
    changed = 0
    for q in osdr[:10]:
        a, _ = retrieval._topk_cosine_from_memmap(vecs, q, 20)
        b, _ = retrieval._topk_cosine_from_memmap(vecs, q, 20, scoring=scoring.WHITENED)
        changed += not np.array_equal(a, b)
    assert changed > 0


def test_identical_rows_still_tie_and_the_lower_row_wins(whitened):
    vecs, _, _ = retrieval._load_archs4_index()
    x = np.asarray(vecs[:400], dtype=np.float32).copy()
    x[300] = x[100]                                    # a profile filed twice
    norms = whitened.row_norms[:400].copy()
    norms[300] = norms[100]
    w = scoring.Whitening(whitened.mu, whitened.W, norms, whitened.meta)
    Q = (x[100] + 1e-3 * x[7]).reshape(1, -1)
    V = scoring.query_operator(Q, w)
    xs = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12) - w.mu
    s = (xs @ V).ravel() / norms
    assert s[100] == s[300]
    order = np.lexsort((np.arange(400), -s))
    assert list(order).index(100) < list(order).index(300)


def test_every_path_carries_plain_cosine_beside_a_whitened_score(whitened, corpus):
    key = corpus["osdr_metadata"]["sample_key"].astype(str).iloc[0]
    hits = retrieval.run_cached_query_retrieval(key, topk=20, scoring=scoring.WHITENED)
    q = retrieval.cached_query_vector(key).astype(np.float64)
    q /= np.linalg.norm(q)
    expected = retrieval.reference_vectors(hits["archs4_index"]) @ q
    np.testing.assert_allclose(hits["cosine"], expected, atol=1e-9)
    assert not np.allclose(hits["score"], hits["cosine"])
    assert hits["score"].is_monotonic_decreasing

    plain = retrieval.run_cached_query_retrieval(key, topk=20)
    np.testing.assert_array_equal(plain["cosine"], plain["score"])


def test_cohort_and_upload_paths_accept_whitening(whitened, corpus, monkeypatch):
    keys = list(corpus["osdr_metadata"]["sample_key"].astype(str).iloc[:4])
    cohort = retrieval.run_cohort_retrieval_with_neighborhood(
        keys, topk=10, scoring=scoring.WHITENED)
    assert "cosine" in cohort.hits and len(cohort.profile) > 0
    qvec = retrieval.cached_query_vector(keys[0])
    monkeypatch.setattr(retrieval, "embed_uploaded_counts", lambda *a, **k: qvec)
    upload = retrieval.run_uploaded_retrieval_with_neighborhood(
        "unused.csv", None, 10, scoring=scoring.WHITENED)
    direct = retrieval.run_cached_query_retrieval(keys[0], 1000, scoring=scoring.WHITENED)
    assert list(upload.hits["gsm"]) == list(direct["gsm"].head(10))


# --- The artifact guard and the fallback --------------------------------------

def test_without_the_artifact_whitening_is_unavailable_and_searches_fall_back(monkeypatch):
    from manifold import paths

    monkeypatch.setattr(paths, "WHITENING_NPZ", paths.CACHE_DIR / "absent.npz")
    scoring.load_whitening.cache_clear()
    assert not scoring.available()
    assert scoring.resolve(scoring.WHITENED) == scoring.COSINE
    assert scoring.resolve("nonsense") == scoring.COSINE
    whitened_option = next(o for o in scoring.control_options() if o["value"] == scoring.WHITENED)
    assert whitened_option["disabled"] is True
    with pytest.raises(retrieval.RetrievalError, match="build_whitening"):
        vecs, _, d = retrieval._load_archs4_index()
        retrieval._topk_cosine_matrix(vecs, np.ones((1, d)), 5, scoring=scoring.WHITENED)


def test_an_artifact_for_another_index_is_refused(whitened, monkeypatch, tmp_path):
    from manifold import paths

    bad = tmp_path / "other.npz"
    np.savez(bad, mu=whitened.mu, W=whitened.W, row_norms=whitened.row_norms[:-1],
             meta=np.array(json.dumps({"index_bytes": 1})))
    monkeypatch.setattr(paths, "WHITENING_NPZ", bad)
    scoring.load_whitening.cache_clear()
    assert scoring.load_whitening() is None


def test_an_index_with_the_same_shape_but_other_rows_is_refused(whitened, monkeypatch):
    """Positional row norms are only valid for the rows they were computed on."""
    vecs, _, _ = retrieval._load_archs4_index()
    shuffled = np.asarray(vecs)[::-1].copy()
    assert shuffled.nbytes == vecs.nbytes
    assert scoring.index_fingerprint(shuffled) != whitened.meta["index_fingerprint"]
    monkeypatch.setattr(retrieval, "_load_archs4_index", lambda: (shuffled, None, shuffled.shape[1]))
    scoring.load_whitening.cache_clear()
    assert scoring.load_whitening() is None


def test_the_transform_has_a_recorded_identity(whitened):
    assert scoring.transform_id() == "fixture"


def test_resolve_honours_an_explicit_choice(whitened):
    assert scoring.resolve(scoring.WHITENED) == scoring.WHITENED
    assert scoring.resolve(scoring.COSINE) == scoring.COSINE
    assert scoring.resolve(" Whitened ") == scoring.WHITENED


# --- What the interface says about the scoring --------------------------------

def _hit(gsm, score, cosine, idx):
    return {"gsm": gsm, "score": score, "cosine": cosine, "archs4_index": idx,
            "gse": "GSE1", "title": "t", "tissue": "Liver", "species": "Mus musculus"}


def test_results_and_csv_carry_both_scores_per_arm():
    from bridge_rna import results as R

    payload = {
        "scoring": "whitened", "mode": "cohort", "sample_id": "C",
        "hits": [_hit("GSM1", 0.61, 0.991, 1), _hit("GSM2", 0.55, 0.990, 2)],
        "comparison": {"hits_b": [_hit("GSM2", 0.70, 0.995, 2)], "query_b": {}},
    }
    res = R.build_results(payload, reference_count=10)
    assert res["scoring"] == "whitened"
    shared = next(r for r in res["rows"] if r["gsm"] == "GSM2")
    # The same sample has a different plain cosine against each cohort's query.
    assert (shared["cosine_a"], shared["cosine_b"]) == (0.990, 0.995)
    csv_text = R.to_csv(res["rows"], res, R.default_filters(), None, {}, "now")
    header = next(line for line in csv_text.splitlines() if not line.startswith("#"))
    assert header.startswith('"rank_a","score_a","cosine_a","rank_b","score_b","cosine_b"')
    assert "Scoring: whitened similarity" in csv_text
    assert "Whitened similarity between embeddings" in csv_text


def test_a_legacy_payload_is_plain_cosine_whatever_the_default():
    from bridge_rna import results as R

    res = R.build_results({"hits": [{"gsm": "GSM1", "score": 0.99}]}, reference_count=1)
    assert res["scoring"] == "cosine"
    assert res["rows"][0]["cosine_a"] == 0.99
    header = next(line for line in R.to_csv(res["rows"], res, None, None, {}, "now")
                  .splitlines() if not line.startswith("#"))
    assert '"cosine"' not in header


def test_the_banner_says_when_whitening_could_not_run():
    from bridge_rna import callbacks

    assert "whitened" in callbacks._scoring_note(scoring.WHITENED, scoring.WHITENED)
    note = callbacks._scoring_note(scoring.WHITENED, scoring.COSINE)
    assert "plain cosine" in note and "precomputed" in note
    assert callbacks._scoring_note(scoring.COSINE, scoring.COSINE) == ""


def test_the_map_and_explorer_name_the_score_that_ranked():
    import pandas as pd

    from bridge_rna import neighborhoods

    frame = pd.DataFrame([_hit("GSM1", 0.61, 0.99, 1)])
    whitened = neighborhoods.build_payload(frame, metric="whitened")
    assert whitened["metric"] == "whitened" and whitened["hits"][0]["metric"] == "whitened"
    assert neighborhoods.build_payload(frame)["metric"] == "cosine"
