"""Reference-fitted corrections must match explicit transformed-space cosine."""
import numpy as np
import pytest


def corpus():
    rng = np.random.default_rng(17)
    x = rng.normal(size=(91, 8)) * np.geomspace(.3, .01, 8) + 3
    return x.astype(np.float32), rng.normal(size=8).astype(np.float32) + 3


@pytest.mark.parametrize("method", ["background", "whiten"])
def test_streamed_scores_equal_explicit_transformation_without_mutation(method):
    from bridge_rna.matching import fit_corrections, reference_norms, corrected_scores
    x, q = corpus()
    before, qb = x.copy(), q.copy()
    fit = fit_corrections(x, seed=8, fit_size=70)
    norms = reference_norms(x, fit, method, chunk=13)
    actual = corrected_scores(x, q, fit, method, norms, chunk=17)
    unit = x / np.linalg.norm(x, axis=1, keepdims=True)
    uq = q / np.linalg.norm(q)
    z = (unit.astype(float) - fit['mean']) @ fit[method]
    zq = (uq.astype(float) - fit['mean']) @ fit[method]
    expected = z @ zq / (np.linalg.norm(z, axis=1) * np.linalg.norm(zq))
    np.testing.assert_allclose(actual, expected, atol=2e-6)
    np.testing.assert_array_equal(x, before)
    np.testing.assert_array_equal(q, qb)


def test_fit_is_reproducible_and_removes_mean_and_one_component():
    from bridge_rna.matching import fit_corrections
    x, _ = corpus()
    a = fit_corrections(x, seed=8, fit_size=70)
    b = fit_corrections(x, seed=8, fit_size=70)
    np.testing.assert_array_equal(a['whiten'], b['whiten'])
    eigen = np.linalg.eigvalsh(a['background'])
    np.testing.assert_allclose(eigen, [0] + [1]*7, atol=1e-8)
    assert np.linalg.eigvalsh(a['whiten']).min() > 0


def test_invalid_and_degenerate_vectors_fail_instead_of_becoming_matches():
    from bridge_rna.matching import fit_corrections, reference_norms, corrected_scores
    x, q = corpus()
    with pytest.raises(ValueError, match='finite|nonzero'):
        fit_corrections(np.zeros((10, 8)))
    x[0, 0] = np.nan
    with pytest.raises(ValueError, match='finite'):
        fit_corrections(x)
    x, _ = corpus()
    fit = fit_corrections(x)
    norms = reference_norms(x, fit, 'whiten')
    with pytest.raises(ValueError, match='nonzero'):
        corrected_scores(x, np.zeros(8), fit, 'whiten', norms)
    with pytest.raises(ValueError, match='method'):
        corrected_scores(x, q, fit, 'unknown', norms)


def test_artifact_refuses_changed_index_and_metadata(tmp_path):
    from bridge_rna.matching import build_artifact, load_artifact
    x, _ = corpus()
    index = tmp_path/'index.mmap'
    x.tofile(index)
    meta = tmp_path/'locations'
    meta.write_bytes(b'row order one')
    artifact = tmp_path/'corrections.npz'
    build_artifact(x, index, meta, artifact, fit_size=70)
    assert load_artifact(artifact, index, meta, x.shape)['metadata']['rows'] == len(x)
    meta.write_bytes(b'row order two')
    with pytest.raises(ValueError, match='changed|mismatch'):
        load_artifact(artifact, index, meta, x.shape)
    meta.write_bytes(b'row order one')
    with index.open('r+b') as f:
        f.write(b'changed')
    with pytest.raises(ValueError, match='changed|mismatch'):
        load_artifact(artifact, index, meta, x.shape)


def test_matching_evidence_names_both_scores_and_preserves_source_links():
    from bridge_rna.matching_ui import matching_evidence
    import json
    from plotly.utils import PlotlyJSONEncoder
    payload = dict(matching=dict(method='whiten', label='Regularized whitening (experimental)',
        overlap=1, topk=2, reference_count=91, percentiles=[0, 100],
        raw_percentiles=[.98, .999], adjusted_percentiles=[-.4, .8],
        raw_top_range=[.998, .999], adjusted_top_range=[.7, .8],
        fit=dict(seed=8, fit_size=70, eigen_floor=.01, index_sha256='abc')),
        hits=[dict(gsm='GSM123', gse='GSE456', score=.8, raw_cosine=.99, raw_rank=20,
                   matching_method='whiten', title='Source title')])
    rendered = json.dumps(matching_evidence(payload), cls=PlotlyJSONEncoder)
    assert 'Original rank' in rendered and 'Adjusted cosine' in rendered
    assert 'GSM123' in rendered and 'acc=GSM123' in rendered
    assert 'biological' in rendered and 'abc' in rendered
    assert matching_evidence({'hits': []}) is None


def test_ai_context_identifies_experimental_score_geometry():
    from bridge_rna.ai import _format_hits_table_text
    import pandas as pd
    text = _format_hits_table_text(pd.DataFrame([dict(gsm='GSM1', score=.8,
        raw_cosine=.99, matching_method='whiten')]))
    assert 'raw_cosine' in text and 'matching_method' in text


def test_corrected_provenance_survives_neighborhood_and_map_overlay(monkeypatch):
    import pandas as pd
    from bridge_rna.neighborhoods import build_payload
    from manifold import callbacks, data, render, layout
    hit = dict(gsm='GSM1', gse='GSE1', score=.8, raw_cosine=.99,
               raw_rank=20, archs4_index=0, matching_method='whiten')
    evidence = build_payload(pd.DataFrame([hit]))
    assert evidence['space'] == 'corrected-512d-whiten'
    assert evidence['hits'][0]['raw_cosine'] == .99
    monkeypatch.setattr(data, 'counts', lambda: (2, 0, 2))
    overlay = callbacks._retrieval_overlay(dict(hits=[hit], matching={'method': 'whiten'}))
    assert overlay['cohorts'][0]['matching_method'] == 'whiten'
    import json
    from plotly.utils import PlotlyJSONEncoder
    detail = json.dumps(layout._sample_detail(evidence['hits'][0]), cls=PlotlyJSONEncoder)
    assert 'Adjusted cosine' in detail and 'Original cosine' in detail
    traces = render._retrieval_traces(np.array([[0., 0.], [1., 1.]]), False, overlay)
    rendered = json.dumps(traces, cls=PlotlyJSONEncoder)
    assert 'adjusted cosine' in rendered and 'whiten' in rendered


def test_corrected_ties_keep_row_order_and_overlap_matches_returned_hits(monkeypatch):
    import pandas as pd
    from bridge_rna import matching, retrieval
    x = np.tile([1., 0.], (30, 1)).astype(np.float32)
    fit = dict(mean=np.zeros(2), whiten=np.eye(2), whiten_norms=np.ones(30), metadata={})
    monkeypatch.setattr(retrieval, 'cached_query_vector', lambda sid: np.array([1., 0.]))
    monkeypatch.setattr(retrieval, '_load_archs4_index', lambda: (x, None, 2))
    monkeypatch.setattr(matching, 'available_artifact', lambda index: fit)
    monkeypatch.setattr(retrieval, '_annotate_from_cache', lambda ids, scores:
                        pd.DataFrame(dict(gsm=[f'GSM{i}' for i in ids], score=scores)))
    hits, neighborhood, evidence = matching.search_corrected('query', 5, 'whiten', depth=30)
    assert hits.archs4_index.tolist() == list(range(5))
    assert neighborhood.archs4_index.tolist() == list(range(30))
    baseline, _ = retrieval._topk_cosine_matrix(x, np.array([1., 0.]), 30)
    assert evidence['overlap'] == len(set(hits.archs4_index) & set(baseline[0, :5]))
