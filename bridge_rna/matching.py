"""Experimental reference-fitted geometry corrections, never learned from queries.

Fit on L2-normalized reference vectors, center, apply one linear map, then
L2-normalize again. Raw reference files and query vectors are never modified.
"""
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import numpy as np

METHODS = {
    'cosine': 'Original cosine',
    'background': 'Remove shared background (experimental)',
    'whiten': 'Regularized whitening (experimental)',
}
SCHEMA = 1
SEED = 20261015
FIT_SIZE = 50000
EIGEN_FLOOR = .01


def unit_rows(values):
    x = np.array(values, dtype=np.float32, copy=True)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError('Embeddings must be a finite matrix.')
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise ValueError('Embeddings must be nonzero.')
    return x / norms


def fit_corrections(index, *, seed=SEED, fit_size=FIT_SIZE):
    if len(index) < 3 or index.shape[1] < 2 or fit_size < 3:
        raise ValueError('Correction fitting needs at least three rows and two dimensions.')
    ids = np.sort(np.random.default_rng(seed).choice(len(index), min(fit_size, len(index)), replace=False))
    x = unit_rows(index[ids]).astype(np.float64)
    mean = x.mean(axis=0)
    centered = x - mean
    eigen, vectors = np.linalg.eigh(centered.T @ centered / (len(x)-1))
    if eigen[-1] <= 1e-12:
        raise ValueError('Reference fit has no usable variance.')
    floor = EIGEN_FLOOR * eigen[-1]
    # Symmetric ZCA whitening with an eigenvalue floor, not unbounded inversion.
    whiten = (vectors * (1 / np.sqrt(np.maximum(eigen, floor)))) @ vectors.T
    background = np.eye(x.shape[1]) - np.outer(vectors[:, -1], vectors[:, -1])
    return dict(mean=mean, whiten=whiten, background=background, fit_indices=ids,
                eigenvalues=eigen)


def reference_norms(index, fit, method, *, chunk=4096):
    if method not in ('background', 'whiten'):
        raise ValueError('Unknown correction method.')
    norms = np.empty(len(index), dtype=np.float64)
    for start in range(0, len(index), chunk):
        z = (unit_rows(index[start:start+chunk]).astype(float) - fit['mean']) @ fit[method]
        norms[start:start+chunk] = np.linalg.norm(z, axis=1)
    if not np.isfinite(norms).all() or np.any(norms <= 1e-10):
        raise ValueError('Correction produced a degenerate reference vector.')
    return norms


def corrected_scores(index, query, fit, method, norms, *, chunk=25000):
    if method not in ('background', 'whiten'):
        raise ValueError('Unknown correction method.')
    if np.shape(norms) != (len(index),) or not np.isfinite(norms).all() or np.any(norms <= 1e-10):
        raise ValueError('Invalid reference correction norms.')
    q = unit_rows(np.asarray(query).reshape(1, -1))[0].astype(float) - fit['mean']
    transform = fit[method]
    zq = q @ transform
    qnorm = np.linalg.norm(zq)
    if not np.isfinite(qnorm) or qnorm <= 1e-10:
        raise ValueError('Query has no nonzero signal after this correction; use original cosine.')
    # (x W) dot (q W) == x dot (W W.T q); precomputed norms avoid an
    # O(n*d*d) transformation on every search. Float64 limits cancellation.
    projected = transform @ (zq / qnorm)
    scores = np.empty(len(index), dtype=np.float32)
    for start in range(0, len(index), chunk):
        end = min(start + chunk, len(index))
        x = unit_rows(index[start:end]).astype(float) - fit['mean']
        scores[start:end] = (x @ projected) / norms[start:end]
    if not np.isfinite(scores).all():
        raise ValueError('Nonfinite corrected scores cannot be ranked.')
    return np.clip(scores, -1, 1)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def build_artifact(index, index_path, locations_path, output, *, fit_size=FIT_SIZE, seed=SEED):
    fit = fit_corrections(index, fit_size=fit_size, seed=seed)
    metadata = dict(schema=SCHEMA, rows=len(index), dim=index.shape[1], seed=seed,
                    fit_size=len(fit['fit_indices']), eigen_floor=EIGEN_FLOOR,
                    background_removed_components=1, whitening_removed_components=0,
                    index_sha256=digest(index_path),
                    locations_sha256=digest(locations_path),
                    preprocessing='L2, reference mean, linear correction, L2')
    for method in ('background', 'whiten'):
        fit[method+'_norms'] = reference_norms(index, fit, method)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix('.tmp.npz')
    np.savez_compressed(temporary, **fit, metadata=json.dumps(metadata, sort_keys=True))
    temporary.replace(output)
    return metadata


def load_artifact(path, index_path, locations_path, shape):
    with np.load(path, allow_pickle=False) as saved:
        data = {key: saved[key] for key in saved.files}
    meta = json.loads(str(data.pop('metadata')))
    if (meta.get('schema'), meta.get('rows'), meta.get('dim')) != (SCHEMA, *shape):
        raise ValueError('Correction artifact schema or shape mismatch; rebuild it.')
    if meta.get('index_sha256') != digest(index_path) or meta.get('locations_sha256') != digest(locations_path):
        raise ValueError('Reference index or row metadata changed; rebuild matching corrections.')
    d = shape[1]
    if data['mean'].shape != (d,) or not np.isfinite(data['mean']).all():
        raise ValueError('Invalid correction mean.')
    for method in ('background', 'whiten'):
        if data[method].shape != (d, d) or not np.isfinite(data[method]).all():
            raise ValueError('Invalid correction matrix.')
        norms = data[method+'_norms']
        if norms.shape != (shape[0],) or not np.isfinite(norms).all() or np.any(norms <= 1e-10):
            raise ValueError('Invalid correction norms.')
    data['metadata'] = meta
    return data


@lru_cache(maxsize=2)
def _cached_artifact(path, index_path, locations_path, shape, stamps):
    return load_artifact(path, index_path, locations_path, shape)


def available_artifact(index):
    from .config import ROOT, EMBEDDING_DIR
    path = ROOT/'cache'/'matching_corrections.npz'
    index_path = Path(index.filename)
    locations = EMBEDDING_DIR/'sample_locations.parquet'
    if not path.exists():
        raise ValueError('Matching corrections are not built. Run python -m precompute.build_matching_corrections.')
    stamps = tuple((p.stat().st_size, p.stat().st_mtime_ns) for p in (path, index_path, locations))
    return _cached_artifact(str(path), str(index_path), str(locations), index.shape, stamps)


def search_corrected(sample_id, topk, method, depth=250):
    """Single cached sample, complete reference universe, baseline evidence."""
    from .retrieval import (cached_query_vector, _load_archs4_index,
                            _topk_cosine_matrix, _annotate_from_cache)
    if method not in ('background', 'whiten'):
        raise ValueError('Unknown correction method.')
    query = cached_query_vector(sample_id)
    if query is None:
        raise ValueError('Experimental matching requires a precomputed OSDR sample. Use original cosine for this sample.')
    index, _, _ = _load_archs4_index()
    fit = available_artifact(index)
    raw = np.empty((1, len(index)), dtype=np.float32)
    baseline_idx, _ = _topk_cosine_matrix(index, query, max(topk, depth), score_sink=raw)
    scores = corrected_scores(index, query, fit, method, fit[method+'_norms'])
    order = np.argsort(-scores, kind='stable')
    raw_order = np.argsort(-raw[0], kind='stable')
    raw_ranks = np.empty(len(index), dtype=np.int32)
    raw_ranks[raw_order] = np.arange(1, len(index)+1)
    ids = order[:max(topk, depth)]
    ranked = _annotate_from_cache(ids, scores[ids])
    ranked['archs4_index'] = ids
    ranked['raw_cosine'] = raw[0, ids]
    ranked['raw_rank'] = raw_ranks[ids]
    ranked['matching_method'] = method
    # Already in stable reference-row tie order. The legacy prefix helper
    # re-sorts with quicksort; do not reorder corrected ties or the overlap set.
    hits = ranked.head(topk).copy().reset_index(drop=True)
    neighborhood = ranked.head(depth).copy().reset_index(drop=True)
    levels = [0, 25, 50, 75, 95, 99, 100]
    evidence = dict(method=method, label=METHODS[method], reference_count=len(index),
                    overlap=len(set(order[:topk]) & set(baseline_idx[0, :topk])), topk=topk,
                    percentiles=levels, raw_percentiles=np.percentile(raw, levels).tolist(),
                    adjusted_percentiles=np.percentile(scores, levels).tolist(),
                    raw_top_range=[float(raw[0, baseline_idx[0, topk-1]]), float(raw.max())],
                    adjusted_top_range=[float(scores[order[topk-1]]), float(scores[order[0]])],
                    fit=fit['metadata'])
    return hits, neighborhood, evidence
