"""Reproducible numerical and descriptive checks, not biological validation."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from bridge_rna.config import ROOT
from bridge_rna.matching import available_artifact, corrected_scores, search_corrected, unit_rows
from bridge_rna.retrieval import (_load_archs4_index, _cached_osdr_embeddings,
                                  run_cached_query_retrieval, cached_query_vector)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'reports'/'matching-validation.json')
    args = parser.parse_args()
    index, _, _ = _load_archs4_index()
    fit = available_artifact(index)
    meta = pd.read_parquet(ROOT/'cache'/'osdr_metadata.parquet')
    ids = _cached_osdr_embeddings()[1]
    # Fixed studies chosen for tissue coverage, before seeing corrected results.
    studies = ['OSD-100', 'OSD-104', 'OSD-194', 'OSD-137', 'OSD-102', 'OSD-613']
    rows = []
    for study in studies:
        candidates = meta[(meta.study == study) & (meta.spaceflight == 'Space Flight')]
        sample = next((r for _, r in candidates.iterrows() if r.sample_key in ids), None)
        if sample is None:
            raise ValueError(f'Missing fixed validation study: {study}')
        sid = sample.sample_key
        base = run_cached_query_retrieval(sid, 30)
        for method in ('background', 'whiten'):
            started = time.monotonic()
            hits, neighborhood, evidence = search_corrected(sid, 30, method)
            raw30 = set(base.archs4_index.astype(int))
            item = dict(sample_id=sid, source_tissue=sample.tissue, method=method,
                seconds=round(time.monotonic()-started, 3),
                overlap_at_5=len(set(hits.archs4_index.head(5)) & set(base.archs4_index.head(5))),
                overlap_at_30=len(set(hits.archs4_index) & raw30),
                raw_top5_spread=float(base.score.iloc[0]-base.score.iloc[4]),
                adjusted_top5_spread=float(hits.score.iloc[0]-hits.score.iloc[4]),
                raw_tissue_categories=base.tissue.value_counts().to_dict(),
                adjusted_tissue_categories=hits.tissue.value_counts().to_dict(),
                raw_largest_gse_count=int(base.gse.value_counts().max()),
                adjusted_largest_gse_count=int(hits.gse.value_counts().max()),
                raw_percentiles=evidence['raw_percentiles'],
                adjusted_percentiles=evidence['adjusted_percentiles'],
                top5=hits.head(5)[['gsm', 'gse', 'score', 'raw_cosine', 'raw_rank', 'tissue']].to_dict('records'))
            # Independent explicit transform check over fixed full-index row positions.
            positions = np.linspace(0, len(index)-1, 97, dtype=int)
            q = cached_query_vector(sid)
            z = (unit_rows(index[positions]).astype(float)-fit['mean']) @ fit[method]
            zq = (unit_rows(q.reshape(1,-1))[0].astype(float)-fit['mean']) @ fit[method]
            expected = z @ zq / (np.linalg.norm(z, axis=1)*np.linalg.norm(zq))
            actual = corrected_scores(index[positions], q, fit, method, fit[method+'_norms'][positions])
            item['max_explicit_score_error'] = float(np.max(np.abs(actual-expected)))
            if item['max_explicit_score_error'] > 2e-6:
                raise ValueError('Streamed and explicit scores disagree.')
            rows.append(item)
            print(study, method, 'overlap@5', item['overlap_at_5'], 'spread',
                  item['raw_top5_spread'], '->', item['adjusted_top5_spread'], flush=True)
    result = dict(status='numerical checks passed; biological benefit unvalidated',
                  limitations=['Tissue categories are existing derived labels, not verified ground truth.',
                               'Six fixed examples are descriptive; they are not a held-out biological benchmark.',
                               'Study duplicates and technical batch effects remain possible.',
                               'Fit parameters were fixed, not selected for these examples.'],
                  fit=fit['metadata'], cases=rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    scalar_rows = [{k: (json.dumps(v, sort_keys=True) if isinstance(v, dict) else v)
                    for k, v in row.items() if k not in ('top5', 'raw_percentiles', 'adjusted_percentiles')}
                   for row in rows]
    pd.DataFrame(scalar_rows).to_csv(args.output.with_suffix('.csv'), index=False)
    print(args.output)


if __name__ == '__main__':
    main()
