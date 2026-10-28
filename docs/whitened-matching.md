# Whitened matching

Requested by Joshua on 2026-10-28, after the benchmark in [`matching-benchmark.md`](matching-benchmark.md) found that whitening the embedding with an eigenvalue floor of 1e-4 raised tissue retrieval accuracy on every benchmark.
This is the implementation doc: the plan, the decisions, the tradeoffs, kept current as it lands.

## Decisions (interview, 2026-10-28)

| Question | Answer |
| --- | --- |
| Default or opt-in | A "Matching" choice, Whitened or Plain cosine. Whitened becomes the default only if `precompute/validate_cohorts.py` passes under it; otherwise it stays opt-in |
| Which searches | Sample, Cohort and Upload, one shared choice. A cohort is pooled exactly as today, and the pooled query is then whitened |
| What the score shows | The whitened score ranks and is the Score column; each match's plain cosine stays in the inspector and the CSV |

## The transform

Every embedding is L2-normalised (invariant 2), centred on the ARCHS4 mean `mu`, and mapped by `W = V diag(1 / sqrt(max(lambda, 1e-4 * lambda_max)))`, where `V, lambda` are the eigenvectors and eigenvalues of the ARCHS4 covariance of the normalised vectors.
The whitened score of reference row `x` against query `q` is the cosine between `W'(x - mu)` and `W'(q - mu)`.

It is fitted on **all 940,455 ARCHS4 rows**, not a subsample.
The benchmark fitted on 100,000 rows disjoint from its queries to avoid leakage, and two different fitting samples agreed to the third decimal; the app has no held-out queries to protect, and a covariance estimated from the whole corpus is the principled estimate.
The floor is relative to the largest eigenvalue, so it does not depend on the sample size.

## Making it cost what a cosine scan costs

Transforming all 940,455 vectors per search would add a 512 x 512 matrix product per row.
It does not have to:

```
score(x) = (x - mu) . v / r_x,   where  u = unit(W'(q - mu)),  v = W u,  r_x = |W'(x - mu)|
```

`v` is computed once per query (a 512-vector), and `r_x` once per reference row, offline.
So a whitened scan is the same single pass over the memmap as a cosine scan: centre the block, one matrix-vector product, divide by the stored norms.
Centring before the product, rather than subtracting `mu . v` afterwards, matters: the normalised embeddings sit close to their mean, so `x . v` and `mu . v` are nearly equal and their difference in float32 would lose most of its digits.
The equivalence is checked against an explicit float64 transform, both in a unit test on synthetic data and on the real corpus by the build script.

## The artifact

`precompute/build_whitening.py` writes `cache/whitening.npz` (about 5 MB): `mu` (512), `W` (512 x 512), `row_norms` (940,455 float32), and a record of the floor, the eigenvalue spectrum summary, the row count and the memmap's size, which the app checks before trusting it.
It is optional, like `archs4_metadata.parquet`: without it the app runs exactly as before, and the Matching control shows Whitened disabled with the command that builds it.
`bridge_rna/scoring.py` is the only module that knows what whitening is.

## What does not change

- The map's coordinates (they are projections of the raw embedding; whitening is a scoring choice, not a new space).
- "Same data as": identical profiles are still found by plain cosine distance below 1e-6, the threshold that was measured.
- A cohort's member list, leave-one-out cosines and outlier flags: they describe the members' own embeddings.
- The comparison's Jaccard overlap: it is computed on whichever matches were retrieved, and both arms always use the same matching.

## The gate

`precompute/validate_cohorts.py --scoring whitened` reruns all six checks with the whitened scan.
Whitened becomes the default only if it passes.

## Status

- [ ] `bridge_rna/scoring.py`, `precompute/build_whitening.py`, artifact built and checked
- [ ] Scan, all three search paths, payload, plain cosine beside each match
- [ ] Matching control, banner phrase, inspector, CSV, plot axis, help entry
- [ ] `validate_cohorts.py --scoring whitened`, then the default
- [ ] Tests, browser suites, guide screenshots, Codex review, docs
