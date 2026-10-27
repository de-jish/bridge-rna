# Which embedding transform separates tissues best?

Requested by Joshua on 2026-10-27, after the in-app help work.
He shared a per-tissue retrieval-accuracy table comparing Bridge ABTT (D=10), Bridge-white cosine, PCA-white cosine, PCA-512 Canberra, Expression Canberra, Expression Pearson and raw Bridge cosine.
The table's source and exact metric are not in this repository, so it is rebuilt here on this corpus, against nulls, before anything is implemented.

## What the screenshot showed

Raw Bridge cosine was weakest on adipose (55.6%), stomach (52.0%), breast (73.8%), spleen (73.3%) and pancreas (85.1%); brain, liver and testis were near 100% for every method.
No method dominated.
ABTT D=10 lifted adipose to 92.5 but dropped breast to 60.3 and spleen to 63.0.
Bridge-white lifted adipose to 75.2 and stomach to 76.3, with breast 66.2 and spleen 71.3.
PCA-white reached 100 on stomach and 88.1 on adipose but fell to 48.2 on breast.

## Questions this answers

1. Per tissue, which transform of the 512-d embedding makes a sample's nearest ARCHS4 neighbours share its tissue most often?
2. Does that hold for the app's actual queries, OSDR spaceflight samples, and not only for ARCHS4 against itself?
3. Does it make results more repeatable (replicates agreeing on their matches), the property cohort pooling was adopted for?
4. Is any gain above what a label-shuffled null gives?

## Methods compared

All are linear maps applied to L2-normalized vectors, followed by cosine, so each is one matrix and a mean and runs on the existing memmap.
Every transform is fitted on ARCHS4 rows only, from a random fitting set disjoint from the benchmark's queries.

| method | transform |
| --- | --- |
| Raw cosine | identity (what the app does today) |
| ABTT D | subtract the mean, project out the top D principal directions (D = 1, 2, 3, 5, 10, 20) |
| Whitening, floor f | subtract the mean, scale every principal direction to unit variance, eigenvalues floored at f x the largest (f = 1e-1, 1e-2, 1e-3, 1e-4) |
| Whitening, top k | whiten only the top k principal directions and drop the rest (k = 64, 128, 256) |

The expression baselines (Canberra, Pearson, PCA-512) need ARCHS4's expression matrices, 113 GB that this pipeline deliberately never downloads, so they are out of scope.
PR #3's whitening used f = 1e-2, which put 479 of 512 directions on the floor; the sweep shows whether that choice mattered.

## Benchmarks

**A. Within ARCHS4.**
Up to 300 query samples per tissue bucket, drawn at random with a fixed seed, excluding "Other" and "Unknown".
For each query, its top 10 neighbours among all 940,455 ARCHS4 samples, **excluding every sample from the query's own GEO series**.
Without that exclusion the task is dominated by same-study batch (a series' samples share lab, platform and processing), and every method looks near-perfect.
Metrics per tissue: kNN accuracy (the majority tissue among labelled neighbours equals the query's) and precision at 10 (the share of neighbours with the query's tissue).
"Other" and "Unknown" neighbours carry no label and do not vote.

**B. OSDR to ARCHS4.**
All 2,108 OSDR samples as queries with their canonical tissue, top 10 ARCHS4 neighbours (no series to exclude), same metrics.

**C. Repeatability.**
For OSDR replicates in the same cohort, the mean top-5 overlap of their matches under each transform, against random pairs from the same study.

**Null.**
Query tissue labels shuffled within each benchmark; any method's accuracy must clear this by a wide margin to count.

## Decision rule

A transform is recommended only if it raises mean per-tissue kNN accuracy on A **and** B over raw cosine, does not drop any tissue with at least 30 queries by more than 5 points, and does not lower replicate agreement (C).
If none qualifies, the recommendation is to keep raw cosine, and the table explains why.

## Results (2026-10-27)

Run: `precompute/benchmark_matching.py --per-tissue 300`, 11,100 ARCHS4 queries over 37 tissues and 2,108 OSDR queries, every method scored over all 940,455 reference samples.
Per tissue and method: [`docs/validation/matching-benchmark.csv`](validation/matching-benchmark.csv); the refit on a second fitting sample: [`matching-benchmark-fitseed7.csv`](validation/matching-benchmark-fitseed7.csv).

Mean per-tissue kNN accuracy (tissues with at least 30 queries), and the largest single-tissue drop against raw cosine:

| method | A: within ARCHS4 | worst drop | B: OSDR to ARCHS4 | worst drop | C: replicate top-5 overlap |
| --- | --- | --- | --- | --- | --- |
| raw cosine (today) | 0.644 | - | 0.956 | - | 0.137 |
| ABTT D=1 | 0.663 | -1.4 | 0.966 | -2.3 | 0.189 |
| ABTT D=3 | 0.668 | -3.0 | 0.967 | 0.0 | 0.208 |
| ABTT D=10 | 0.672 | -2.8 | 0.961 | -7.0 | 0.220 |
| ABTT D=20 | 0.682 | -1.7 | 0.961 | -14.0 | 0.248 |
| whitening, floor 1e-2 (PR #3) | 0.676 | -1.5 | 0.965 | -4.7 | 0.227 |
| whitening, floor 1e-3 | 0.684 | -1.5 | 0.970 | 0.0 | 0.226 |
| **whitening, floor 1e-4** | **0.693** | **-1.5** | **0.979** | **0.0** | 0.209 |
| whitening, floor 1e-6 (refit) | 0.702 | -2.2 | 0.978 | 0.0 | 0.207 |
| whitening, top 256 | 0.692 | -1.3 | 0.976 | -2.3 | 0.213 |
| label-shuffled null | 0.025 | | 0.07 | | |

What it shows:

- **Every transform beats raw cosine, and near-full whitening beats them all.** Whitening with the eigenvalue floor at 1e-4 of the largest raises within-ARCHS4 accuracy by 4.9 points and OSDR accuracy by 2.3, lowers no tissue by more than 1.5 points on A, and lowers none at all on B.
- **The gains land on exactly the confusable tissues.** Thyroid 53.4 to 74.4, esophagus 37.8 to 55.3, salivary gland 42.5 to 61.5, stomach 50.6 to 62.2, bladder 50.5 to 60.7, thymus (OSDR) 67.1 to 81.0. Tissues that were already near 90% barely move.
- **It is not luck of the fit.** Refitting on a different 100,000 rows gives 0.693 and 0.978 again. Below 1e-4 the floor stops mattering (1e-5 and 1e-6 land within a point), so 1e-4 is the regularised end of a plateau rather than a tuned peak.
- **PR #3's floor of 1e-2 was too high.** It flattened 479 of 512 directions and gives up half the gain on A and two thirds on B.
- **ABTT does what the screenshot showed it doing: it helps the weak tissues and hurts others.** At D=10 and D=20 it costs OSDR tissues 7 to 14 points (Breast / mammary, Thymus), because the top directions it removes carry tissue biology, not only nuisance.
- **Replicates agree more under every transform** (0.137 to about 0.21), but so do unrelated animals from the same study (0.125 to about 0.19). The transform does not make a cohort's matches distinctive from its study's; that remains cohort pooling's job.

How this differs from the screenshot: its taxonomy and metric are unknown here and its expression baselines are out of reach, so the numbers are not comparable one to one.
The direction agrees for whitening and ABTT's trade-off; unlike the screenshot, ABTT D=10 did not lower spleen here (48.3 to 52.7 on A).

**Recommendation: whitening with the eigenvalue floor at 1e-4, fitted on ARCHS4.** It passes the decision rule on all three benchmarks.

## Status

- [x] Benchmark script (`precompute/benchmark_matching.py`), one streaming pass per method
- [x] Results table and recommendation
- [ ] Implementation of the recommended transform, opt-in or default (Joshua's decision)
