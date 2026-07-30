# Cohort pooling: querying with an experimental group instead of one sample

**Status: measured and specified, not built.**
Every number here was measured on the real corpus - the 2,108 cached OSDR embeddings and the 940,455-row ARCHS4 memmap - on 2026-07-30.
The scan used to produce them reproduces `bridge_rna.retrieval._topk_cosine_from_memmap` exactly, indices and scores identical to six decimal places, so these are the app's own answers and not an approximation of them.

## The question

The Retrieve view answers for one OSDR sample at a time.
A spaceflight study does not have one sample; it has an experimental group - six to thirty-eight animals sharing a study, a tissue, and a flight-or-ground condition.
Josh's mentor's proposal: combine a group into one query so the answer describes the group rather than whichever animal happened to be picked, because any single animal could be an outlier.

Three questions follow. Is mean pooling the right operation? Does averaging embeddings mean anything biologically? Is it even possible here?

The answers are yes, yes, and yes-cheaply. But the measurements also change *why* the feature is worth building, and that turns out to matter more than the feature itself.

## 1. What a cohort is, and how many there are

The natural grouping is the one OSDR already curates: `(study, tissue, spaceflight)` - the ISA-Tab factor grouping, all three columns of which are already in `cache/osdr_metadata.parquet`.

| | |
| --- | --- |
| Cohorts (study x tissue x condition) | **215** across 70 studies |
| Cohorts with at least 3 members | **207** |
| OSDR samples living in one | **2,095 of 2,108 (99.4%)** |
| Median cohort size | **9**; mean 9.8, max 38, only 3 singletons |

So this is not a niche feature. Essentially every retrievable OSDR sample belongs to a group of about nine.

## 2. Mean pooling is exactly "average the similarity scores"

Retrieval ranks ARCHS4 samples by cosine to the query. For a pooled query `m = (1/k) Σ v_j`:

```
cos(m, x) = (m · x) / (|m| |x|) = (1 / (k |m|)) Σ_j |v_j| · cos(v_j, x)
```

`|m|` does not depend on `x`, so it cannot change the ranking. **Ranking by the cohort mean is identical to ranking by the average of the members' own cosine scores, weighted by each member's L2 norm.**

Mean pooling is therefore not an exotic operation on a black-box vector. It is exactly the sensible thing stated in the obvious way: ask every animal, then average the votes.

### The weights are wrong, and this repo already knows why

That norm weighting is an accident, not a design. Invariant 2 records that raw embedding norms are **not** a nuisance scale but a *transcriptome-concentration* axis (r = +0.987 with the share of expression held by the top 100 genes), spanning 3.9x across the corpus - liver 13.6, brain 8.3. Averaging raw vectors therefore lets the members with the most concentrated transcriptomes cast the loudest votes, for no stated reason.

**L2-normalize each member before averaging, then normalize the result.** Then the ranking is an unweighted mean of the members' cosines, one animal one vote. This is the same argument invariant 2 makes for normalizing before any reduction, applied one level up.

It is also the textbook estimator rather than an ad-hoc fix: for data compared by cosine, the natural distribution is von Mises-Fisher, and the maximum-likelihood estimate of its mean direction is precisely the normalized sum of unit vectors. Its resultant length `R̄ = |Σ v̂_j| / k ∈ [0, 1]` is the standard concentration statistic and comes free - a ready-made honesty readout.

### How much does the normalization actually change? Almost nothing, here

Measured over all 207 cohorts:

| | |
| --- | --- |
| L2-norm spread *within* a cohort | median **1.09x** (corpus-wide: 3.9x) |
| cos(spherical centroid, raw mean) | median **0.9999994**, worst 0.99951 |
| Top-5 hits identical between the two | **13 of 14** cohorts tested |

Members of one cohort are the same tissue from the same study, so their concentrations already match and the weighting has almost nothing to bite on. Normalize anyway - it costs one line, it is the defensible estimator, and it is what keeps the operation correct if anyone ever pools across tissues, where the 3.9x spread is real.

## 3. Does averaging embeddings make biological sense? Yes, and it was checked against the biological alternative

The objection worth taking seriously: an average of embeddings need not be the embedding of anything. It could land off the manifold of vectors the encoder actually produces, which would make its cosine scores incomparable to the index's.

The biologically standard alternative is **pseudo-bulk**: sum the cohort's raw counts, then embed once. That is guaranteed to be a real transcriptome, and it is what a bench biologist means by "combine the samples". The two routes were run against each other on five real cohorts - counts summed and embedded live through `precompute/embed_upload.py`, versus the spherical centroid of the cached vectors:

| cohort | k | cos(pseudo-bulk, centroid) | cos(centroid, members) |
| --- | --- | --- | --- |
| OSD-194 Left retina, flight | 5 | 0.99970 | 0.99924 |
| OSD-580 Heart, vivarium | 8 | 0.99972 | 0.99844 |
| OSD-105 Left tibialis anterior, flight | 6 | 0.999994 | 0.999880 |
| OSD-240 dorsal skin, flight | 10 | 0.99900 | 0.99652 |
| OSD-99 EDL, flight | 6 | 0.999996 | 0.999902 |

**In all five, the pseudo-bulk embedding is closer to the spherical centroid than the cohort's own members are.** The two independent definitions of "the cohort's transcriptome" - one computed in expression space through the model, one computed in embedding space without it - land on top of each other, inside the spread of the thing they describe. The centroid's norm matches the members' mean norm too (7.60 vs 7.62, 12.78 vs 13.16), so its implied transcriptome concentration is preserved rather than invented.

The centroid is not an off-manifold artifact. Averaging embeddings is a legitimate stand-in for pooling the biology, and it needs no model, no counts files, and no subprocess.

## 4. The finding that reframes the feature

The premise was "one sample could be an outlier". The measurements say the problem is both worse and different.

**Within a cohort, members are nearly identical:** mean pairwise cosine **0.9933** (median 0.998), against **0.8826** for random same-size groups drawn from OSDR. Cohort resultant `R̄` has median 0.9991. These are tight, well-behaved groups with no outlier problem to speak of.

**And yet their retrieval results barely overlap.** Top-5 Jaccard between two members of the same cohort, averaged over pairs:

| | top-5 | top-20 | top-100 | top-500 |
| --- | --- | --- | --- | --- |
| member vs member | **0.13** | 0.22 | 0.28 | 0.45 |
| centroid vs leave-one-out centroid | **0.78** | 0.81 | 0.87 | 0.92 |

In every cohort tested there exists a pair of replicates whose top-5 lists share **nothing at all**.

The reason is a scale mismatch, and it is stark:

| | typical | range |
| --- | --- | --- |
| Cosine gap between two replicates of one cohort | **0.0030** | 0.0004 - 0.019 |
| Cosine gap from ARCHS4 rank 1 to rank 5 | **0.0005** | 0.00005 - 0.0009 |
| Cosine gap from ARCHS4 rank 1 to rank 500 | **0.0071** | 0.0003 - 0.017 |

**The entire top-500 of a 940,455-sample index spans a cosine range comparable to the distance between two animals in the same cage.** For OSD-168 liver the whole top-500 fits inside 0.0003 cosine while replicates differ by 0.0019 - replicate noise is six times wider than the band the entire result list lives in. The top-5 ordering is decided far below the noise floor of the biology.

This is the same phenomenon already recorded in `CLAUDE.md` for the rejected kNN tissue-transfer candidate, where the winning sample beat the runner-up by a median of 0.00089 cosine and the winner was "essentially arbitrary". It is a property of a dense 940k-point index in a 512-d space, not a defect in any one query.

**So the case for pooling is stronger than the mentor's argument, and rests on different ground.** Not "protect against the rare outlier" - there barely are outliers. Rather: *a single-sample top-5 is not a stable measurement at all, and averaging over the cohort is what turns it into one.* Measured, pooling raises result stability from 0.13 to 0.78, a six-fold improvement, and that is the number that justifies the feature.

### What pooling does not fix

Two limits are worth stating plainly, because the feature would otherwise over-promise.

**It does not make the top-5 a fact.** Leave-one-out stability is 0.78, not 1.0. The honest reading of a pooled result is "these came from the right neighbourhood", not "these are the five closest samples on Earth". Depth helps: agreement is 0.92 at top-500. Any UI should show the rank-1-to-rank-k score gap so a user can see how thin the margin is.

**It concentrates batch, not just biology.** Decomposing the tightness:

| grouping | mean pairwise cosine |
| --- | --- |
| random OSDR samples, same group size | 0.8826 |
| random samples from the same **study** | 0.9805 |
| the actual cohort (study x tissue x condition) | 0.9933 |

Going from random to same-study closes **84%** of the distance to a real cohort. Most of what makes a cohort coherent is the study it came from, not its tissue or its flight condition. Averaging within a study cancels the per-animal noise (as `1/sqrt(k)`) but leaves the study's batch signature untouched and therefore *relatively louder*. This is the same cross-corpus batch effect already documented at 54x above chance. A pooled query is a cleaner measurement of "this study's samples" than of "this biology".

## 5. Recommended implementation

### The estimator

```python
def cohort_query_vector(rows: np.ndarray) -> np.ndarray:
    """Spherical (von Mises-Fisher) mean of a cohort's cached embeddings."""
    u = rows / np.linalg.norm(rows, axis=1, keepdims=True)   # one animal, one vote
    m = u.mean(axis=0)
    return m / np.linalg.norm(m)
```

Report alongside it, from the same `u`:

- `R̄ = |u.mean(axis=0)|` - cohort concentration, the vMF resultant.
- Each member's cosine to the **leave-one-out** centroid, which is the correct outlier statistic. Using the full centroid instead lets an outlier pull the reference towards itself and hide.

**Do not use the medoid.** It was measured: the medoid agrees with the centroid on only 0.46 of the top-5 on average, and it is one sample, so it inherits exactly the single-sample instability the feature exists to remove.

**Do not auto-drop outliers.** Flag them and let the user exclude, matching how the app already treats unavailable samples: shown, disabled, with the reason, rather than silently hidden.

### Where it goes

It is a **fifth query-vector source**, the same shape as file ingestion (`docs/file_ingestion.md`), and cheaper than any existing path because every member's vector is already cached:

```python
def run_cohort_retrieval(sample_ids, topk, ...):
    rows  = np.stack([cached_query_vector(s) for s in sample_ids])
    q_vec = cohort_query_vector(rows)
    idx, score = _topk_cosine_from_memmap(index_vecs=..., q_vec=q_vec, k=topk)
    hits = _annotate_from_cache(idx, score)
    hits["archs4_index"] = idx.astype(int)
    return hits
```

No model, no subprocess, no torch, no new artifact. One memmap scan, so the same ~0.5 s as the cached path, and the hits carry the identical schema and the same `archs4_index` map join.

| file | change |
| --- | --- |
| `bridge_rna/retrieval.py` | `cohort_query_vector`, `run_cohort_retrieval`, `cohort_members`, mode `"cohort"` |
| `bridge_rna/callbacks.py` | a `"cohort"` entry in `_retrieval_phrase`, a synthesized cohort query row |
| `bridge_rna/layout.py` | a "Query the whole group (k samples)" toggle under the sample picker |
| `bridge_rna/panels.py` | cohort membership, `R̄`, and the outlier list in the inspector |

The banner must name the path, as it must for every other mode. The query node should be labelled with the cohort and its size, never with one member's name.

### What must be measured before shipping

Mirroring how every other candidate in this repo was accepted or rejected:

1. **Leave-one-out stability** of the pooled top-k, per cohort. Already measured at 0.78 for top-5. Ship the number in the UI; it is the feature's own honesty statistic.
2. **A structure-free null.** A pooled query over `k` *random* OSDR samples must be measurably worse than a pooled query over a real cohort of size `k`. If a random group is just as stable, pooling is only averaging and the cohort definition is doing nothing.
3. **A within-study null**, which is the sharper test given that study batch supplies 84% of a cohort's coherence: pool `k` random samples *from the same study* and compare. This isolates what tissue and condition contribute.

Point 3 is the one most likely to be uncomfortable, and it is the one worth running first.

### One idea to leave alone for now

Pooling flight and ground arms separately and taking the difference, `Δ = centroid(flight) - centroid(ground)`, cancels the shared study batch and is the standard differential-expression move. It is tempting and it does not belong in this view, for two reasons.

A difference of two unit vectors is not a transcriptome. Cosine-ranking ARCHS4 against it asks which GEO sample's *absolute* expression profile most resembles a *change*, which is a category error - the index holds profiles, not contrasts.

And the corpus-level version of this was already built, measured, and rejected: the flight-minus-ground "spaceflight-likeness" axis correlated r = -0.990 with PC1, which is the transcriptome-concentration axis, and one in ten random flight/ground relabelings beat it. A within-study paired contrast is a better-posed object than that global axis, but it answers a different question than "what on Earth looks like this", and it would need its own within-study permutation null before anyone trusted it.

## Reproducing these numbers

The three probes that produced this document are single-file scripts, in order: cohort structure and geometry plus a 492-query streaming scan; a verification of that scan against the app's own top-k plus agreement at depths 5/20/100/500; and the pseudo-bulk comparison, which shells out to `precompute/embed_upload.py` exactly as the upload path does. Each streams the ARCHS4 memmap in 50,000-row blocks keeping a running top-k, which is the same technique `precompute/validate_artifacts.py --mixing` uses and the reason no ANN index is needed.
