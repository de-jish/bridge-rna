# Bridge RNA design notes

Every design decision in this repository that needed more than a code comment, in one file.

There used to be eight of these, one per feature, and they cross-referenced each other twenty times over.
The split cost more than it bought: a decision about the cohort estimator lived in one file, the measurement that justified it in a second, and the panel that displays it in a third, so following one thread meant opening three documents that each restated the other two.
Nothing has been dropped in the merge - every measurement, every rejected alternative, and every "and here is why the obvious thing is wrong" survives - and the cross-references are now anchors in this file.

**What this is for.** Each section records what was measured, what was built, and what was deliberately *not* built.
The last of those is the part worth keeping: several of the features below were built in full, measured on the real corpus, and then deleted, and without the numbers written down somebody will build them again.

`CLAUDE.md` is the summary and the working brief; this is the evidence behind it.
`REFERENCE.md` holds verified ground-truth facts about the artifacts themselves.
`progress.md` is the running log.

## Contents

| section | what it decides |
| --- | --- |
| [Cohort pooling](#cohort-pooling) | whether pooling a group beats querying one sample, and by how much |
| [Cohort retrieval](#cohort-retrieval) | what a cohort is, and how the pooled query is built and shown |
| [Live stability](#live-stability) | measuring a result's stability on the query that just ran, not from a curve |
| [The even split](#stability-panel-even-split) | laying a comparison's two arms out as equals |
| [The map key](#map-key) | decoding every mark the map can draw |
| [Finding a sample](#finding-a-sample-on-the-map) | locating one sample among 942,563, and the probe that was cut |
| [Completing an identifier](#find-autocomplete) | suggesting what is being typed, without becoming a free-text search |
| [One reset for every framing](#reset-view) | undoing a frame, whoever narrowed the view |
| [The OSDR-only color-bys](#osdr-only-color-bys) | why nine fields were removed and what the machinery is still for |
| [File ingestion](#file-ingestion) | embedding an uploaded counts file and retrieving against it |
| [README screenshots](#readme-screenshots) | capturing the two images by measurement rather than by eye |

---

<a id="cohort-pooling"></a>

## Cohort pooling: querying with an experimental group instead of one sample

**Status: measured and specified, not built.**
Every number here was measured on the real corpus - the 2,108 cached OSDR embeddings and the 940,455-row ARCHS4 memmap - on 2026-07-30.
The scan used to produce them reproduces `bridge_rna.retrieval._topk_cosine_from_memmap` exactly, indices and scores identical to six decimal places, so these are the app's own answers and not an approximation of them.

### The question

The Retrieve view answers for one OSDR sample at a time.
A spaceflight study does not have one sample; it has an experimental group - six to thirty-eight animals sharing a study, a tissue, and a flight-or-ground condition.
Josh's mentor's proposal: combine a group into one query so the answer describes the group rather than whichever animal happened to be picked, because any single animal could be an outlier.

Three questions follow. Is mean pooling the right operation? Does averaging embeddings mean anything biologically? Is it even possible here?

The answers are yes, yes, and yes-cheaply. But the measurements also change *why* the feature is worth building, and that turns out to matter more than the feature itself.

### 1. What a cohort is, and how many there are

The natural grouping is the one OSDR already curates: `(study, tissue, spaceflight)` - the ISA-Tab factor grouping, all three columns of which are already in `cache/osdr_metadata.parquet`.

| | |
| --- | --- |
| Cohorts (study x tissue x condition) | **215** across 70 studies |
| Cohorts with at least 3 members | **207** |
| OSDR samples living in one | **2,095 of 2,108 (99.4%)** |
| Median cohort size | **9**; mean 9.8, max 38, only 3 singletons |

So this is not a niche feature. Essentially every retrievable OSDR sample belongs to a group of about nine.

### 2. Mean pooling is exactly "average the similarity scores"

Retrieval ranks ARCHS4 samples by cosine to the query. For a pooled query `m = (1/k) Σ v_j`:

```
cos(m, x) = (m · x) / (|m| |x|) = (1 / (k |m|)) Σ_j |v_j| · cos(v_j, x)
```

`|m|` does not depend on `x`, so it cannot change the ranking. **Ranking by the cohort mean is identical to ranking by the average of the members' own cosine scores, weighted by each member's L2 norm.**

Mean pooling is therefore not an exotic operation on a black-box vector. It is exactly the sensible thing stated in the obvious way: ask every animal, then average the votes.

#### The weights are wrong, and this repo already knows why

That norm weighting is an accident, not a design. Invariant 2 records that raw embedding norms are **not** a nuisance scale but a *transcriptome-concentration* axis (r = +0.987 with the share of expression held by the top 100 genes), spanning 3.9x across the corpus - liver 13.6, brain 8.3. Averaging raw vectors therefore lets the members with the most concentrated transcriptomes cast the loudest votes, for no stated reason.

**L2-normalize each member before averaging, then normalize the result.** Then the ranking is an unweighted mean of the members' cosines, one animal one vote. This is the same argument invariant 2 makes for normalizing before any reduction, applied one level up.

It is also the textbook estimator rather than an ad-hoc fix: for data compared by cosine, the natural distribution is von Mises-Fisher, and the maximum-likelihood estimate of its mean direction is precisely the normalized sum of unit vectors. Its resultant length `R̄ = |Σ v̂_j| / k ∈ [0, 1]` is the standard concentration statistic and comes free - a ready-made honesty readout.

#### How much does the normalization actually change? Almost nothing, here

Measured over all 207 cohorts:

| | |
| --- | --- |
| L2-norm spread *within* a cohort | median **1.09x** (corpus-wide: 3.9x) |
| cos(spherical centroid, raw mean) | median **0.9999994**, worst 0.99951 |
| Top-5 hits identical between the two | **13 of 14** cohorts tested |

Members of one cohort are the same tissue from the same study, so their concentrations already match and the weighting has almost nothing to bite on. Normalize anyway - it costs one line, it is the defensible estimator, and it is what keeps the operation correct if anyone ever pools across tissues, where the 3.9x spread is real.

### 3. Does averaging embeddings make biological sense? Yes, and it was checked against the biological alternative

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

### 4. The finding that reframes the feature

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

#### What pooling does not fix

Two limits are worth stating plainly, because the feature would otherwise over-promise.

**It does not make the top-5 a fact.** Leave-one-out stability is 0.78, not 1.0. The honest reading of a pooled result is "these came from the right neighbourhood", not "these are the five closest samples on Earth". Depth helps: agreement is 0.92 at top-500. Any UI should show the rank-1-to-rank-k score gap so a user can see how thin the margin is.

**It concentrates batch, not just biology.** Decomposing the tightness:

| grouping | mean pairwise cosine |
| --- | --- |
| random OSDR samples, same group size | 0.8826 |
| random samples from the same **study** | 0.9805 |
| the actual cohort (study x tissue x condition) | 0.9933 |

Going from random to same-study closes **84%** of the distance to a real cohort. Most of what makes a cohort coherent is the study it came from, not its tissue or its flight condition. Averaging within a study cancels the per-animal noise (as `1/sqrt(k)`) but leaves the study's batch signature untouched and therefore *relatively louder*. This is the same cross-corpus batch effect already documented at 54x above chance. A pooled query is a cleaner measurement of "this study's samples" than of "this biology".

### 5. Recommended implementation

#### The estimator

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

#### Where it goes

It is a **fifth query-vector source**, the same shape as file ingestion ([File ingestion: embed an uploaded OSDR sample live and retrieve its Earth analogs](#file-ingestion)), and cheaper than any existing path because every member's vector is already cached:

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

#### What must be measured before shipping

Mirroring how every other candidate in this repo was accepted or rejected:

1. **Leave-one-out stability** of the pooled top-k, per cohort. Already measured at 0.78 for top-5. Ship the number in the UI; it is the feature's own honesty statistic.
2. **A structure-free null.** A pooled query over `k` *random* OSDR samples must be measurably worse than a pooled query over a real cohort of size `k`. If a random group is just as stable, pooling is only averaging and the cohort definition is doing nothing.
3. **A within-study null**, which is the sharper test given that study batch supplies 84% of a cohort's coherence: pool `k` random samples *from the same study* and compare. This isolates what tissue and condition contribute.

Point 3 is the one most likely to be uncomfortable, and it is the one worth running first.

#### One idea to leave alone for now

Pooling flight and ground arms separately and taking the difference, `Δ = centroid(flight) - centroid(ground)`, cancels the shared study batch and is the standard differential-expression move. It is tempting and it does not belong in this view, for two reasons.

A difference of two unit vectors is not a transcriptome. Cosine-ranking ARCHS4 against it asks which GEO sample's *absolute* expression profile most resembles a *change*, which is a category error - the index holds profiles, not contrasts.

And the corpus-level version of this was already built, measured, and rejected: the flight-minus-ground "spaceflight-likeness" axis correlated r = -0.990 with PC1, which is the transcriptome-concentration axis, and one in ten random flight/ground relabelings beat it. A within-study paired contrast is a better-posed object than that global axis, but it answers a different question than "what on Earth looks like this", and it would need its own within-study permutation null before anyone trusted it.

### Reproducing these numbers

The three probes that produced this document are single-file scripts, in order: cohort structure and geometry plus a 492-query streaming scan; a verification of that scan against the app's own top-k plus agreement at depths 5/20/100/500; and the pseudo-bulk comparison, which shells out to `precompute/embed_upload.py` exactly as the upload path does. Each streams the ARCHS4 memmap in 50,000-row blocks keeping a running top-k, which is the same technique `precompute/validate_artifacts.py --mixing` uses and the reason no ANN index is needed.

---

<a id="cohort-retrieval"></a>

## Cohort retrieval: querying with an experimental group

**Status: built, measured on the real corpus, and tested, 2026-08-05.**
**Amended 2026-08-06:** the confidence readout this document put on the rail is now measured per query and reported on the right; [Result stability, measured on the query that just ran](#live-stability) is the current design, and the passages it supersedes are marked in place rather than rewritten.
This is the implementation document for the feature [Cohort pooling: querying with an experimental group instead of one sample](#cohort-pooling) specified and measured.
That document is the prior evidence; this one is the build, and it carries its own measurements.
Every number below was produced by `precompute/validate_cohorts.py` against the real 963 MB ARCHS4 memmap and the real 2,108 cached OSDR embeddings, over **all 212 cohorts**, not a sample of them.

| | |
| --- | --- |
| Cohorts under the default definition | **212** with two or more samples, across 70 studies (215 including 3 singletons) |
| Size | median 10, mean 9.9, max 38; 2,105 of 2,108 samples grouped |
| Pooled top-5 leave-one-out stability | **0.738**, against **0.161** for one sample: a **4.6x** gain |
| Against a structure-free null | 0.331, so the cohort definition is worth **+0.407** |
| Against a within-study null | 0.683, so tissue and arm are worth **+0.055** on top of the study |
| Cost of a pooled query | one memmap pass at any k. Since 2026-08-06 that pass also measures the result's stability, so it carries `2k+1` query vectors: 0.44 s for the pooled query alone, 1.00 s at the 38-animal maximum |
| Tests | 55 unit tests, 146 browser checks, 6 corpus-scale validation checks |

### Why this exists, in one paragraph

The Retrieve view answers for one OSDR sample at a time, and a single-sample top-5 is not a stable measurement.
Two replicates from the same cage share on average **0.13** of their top-5 ARCHS4 hits, and in every cohort tested there is a pair whose top-5 lists share nothing at all.
The cause is a scale mismatch rather than an outlier problem: the entire top-500 of a 940,455-sample index spans a cosine range comparable to the gap between two animals in the same cage, so the ordering of the result list is decided below the noise floor of the biology.
Pooling the cohort raises leave-one-out top-5 agreement from 0.13 to **0.78**, a six-fold gain.
That is the case for this feature, and it is a different case from the one that motivated it.

### 1. What a cohort is

#### The default

A cohort is a set of OSDR samples that share a **study**, a **tissue**, and a **spaceflight arm**.
This is the ISA-Tab factor grouping OSDR already curates, so the tool is reading a grouping that exists rather than inventing one.
Measured on the shipped metadata: **215 cohorts across 70 studies**, median size 9, mean 9.8, max 38, and 2,095 of the 2,108 embedded samples live in one.

The arm is the **raw** OSDR value, not the binary Flight-vs-Ground collapse.
`manifold/data._flight_status` already records why: a basal animal was sacrificed at experiment start and a vivarium animal never entered flight hardware, so the seven control arms are not interchangeable.
Pooling a Vivarium Control with a Basal Control would average two different experiments and call the result one group.

#### Why study is pinned and cannot be unticked

Random samples drawn from the same *study* already reach mean pairwise cosine 0.9805, against 0.9933 for a real cohort and 0.8826 for random OSDR samples.
Same-study membership therefore supplies **84%** of what makes a cohort coherent, and most of that is batch rather than biology.
Pooling *across* studies would average across the strongest batch boundary in the corpus, so study is a fixed facet.
The UI shows it as a pinned chip that cannot be removed, with the reason on hover, rather than hiding it.

#### What the user controls

Three facets are available, drawn from `cache/osdr_metadata.parquet`, which is exactly the table of the 2,108 samples that have a cached embedding.

| facet | default | note |
| --- | --- | --- |
| Study | **pinned on** | cannot be removed, for the reason above |
| Tissue | on | |
| Spaceflight arm | on | raw arm, seven values |

Unticking one merges cohorts and makes them larger: dropping Tissue pools a study's organs, dropping Spaceflight arm pools its flight animals with its controls.
The cohort count and the size distribution update live under the chips.

**Six further facets were offered and were removed on 2026-08-05: sex, strain, genotype, habitat, mission duration, diet.**
They came from the same parquet and cost nothing to offer, which is exactly why they were there, and that turned out to be the wrong reason.
Every one of them could only ever *split* a cohort, and the measured stability curve in section 3 is a function of size: a cohort of 10 reads 0.81 and a cohort of 3 reads 0.51.
So the six controls did nothing but trade away the quantity this whole feature exists to buy, in exchange for a contrast the two-arm comparison in section 4 answers directly and attributably.
They also made the *default* harder to read, because nine chips of which six are off looks like a definition you are expected to tune rather than the curated grouping OSDR already publishes.
The columns are still in the parquet, still resolved by the map's color-by registry, and re-adding one is a single line in `bridge_rna/cohorts.FACETS`; nothing else in the app hard-codes a facet.

A fourth control sits below the cohort picker: the **member list**, where any individual sample can be excluded from the pool.
Nothing is ever auto-excluded.

### 2. The estimator

```python
def cohort_query_vector(rows: np.ndarray) -> np.ndarray:
    """Spherical (von Mises-Fisher) mean of a cohort's cached embeddings."""
    u = rows / np.linalg.norm(rows, axis=1, keepdims=True)   # one animal, one vote
    m = u.mean(axis=0)
    return m / np.linalg.norm(m)
```

Each member is L2-normalized **before** averaging.
Without that, `cos(mean, x)` is the members' cosines weighted by their L2 norms, and invariant 2 establishes that the norm is a transcriptome-concentration axis spanning 3.9x across the corpus, so the most concentrated transcriptomes would cast the loudest votes for no stated reason.
Within a real cohort the spread is only 1.09x and the two estimators agree to a median cosine of 0.9999994, so this changes almost nothing today.
It is done anyway because it is the maximum-likelihood estimator for data compared by cosine, and because it stays correct if anyone ever unticks Tissue and pools across organs, where the 3.9x spread is real.

Two statistics fall out of the same `u`, and both are shown:

- **Each member's cosine to the leave-one-out centroid.** This is the correct outlier statistic. Using the full centroid instead lets an outlier pull the reference towards itself and hide inside it.
- **Expected top-5 stability given k**, read off the measured curve in `precompute/validate_cohorts.py`. *This second one was replaced on 2026-08-06 by the same statistic measured on the query that just ran, at the depth on screen; see section 3 and [Result stability, measured on the query that just ran](#live-stability). It is no longer a function of the cohort's size, because it is no longer an estimate.*

A third was shown and **was removed on 2026-08-05**: `R̄ = |u.mean(axis=0)|`, the vMF resultant length, labelled "Group tightness" on the card.
It is a real statistic and it is measured in [Cohort pooling: querying with an experimental group instead of one sample](#cohort-pooling), but as a readout it was inert.
Across all 212 real cohorts its median is **0.9991**, and it is no lower for a cohort of two than for one of thirty, so it never separated a group worth trusting from one that was not.
A number that is always within a thousandth of its maximum, sitting on a card beside a number that genuinely varies, is read as a grade rather than as a constant, which is the opposite of what it says.
The per-member leave-one-out cosine stays, because that one does vary within a cohort and names an individual animal a user can act on.

The medoid was measured and rejected: it agrees with the centroid on only 0.46 of the top-5, and being one sample it inherits exactly the single-sample instability the feature exists to remove.

### 3. Low N, and what the interface says about it

> **Superseded in part on 2026-08-06.** The stability figure this section put on the rail's cohort card is gone; it is measured per query and reported on the right after the search instead. [Result stability, measured on the query that just ran](#live-stability) is the current design and carries the reason. What survives here is the size treatment in the picker, which is the one thing that can honestly be said before a search has run.

A cohort of two is still a cohort, and it is still better than one sample.
It is not as good as a cohort of nine, and the picker has to say so without either hiding the result or crying wolf.

Three states, and the threshold comes from the measured stability-versus-k curve rather than from taste:

| k | state | treatment |
| --- | --- | --- |
| 1 | not a cohort | disabled in the picker, reason shown: pooling needs at least two samples |
| 2 to 4 | low N | selectable and searchable, marked "low N" in the picker's option label |
| 5 and up | normal | no mark |

That is a statement about **size**, made in the one place where size is all that is known.
The card underneath states the size and nothing else.

The amber flag moved with the number: it now fires on the *measured* stability falling under `STABILITY_FLOOR`, which is the same 0.70 that picked `LOW_N_THRESHOLD`, and it appears in the stability panel beside the number that triggered it.
Amber rather than red, for the same reason the map's coverage bar is amber: a result that moves when you drop an animal is reporting correctly, not failing.

#### The curve, and why it is bucketed

Measured leave-one-out top-5 agreement, over all 212 cohorts:

| k | cohorts | stability | sd |
| --- | --- | --- | --- |
| 2 | 5 | 0.34 | 0.20 |
| 3 | 22 | 0.51 | 0.27 |
| 4 | 8 | 0.55 | 0.17 |
| 5-9 | 70 | 0.72 | 0.18 |
| 10-14 | 70 | 0.81 | 0.12 |
| 15+ | 37 | 0.86 | 0.11 |

`LOW_N_THRESHOLD` is 5 because k >= 5 is the first bucket to reach 0.70.
It is one constant in `bridge_rna/cohorts.py` carrying the measurement in its comment, so it cannot drift from the evidence, and `validate_cohorts.py` check 5 exits non-zero if a full sweep moves the knee away from it.

**This curve set that threshold and no longer does anything else.**
It was `cohorts.STABILITY_BY_K` until 2026-08-06, quoted on the card, and the sd column above is why it could not stay there: at 0.18 within the 5-9 bucket, "0.72" covers real cohorts measuring 0.316 and 0.849.
A population average printed beside one cohort's name is read as a property of that cohort.

Two things about this table were corrections rather than choices, and both are worth keeping.

**The first sweep quoted per-size figures and they were noise.** Sampling two cohorts per size produced 0.38 at k=5 sitting beside 0.90 at k=6, which is a fact about which two cohorts were drawn and not about size. A number quoted in the interface has to stand on enough cohorts that it does not swing by half its range when the seed changes, so sizes are pooled into buckets and each bucket reports its count and its spread.

**Adjacent buckets inverted, and merging them is the honest repair.** Even over all 212 cohorts, 5-6 scored 0.736 and 7-9 scored 0.696, an inversion of 0.04 against a within-bucket sd of 0.18. A larger cohort must never be reported as less trustworthy than a smaller one, so `validate_cohorts.py` merges adjacent buckets that invert before printing the curve, which is what produced the 5-9 row. Clamping the number instead would have invented monotonicity; shipping the raw pair would have told a researcher their seven-animal cohort was worse than a five-animal one. The unit test that pinned the shipped curve monotone went with the curve on 2026-08-06; `validate_cohorts.py` still does the merging, and what it now gates is that the knee still lands on `LOW_N_THRESHOLD`.

### 4. Two arms, run as two queries

An optional **compare against** picker runs a sibling cohort as a second, independent pooled query, and draws both on one network.
The number it produces is the **overlap between the two hit sets**, which answers a real question: do this study's flight animals and its ground controls land in the same part of Earth's transcriptome space, or different parts?

What it deliberately is **not** is the difference vector `centroid(flight) - centroid(ground)`.
That is the standard differential-expression move and it does not belong here for two reasons.
A difference of two unit vectors is not a transcriptome, so cosine-ranking ARCHS4 against it asks which GEO sample's *absolute* profile most resembles a *change*, which is a category error against an index that holds profiles.
And the corpus-level version was already built, measured and rejected: the flight-minus-ground axis correlated r = -0.990 with PC1, which is the transcriptome-concentration axis, and one in ten random flight/ground relabelings beat it on spatial structure.

### 5. Where the code goes

#### A fifth query-vector source, not a new pipeline

The cosine scan, `_annotate_from_cache`, and the `archs4_index` map join are the cached path's, reused unchanged, exactly as file ingestion reuses them.
So a cohort hit carries the same schema as a single-sample hit, and everything downstream of the hits frame keeps working without knowing a cohort produced it.

```python
def run_cohort_retrieval(sample_ids, topk):
    rows = np.stack([cached_query_vector(s) for s in sample_ids])
    # Row 0 answers the query; the rest exist only to measure how far it
    # survives dropping one member, and are scored in the same pass.
    q_mat = np.concatenate([cohort_query_vector(rows).reshape(1, -1),
                            leave_one_out_vectors(rows), rows])
    idx, score = _topk_cosine_matrix(index_vecs=..., q_mat=q_mat, k=topk)
    hits = _annotate_from_cache(idx[0], score[0])
    hits["archs4_index"] = idx[0].astype(int)
    return hits, rows, measure_stability(sample_ids, idx[0], ..., depth=topk)
```

No model, no subprocess, no torch and no new artifact, and still **one** memmap scan whatever the cohort size.
The scan carries `2k+1` query vectors rather than one since 2026-08-06, which is what makes result stability a measurement instead of a lookup: 0.44 s for the pooled query alone, 0.50 s at the median cohort size of 10, and 1.00 s for the 77 vectors a 38-animal cohort needs.
[Result stability, measured on the query that just ran](#live-stability) section 4 has the full table and the reason the extra queries are nearly free.

| file | change |
| --- | --- |
| `bridge_rna/cohorts.py` | **new.** Facet registry, cohort construction, geometry, low-N tiering. The only file that knows what a cohort is. |
| `bridge_rna/retrieval.py` | `run_cohort_retrieval`, mode `"cohort"` |
| `bridge_rna/callbacks.py` | a `"cohort"` entry in `_retrieval_phrase`, the mode switch, the cohort callbacks, a synthesized cohort query row |
| `bridge_rna/layout.py` | the segmented Sample / Cohort / Upload switch and the cohort panel |
| `bridge_rna/panels.py` | the cohort inspector: membership, per-member LOO cosine, the overlap readout |
| `bridge_rna/figures.py` | a pooled query node, and a two-query network for the compare case |
| `manifold/callbacks.py` | `_retrieval_overlay` draws every pooled member on the map, not one |
| `precompute/validate_cohorts.py` | **new.** The honesty gate. |

`bridge_rna/cohorts.py` is a separate module rather than more of `retrieval.py` because it depends on no embedding and no memmap at all.
It is pure metadata grouping plus 512-d arithmetic, it can be tested against the fixture corpus without either artifact, and keeping it apart is what stops `retrieval.py` from growing a second responsibility.

#### The status banner

`_retrieval_phrase("cohort")` must name the path, as it must for every mode.
The invariant this repo already broke once, when every cached result was announced as demo-script output, is that the interface always says which path answered.
A pooled result must never be labelled with one member's name; the query node carries the cohort's name and its size.

### 6. How we know it works

`precompute/validate_cohorts.py`, run against the real 963 MB memmap, and mirroring how every other candidate in this repo was accepted or rejected.
It computes every query vector it needs up front and streams the memmap **once**, keeping a running top-k per query, which is the same technique `validate_artifacts.py --mixing` uses and the reason hundreds of queries cost one pass rather than hundreds.

Six checks, all passing as of 2026-08-05.
The whole run is 9,270 query vectors scored in a single 73-second pass over the memmap.

**1. Identity, and what it taught.**
A pooled query over one sample must reproduce that sample's cached-path result, or the pooling code is not sitting on the path it claims to reuse.

The first version demanded an identical top-100 and **failed**, and the reason turned out to be worth more than the check.
Pooling one sample normalizes it twice, once inside `cohort_query_vector` and once inside the scan, so the query vector differs from the plain one by **7.45e-9**, a single float32 ulp, at cosine 1.0.
Scores then differ by at most **1.19e-7**, which is float32 epsilon at magnitude 1.
That is enough to reorder the list: the first differing rank is 23, and the score gap between rank 23 and rank 24 there is **exactly 0.0**.
The two runs are permuting an exact tie rather than disagreeing, and through rank 50 they are still the same set.

So the gate is that scores agree to float32 and the depth a user actually reads is identical in order, and the measured divergence depth is printed rather than hidden.
Demanding more would be demanding that float32 have more precision than it has, and it would be the same over-reading of a hairline score gap that this whole feature exists to stop.
*Correctness: a failure exits non-zero.*

**2. Leave-one-out stability.**
Full-cohort top-k against each leave-one-out top-k, versus member-against-member:

| depth | pooled | member vs member | gain |
| --- | --- | --- | --- |
| top-5 | **0.738** | 0.161 | 4.6x |
| top-20 | 0.778 | 0.214 | 3.6x |
| top-100 | 0.826 | 0.302 | 2.7x |

**3. A structure-free null.**
k random OSDR samples pooled score **0.331**, so a real cohort beats an arbitrary group of the same size by **+0.407**.
Pooling is not merely averaging, and the cohort definition is doing most of the work.
*Correctness: a failure exits non-zero.*

**4. A within-study null.**
k random samples drawn from the cohort's own study, ignoring tissue and arm, score **0.683**.
So tissue and arm together are worth **+0.055** on top of what same-study membership already supplies.

This is the uncomfortable number, and it is the one to quote honestly rather than bury.
It is consistent with the earlier finding that study alone closes 84% of the distance to a real cohort.
It does not undermine the feature, since a pooled query is still 4.6x more stable than a single sample and that is the claim the interface makes.
It does mean a pooled result is a cleaner measurement of "this study's samples" than of "this biology", and the docs say so rather than letting the gain be read as purely biological.

**5. Stability versus k.**
The bucketed curve in section 3, which sets `LOW_N_THRESHOLD` and nothing else since 2026-08-06.
The check now compares the measured knee against the shipped threshold and fails on a mismatch, but only on a full sweep: a `--cohorts` sample draws one or two cohorts per size, which is the regime that produced 0.38 at k=5 beside 0.90 at k=6 in the first run of this script.

**6. Normalization.**
Over all 212 cohorts, `cos(spherical mean, raw mean)` has median **0.9999995** and worst case **0.99951**, and the two agree on the exact top-5 in 112 of 212 cohorts.
Exactly as predicted: within one cohort the L2-norm spread is about 1.09x, so the weighting has almost nothing to bite on.
Spherical is kept because it becomes the correct estimator the moment anyone unticks Tissue and the corpus-wide 3.9x spread is real.

### 7. Interface

The left rail gets a segmented control at the top: **Sample / Cohort / Upload**.
Only one query source is visible at a time, so the rail gets *shorter* than the current stacked "Query sample" plus "Or upload a sample" arrangement rather than longer.
The canvas, the inspector, the AI hypothesis, the top-k slider and the map hand-off are all shared and unchanged, because a cohort is a query like any other.

Cohort mode, top to bottom:

1. **OSDR study** dropdown, the same one Sample mode uses.
2. **Group by**, a row of facet chips. Study is pinned. A line under it reports how many cohorts the current definition produces and their size range.
3. **Cohort** dropdown, listing this study's cohorts with size and confidence state. Singletons are disabled with the reason, matching how the sample picker treats an unretrievable sample.
4. **The cohort card**: k pooled, and since 2026-08-06 nothing else.
5. **Members**, a collapsed disclosure listing every sample with its leave-one-out cosine, each with a checkbox. Any member flagged as an outlier is marked, never removed.
6. **Compare against**, an optional sibling-cohort picker, empty by default.
7. **A second cohort card**, when one is armed. Added 2026-08-06.
8. **Search cohort**.

The right-hand inspector gained a **stability panel** above the details panel on 2026-08-06, populated after the search, carrying one block per pooled query.
It is not on the rail because it is a property of the result rather than of the selection, and it does not exist until the query runs.

The rail's existing rule still holds where it can: the fact that qualifies a control sits directly under that control.
The cohort-count line hangs under the facet chips, and each cohort's size sits under the picker that chose it.

Step 7 exists because a comparison runs **two** independent pooled queries and the rail described one of them.
That is not a symmetry argument. Stability is a property of each arm on its own, so an overlap of 0.25 between a 12-animal cohort measuring 0.86 and a 2-animal cohort measuring 0.31 means something quite different from the same 0.25 between two cohorts of twelve - and the number that decides how much of the overlap to believe was on screen for the first arm only, while the second got a color in the network figure and a mark on the map.
The stability panel splits along the same line, for the same reason.
It sits under "Compare against" rather than beside the first card for the rule above, and because that ordering keeps the member ticks - which belong to cohort A alone - next to cohort A's card.
Both cards take a role line (`● COHORT A`, `● COHORT B · differs by spaceflight arm`) **only when there are two**; a lone cohort gets no letter, since there is nothing to tell it apart from.
The contrast facet is stated once, on the second card, because it is a property of the pair.

### 8. Testing

Three layers, each answering a question the others cannot.

**`tests/test_cohorts.py`, 55 tests, against the synthetic fixture corpus.**
Facet grouping, the estimator, leave-one-out cosines, low-N tiering, the pinned-study rule, and the sibling relation.
Two are worth calling out because they pin claims made in prose everywhere else.
`test_pooled_ranking_is_the_mean_of_the_members_own_cosines` checks the central algebraic claim directly: ranking by cosine to the spherical mean is identical to ranking by the unweighted average of the members' own cosines, which is what "ask every animal, then average the votes" means.
`test_one_animal_one_vote_regardless_of_transcriptome_concentration` scales one member's norm by 10 and asserts the pooled direction is unchanged, and then asserts that the *raw* mean would have been dragged, so the test cannot pass vacuously.

**`precompute/validate_cohorts.py`, 6 checks, against the real corpus.**
Section 6. This is the only layer that can speak to whether pooling works, as opposed to whether it computes what it says.

**`tests/e2e_cohort_check.py`, 146 browser checks, against the real app and the real cache.**
Define a cohort, retick facets, watch the count change, read the confidence card, open the member list, pool and search, open the inspector, exclude a member and watch every number restate, compare two arms, and follow the whole cohort to the map.
It asserts on what the page reports about itself, and two of its checks exist because this feature shipped those exact regressions and had them fixed: callbacks firing at page load so the canvas greeted a visitor with "Cohort retrieval failed", and the legend continuing to advertise a GSE column while a comparison that draws none was on screen.

**One trap, recorded because it cost a debugging cycle.**
The obvious wait predicates for a search - "the network has nodes", "the spinner is idle" - are both already true when a *second* search starts, so waiting on them returns instantly and the check then reads the previous result's banner.
The comparison step appeared to fail while the app was doing exactly the right thing.
`run_cohort_search` now waits for the status banner to change, and uses the spinner only as a secondary settle.

### 9. Both cohorts on the map

**Status: designed and built 2026-08-05.**

Until now a comparison was invisible on the map.
`_retrieval_overlay` read `member_ids` and `hits`, both of which describe cohort A only, and ignored `payload["comparison"]` entirely.
So a user who ran flight against ground and then walked to the map saw one arm, one set of hits, a badge reading "Showing retrieval: 5 hits", and a rail line naming cohort A - with nothing anywhere saying a second cohort had been retrieved and was not being drawn.

That is not a limitation, it is an omission, and the distinction matters because option A - keep it main-only and say so - would have cost the same UI work as fixing it.
The choice was between spending that work on a disclaimer and spending it on the answer.

#### Why the map is the right place to settle a comparison

The comparison reports a **Jaccard overlap between two hit sets**, and section 4 states the question it stands for: do this study's flight animals and its ground controls land in the same part of Earth's transcriptome space?
Set overlap and spatial coincidence come apart in both directions, and this corpus makes that the common case rather than an edge case.
Two cohorts can share **zero** hits and sit in one tissue neighbourhood, each retrieving different GEO samples from the same crowd - Jaccard says 0.00 and the honest answer is "the same place, resolved finer than k=5".
The reverse happens too: an overlap of 0.4 where the shared hits are generic and each cohort's exclusive hits sit in different territories.
Neither reading is available from the network figure, which has no space in it.
Drawing both cohorts is the structure-free check on the headline number, which is how every other claim in this repo was accepted or rejected.

#### Hue tells you the cohort; shape tells you who retrieved a hit

The obvious move - lift the comparison network's blue, warm and teal onto the map - was measured and is dead.
Against the worst of the eleven `CATEGORICAL` tissue hues on `PLOT_BG`, the network's cohort-A `#2b7fff` measures **1.03:1**, its cohort-B `#d9791b` **1.00:1**, and its shared `#0bab9f` **1.07:1**.
All three vanish over large tissue buckets.
That is the same measurement `manifold/theme.py` already records as the reason a hit ring is white at all: `#2b7fff` against `CATEGORICAL[0]` (the 155,761-point Blood / immune bucket, 16.6% of the corpus) is 1.03:1, and white is 3.64:1 against it and 16.9:1 against the background.
A two-cohort map can afford a lost hit even less than a one-cohort map, because a lost hit is now also a lost group assignment.

So the two channels are split by what each can carry:

| what | channel | A | B |
| --- | --- | --- | --- |
| pooled member | **fill hue** | teal `RETRIEVAL_QUERY` | gold `RETRIEVAL_QUERY_B` |
| retrieved hit | **ring shape**, always white | `circle-open`, size 20 | `square-open`, size 27 |

Hue is safe on a member because a member is a *filled* mark with a 2 px white outline: its findability comes from the outline, so the fill only has to be discriminable from the other cohort's fill.
Gold `#ffc233` against teal `#0bab9f` measures CIEDE2000 **43.4** in normal vision and **31.7 / 45.0 / 48.0** under protanopia, deuteranopia and tritanopia, every one far above the 8.4 CVD bar the categorical palette was validated to.
It is 10.5:1 against `PLOT_BG` and 17.0 dE from the nearest categorical hue, so it cannot be read as a legend row.
`ACCENT_WARM #d9791b`, the network's own cohort-B color, was rejected for the map specifically: it sits 0.3 dE from `CATEGORICAL[3]` under deuteranopia.

Cohort A keeps teal deliberately.
A comparison is then the single-cohort case plus a second thing, rather than a new color scheme, so a single-sample map, a single-cohort map and the first arm of a comparison all draw the same mark.

**A shared hit is emergent, never computed.** It is in both hit lists, so it receives both traces: a 20 px ring inscribed in a 27 px square. There is no third symbol and no set intersection in the renderer, which means the shared set drawn on the map cannot drift from the `comparison["shared"]` the banner quotes, because it is not calculated twice.

`square-open` and `circle-open` are both in `Scatter3d`'s small symbol vocabulary, so the hit encoding is identical in 2-D and 3-D - which no hue scheme would have managed, since `star` is rejected outright in 3-D and already falls back to `diamond`.

#### Three things this fixed on the way

**The halo scaled with membership and should not have.** Every query point got a 46 px ring at 0.50 alpha, so a 38-animal cohort composited into a teal disc - before any comparison existed. The star already shrank to 0.7x when pooled; the halo never got the same treatment. Alpha is now `0.50 / sqrt(k)` clamped to `[0.14, 0.50]` and the ring narrows to 32 px when pooled, so total halo ink stays roughly constant instead of growing with the cohort.

**The map-rank hover line was measured from an arbitrary animal.** It took `query_points[0]`, which for a cohort is whichever member happened to be first in metadata order, and for cohort B would have been a member of the wrong cohort entirely. Each hit is now ranked from **the nearest drawn member of the cohort that retrieved it**, and the hover says so.

**Rank numerals are dropped in a comparison.** Two competing numeral sets over the same few hundred pixels is illegible, and prefixing them is worse at 9 px. The hover carries strictly more: for a shared hit it names both cohorts, both 512-d ranks and both cosines.

#### What the user controls

The map rail's "Show it on the map" checkbox becomes **one tick per cohort**, labelled with each cohort's own name, both on by default.
It is the same control rather than a new one, so nothing else about the rail changes, and unticking one is the escape hatch when two 38-member cohorts crowd the same region.

Framing follows the ticks, so "Frame the retrieval" frames what is actually drawn.

**Superseded on 2026-08-06.** A color key sat directly under those ticks, naming each cohort against a swatch and saying in prose that a ring inside a square is a hit both arms found.
It moved onto the plot, into the floating key, and the reason it moved is the reason this section gave for putting it on the rail: put the fact where the misreading happens, and the misreading happens at the glyph.
Two further things were wrong with it there.
The ticks immediately above already carry each cohort's name, so the rail named them twice; and the swatches keyed only the *member* hues, while the hits - which outnumber the members and are what a comparison is about - were encoded by a shape the rail never mentioned.
What is left on the rail is one line of the control's own feedback ("Both arms drawn, **10** hits, **2** of them retrieved by both").
[The key on the map, and describing both pooled queries](#map-key) is the design document for what replaced it.

#### The cross-view color swap

`GRAPH_THEME` expressed the comparison network's A / B / shared language through keys named `gsm`, `gse` and `query`, which mean something else in `build_network_figure`, and it gave "retrieved by both" `#0bab9f` - the exact hex the map uses for the query.
Teal therefore meant "the query" in one view and "a shared hit" in the other, for the same search, and running a comparison silently recolored the query star that the search a minute earlier had drawn teal.

Two literals fix it: cohort A becomes teal and "retrieved by both" becomes blue.
Both views now agree that teal is cohort A and warm is cohort B, while each renders "both" the way its own canvas supports - a color on white, a doubled mark on navy.
`GRAPH_THEME` gains `cohort_a` / `cohort_b` / `cohort_shared` keys so the retrieval view names what it means.

---

<a id="live-stability"></a>

## Result stability, measured on the query that just ran

**Status: built, measured on the real corpus, and tested, 2026-08-06.**
This document replaces the confidence readout described in section 3 of [Cohort retrieval: querying with an experimental group](#cohort-retrieval).
That readout quoted `STABILITY_BY_K`, a bucketed curve measured offline over all 212 cohorts and looked up by cohort size.
It is now a measurement of the cohort actually on screen, taken during the search, and it appears only after the search.

| | |
| --- | --- |
| What is measured | mean Jaccard overlap between this query's top-d and the top-d it produces with any one member dropped |
| Depth | the retrieval depth on screen, not a fixed 5 |
| Baseline, also measured live | mean pairwise Jaccard between the members' own single-sample top-d |
| Cost | one memmap pass per cohort, scoring `2k+1` query vectors instead of 1 |
| Measured cost | 0.44 s at 1 query, 0.50 s at 11, 1.00 s at 77, against a 963 MB read that dominates all of them |
| Where it appears | the inspector, on the right, after the search. Never on the control rail. |

### 1. Why the precomputed number had to go

`STABILITY_BY_K` was honest about what it was and still misleading about what it looked like.
It is a population average: bucket every one of the 212 cohorts by size, take the mean leave-one-out top-5 agreement within each bucket, and read the bucket floor at or below `k`.
A cohort of 7 read the 5-9 bucket and was told 0.72.

The problem is that the spread inside a bucket is most of the range.
Measured live over 22 real cohorts, one per distinct size, at top-5:

| k | this cohort's measured stability | what `STABILITY_BY_K` said |
| ---: | ---: | ---: |
| 4 | **0.339** | 0.55 |
| 6 | **0.849** | 0.72 |
| 7 | **0.316** | 0.72 |
| 8 | **0.668** | 0.72 |
| 10 | **1.000** | 0.81 |
| 18 | **0.619** | 0.86 |
| 35 | **1.000** | 0.86 |

A cohort of 7 whose result does not survive a single dropped animal (0.316) and a cohort of 6 whose result barely moves (0.849) were both told the same thing, and the one told 0.72 while measuring 0.316 was told it more confidently than it deserved.
Reading a number off a curve fitted to other people's cohorts, and printing it beside *this* cohort's name, is the same class of error the status banner made when it announced every cached result as subprocess output: the interface asserting something about itself that is not true of the thing on screen.

The curve is not wrong, and it is not deleted from the record.
It remains the right way to answer "how large should a cohort be", which is why `LOW_N_THRESHOLD` still comes from it and `precompute/validate_cohorts.py` still measures it.
It was the wrong way to answer "how far should I trust *this* list", which is the question a number sitting next to a selected cohort is read as answering.

### 2. Why it moved to the right, and why after the search

The rail's standing rule is that the fact qualifying a control sits directly under that control.
A precomputed number obeyed that rule, because it was a function of the selection and nothing else: pick a cohort, read its size, look up the curve.

A measured number cannot obey it, because it does not exist until the query runs.
It is a property of the result, not of the selection, so it belongs with the result.
Putting a live measurement under the picker would mean either running the scan on every selection change, which turns a dropdown into a 1 s query, or showing a stale number from the previous cohort, which is worse than the curve was.

So the cohort card on the rail now states only what is true before the search: the role, the cohort's name, and how many samples are about to be pooled.
That is what the user asked for, and it is also the only honest thing the rail can say at that moment.

### 3. What is measured

For a cohort with members `1..k`, cached vectors `rows`, and retrieval depth `d`:

- `F` is the top-d of the pooled query, which is the list on screen.
- `L_i` is the top-d of the same cohort with member `i` dropped and the rest pooled.
- `M_i` is the top-d of member `i` on its own.

Then

- **Result stability** is `mean_i Jaccard(F, L_i)`.
- **The single-sample baseline** is `mean_{i<j} Jaccard(M_i, M_j)`.
- **The gain** is the ratio of the two.

Jaccard rather than the overlap fraction, because it is the statistic `precompute/validate_cohorts.py` has always used for this, and because the two-arm comparison banner already reports a Jaccard overlap.
One vocabulary for "how much do two hit lists share", used in both places it appears.
The previous card's note said "the measured share of the top 5 that survives", which described the overlap fraction while printing the Jaccard; that wording is gone with the number it described.

**The depth is the depth on screen.** The offline curve was fixed at top-5 because it had to pick one, but the list a user is reading is `topk` deep, and that is the list whose stability they want to know. The drift is small and is recorded here so nobody has to guess at it: across the same 22 cohorts, mean stability is 0.745 at top-5, 0.774 at top-20 and 0.791 at top-30. Deeper lists are slightly more stable, which is what a set-overlap statistic does as the sets grow.

**The baseline is measured too, and that is what makes the headline readable.** A bare 0.62 has no scale. The old card supplied one by quoting `SINGLE_SAMPLE_STABILITY = 0.16`, a corpus-level constant measured at top-5, which put a top-5 constant beside what is now a top-20 or top-30 measurement. Scanning each member on its own costs `k` more query vectors in a pass that is already running, so the scale is measured on the same cohort, at the same depth, in the same pass. Across those 22 cohorts the live baseline is 0.101 at top-5, and the gain from pooling ranges from 1.5x on a pair to 58x on a cohort of 35.

**A zero baseline is reported as a zero, not as a ratio.** One cohort of 4 measured a single-sample baseline of exactly 0.000: no two of its members share a single hit. The gain there is not "339285714x", it is undefined, and the panel says the members retrieve four disjoint lists instead of printing a ratio.

**The weakest member is named.** `Jaccard(F, L_i)` is already computed per member, so the member whose removal moves the list furthest is known for free. It is named on the panel for the same reason the per-member leave-one-out cosine stayed in the member list when `R̄` was deleted: it varies, and it points at a specific animal.

### 4. Cost, and why it is affordable

The whole design turns on one measurement, taken against the real 963 MB memmap:

| query vectors in one pass | wall clock |
| ---: | ---: |
| 1 | 0.44 s |
| 11 | 0.50 s |
| 21 | 0.58 s |
| 41 | 0.77 s |
| 77 | 1.00 s |

The read and the float16-to-float32 normalization dominate; the matrix multiply against `m` queries is nearly free next to them.
The largest cohort in the corpus has 38 members, so the worst case is 77 query vectors and about 0.56 s more than the 0.44 s a single pooled query already costs.
A two-arm comparison runs one such pass per arm, which is the same number of passes it ran before.

This is the same technique `precompute/validate_cohorts.py` and `validate_artifacts.py --mixing` already use: stack every query vector into one `(m, 512)` matrix, stream the index in blocks, and keep a running top-k per query.
`retrieval._topk_cosine_matrix` is now the single implementation of the scan, and `_topk_cosine_from_memmap` is a one-row wrapper over it, so the pooled query and its leave-one-out variants are scored by exactly the same code that scores a single sample.
Verified against the shipped scan on the real corpus before the change was made: identical top-30 in order, maximum score difference **0.0**.

### 5. What was deleted

Deleted rather than left unused, which is what this repository did with `resultant_length` for the same reason:

- `cohorts.STABILITY_BY_K`, the bucketed curve.
- `cohorts.expected_stability(k)`, its lookup.
- `cohorts.SINGLE_SAMPLE_STABILITY`, the 0.16 constant, now measured live.
- `CohortGeometry.stability`, the property that fed the card.
- The `.cohort-stat` block, the meter, and the low-N flag inside `build_cohort_card`.
- The "Result stability" row in `build_cohort_details`, because the measurement now has one home on the right and two copies of one number stacked vertically is noise.

Kept, with its justification intact:

- `cohorts.LOW_N_THRESHOLD = 5`, still the first bucket in the measured curve to reach 0.70, still the reason the cohort picker marks a small cohort "low N" before you commit to it. It is a statement about size, made where size is the only thing known.
- `precompute/validate_cohorts.py` check 5, which measures the curve. Its output no longer says "paste into `STABILITY_BY_K`"; it reports the knee and **fails the run** if the knee has moved away from `LOW_N_THRESHOLD`, which is the one thing the curve still sets.

That gate applies to a **full** sweep only, and the restriction was itself a correction.
The first version failed on any run, and a `--cohorts 40` sample promptly failed it: with one or two cohorts per size bucket the knee landed at k >= 10 rather than 5.
That is the same regime that produced 0.38 at k=5 beside 0.90 at k=6 in the first run of this script, and failing a build on it would be enforcing exactly the noise the bucketing exists to survive.
Over all 212 cohorts the knee is 5 and check 5 passes.

`cohorts.STABILITY_FLOOR = 0.70` is new and is the same 0.70.
The knee was chosen as the first bucket whose *measured* stability reached 0.70; the floor applies that threshold to the measurement itself rather than to size as a proxy for it.
Below it the panel says so, in amber rather than red, for the reason the map's coverage bar is amber: a cohort whose result moves when you drop an animal is reporting correctly, not failing.

### 6. Where the code goes

`bridge_rna/cohorts.py` keeps its promise to open no embedding and no memmap.
It gains the pure arithmetic only: `top_k_agreement`, `leave_one_out_vectors`, and `StabilityMeasurement`, which is built from rankings the caller supplies.
Every one of them is testable against the fixture corpus on a machine with neither artifact, which is what the module exists for.

`bridge_rna/retrieval.py` owns the scan and the wiring.
`run_cohort_retrieval` now returns `(hits, rows, stability)`, builds every query vector it needs before touching the memmap, and scores them in one pass.

`bridge_rna/panels.py` gains `build_stability_panel`, which renders one measurement or two.
The two-cohort case reuses the role dot and the cohort's own name, because [The key on the map, and describing both pooled queries](#map-key) established that cohort B's hex cannot agree across the views and the binding is therefore the name.

### 7. Making two measurements fit on screen at once

The first build of the panel labelled every block "RESULT STABILITY" and spelled out the full definition under each number, then added a three-line amber caution when the number was low.
On a single cohort that reads fine.
On a comparison it does not: measured in the browser, the two blocks wanted **644 px** inside a panel that had **389 px**, so cohort B's entire measurement sat below the fold.
A second measurement nobody can see is a second measurement nobody made, which is the whole thing this feature exists to prevent.

Three changes fixed it, and each one is a rule rather than a nudge.

**What is shared is said once.** The panel heading names the statistic and the subtitle states what was measured and at what depth, so a block carries only what differs between the two arms: the name, the number, the meter, the size, the baseline, and the member that moves it most. The per-block label was pure repetition and the definition was identical in both.

**The caution is one line.** It was a title over a three-line body, and the body said the same thing twice on a comparison. It now reads "Under 70%: read these as a neighbourhood, not a ranking", with the threshold interpolated from `STABILITY_FLOOR` so the sentence cannot drift from the rule that fires it.

**The details panel yields first.** `.details-panel` had `flex-shrink: 20` against the stability panel's 1. With equal shrink factors the overflow was split between them and the stability panel lost 165 px it needed; the details panel is the right one to give way, because it scrolls a reference list where the panel above it carries two numbers that only mean anything side by side.

That mechanism was replaced on 2026-08-06 and the replacement is worth stating here, because the `20` was a workaround rather than a rule.
With `flex-basis: auto` the details panel asks for the ~506 px its content measures - a height it will never get and does not need - so every layout pass began in overflow and ended by taking that overflow back off *both* panels in proportion.
The 20 made the details panel's share large; it never made the stability panel's share zero.
`.details-panel` is now `flex: 1 1 0`, so it claims the leftovers instead of claiming its content and giving it back, and there is no overflow to divide.
The degradation path is unchanged and now falls out of the same rule: a zero-basis item contributes nothing to shrink, so on a column too short for all three the details panel stops at its 120 px floor and the stability panel is the one that scrolls.
[Splitting the stability panel's two cohort sections evenly](#stability-panel-even-split) has the measurements.

`tests/e2e_cohort_check.py` measures this rather than trusting it: after a two-arm search it reads the panel's scroll box and content box and asserts that nothing is clipped and that the two arms are the same size.
The first version of that check compared the last block against `bounding_box()`, which is the *border* box, so it allowed a block to run through the panel's own 20 px bottom padding - and it passed for weeks while cohort B's last row was clipped at every viewport. That is recorded in section 2 of [Splitting the stability panel's two cohort sections evenly](#stability-panel-even-split).

**The three cuts above were not enough, and the layout changed the same day.**
Saying the shared parts once, cutting the caution to one line and making the details panel yield brought a comparison from 644 px down to 456 px in a 445 px box, which is a great deal better and is still 11 px short.
Cohort B's last row stayed clipped at every viewport the app is used at, and the two arms were never the same height - the gap between them was a property of how their sentences and member keys happened to wrap, so it moved from one search to the next.
The two arms are drawn as even columns with their rows aligned by `subgrid` now, at 354 px, and [Splitting the stability panel's two cohort sections evenly](#stability-panel-even-split) is the whole account of it.

One number in this section is worth correcting rather than leaving to be re-derived: `max-height: 65%` was never what constrained the panel.
At the inspector heights measured it allows about 581 px against a natural height of 458 px, so it did not bind at any of them.
Flex shrink did all of the constraining, which is why the fix was a flex basis rather than a taller cap.

### 8. Alternatives considered and rejected

**Keep the curve on the rail and add the measurement on the right.**
Rejected because the two disagree by up to 0.4 on a real cohort, and a reader who saw 0.72 before the search and 0.32 after it has been told the interface cannot be trusted, which is true and is not the intended lesson.

**Compute the measurement on selection, not on search.**
This would let the number stay under the picker and keep the rail's layout rule intact.
Rejected on cost and on honesty: it turns every dropdown change into a memmap pass, and it would report the stability of a query that has not run at the depth the slider currently sits at, which changes under it.

**Report the standard deviation of the per-member agreements beside the mean.**
Rejected as a second headline. `R̄` was removed for looking like a grade while grading nothing, and the lesson taken from it was that this card supports exactly one number. The per-member spread is expressed instead by naming the member that moves the list most, which is actionable where a spread is not.

**Estimate stability by bootstrapping the member set rather than by leave-one-out.**
More statistically flexible, and it would give a confidence interval.
Rejected because leave-one-out is what `validate_cohorts.py` measures over all 212 cohorts, and having the live number and the corpus-scale gate compute the same statistic is worth more than a confidence interval on a number whose whole purpose is to be read in two seconds.

### 9. How it is tested

Against the fixture corpus, in pytest, with no real artifact:

- The fused `m`-query scan reproduces a naive per-query scan row for row, in index order and in score.
- A cohort whose members are identical vectors measures stability exactly 1.0, because every leave-one-out pool is the same vector.
- The statistic's bounds, its symmetry, and its behaviour on disjoint and identical lists.
- `run_cohort_retrieval` returns a measurement whose size, depth, and per-member length match the cohort and the slider.
- The rail card renders no stability, no meter, and no flag; the panel renders the measured value.
- The deleted names stay deleted.

One thing the units cannot see, caught by an adversarial review of the diff rather than by any of them: `gain` guards a baseline of exactly zero, but the panel printed the baseline at two decimals, so the one real corpus combination with a baseline of 0.0048 (OSD-612's 4-animal cerebral hemisphere at top-18) rendered "one alone overlaps another by 0.00, a 88.6x gain".
`panels._share` drops to three decimals below 0.005 rather than suppressing the ratio, because a near-zero baseline is exactly when the gain is worth stating.

Against the real corpus, in the browser:

- `tests/e2e_cohort_check.py` now gains a `--loops` flag and asserts that the panel is absent before a search, present after it with a number, carries both arms of a comparison, and reports a stability that is not the value the deleted curve would have produced for that size.

---

<a id="stability-panel-even-split"></a>

## Splitting the stability panel's two cohort sections evenly

**Status: in progress, 2026-08-06.**
A follow-on to [Result stability, measured on the query that just ran](#live-stability), which built the panel this document re-lays-out.
Section 7 of that document, "Making two measurements fit on screen at once", is the direct ancestor: it fixed a 644 px panel into a 389 px box by cutting what each block says.
This document is what happened next, which is that the cut was not quite enough.

### 1. The defect, measured

Driving the real app in Chromium against study OSD-100, cohort `left eye · Ground Control` compared against `left eye · Space Flight` (differs by spaceflight arm), top-k 5, cohort A and cohort B do not get the same treatment.

| viewport | `#stability-panel` client height | its content height | cohort B's "Moves it most" row is clipped by |
| --- | ---: | ---: | ---: |
| 1680 x 1050 | 447 px | 456 px | **7.8 px** |
| 1600 x 1000 | 445 px | 456 px | **9.9 px** |
| 1512 x 982 | 444 px | 456 px | **10.7 px** |
| 1440 x 900 | 441 px | 456 px | **14.2 px** |
| 1280 x 800 | 389 px | 456 px | **65.6 px** |

At every viewport tested, cohort A's block is complete and cohort B's is not.
The row that goes is the one naming the animal whose absence moves the result furthest, which is the only actionable line in the block.

The two blocks are also unequal before any clipping: cohort A measures 148.3 px and cohort B measures 160.7 px.
That 12.4 px is worth decomposing, because the obvious culprit is not one of the terms and the real shape of it is the argument for the fix.

| term | px |
| --- | ---: |
| the separator box that exists on the second block only (`padding-top: 12` + `border-top: 1`) | **+13.00** |
| cohort A's scale sentence wrapping to two lines where cohort B's fits on one | **-15.94** |
| cohort B's 28-character member key wrapping where cohort A's 27-character key does not | **+15.28** |
| | **+12.34** |

The `differs by spaceflight arm` phrase costs nothing: at 322 px it sits on cohort B's role line beside the letter and adds no height at all.
What is left is one structural offset and **two content terms that happened to nearly cancel**, each swinging between roughly -16 px and +29 px depending on how a given cohort's sentence and member name wrap.
So the layout was not stably asymmetric, it was *metastable*: the gap between the two arms was a property of the strings in them, and it moved from one search to the next.
That is the case for an arrangement in which the two arms are the same size because of how they are laid out rather than because of what they happen to contain.

The panel is not the only thing starved.
In the same measurements `#details-panel` had 506 px of content and was given between 118 px and 310 px, so the inspector below is scrolling hard while the panel above it holds 447 px to say two numbers.

### 2. Why the shipped guard did not catch it

`tests/e2e_cohort_check.py` already asserts that both arms fit, and it passes:

```python
box = stab.bounding_box() or {}
last = stab.locator(".stability-cohort").last.bounding_box() or {}
c.ok(bool(box) and bool(last)
     and last["y"] + last["height"] <= box["y"] + box["height"] + 1,
     ...)
```

`bounding_box()` returns the **border box**, so the comparison allows the last block to run into the panel's own 20 px bottom padding and stop 1 px short of its border.
At 1600 x 1000 the last block's bottom is 9.9 px below the content box and 10.1 px above the border box, so the assertion is satisfied by the exact margin that hides the failure.
The panel is also a scroll container, and the check never reads `scrollHeight` against `clientHeight`, so 11 px of unreachable content is invisible to it.

That is the same class of error [Result stability, measured on the query that just ran](#live-stability) section 1 describes for `STABILITY_BY_K`: an assertion that is true about a thing adjacent to the one that matters. The guard is tightened as part of this change.

### 3. What "split evenly" has to mean

The request was "can these 2 sections in the cohort feature be split evenly UI wise", against a screenshot of the panel with cohort A complete and cohort B truncated mid-block.

Any fix has to satisfy three things at once, and the measurements above are why:

- **The two arms get the same amount of the panel.** Not approximately: a comparison exists to be read as a comparison, and one arm rendered complete beside one arm rendered partial tells the reader the first is the finding and the second is a footnote.
- **Neither arm is clipped at any viewport the app is used at**, down to 1280 x 800.
- **The panel stops starving the inspector below it.** 447 px for two numbers, against 506 px of definition, members and estimator prose compressed into 264 px, is the wrong division of a 934 px column.

### 4. What shipped: two even columns with their rows aligned

The two arms sit side by side in equal columns, and their rows line up across the gap.

```
Result stability
Measured on this query: how much of these 5 hits comes back when any
one pooled sample is dropped, averaged over all of them.
The two arms differ by spaceflight arm.

● COHORT A                      ● COHORT B
left eye · Ground Control       left eye · Space Flight
0.89  of 5                      0.94  of 5
━━━━━━━━━━━━━━━━━━━━━━━╌╌       ━━━━━━━━━━━━━━━━━━━━━━━━╌
6 pooled, and one alone         6 pooled, and one alone
overlaps another by 0.32,       overlaps another by 0.60,
a 2.8x gain.                    a 1.6x gain.
- - - - - - - - - - - - -       - - - - - - - - - - - - -
MOVES IT MOST      0.67         MOVES IT MOST      0.67
Mmus_C57-6J_EYE_GC_Rep1_M33     Mmus_C57-6J_EYE_FLT_Rep4_M26
```

Measured on the real app, same query as section 1:

| | before | after |
| --- | ---: | ---: |
| `#stability-panel` content height | 456 px | **354 px** |
| internal overflow at 1600 x 1000 | 11 px | **0 px** |
| internal overflow at 1280 x 800 | 67 px | **0 px** |
| cohort A block | 148.3 px | **155 px wide, equal height** |
| cohort B block | 160.7 px | **155 px wide, equal height** |
| `#details-panel` height at 1600 x 1000 | 264 px | **355 px** |
| `#details-panel` height at 1280 x 800 | 120 px (its floor) | **155 px** |

Five things about it are load-bearing.

**The alignment is `subgrid`, and the rows are assigned by class rather than by child order.**
Each arm is a grid whose rows are the pair grid's own, so the name row, the number row, the member row and the flag row line up across both columns however many lines any of them wraps to.
That is what puts 0.89 on the same baseline as 0.94 instead of 160 px below it, which is the comparison the panel exists to support.
Rows are addressed by class because either of the last two can be missing from either arm, and counting children would let cohort B's flag land in the row holding cohort A's member name.
Where `subgrid` is unsupported the declaration is dropped and each arm falls back to its own four-row grid: the columns stay even and only the cross-column baselines go.

**The tracks are `minmax(0, 1fr)`, never `1fr`.**
`1fr` is `minmax(auto, 1fr)`, which floors a track at its min-content width.
Sample keys are mono and run to 39 characters across the corpus (`Mmus_C57-6J_LVR_RR1_BSL_noERCC_Rep5_M10`; median 26, p95 37, over all 2,108), and any unbreakable run would push its column wider than its twin, silently undoing the even split the rule exists to make.

**"differs by *facet*" moved from cohort B's role line to the panel header.**
It is the one fact in the panel that belongs to neither arm: it describes the pair.
Hanging it under B's letter made B's name start a line below A's, which is precisely the ragged edge that even columns exist to remove.
This is the panel's own existing rule - [Result stability, measured on the query that just ran](#live-stability) section 7, "what is shared is said once" - applied to the one line that had escaped it.

**The "moves it most" row is always drawn, and says so when there is nobody to name.**
`cohorts.weakest_member` returns `None` when every member's absence moves the list equally far, and that is an answer rather than a gap.
It now reads "Moves it most / every member equally", in the text face rather than the mono one, because it is prose and not an accession.
An absent row and a clipped row look identical on screen, and a clipped row on this exact line is the defect being fixed, so silence was not available as an answer here.

**The low-stability flag is deliberately *not* equalized.**
It appears under the shaky arm only, leaving a gap under the healthy one.
Equalizing it would need either a blank cell, which is the empty promise the missing "moves it most" row just stopped being, or an "above 70%" counterpart badge - and a pass mark for a healthy cohort is exactly the grade `R̄` was deleted for being ([Cohort pooling: querying with an experimental group instead of one sample](#cohort-pooling)).
A caution that appears only when there is something to be cautious about is working correctly.

The three-line weakest row is a consequence rather than a choice.
Label, score and a 27-character key shared one baseline while the panel was a single 322 px column; in a 155 px one they cannot, since the label and value are fixed width and left about 29 px for the key, which - because it wraps rather than truncates, deliberately - broke one character per line into a 400 px column.
Label and score now share a line and the key sits beneath them.

#### The flex fix underneath it

Making the panel shorter was not sufficient, and the residue is instructive.
With the columns in place the panel still lost 3 px at 1680 x 1050 and 11 px at 1280 x 800, which is enough to cut the descenders off the last line of a sample key.

The cause was `.details-panel { flex: 1 20 auto }`.
With a content basis that panel asks for the ~506 px its definition, member list and estimator prose measure - a height it will never get and does not need, since it scrolls by design - so every layout pass began in overflow and ended by taking that overflow back off both panels in proportion.
`flex-shrink: 20` made the details panel's share large; it never made the stability panel's share zero.

`.details-panel` is now `flex: 1 1 0`: it claims the leftovers instead of claiming its content and giving it back, so there is no overflow to divide.
The degradation path is unchanged and now falls out of the same rule rather than from a tuned constant - a zero-basis item contributes nothing to shrink, so on a column too short for all three the details panel stops at its 120 px floor and the stability panel is the one that scrolls internally.

`.stability-panel`'s `max-height: 65%` stays, demoted to the backstop it was always meant to be.
At 1600 x 1000 the panel asks for 354 px of an allowed 607 px, so it never binds; it still earns its place on a genuinely short window, where two flagged arms and long labels reach about 409 px.

**`flex: 1 1 0` is right only while the column's height is fixed, and getting that wrong regressed every width below 1180 px.**
There the app grid collapses to one column and the document scrolls, so the inspector's height comes from its contents - and an item with a zero basis contributes nothing to that height.
The column therefore sized itself to the other two panels and left the details panel sitting on its 120 px floor, scrolling internally with **372 px hidden at 900 px wide and 388 px at 390 px**, where with a content basis it had stood at its full 491 px and let the page scroll.
Measured both ways against the running app rather than reasoned about.
The `@media (max-width: 1180px)` block now restores `flex: 1 1 auto` and `overflow: visible`, alongside the two rules already there that lift the caps on `.stability-panel` and `.ai-panel` for exactly the same reason: once the document scrolls, a panel should be as tall as what it holds.

#### The narrowest phones

The label and the score of the "moves it most" row are 97.8 px and 26.4 px, so with the 8 px gap the pair needs 132.2 px on one line.
A column is half the page less 78 px of padding, border and gutter, so the two stop fitting below about 342 px of page width - reached only by the narrowest phones, and measured at 320 px, where a column is 121 px and cohort A's score sat against cohort B's label.
`.stability-weakest-label` is now `flex: 0 1 auto` with `min-width: 0` rather than `flex: none`, so it wraps to two lines instead, identically in both columns.
Stacking the arms under a breakpoint was the alternative and was rejected: it buys the same fix by making the phone the one place the two arms cannot be compared, and the even split survives 320 px without it.

### 5. Alternatives considered and rejected

Three layouts were specified in full and judged on information design, fidelity to this repository's recorded decisions, and implementation risk.
All three judges ranked the aligned comparison first.

**Whole blocks side by side, without shared row baselines.**
The same two columns, but each arm self-contained, its rows falling where its own content puts them.
Rejected because it gives up the thing the columns were for: with A's note wrapping to three lines and B's to two, the two headline numbers stop sharing a baseline and the reader is back to comparing across a vertical offset, just a smaller one.
Its own specification also proposed holding the columns even by padding each arm to a fixed five rows with inert spacer divs, which reintroduces positional counting - the failure mode that `subgrid` plus class-addressed rows exists to avoid.

**Keep the stack and equalize the two blocks.**
The conservative option: same vertical arrangement, identical row structure in both arms, equal heights.
Rejected because it does not improve the reading the panel exists to support - the two headline numbers stay about 180 px apart with five intervening rows, so comparing them still costs memory rather than a glance - and because it resolves the space problem in the wrong direction, growing the panel to about 511 px and dropping the details panel to about 200 px against a details panel already hiding 244 px of its 506.
It is also not what was asked for.
Its diagnosis of the flex bug was correct and is kept; so is its argument that a vanished row and a clipped row look the same, which is why "every member equally" is now printed.

**Stack the two columns again below 680 px, or below 360 px.**
Rejected on measurement: at a 390 px phone width the two columns are 156 px each, which is *wider* than the 155 px they get on a 1600 px desktop, because the inspector goes full-bleed when the app grid collapses to one column.
The one width where anything did break was 320 px, and letting the row's label wrap fixes it without a breakpoint.
A breakpoint would have made the phone the only place the two arms cannot be compared.

**Give cohort B's meter the warm hue its dot carries.**
Rejected because the meter's fill already encodes something else: it turns amber when the measurement is below `STABILITY_FLOOR`.
Two meanings on one channel would make a healthy cohort B indistinguishable from a flagged cohort A.
Identity stays bound to the dot and the name, which is the rule [The key on the map, and describing both pooled queries](#map-key) settled for the same reason.

### 6. How it is tested

In pytest, against the component tree rather than a screenshot:

- Two arms are wrapped in one `.stability-pair.is-pair`; a lone cohort gets the wrapper without the modifier, because a single block in a two-column grid is a half-empty table.
- Every row of an arm is a direct child carrying its own class, checked on a comparison where arm A has a member to name and arm B has a flag instead - the mirror-image case that positional row assignment would get wrong.
- The weakest row is present on both arms whatever either measured, and says "every member equally" with no score when there is nobody to name.
- The facet is stated once, in the header, and the string appears exactly once in the whole panel.

In the browser, against the real corpus, in `tests/e2e_cohort_check.py`:

- The panel does not scroll internally (`scrollHeight - clientHeight <= 1`) and the last arm's last row sits inside the panel's **content** box, not its padding.
- The two arms start on the same line and are the same height, both within 1 px.
- The two headline numbers share a baseline.
- The two arms are laid out as one even pair rather than stacked.

Across ten payload shapes rendered through the shipping `build_stability_panel` and measured in Chromium at 1700 px and at 390 px - both arms named, one arm with no weakest member, neither arm with one, one flagged, both flagged, the corpus's longest sample key in both columns, long cohort labels at top-k 50, a zero baseline, and a lone cohort with and without a named member - every case has equal column widths, equal block heights, aligned names, aligned numbers, and no horizontal overflow.

---

<a id="map-key"></a>

## The key on the map, and describing both pooled queries

**Status: built, run against the real corpus, and tested, 2026-08-06.**
This is the implementation document for a copy pass over both views plus one real design change: the map now keys every mark it draws, and a two-arm cohort comparison describes both of its pooled queries instead of one.

| | |
| --- | --- |
| Encodings the map draws at once | **4** - corpus hue, member fill hue, hit ring shape, corpus glyph shape |
| Encodings it explained before this | **1** |
| Defects found on the way and fixed | **5** - two in the map as it stood, three in this change, all found by audit or review rather than by a test |
| Sentences removed from the interface | **7** |
| Tests | 330 unit tests, 266 browser checks across 3 suites |

### 1. What was asked for

Seven pieces of copy to delete, one to reword, American spelling in the map, a description for the second pooled cohort, and this:

> I want the key/legend of the mapping to be much more clear, especially when 2 different sets of vectors are pooled. The UI has to be clean and easy to follow.

The first six are edits.
The last two are the design work, and they turned out to be the same problem seen from the two views: **a comparison runs two pooled queries, and the interface consistently described only one of them.**

### 2. The copy that went, and why each one earned its deletion

Every removal below leaves the fact it carried recorded somewhere that outlives microcopy - the docs, or the code comment beside the thing it describes.
Deleting a sentence from a rail is not deleting a decision.

| removed | where it was | why it goes |
| --- | --- | --- |
| "Not a difference vector." | cohort compare hint | Defines the feature by what it is not, to a reader who never suspected it was. The claim is load-bearing and stays in `CLAUDE.md` and [Cohort retrieval: querying with an experimental group](#cohort-retrieval) §4. |
| "Nothing is dropped for you." | cohort member list | Reassurance against a fear the interface never raised. "Untick a sample to leave it out of the pool" already says the pool is yours. |
| "Adds roughly two seconds per hit. Off, a search is local and instant, and abstracts are fetched for a hit when you open it or when the AI hypothesis needs them." | metadata enrichment | Three lines of rail explaining a control that sits inside a disclosure already labelled **Optional** and defaults to off. |
| "Colours all 942,563 points." | color-by coverage | Restates a full bar in words. The readout exists to answer "why is most of my map not colored?", which a whole-map field never raises - so it now says nothing and the bar carries it. |
| "Both corpora folded onto one anatomical vocabulary, so a liver in GEO and a NASA liver share a colour." | Tissue color-by hint | A paragraph of method under a control that had already answered the question; the menu says "whole map" and the legend names every bucket. |
| "One glyph per sample; zoom re-samples the visible window." | ARCHS4 point budget | Describes the mechanism of a control whose four pills already read 100k / 250k / 500k / All. |
| "Each glyph is a pooled member; the query itself is a mean of them and has no position here." | map retrieval key | Replaced by a key row that *says* `pooled member` beside the glyph it names. Showing beats telling. |
| "Hits are ranked by cosine distance in 512 dimensions. This map is a projection into two or three of them and does not preserve those distances, so how far a hit sits from the query here is not its rank, and no line is drawn between them." | map retrieval caveat | Four clauses to protect against one misreading. |

The caveat's last sentence was kept and reworded, because it had to stand without the paragraph that used to precede it:

> **Hover a hit for its rank in the search and its rank on the map. The two disagree: a projection cannot preserve 512-dimensional distances.**

That is the whole honest claim in two sentences: there are two orderings, they differ, and here is where to see both.
It said "a projection of 512 dimensions into two" first, which §5 records as one of the three things this change got wrong.

`tests/test_app.py::test_the_removed_copy_stays_removed` pins every one of them.
Prose regresses silently - nothing else in the suite would notice a paragraph coming back, and several of these had already outlived the thing they described.

#### Spelling

The app now spells `color` throughout, in strings, comments and identifiers, and `tests/test_app.py::test_the_app_spells_color_the_american_way` keeps it that way.
This was drift rather than a choice: the package is `colorby.py`, the control is `#color-by`, the functions are `color_for_index` and `covers_corpus`, and the one private function spelled the other way, `_colour_plan`, is now `_color_plan`.
Only the map was asked for, but the retrieval view had one user-visible instance left - `", colour = which cohort retrieved it"` in the comparison legend strip - which would have put both spellings on screen for the same two-cohort search. Everything else there was comments and one local variable named `colour` sitting inside `{"color": colour}`.

### 3. The map key

#### The problem, stated exactly

With a two-cohort comparison drawn, the map carries four encodings at once:

| channel | what it distinguishes | where it was explained |
| --- | --- | --- |
| corpus glyph **hue** | tissue, species, flight arm, … | the floating legend |
| member fill **hue** | which cohort a pooled sample belongs to | a swatch on the rail, 800 px from the glyph |
| hit ring **shape** | which cohort retrieved a hit | **nowhere** |
| corpus glyph **shape** | ARCHS4 circle versus OSDR diamond | **nowhere** |

The audit that produced that table is in §7.
The third row is the worst of them: in a comparison the hits outnumber the members and are what the whole feature is about, and their channel had no key at all.
A viewer saw two sizes of white ring and two colors of star with nothing on screen tying them together.

#### What was built

**One panel on the plot, three sections, ordered by how transient each is.**

```
┌──────────────────────────────┐
│ POOLED MEMBERS               │   only when two cohorts are drawn
│  ★  Liver · Space Flight   8 │   teal star
│  ★  Liver · Ground Control 6 │   gold star
│ RETRIEVED HITS               │
│  ○  Liver · Space Flight  10 │   white circle ring
│  □  Liver · Ground Control 10│   white square ring
│  ▣  retrieved by both      3 │   ring inside a square
├──────────────────────────────┤
│ COLOR · TISSUE               │   the only section that scrolls
│  [ filter categories…      ] │
│  ■  Blood / immune   155,761 │
│  …                           │
├──────────────────────────────┤
│  ●  ARCHS4                   │   shape, not hue
│  ◆  OSDR                     │
└──────────────────────────────┘
```

The retrieval is on top because it is what the reader asked for a moment ago and what will be gone next search.
The color list is the standing state of the map and is the only part that can grow, so it is the only part that scrolls.
The corpus shapes are a footnote that never changes.

**A comparison is grouped by role, not by cohort, and that is the whole design.**
The encoding has two factors - which cohort, and member versus hit - and they do not use the same channel.
Listing each cohort's two marks together buries that.
Listing the two member rows adjacent and the two hit rows adjacent shows it: one pair differs only in hue, the next differs only in shape, and the reader sees each channel vary with the other held fixed.
No sentence has to assert the rule, which is what makes it readable at a glance rather than after a paragraph.

**A single query does not pay for the comparison.**
A plain search gets two rows, no headings and no cohort names: with one query on screen there is nothing for a name or a group to distinguish it from.
An uploaded sample gets one row, because it was never embedded into this map, has no coordinate, and draws no member mark - a row for it would key a glyph that is not there.

**Marks are drawn as marks.**
Naming a square ring in words, in a panel that could simply draw one, is the key doing less than it could.
Shapes are CSS; **hues come from `manifold/theme.py` through an inline style**, which removes the last place a cohort color was written twice.
The rail's old swatches mirrored `RETRIEVAL_QUERY` and `RETRIEVAL_QUERY_B` into `map.css` by hand, because Plotly cannot read a CSS variable - two spellings of one hex, and a drifted swatch would have labelled the wrong cohort.
`test_the_map_key_reads_its_hues_from_the_theme` asserts both that the key carries the theme's values and that neither hex appears in `map.css` again.

**The key follows the plot into 3-D.**
`Scatter3d` rejects `star` outright, so a member draws as a diamond there and the key draws a diamond too.
A key that kept showing a star would assert a mark 3-D does not draw, which is worse than the silence it replaced.
The hit shapes are deliberately unchanged between 2-D and 3-D; that is why shape, not hue, carries cohort identity in the first place.

**An unticked arm keeps both of its rows**, receded, with `hidden` where the count would be.
Dropping them would leave a gold glyph on screen a moment later with nothing to look it up in, and would make the key disagree with the ticks that produced it.
The `retrieved by both` row leaves, because with one arm hidden the doubled mark cannot exist.

#### What the rail gave up

The rail's two-cohort key - a swatch per cohort and a paragraph on what a ring inside a square means - is gone.
The ticks directly above it already carry each cohort's name, so the rail was naming them twice and explaining glyphs nowhere near them.
What is left is one line of the control's own feedback:

- *"**Liver · Space Flight** and its 5 nearest ARCHS4 neighbors, drawn where they sit in the space."*
- *"Both arms drawn, **10** hits, **2** of them retrieved by both."*
- *"**Liver · Space Flight** only. Tick both to see which samples they share."*

This supersedes the placement recorded in [Cohort retrieval: querying with an experimental group](#cohort-retrieval) §9 ("What the user controls"), and the reason it supersedes it is the reason that section gave for putting it there: put the fact where the misreading happens.
The misreading happens at the glyph.

#### What was deliberately not done

**The plot badges stay.** Two of the three designs proposed folding them into the key. They answer a different question - *what is drawn right now*, which changes on every zoom - where the key answers *what does this mark mean*. A key that changed on zoom would be a worse key.

**The halo gets no row.** It is always concentric with a member mark, carries no value independent of it, and its alpha varies with cohort size, so a row would have to explain a quantity that means nothing alone.

**Residual swatches are not dimmed to match their glyphs.** "Other" is drawn at 0.26 opacity and 82% size on the plot while its legend swatch is full strength. Matching them was considered and dropped: the recession is a deliberate ranking of informative categories over uninformative ones, and a dimmed swatch reads as a disabled row.

**The context cloud still gets no legend row**, per invariant 5. One design proposed a subtitle for it inside the key; the coverage readout and the plot badge already carry that state, and both say the points are scenery rather than a category.

### 4. Two pooled queries, two descriptions

Arming a comparison runs a **second** independent pooled query.
The rail described the size and stability of the first and said nothing at all about the second, while both the network figure and the map gave it a color and drew it.

That matters for a specific reason rather than for symmetry: cohort B can easily be the smaller and shakier arm, and result stability is a property of each arm on its own.
An overlap of 0.25 between a 12-animal cohort measuring 0.86 and a 2-animal cohort measuring 0.31 means something different from the same 0.25 between two cohorts of twelve.
The number that decides how much of the overlap to believe was not on screen.

**Each pooled query now gets a card under its own picker**: the selected cohort under the Cohort dropdown, the sibling under "Compare against".
That is the rail's standing rule - the fact that qualifies a control sits directly under that control - and it keeps the member ticks, which belong to cohort A alone, next to cohort A's card.

Each card gains a role line only when there are two: `● COHORT A` and `● COHORT B · differs by spaceflight arm`, with a left rule in the same hue.
A lone cohort gets no letter, because with no second arm on screen there is nothing to distinguish it from.
The contrast facet is stated once, on the second card, because it is a property of the pair.

The dot and the rule are `--accent-teal` and `--accent-warm`, which **are** `GRAPH_THEME["cohort_a"]` and `["cohort_b"]`, so a card and the star it describes in the network figure cannot disagree.

#### The one thing that cannot be made consistent, and what carries it instead

Cohort B is `#d9791b` in the retrieval network and `#ffc233` on the map, and this is not fixable by choosing better.
`#ffc233` on white measures about 1.8:1, unusable in the retrieval view; `#d9791b` sits 0.3 dE from `CATEGORICAL[3]` under deuteranopia, unusable on the map's navy canvas next to eleven tissue hues.
`manifold/theme.py` records both measurements.

So **the binding across the two views is the cohort's name, not its hue**, and both surfaces now print the name next to their own mark: the card, the network's band labels, the map's tick, and the map's key rows.
Cohort A survives the trip unchanged at `#0bab9f`, which is what made the difference look like a different arm rather than a different surface.

### 5. Three defects this change introduced, and what caught them

An adversarial review of the finished branch raised 27 candidates, of which these three survived a refutation pass. None was visible to any of the 330 tests or the 266 browser checks, because all three were *statements* rather than crashes.

**The single-query key ignored the show/hide tick.** `retrieval_key_children`'s comparison branch reads `roles` and recedes a hidden arm; the single-query branch never read it. So unticking "Show it on the map" for a plain search took the star and the rings off the plot while the key went on reading "the query sample 1 / retrieved hit 5" - the exact failure the count rule exists to prevent, in the commonest state the map has. The rail sentence beside it had the same gap, under a docstring this change had just rewritten to claim it read the ticks.

**A hit retrieved by both cohorts was counted twice.** `hit_points` is a concatenation across the arms, so the rail said "10 hits, 2 of them retrieved by both" for the same comparison whose banner on the other view said "share 2 of 8 retrieved samples". Two surfaces, one search, two numbers, and a subset relation that holds under neither reading. Both now count distinct samples, and the badge says "samples" rather than "hits" so the unit is unambiguous.

**The reworded caveat said "two" while you could be looking at three.** "a projection of 512 dimensions into two" is a static string inside a group that stays on screen in 3-D. The copy it replaced read "into two or three of them", correct in both. It is now "a projection cannot preserve 512-dimensional distances", which is true of any projection and needs no dimensionality to check itself against - the alternative being to make one hint dims-aware for one word.

Each is now pinned by a test, including a browser check that unticks a single query and asserts the plot, the key and the rail all agree.

### 6. Two defects found on the way

Neither was in scope. Both were found by auditing what the map draws against what anything on screen explains, and both make a claim the code already made in prose actually true.

**3-D silently deleted the OSDR overlay's visual identity.**
`_scatter`'s `Scatter3d` branch passed no `symbol` and hard-coded `line=dict(width=0)`, so `OSDR_SYMBOL="diamond"` and `OSDR_OUTLINE="#ffffff"` were both discarded.
In 2-D the overlay is a white-ringed diamond at size 8.5; in 3-D it arrived as a plain circle at 4.25 with no ring, in the same palette hue as the 940,455 glyphs beneath it.
Only hover could tell 2,108 spaceflight samples from the corpus they sit in, which is the one thing `theme.py` says this map may not do, and `render.py`'s own module docstring asserted the opposite.
`test_osdr_markers_are_visually_distinct_from_the_cloud` never caught it because it only ever ran `("pca", "2d")`.

Both channels survive the trip: `diamond` is in `Scatter3d`'s eight-symbol vocabulary and gl-scatter3d honours `marker.line` as a per-point border.
This was also a **prerequisite** for the corpus key: a key asserting a diamond that 3-D does not draw is worse than the silence it replaced.

**A hit retrieved by both cohorts named only one of them on hover.**
Two traces sit at the identical coordinate and Plotly resolves exactly one tooltip per position, so building each cohort's rows from its own hit list alone made one arm's rank and cosine unreachable - for the very points a comparison exists to show.
`render.py` justified dropping the rank numerals in a comparison on the grounds that "the hover says strictly more (a shared hit names both cohorts, both ranks and both cosines)".
No code produced that tooltip.

A pre-pass now indexes every drawn point by the cohorts that retrieved it, and each arm carries its own map rank, because a shared hit is one coordinate with two nearest members and therefore two valid answers.
**The marks stay emergent.** A hit is still drawn twice because it is in two hit lists; nothing computes an intersection in order to draw. Only the hover reads across the arms.

### 7. Every mark the map can draw

The audit this work was built on. Each row is what the mark is, its 2-D and 3-D symbols, and where a viewer can decode it **now**.

| mark | 2-D | 3-D | keyed by |
| --- | --- | --- | --- |
| ARCHS4 corpus glyph | circle 3.4 | circle 1.7 | color legend + corpus key |
| residual (Other / Unknown) | circle 2.8 @ 0.26 | same | color legend + corpus key |
| ARCHS4 context cloud | circle 2.6 @ 0.35 | same | coverage readout + plot badge (no legend row, invariant 5) |
| OSDR overlay | diamond 8.5, white ring | **diamond 4.25, white ring** (was a plain circle) | corpus key |
| query halo | circle-open 46 / 32 | half size | nothing, deliberately (§3) |
| pooled member, cohort A | star, teal | diamond, teal | retrieval key |
| pooled member, cohort B | star, gold | diamond, gold | retrieval key |
| hit ring, cohort A | circle-open 20, white | circle-open 10 | retrieval key |
| hit ring, cohort B | square-open 27, white | square-open 13.5 | retrieval key |
| retrieved by both | both of the above | both | retrieval key |
| rank numerals | text above the ring | same | hover |

Two things in that table are known and unchanged.
`OSDR_HIGHLIGHT` is currently unreachable - every `ColorBy` has OSDR in its scope and `covers()` only ever strips ARCHS4 - and is kept as the defensive branch for a future ARCHS4-only field.
`RETRIEVAL_MAX_NUMERALS` is 25 while the top-k slider goes to 30, so at k > 25 the last rings carry no numeral; the ring and the hover both still work, and numbering past 25 was measured to be unreadable overlapping text.

### 8. Testing

**`tests/test_app.py`, 10 new tests, and `tests/test_cohorts.py`, 4 more.**
The key's structure for each of the five states (plain search, pooled cohort, upload, comparison, one arm hidden), the 3-D symbol substitution, the corpus key following the Layers ticks, that every glyph shape has a stylesheet rule, that the key reads its hues from `theme`, and the two copy guards.

The glyph-rule test exists because `test_every_classname_used_in_python_exists_in_some_stylesheet` cannot catch these: the class is built by string interpolation from the shape name, so it never appears in a `className` literal, and a shape with no rule renders as an empty 14 px box - a key row pointing at nothing.

**`tests/e2e_check.py`** gains the corpus key, its response to the Layers ticks, and the silent whole-map coverage readout.

**`tests/e2e_cohort_check.py`** gains both cohort cards with their role labels and the contrast facet, the role-grouped key with its shapes and its two hues, the hidden-arm rows, the 3-D diamond substitution, the OSDR overlay keeping its symbol and ring in 3-D, and a shared hit's hover naming both arms.

**One trap, recorded.**
A browser check that navigates with `page.goto("/map")` loses `hits-store` and sees an empty map. Use the in-app link; it is also the real user path.

**One bug the tests did not catch, and now do.**
The divider between the retrieval key and the color list was written as `.bm-key:not(:last-child)`, which can never fire - each key is the only child of its own slot div, so it is always a last child. It was rewritten as a modifier class, and then landed on only one of the two return statements, because the comparison branch returns from a different indent level than the single-query branch. Both key tests now assert the modifier is present. Neither failure was visible to any assertion; both were visible in a screenshot.

---

<a id="finding-a-sample-on-the-map"></a>

## Finding a sample on the map

The map draws 942,563 glyphs and, until 2026-08-11, offered no way to ask about a specific one.
A researcher holding OSD-100, or a GEO series they already know, could only hunt for it by eye.

Type an identifier into the box on the rail; the matching points are marked with a white X, and a button offers to frame them.
`manifold/find.py` is the whole of the lookup and opens no embedding, builds no figure and imports nothing from Dash, so it is unit-testable against the fixture corpus on a machine with neither the memmap nor the real cache.

### 1. What resolves, and what deliberately does not

| input | resolves to | measured scale |
| --- | --- | --- |
| `GSM…` | one ARCHS4 point | 940,455 accessions, all `GSM`-prefixed, no duplicate numbers |
| `GSE…` | every sample in that series | 51,284 series; median 9, p95 55, max 8,764 |
| `OSD-###` | every sample in that study | 70 studies; median 20, max 192 |
| an OSDR sample name, or a full `<study>\|<name>` key | one OSDR point | 2,108 keys, unique |

Matching is case-insensitive, trims whitespace, and accepts `OSD100` for `OSD-100`, because those are the same study to everyone except a string compare.
The echoed label is the canonical identifier rather than what was typed.

**Free text over titles and characteristics is refused.**
`archs4_metadata.parquet` carries `title`, `source_name` and `characteristics`, and a substring scan of 940,455 rows costs about 200 ms, so this is a decision and not a limitation.
It would return hundreds of rows for "liver" and read as a biological query when it is a string match, on the one map whose entire design is that a field declares what it does and does not describe.
"Where is liver" is the Tissue color-by's question and `manifold/colorby.py` already answers it across both corpora, so a `shape` miss says exactly that and points at the control that does the job.

An expansion-minded reviewer proposed searching the 51,284 *series titles* as a disambiguator, returning a picklist so the resolver still receives an exact accession.
It is the strongest version of the idea and it is still not built: it re-admits free text through a second door, and GSE was already refused as a color-by for being a pure batch label with 333x lift.

### 2. A miss is four outcomes, not one

Answering "liver" and "GSM999999999" with the same "not found" tells the first user their search is broken when they wanted a different control, and tells the second nothing about whether this machine could have looked it up at all.

| reason | when | what the rail says |
| --- | --- | --- |
| `empty` | nothing typed | nothing |
| `shape` | matches no grammar | what to type, and points at Color by |
| `absent` | well-formed, and this corpus lacks it | what the corpus does hold |
| `no_geo_metadata` | a GEO id, and the optional join was never fetched | `colorby.ARCHS4_META_HINT`, the same sentence the color-by uses |

That last row is the one an adversarial review caught, and it was a real hole.
`cache/archs4_metadata.parquet` is optional - `CLAUDE.md` says so, and a fresh clone starts without it - and without it **940,455 of the 942,563 points cannot be addressed at all**.
Reporting that as "absent" would tell a user their accession does not exist when the truth is that this machine cannot look it up, which is invariant 5 by another route.
`find.searchable()` reads `data.archs4_metadata_available` itself rather than re-deriving the path, because a second source of truth for that file was already a real bug once.

A well-formed OSDR sample key the map lacks is `absent`, not `shape`.
`OSD-141|Mmus_C57-6J_SPL_cells_Rep1_SP1` is a perfectly good key for one of the **788** samples the retrieval catalog lists and the embedding never covered (2,896 listed, 2,108 embedded), and telling its owner "that is not an identifier, try the color-by" would be wrong twice.
The map deliberately does **not** distinguish those 788 from a genuine typo: the catalog that knows they exist lives behind `BRIDGE_RNA_ROOT`, which the map view opens nothing in, and only **1 of the 788** has a study that is itself on the map, so inferring it from map-side data would sharpen one message in 788 and mislead about the rest.

### 3. The index is integer-keyed, and the 839 blank rows are why it is filtered

Both GEO accession spaces are a prefix plus digits, so the index is the parsed integer, sorted once, queried with `np.searchsorted`.

Measured on the real corpus: **15.0 MB retained, 460 ms to build once** on the first search, then about 0.8 ms a lookup.
`pd.Index(accessions).get_loc` retains **96.8 MB** on an app whose whole working set is 80.8 MB, and is rejected on that.
Two corrections to earlier figures, both instructive. An initial "90 ms" was the integer parse alone and excluded materializing two 940,455-row string columns, which is most of the cost; and a "5,963 ms" reading was taken with `tracemalloc` running, which inflates allocation-heavy code several-fold. The parse itself is about 200 ms whether written as a regex, a slice or a Python loop - all three within 20 ms of each other - so there is nothing to win by making it cleverer.

**839 rows of the real join carry an empty `series_id`** and are skipped before parsing.
They are the samples present in the release-matched v2.5 metadata and absent from the v2.latest the API serves.
Slicing the digits off one raises `ValueError` and takes the whole index build with it; coercing it to 0 would file those 839 under a series that does not exist.
They stay addressable by their own GSM and by nothing else.
`tests/fixture_corpus.py` now blanks every 53rd `series_id` for the same reason: without such a row this path was only ever tested against the easy half of its input.

### 4. The mark, capped and keyed

A white **X** - `x-open` in 2-D, `x` in 3-D, because `Scatter3d`'s eight-symbol vocabulary rejects the first outright, which is the error that took the figure callback down with a 500 the first time 3-D met a retrieval.
White for the measured reason the hit rings are white: no hue clears 3:1 against the worst categorical tissue bucket on the navy canvas.
Shape is the free channel and an X collides with nothing already drawn - the corpus is circles and diamonds, the query a star, a hit an open circle or an open square.

It is `Scattergl` in 2-D where the retrieval overlay uses plain `Scatter`.
That choice is right for at most k+2 points needing `markers+text` centred and wrong for a series of 8,764 marks that draws no text.
Marks shrink to 0.7 when there are several, the same rule and the same constant a pooled cohort's members follow: OSD-100's twelve samples frame into 1.08 units of x and at full size composite into one blot.

**The cap is `theme.FIND_MAX_MARKS` = 500, and it states what it dropped** - in the status line, the plot badge, and the key row - following `RETRIEVAL_MAX_NUMERALS` and `COMPARISON_MAX_LABELS` and the standing rule against silent caps.
Past a few hundred X's the marks stop locating individual samples and start painting a region, which is a color-by's job, and specifically the job GSE was refused as a color-by for.
The first 500 in point order is not a sample: point order is the corpus's own order, so the marked subset is reproducible between identical searches rather than shifting on every redraw.

The key row counts what is **drawn**, not what exists, which is the key's standing rule and most obviously checkable here: 8,764 beside 500 visible marks would be that rule broken in plain sight.
Per [The key on the map](#map-key) the mark earns a row and a `.bm-key-glyph.is-found` stylesheet rule; unlike the member mark it needs no dimensionality argument, because both spellings draw the same X.

### 5. Framing is a button, and never automatic

This is the decision the design review changed, and there are two reasons, the second stronger than the first.

**A found set can span the map.** The pad in `frame_points` is a share of the span, which is right for a retrieval's handful of neighbouring points and wrong for a set drawn from one experiment but not from one region. Measured before the guard went in: **OSD-457's 192 samples framed to 1.22x the corpus width and GSE228590's 8,764 to 1.03x**, against 0.02x for a typical study and 0.06x for a typical series. So "framing" would have zoomed the user *out*, silently, as the result of typing.

**And a 2-D neighbourhood is not a similarity neighbourhood.** The map's 20 nearest points to a query overlap the true cosine top-20 by a mean of 2.7 of 20 and a **median of 0**. Dropping someone into a zoomed view of their sample's surroundings invites reading those surroundings as related when they are not. Marking answers "where is it"; zooming asserts something further, so it waits to be asked for. `_frame_for`'s own docstring had already made this call for the retrieval, and the find must not quietly reverse it - `test_finding_something_never_moves_the_viewport_by_itself` pins it structurally, by asserting `find-store` is not an Input of the viewport callback, because that is the thing that would make framing automatic.

`_clamped_to_corpus` fixes the first problem for both callers: each edge is clamped to the corpus extent, so a spread set frames at exactly the whole map - the honest answer for a set that really is everywhere - and every compact set is untouched. After it: OSD-457 1.00x, GSE228590 0.87x, OSD-100 unchanged at 0.02x.

The button is hidden in 3-D, where framing pins the 2-D axis ranges and the camera ignores them, exactly as the frame-retrieval button is.
The viewport callback returns `no_update` and never `None` on a miss, because `None` is that callback's "reset to the whole corpus" and a dead click would otherwise have zoomed the user out.

### 6. One record panel, two ways in

Clicking an OSDR diamond and finding a sample ask the same question, so they share `picked-group` rather than growing a second panel beside it.
The click path is turned into a find: the overlay's `customdata` already carries the sample key and `find.find(key)` resolves exactly that point, so there is one lookup, one record builder and one panel instead of two of each drifting apart.
A click on the cloud returns no `customdata` and leaves the panel alone rather than clearing it - a stray click on empty space should not wipe the record the user just searched for.

The action at the bottom differs by corpus and the difference is not cosmetic.
An OSDR sample offers a retrieval, an in-app navigation and therefore a `dcc.Link`.
A GEO sample offers its NCBI record, an external URL and therefore an `<a>`: **a `dcc.Link` pointing off-site hands its href to Dash's client-side router**, which tries to resolve `ncbi.nlm.nih.gov` as an application route instead of leaving the page.
An OSDR study gets no GEO link at all, because a study is an OSDR concept and the link would 404.
A set reports its count and does not repeat its own name as a row under itself.

`find.describe` lives beside the lookup rather than in `layout` because it is a metadata read and not a rendering decision, so it is testable without Dash and both entry points share it.

### 7. What was cut: the click probe

The sibling feature - click any point in the ARCHS4 cloud and inspect it - was specified, spiked, and dropped.
It is worth recording why, because the obvious implementation looks fine and is not.

**The cloud emits no click event at all.** `render._scatter` sets `hoverinfo="skip"` on any trace with no hover lines, and `skip` suppresses click picking as well as hover. Measured by attaching a `plotly_click` listener and clicking a dense ARCHS4 pixel 19.3 data units from the nearest OSDR diamond: zero events. The same click on a diamond returns a point.

**Turning hover on to get the click costs 240 ms per mouse move.** With `hoverinfo="x+y"` restyled onto all 13 cloud traces the point becomes clickable and synchronous mousemove handling goes to a **median of 239.8 ms, max 243.7 ms** at 942,563 points, sustained rather than a one-time index build. `hoverinfo="none"` was tried on the theory that it keeps events while dropping the label; it does not restore click picking either.

**A workable route exists.** A plain DOM listener on the `.nsewdrag` layer converts the click's pixel position through Plotly's own `xaxis.p2d()` - round-tripped at an error of 1.4e-4 data units, or 0.004 px - and the server resolves the nearest drawn point in 0.5-0.9 ms against a candidate set recomputed deterministically in 2.8-19.2 ms. No hover, no `customdata`, no per-frame cost.

**But a click at full-corpus zoom is ambiguous, and the number is large.** The end-to-end spike resolved a click to within 0.09 px of the target and found **758 drawn points within 3 px of it**; zoomed to a 3-unit window, still 35. At 0.0404 data units per pixel a point's nearest neighbours are 0.0019 to 0.0092 units away, so hundreds of samples share a pixel. A probe that prints one accession for a click on 758 overlapping samples is the same class of error as a population stability curve quoted beside one cohort's name.

3-D is the opposite case and is the more promising one: picking there is GPU-based and costs **0.1 ms** per mouse move at the 40,000-point cap, though `plotly_click` could not be made to fire on a `Scatter3d` cloud point in a headless driver during the spike.

**A guard survives the cut.** A test asserts the ARCHS4 traces still carry `hoverinfo="skip"`, so a future change that "fixes" picking by enabling hover fails in the suite rather than in someone's hand at 240 ms per mouse move.

Note that the *record* half of the probe shipped anyway, through the find box: looking a GSM up by accession renders exactly the GEO record a click would have shown. What was cut is the pointing, not the reading.

### 8. Deferred

**Deep links.** `/map?find=GSE143281` would make a 942,563-point instrument citable, and a found set is stable across projections so the URL would be durable. Not built: it collides with `app.navigation_for`'s decline-to-repaint rule, which already caused one live bug, and it is scope beyond the box that was asked for.

**A composition readout for a found set** - its spread, its tissue buckets - was proposed and rejected outright rather than deferred. It reinstates the lasso's 512-d statistical readout, which `CLAUDE.md` lists among the features that are gone and must not come back as current behavior.

### 9. How it is tested

35 unit tests in `tests/test_find.py`, all against the fixture corpus: the four grammars, normalization, each of the four miss reasons, the blank-`series_id` filter, index dtypes and one-time construction, the coverage degradation with the join removed, the mark's cap and its WebGL trace type, the key row's count, the panel's component types for internal versus external links, and the structural check that a search cannot move the viewport.

20 browser checks in `tests/e2e_check.py` against the real corpus, taking it from 51 to 71: a GSM marking exactly one point, the 8,764-sample series capped at 500 with the cap stated in three places, free text pointed at the color-by, an absent accession distinguished from it, the viewport unmoved until the button is pressed, and 3-D marking the set while hiding a frame button its camera would ignore.

---

<a id="find-autocomplete"></a>

## Completing an identifier, without becoming the free-text search that was refused

### What this adds

The find box took a whole identifier or nothing.
That is a reasonable contract for `GSM4256053`, which a reader pastes, and a poor one for `OSD-137`, which they half-remember, and a bad one for `Mmus_BAL-TAL_LVR_FLT_Rep1_F1`, which nobody types correctly twice.
As of 2026-08-12 the box suggests completions while it is being typed: a bounded, scrollable listbox with full combobox semantics, driven by mouse, touch and keyboard.

### 1. The rule that keeps it from being a search

`manifold/find.py` refuses free text, and that refusal is a decision with a measurement behind it: `archs4_metadata.parquet` carries `title`, `source_name` and `characteristics`, and a substring scan of 940,455 rows costs about 200 ms - affordable, and still wrong, because "liver" would return hundreds of rows and read as a biological query on the one map whose whole design is that a field declares what it does and does not describe.

Suggestions could have quietly reintroduced exactly that, as a dropdown.
They do not. **Every suggestion is a prefix of one of the four grammars `find()` already resolves**, so a suggestion is a completion of what is being typed and never a guess at what it might have meant.
Typing `liver` produces no list at all, and the sentence under the box still points at the Tissue color-by.

The one apparent exception is not one. An OSDR sample *name* is matched by substring rather than by prefix, because a name is itself an identifier, there are 2,108 of them rather than 940,455, and nobody remembers whether the one they want begins `Mmus_C57-6J_LVR` or ends `_FLT_Rep1_M23`.

### 2. What each grammar costs

| typed | how it is resolved | measured, warm |
| --- | --- | --- |
| `GSM4256` | numeric-prefix scan of the sorted int32 accession index | ~2 ms |
| `GSE1652` | the same, plus `np.unique` on the slice for each series' sample count | ~2 ms |
| `OSD-10` | scan of the 70-key study index | ~8 ms |
| `Mmus_C57` | substring scan of 2,108 OSDR keys | <1 ms |

The accession index stores numbers, not strings - 15.0 MB of int32 against 96.8 MB for a pandas string index, which is the measurement that made the find box affordable in the first place.
A numeric prefix is therefore not a string compare but a union of decade ranges: the integers beginning "42" are `[42, 43)`, then `[420, 430)`, then `[4200, 4300)`, and so on.
Nine `searchsorted` pairs cover every GEO accession ever issued.
The first call of a session pays the index's one-time 460 ms build, which is the same build the first *search* has always paid for, now paid a few keystrokes earlier.

### 3. Typing must not run a search, and twice it did

`dcc.Input`'s `debounce=True` means "publish the value on Enter or on blur", so with it on there is no per-keystroke value to complete.
It is off, and the commit is made explicit instead: `n_submit` and `n_blur` are exactly the two events `debounce` was firing on, and they are Inputs to `resolve_find`.
That split is what keeps a 942,563-point figure from being rebuilt on every letter of `GSM4256053`.

Two defects got round it, and both were found in the browser rather than in the suite.

**A pattern-matching `ALL` input fires when its family is re-rendered, not only when a member is clicked.** The suggestion rows are re-rendered on every keystroke with every `n_clicks` back at zero, so reading that as "no identifier, use whatever is in the box" ran a real search per letter - the exact cost the split exists to avoid, smuggled in through the other input.

**Worse: Dash discards the response to a request that a newer request for the same callback supersedes.** With the family's input on the callback that searches, a keystroke's no-op reliably overtook the Enter the server had *already answered correctly*. Measured on the real app: the first find of every session vanished, and the rest survived on timing alone. The fix is structural rather than defensive - the family's input moved to `choose_suggestion`, which writes `find-input.value` and a `find-chosen` store, and `find-store` is now written by a callback with **no per-keystroke input at all**. `tests/test_app.py::test_nothing_that_changes_per_keystroke_can_reach_the_search` asserts that property directly rather than asserting the symptom.

A third, smaller one is recorded because it looks like a defect and is not. `page.fill` followed immediately by `page.press("Enter")` commits the *previous* text, because the two are CDP calls with no event-loop tick between them and Dash's Input publishes `value` one tick after the change. Measured: a gap of 0 ms is stale and **every gap from 5 ms up is correct**, against a fastest realistic keypress-to-keypress of tens of milliseconds. It is not reachable from a keyboard, and the harness now reproduces the ordering a real one has.

### 4. Who owns whether the list is open

Two independent facts, one owner each, composed by CSS into a single `display`:

| fact | owner | mechanism |
| --- | --- | --- |
| are there any rows | the server | `offer_suggestions` renders them; `.bm-suggest:empty` hides an empty list |
| has the reader dismissed it | the browser | one `is-closed` class on the group |

No callback writes a `style` here and `assets/find-suggest.js` never removes a row, so the two cannot race over one property.
That file is the map's only JavaScript, and it exists for the one thing Dash cannot express: a keystroke. `dcc.Input` publishes `value`, `n_submit` and `n_blur`, and none of those is Up, Down or Escape.

Selection goes **through** Dash rather than around it. A row is a real component with a pattern-matching id carrying its own identifier, so a click, a tap and Enter all end in the same `n_clicks` and the same server callback - Enter is implemented as `activeRow.click()` for exactly that reason. Nothing pokes a value into the input, which React would not see and which would give a search two ways to be committed.

Two smaller decisions in that file are load-bearing.
`mousedown` inside the list is `preventDefault`ed, because without it the mousedown blurs the input, `n_blur` commits the fragment in the box, and the click commits the real identifier a moment later - two searches for one gesture, the first of them wrong. It is also what makes a tap work, since a touch synthesises that mousedown before the blur.
And Enter is stopped in the **capture** phase when a row is active, so Dash's own Enter handler never sees it; with no row active it is deliberately left alone and Enter means what it always did.

### 5. The row is two lines, and that was measured

Side by side on a 268 px rail the identifier and its detail compete for about 220 px, and which one loses is decided by CSS rather than by which matters.
With both at the default shrink, `GSM5028824` beside `GSE165242 · Embryo / stem cell` ellipsed to `GSM502…` - the one string in the row that cannot be inferred from the rest of it.
Weighting the shrink towards the detail helped and did not fix it: any shrink at all on a 78 px label truncates it, and `flex-shrink: 0` instead spends the detail down to `OSD-100 · l…`.
Neither half is expendable - the OSDR sample names differ in their *last* characters (`Rep1_M23` against `Rep2_M24`), which is exactly what an ellipsis eats.

Stacked, both are whole at every width tested (1680, 1280, 1100, 860, 600 and 393 px), at about 11 px per row.
The list holds ten suggestions in a 196 px window, so four rows and part of a fifth are visible: a window sized to a whole number of rows looks complete when it is not, and the cut row is the affordance that says it scrolls.

The list is **inline** rather than floating, because the rail is itself a scroll container and an absolutely-positioned popup would either be clipped by it or detach from the field it belongs to.

### 6. What a miss says

A miss carries the reason `find()` already distinguishes, and the interface answers each one differently.
A well-formed accession this map does not carry gets one flat, unselectable row ("Nothing on this map matches that."), because that is worth saying.
Free text gets **no list at all** - a dropdown volunteering "no matches" under the word "liver" would contradict the sentence below the box, which correctly says that free text is not what this control takes.

### 7. How it is tested

20 unit tests in `tests/test_find.py`: each grammar's completion, exact-match-first ordering, the series sample count, substring matching on OSDR names, studies ranked above their own samples, the free-text refusal, each miss reason, the one-character floor, the limit, the degraded no-GEO-join state, the blank-`series_id` rows that would crash a naive prefix parse, and the contract that **every suggested value resolves through `find()`** - which is what keeps one resolution path rather than two.

Three callback-level tests in `tests/test_app.py` pin the wiring: that a re-rendered family is not a click, that a component merely appearing is not a commit, and that nothing changing per keystroke can reach the search callback.

27 browser checks in `tests/e2e_check.py`, plus a 40-check interactive sweep run during development covering partial and exact queries, case, rapid typing, clearing and reopening, mouse, touch and keyboard selection, Escape, click-outside, scrolling past the visible height, and the six supported viewports.

---

<a id="reset-view"></a>

## One reset for every framing

### The question that decided where it goes

Three things narrow the map's viewport: framing a find, framing a retrieval, and the reader's own scroll or drag.
The first question was whether those share state. They do, completely: all three write `viewport-store`, which one callback owns, and `None` is that callback's word for the whole corpus.
By the time it is written, a framed study and a scroll zoom are indistinguishable.

So there is one thing to undo, and it takes one control.
A button beside "Frame it" and a second beside "Frame the retrieval" would have been two handlers for one piece of state - and neither would have answered a plain scroll-zoom, which is how most readers narrow the view in the first place.

**It sits on the plot rather than on the rail**, and that follows the rail's own rule rather than breaking it: a control's qualifier sits with that control, and this one qualifies the viewport, which belongs to the plot. It shares the badge strip, which already reports what is drawn right now.

It is labelled **Reset view** rather than "Unframe" because it also undoes a scroll zoom, which was never a frame.

### The part that would have failed silently

`theme.base_figure_layout` sets `uirevision="keep"`, which asks Plotly to preserve the reader's own pan and zoom across a figure update **unless the incoming figure changes the attribute in question**.
Framing works on that rule: it sets a range where the previous figure had none.
Resetting has to go the other way, and an absent key is not a changed value - leaving the range off and expecting Plotly to autorange leans on the one thing `uirevision` exists to prevent.

`callbacks.viewport_axes` therefore says `autorange` outright.
Measured on the real corpus, the reset restores the whole map exactly: x span 60.17 → 1.13 → 60.17 in UMAP, 116,882 → 1.46 → 116,882 in t-SNE, 2.22 → 0.51 → 2.22 in PCA.

### What it clears, and what it must not

Exactly one store, so there is nothing else it *could* clear.
Verified end to end for the find (study, series and single GSM, in all three projections) and for all four retrieval paths (sample, cohort, comparison, upload): after a reset the query text, the found marks, both cohorts' members, every hit and the frame button are all still there, and framing again gives the identical window rather than a stale one.

3-D never offers it. `frame_points` declines there and the viewport does not drive a `Scatter3d` camera, so a reset would be a click with no visible effect - which is what the lasso was removed for.

---

<a id="osdr-only-color-bys"></a>

## The nine OSDR-only color-bys, and what their machinery is still for

Removed on 2026-08-12: Flight vs Ground, Spaceflight arm, Strain, Sex, Genotype, Study, Habitat, Mission duration, Diet.
None of them separated anything on this map. They coloured 2,108 of 942,563 points - 0.2% - and the 2,108 are scattered through a corpus whose structure is set by the 940,455 they sit among, so a spaceflight arm or a mouse strain drew a handful of differently-coloured diamonds with no spatial pattern to read.
Tissue and Species remain, and both colour all 942,563 points.

`manifold/data._flight_status` went with them: it derived a Flight/Ground collapse that existed for one colour-by and had no other reader. The raw `spaceflight` column is untouched, because `find.py` prints it in the record panel and the retrieval half groups cohorts by it - on the raw arms rather than the collapse, since a basal animal was sacrificed at experiment start and a vivarium animal never entered flight hardware.

**The coverage machinery stays, and is not vestigial.** It looks over-engineered for a two-entry registry and is not, because **Tissue becomes an OSDR-only field on a machine that never fetched `cache/archs4_metadata.parquet`** - which is the state a fresh clone starts in. On that path the menu labels it "OSDR only", the coverage bar goes amber and states the exact count, the fix is named under the control, and the renderer draws ARCHS4 as a faint context cloud rather than as a grey category. That is invariant 5, and it is now the only route to it, so the tests that used to assert it through `flight_status` assert it through degraded Tissue instead - a stronger test, since it is a state a real user reaches.

One consequence worth recording: the fixture corpus had exactly one field that overflowed the eleven-slot palette, and it was Study. With Study gone the fixture's Tissue spanned seven buckets and nothing anywhere in it could overflow a legend. `tests/fixture_corpus.py` now draws ARCHS4's raw source names from eighteen strings that canonicalize to eighteen distinct buckets, three per synthetic cluster, so Tissue overflows there the way it does on the real corpus - 39 buckets against eleven slots.

---

<a id="file-ingestion"></a>

## File ingestion: embed an uploaded OSDR sample live and retrieve its Earth analogs

### What this adds

The Retrieve view can already answer for the 2,108 OSDR samples the manifold precompute embedded.
This feature lets a user bring an OSDR sample the corpus has never seen - upload its counts, embed it live, and get the identical output (network graph + inspector + optional LLM summary) the picker produces, computed against the same 940,455-sample ARCHS4 index.

### The one idea that makes this small

`bridge_rna/retrieval.py` is already built around a single fact: the cosine scan (`_topk_cosine_from_memmap`) is shared, and the three existing paths (cached, precomputed, demo) differ *only in where the 512-d query vector comes from*.
File ingestion is a **fourth query-vector source**.
Everything downstream - the top-k scan, the offline annotation (`_annotate_from_cache`), the `archs4_index` column, the figure, the inspector, the summary - is reused unchanged.

So the output for an uploaded sample is the same schema as the cached path, annotated from the same local `archs4_metadata.parquet` (gse / title / source_name / characteristics / tissue / species), with the same `archs4_index` that lets a hit be located on the Map.

### Where the embedding runs, and why it is a subprocess

Invariant: the serving app never imports `torch` at module scope, and `tests/test_app.py::test_the_serving_app_does_not_import_the_scientific_stack` pins that.
The existing `demo` path already embeds live, and it does so by shelling out (`run_real_retrieval` -> `demo_osdr_top5.py`).
File ingestion follows the same pattern: a new CLI, `precompute/embed_upload.py`, loads the checkpoint, embeds one counts file, writes a 512-d `.npy`, and exits.
`bridge_rna/retrieval.run_uploaded_retrieval` invokes it with `sys.executable`, loads the vector, and runs the shared scan.
No torch import enters the serving process, and the app still starts on a machine with no model.

### The preprocessing is the audited one, not a new one

Embedding an OSDR sample wrong is a silent scientific error, so `embed_upload.py` reuses the exact symbols already funnelled through `manifold/bridge_rna.py` (`load_bridge_rna_symbols`) rather than re-implementing preprocessing:

1. Read counts CSV, `index_col=0` (mouse Ensembl gene IDs), strip version suffix.
2. Map mouse Ensembl -> human ortholog symbol (`build_mouse_to_human_maps`), keep mappable, sum duplicates.
3. Reindex to the 15,165 canonical genes, fill 0.
4. TPM-normalize (`normalize_counts_to_tpm_single`), then `log1p(max(., 0))`.
5. `model.encode(x, None, normalize=False)` -> 512-d, exactly as `embed_osdr.py` and `demo_osdr_top5.py` do (species passed as `None`; the OSDR embedding path does not use the species embedding).

**Invariant 1 (gene-digest gate) is enforced**: `embed_upload.py` computes `canonical_gene_order_digest(genes)` and aborts unless it equals `CANONICAL_GENES_SHA256`.
This is byte-for-byte the same gate `embed_osdr.py` runs, so an uploaded sample is embedded in the same gene order as the corpus it is compared against, or the build refuses.

### Input contract

- Format: CSV (optionally `.tsv`/gzip), genes in rows, samples in columns, first column = mouse Ensembl gene IDs (version suffixes tolerated), matching the OSDR counts matrices already in `data/osdr`.
- Species: mouse. OSDR spaceflight data is Mus musculus (the shipped metadata is 100% mouse), and the preprocessing maps mouse Ensembl -> human ortholog space. A human-indexed matrix is out of scope for this pass and rejected with a clear message rather than silently mis-mapped.
- Sample column: if the matrix has one sample column it is used; if several, the user picks which column, defaulting to the first.
- A file that maps zero genes through the ortholog table (e.g. human Ensembl IDs, or symbols) is rejected with the reason, never embedded into a meaningless vector.
- Size: capped at `MAX_UPLOAD_BYTES`, 200 MB, which is generous headroom over the few MB a whole-transcriptome counts file actually costs.
  The cap is enforced twice, because one of the two places is too late on its own: as Flask's `MAX_CONTENT_LENGTH` (`app.py`), so an oversized body fails at the request layer with a 413 rather than arriving truncated and being parsed into a malformed matrix; and again in the callback after the base64 decode, where the user is told the actual size and the limit.

**No metadata is required or accepted alongside the counts, so this path collects nothing about the sample or about whoever uploaded it.**
That is a consequence of the design rather than a policy bolted on: the query vector is a pure function of one counts column, so tissue, flight-vs-ground and accession could only fill the inspector and the summary prompt, never a hit or a score.
Combined with the default loopback bind, an upload in the normal local setup does not cross a network at all.

### Failure handling

Every failure is surfaced as one clean line in the status banner, with the full detail logged server-side (mirrors `run_real_retrieval`):
- missing model prerequisites (checkpoint / orthologs / canonical genes / exon lengths not resolved, or an unresolved LFS pointer),
- unreadable or empty counts file,
- no ortholog-mappable genes,
- gene-digest mismatch (refuses rather than embedding wrong).

### The retrieval mode

`search_hits`-style contract: the uploaded path returns mode `"uploaded"`.
The status banner must name it, exactly as the existing invariant requires the interface to always say which path ran.
This is added to the banner's mode->label map, not special-cased.

### Files

| file | change |
| --- | --- |
| `precompute/embed_upload.py` | NEW - embed one counts file -> 512-d npy, gene-digest gated |
| `bridge_rna/config.py` | add `UPLOAD_EMBED_SCRIPT_PATH`, upload size cap |
| `bridge_rna/retrieval.py` | add `embed_uploaded_counts` + `run_uploaded_retrieval`, mode `"uploaded"` |
| `bridge_rna/layout.py` | `dcc.Upload` + sample-column control in the Retrieve rail |
| `bridge_rna/callbacks.py` | upload -> temp file -> retrieval -> `hits-store` -> same render; banner label |
| `tests/test_upload_ingestion.py` | NEW - preprocessing parity, gene-digest gate, mode, annotation schema |
| docs | IMPLEMENTATION.md, REFERENCE.md, progress.md, CLAUDE.md |

### Testing

- **Preprocessing parity**: an uploaded file built from a known OSDR sample's own counts column must embed to the same 512-d vector (to float tolerance) as that sample's precomputed cached vector - the strongest possible check that the live path matches the corpus.
- **Gene-digest gate**: a shuffled canonical gene order aborts the embed.
- **Annotation schema**: uploaded hits carry the same columns as cached hits, including `archs4_index`.
- **Mode**: the uploaded path reports `"uploaded"` and the banner names it.
- **Serving-import invariant still holds**: `embed_upload.py` is in `precompute/`, so the app-import test is unaffected.

### Verified in a browser, in a loop (2026-07-30)

`tests/e2e_upload_check.py` drives the real app through the real flow: 97 checks, three cycles through one long-lived page, about eight minutes.
The loop is the design. A one-shot upload check passes on a page that leaks state between runs, so every fixture runs three times and every cycle must reproduce cycle 1 step for step.
Correctness is anchored to the catalog path rather than to a golden file: `examples/osdr_upload_example.csv` is two columns of OSD-100, both cached OSDR samples, so an uploaded column must return exactly what the picker returns for that sample. Format variations (single-column, version-suffixed IDs, TSV, gzip) must return that same answer, rejections must draw nothing and say why, and a valid upload straight after a rejection must be correct again.

Two defects it found, both fixed:

- **Staged uploads accumulated forever.** Every upload wrote a `delete=False` temp file that nothing removed; one looped run left 32 files and 29 MB behind, surviving process exit. See `_upload_dir` / `_discard_upload` / `_sweep_abandoned_upload_dirs` in `bridge_rna/callbacks.py`, and the note there on why the reaping is PID-tagged rather than signal-based.
- **The inspector claimed an OSDR study for an uploaded sample**, repeating the "Uploaded file" study ID already under Identity and sending a lookup for a study that does not exist.

### Staging, and why it is not just a temp file

The embedder is a subprocess and takes a path, so an upload has to reach disk and survive past the callback that received it. That is the whole reason it cannot live in a `with` block. Three bounds keep it from being a leak: one process-owned directory removed at exit, a session's previous file unlinked when its next upload arrives, and PID-tagged reaping of directories whose owner is gone. Steady state is one file per active session.

---

<a id="readme-screenshots"></a>

## The two README screenshots, and why they are measured rather than framed by hand

`README.md` embeds two images, one per view: `docs/bridge-rna-interface.png` and `docs/bridge-rna-map.png`.
`tests/screenshot_readme.py` captures exactly those two and nothing else.
It is a sibling of `tests/screenshots.py` with a narrower job: that one walks both views and composes a fourteen-frame gallery at a fixed 1680x1010, this one produces the two frames the README ships and refuses to accept a fixed viewport.

### The defect this was written to fix

The shipped `docs/bridge-rna-interface.png`, captured 2026-07-22, was cut off.
The inspector ended mid-record, so the top hit's publication, journal and DOI were missing from the frame, and the retrieval network's lowest node was sliced in half by the bottom of the image.

That is not a framing mistake, it is a property of the layout meeting a viewport that was too short.
Both views are fixed-height instruments that scroll internally: `assets/01-shell.css` gives the shell `height: 100%`, and every panel under it carries `min-height: 0` with its own `overflow-y: auto`.
A page like that never grows past the window, so a window shorter than the content does not produce a scrollable screenshot.
It produces a silently clipped one, and the clipping is invisible in the capture script's output.

Measured on the real app at the viewport the old capture used:

| viewport | `.sidebar` | `#details-panel` |
| --- | --- | --- |
| 1680x1010 | 26 px hidden | 410 px hidden |
| 1680x1400 | fits | 20 px hidden |
| 1680x1444 | fits | fits |

The 410 px is the same 410 px named in the comment at `assets/retrieve.css:21-28`, which records the day the row track became `minmax(0, 1fr)` so a long GEO record would scroll the inspector instead of pushing the whole page below the fold.
The CSS fix was correct and is still in place.
The screenshot simply predated it and was never retaken.

### Two ways a frame can be cut, and only one of them is visible to the DOM

**A panel can clip its content.** `scrollHeight > clientHeight` on any scroll container says so exactly, and `fit_viewport` grows the window until no container reports it.
The window is re-measured after each step rather than computed once, because a taller window changes what the panels lay out.

Two false positives have to be excluded or the loop never converges.
`.visually-hidden` clips a 1x1 box on purpose, which is how a Dash control that renders no labelable element gets an accessible name, and Dash's own checkbox wrappers are 1x1 for the same reason.
Both report overflow forever and neither is visible, so the check ignores anything under 40 px in either dimension.

**A figure can run off its canvas, and the DOM cannot see it at all.**
A Plotly canvas is exactly as big as its container whether or not the drawing inside it fits.
The first capture of the map made this concrete: a 3-D camera dollied in by two mouse-wheel events reported no overflow anywhere on the page while the point cloud and the bottom row of tick numerals ran off the bottom edge.

So the figure is measured as pixels.
`EDGE_INK_JS` re-renders it through `Plotly.toImage` and counts how much of the outer band differs from the paper colour, which is read out of the corner of the image rather than assumed so the same code works on the map's navy canvas and the retrieval network's white one.
Rendering through `toImage` rather than reading the live canvas is deliberate: it returns the figure alone, so the floating key and the plot badges sitting over the map do not count as ink.

Two bands, because "cut off" and "uncomfortably close" are different faults and only one is a defect:

- `CUT_BAND`, 3 px. Ink here means the drawing continues past the boundary, which is a glyph or a numeral with its other half missing. This is the hard failure, asserted after the shot is taken.
- `COMFORT_BAND`, 14 px. This is the margin the framing aims for. Failing it costs a wider camera, not a failed run.

### Framing the 3-D scene

The camera is set outright with `Plotly.relayout`, not dollied with wheel events.
A wheel step is a fixed fraction of the current distance, so a loop of wheel events cannot ask for a particular framing and cannot be repeated after a resize, which is exactly how the first attempt overshot.

`eye` is a unit direction times a distance, and only the distance is searched.
The winner is the framing with the most ink on the canvas among those leaving the comfort band clean, which is "as large as it fits" stated as a number rather than as a judgement.
Measured at 1680x1010, tissue colouring, 40k ARCHS4 budget:

| distance | canvas carrying ink | ink in the outer 14 px | |
| --- | --- | --- | --- |
| 1.60 | 7.97% | 0.247% | touches the edge |
| 1.75 | 7.20% | 0.166% | touches the edge |
| 1.90 | 6.52% | 0.126% | touches the edge |
| 2.05 | 5.94% | 0.089% | touches the edge |
| **2.20** | **5.44%** | **0.000%** | **chosen** |
| 2.40 | 4.87% | 0.000% | smaller for nothing |

The direction is fixed at `(0.68, 0.68, 0.28)` rather than searched.
It was searched first, over three candidates at three canvas heights, and the three scored within 0.01% of each other on fill, which is noise.
Letting noise pick the camera angle means the frame changes shape between runs for no reason, so the angle is a stated compositional choice: slightly above the cloud and off-square, so the corpus reads as a volume rather than as a flat sheet.

Canvas height was searched at the same time and does not earn its cost.
Growing the map window from 1010 px to 1450 px moves fill from 5.44% to 5.83%, because what limits the 3-D frame is the sprawl of the x and y tick numerals across the bottom, not the height of the canvas.
The map is therefore captured at its natural fitted height and the retrieval frame is the tall one.

**Only what a user could do.** The camera is a user action, since rotating and zooming the scene is what the modebar and the drag handle are for.
`scene.domain`, axis visibility and marker sizes are not, so none of them are touched.
A screenshot that reframes the app by editing the figure is no longer a screenshot of the app.

### What the two frames land at

| frame | viewport | image | edge band |
| --- | --- | --- | --- |
| `bridge-rna-interface.png` | 1680x1444 | 3360x2888 | 0.000% |
| `bridge-rna-map.png` | 1680x1010 | 3360x2020 | 0.000% |

Both at `device_scale_factor=2`, so the type is retina-sharp at the width a README renders them.
The two heights differ because the two views need different amounts of room, and forcing them to match would mean either clipping the inspector again or padding the map with empty canvas.

Each frame is captured in its own browser context, so the map is a clean map rather than one carrying the retrieval the previous frame left in `hits-store` on the shell.

### One truncation that is not a defect

The rail's OSDR sample dropdown reads `Mmus_C57-6J_EYE_FLT_Re...`.
That is a `text-overflow: ellipsis` inside a fixed 288 px rail, not a clipped frame: sample keys are arbitrarily long, the control is doing the right thing, and the full key `Mmus_C57-6J_EYE_FLT_Rep1_M23` is printed in full in the query card immediately below it and again as the query node's label in the network.
Do not widen the rail to make it go away.

### Rejected

**`full_page=True`.** It captures the document, and on a `height: 100%` shell with `overflow: hidden` the document *is* the viewport. It would have reproduced the clipped frame exactly.

**Cropping or scaling the image afterwards.** A README screenshot is evidence about the app. Anything that edits the pixels after capture makes it evidence about the editing.

**Shrinking the page with CSS `zoom` to make the content fit.** It fits the content by making the app's type smaller than the app's type, which misrepresents the interface at exactly the moment the image is meant to represent it.

**Capturing the map in 2-D, which fills its canvas far better.** It does, and it is a different claim: the README's map paragraph is about a corpus you can rotate. 2-D is one pill away in the app and the README already says so.

### Running it

```bash
/Users/josh/Bridge-RNA/.venv/bin/python tests/screenshot_readme.py           # both, straight into docs/
/Users/josh/Bridge-RNA/.venv/bin/python tests/screenshot_readme.py --only map
/Users/josh/Bridge-RNA/.venv/bin/python tests/screenshot_readme.py --out /tmp/shots --headed
```

It boots `app.py` against the real `cache/`, so it needs the artifacts and the LFS objects, and it takes about three minutes.
It exits non-zero if either frame ends up with a clipped panel, a figure touching its canvas edge, or a console error, so a layout change that breaks the images fails the capture rather than shipping a cut one.
