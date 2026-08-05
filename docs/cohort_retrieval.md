# Cohort retrieval: querying with an experimental group

**Status: built, measured on the real corpus, and tested, 2026-08-05.**
This is the implementation document for the feature `docs/cohort_pooling.md` specified and measured.
That document is the prior evidence; this one is the build, and it carries its own measurements.
Every number below was produced by `precompute/validate_cohorts.py` against the real 963 MB ARCHS4 memmap and the real 2,108 cached OSDR embeddings, over **all 212 cohorts**, not a sample of them.

| | |
| --- | --- |
| Cohorts under the default definition | **212** with two or more samples, across 70 studies (215 including 3 singletons) |
| Size | median 10, mean 9.9, max 38; 2,105 of 2,108 samples grouped |
| Pooled top-5 leave-one-out stability | **0.738**, against **0.161** for one sample: a **4.6x** gain |
| Against a structure-free null | 0.331, so the cohort definition is worth **+0.407** |
| Against a within-study null | 0.683, so tissue and arm are worth **+0.055** on top of the study |
| Cost of a pooled query | one memmap pass, the same ~0.5 s as a single sample, at any k |
| Tests | 34 unit tests, 60 browser checks, 6 corpus-scale validation checks |

## Why this exists, in one paragraph

The Retrieve view answers for one OSDR sample at a time, and a single-sample top-5 is not a stable measurement.
Two replicates from the same cage share on average **0.13** of their top-5 ARCHS4 hits, and in every cohort tested there is a pair whose top-5 lists share nothing at all.
The cause is a scale mismatch rather than an outlier problem: the entire top-500 of a 940,455-sample index spans a cosine range comparable to the gap between two animals in the same cage, so the ordering of the result list is decided below the noise floor of the biology.
Pooling the cohort raises leave-one-out top-5 agreement from 0.13 to **0.78**, a six-fold gain.
That is the case for this feature, and it is a different case from the one that motivated it.

## 1. What a cohort is

### The default

A cohort is a set of OSDR samples that share a **study**, a **tissue**, and a **spaceflight arm**.
This is the ISA-Tab factor grouping OSDR already curates, so the tool is reading a grouping that exists rather than inventing one.
Measured on the shipped metadata: **215 cohorts across 70 studies**, median size 9, mean 9.8, max 38, and 2,095 of the 2,108 embedded samples live in one.

The arm is the **raw** OSDR value, not the binary Flight-vs-Ground collapse.
`manifold/data._flight_status` already records why: a basal animal was sacrificed at experiment start and a vivarium animal never entered flight hardware, so the seven control arms are not interchangeable.
Pooling a Vivarium Control with a Basal Control would average two different experiments and call the result one group.

### Why study is pinned and cannot be unticked

Random samples drawn from the same *study* already reach mean pairwise cosine 0.9805, against 0.9933 for a real cohort and 0.8826 for random OSDR samples.
Same-study membership therefore supplies **84%** of what makes a cohort coherent, and most of that is batch rather than biology.
Pooling *across* studies would average across the strongest batch boundary in the corpus, so study is a fixed facet.
The UI shows it as a pinned chip that cannot be removed, with the reason on hover, rather than hiding it.

### What the user controls

Nine facets are available, drawn from `cache/osdr_metadata.parquet`, which is exactly the table of the 2,108 samples that have a cached embedding.

| facet | default | note |
| --- | --- | --- |
| Study | **pinned on** | cannot be removed, for the reason above |
| Tissue | on | |
| Spaceflight arm | on | raw arm, seven values |
| Sex | off | |
| Strain | off | |
| Genotype | off | |
| Habitat | off | |
| Mission duration | off | |
| Diet | off | |

Adding a facet splits cohorts and makes them smaller and more homogeneous.
Removing one merges them and makes them larger and more heterogeneous.
Both directions are legitimate questions and both are one click, so the cohort count and the size distribution update live under the chips.

A tenth control sits below the cohort picker: the **member list**, where any individual sample can be excluded from the pool.
Nothing is ever auto-excluded.

## 2. The estimator

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

Three statistics fall out of the same `u` and all three are shown:

- **`R̄ = |u.mean(axis=0)|`**, the vMF resultant length, in [0, 1]. How tightly the cohort agrees on a direction.
- **Each member's cosine to the leave-one-out centroid.** This is the correct outlier statistic. Using the full centroid instead lets an outlier pull the reference towards itself and hide inside it.
- **Expected top-5 stability given k**, read off the measured curve in `precompute/validate_cohorts.py`. This is the honest confidence number, and it is a property of the cohort's size rather than of its tightness.

The medoid was measured and rejected: it agrees with the centroid on only 0.46 of the top-5, and being one sample it inherits exactly the single-sample instability the feature exists to remove.

## 3. Low N, and what the interface says about it

A cohort of two is still a cohort, and it is still better than one sample.
It is not as good as a cohort of nine, and the interface has to say so without either hiding the result or crying wolf.

Three states, and the threshold comes from the measured stability-versus-k curve rather than from taste:

| k | state | treatment |
| --- | --- | --- |
| 1 | not a cohort | disabled in the picker, reason shown: pooling needs at least two samples |
| 2 to 4 | low confidence | selectable and searchable, with an amber flag naming the measured stability at that k |
| 5 and up | normal | the stability figure is still shown, without the flag |

Amber rather than red, for the same reason the map's coverage bar is amber: a small cohort is working correctly, not failing.
The flag names the number rather than a word, because "low confidence" alone tells a researcher nothing they can act on while "3 samples, measured top-5 stability 0.51, against 0.86 at 15 or more" tells them how far down the list to stop reading.

### The curve, and why it is bucketed

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
It is one constant in `bridge_rna/cohorts.py` carrying the measurement in its docstring, so it cannot drift from the evidence.

Two things about this table were corrections rather than choices, and both are worth keeping.

**The first sweep quoted per-size figures and they were noise.** Sampling two cohorts per size produced 0.38 at k=5 sitting beside 0.90 at k=6, which is a fact about which two cohorts were drawn and not about size. A number quoted in the interface has to stand on enough cohorts that it does not swing by half its range when the seed changes, so sizes are pooled into buckets and each bucket reports its count and its spread.

**Adjacent buckets inverted, and merging them is the honest repair.** Even over all 212 cohorts, 5-6 scored 0.736 and 7-9 scored 0.696, an inversion of 0.04 against a within-bucket sd of 0.18. A larger cohort must never be reported as less trustworthy than a smaller one, so `validate_cohorts.py` merges adjacent buckets that invert before printing the curve, which is what produced the 5-9 row. Clamping the number instead would have invented monotonicity; shipping the raw pair would have told a researcher their seven-animal cohort was worse than a five-animal one. A test pins the result monotone.

## 4. Two arms, run as two queries

An optional **compare against** picker runs a sibling cohort as a second, independent pooled query, and draws both on one network.
The number it produces is the **overlap between the two hit sets**, which answers a real question: do this study's flight animals and its ground controls land in the same part of Earth's transcriptome space, or different parts?

What it deliberately is **not** is the difference vector `centroid(flight) - centroid(ground)`.
That is the standard differential-expression move and it does not belong here for two reasons.
A difference of two unit vectors is not a transcriptome, so cosine-ranking ARCHS4 against it asks which GEO sample's *absolute* profile most resembles a *change*, which is a category error against an index that holds profiles.
And the corpus-level version was already built, measured and rejected: the flight-minus-ground axis correlated r = -0.990 with PC1, which is the transcriptome-concentration axis, and one in ten random flight/ground relabelings beat it on spatial structure.

## 5. Where the code goes

### A fifth query-vector source, not a new pipeline

The cosine scan, `_annotate_from_cache`, and the `archs4_index` map join are the cached path's, reused unchanged, exactly as file ingestion reuses them.
So a cohort hit carries the same schema as a single-sample hit, and everything downstream of the hits frame keeps working without knowing a cohort produced it.

```python
def run_cohort_retrieval(sample_ids, topk):
    rows  = np.stack([cached_query_vector(s) for s in sample_ids])
    q_vec = cohort_query_vector(rows)
    idx, score = _topk_cosine_from_memmap(index_vecs=..., q_vec=q_vec, k=topk)
    hits = _annotate_from_cache(idx, score)
    hits["archs4_index"] = idx.astype(int)
    return hits
```

No model, no subprocess, no torch, no new artifact, and one memmap scan, so the same ~0.5 s as the cached path.

| file | change |
| --- | --- |
| `bridge_rna/cohorts.py` | **new.** Facet registry, cohort construction, geometry, low-N tiering. The only file that knows what a cohort is. |
| `bridge_rna/retrieval.py` | `run_cohort_retrieval`, mode `"cohort"` |
| `bridge_rna/callbacks.py` | a `"cohort"` entry in `_retrieval_phrase`, the mode switch, the cohort callbacks, a synthesized cohort query row |
| `bridge_rna/layout.py` | the segmented Sample / Cohort / Upload switch and the cohort panel |
| `bridge_rna/panels.py` | the cohort inspector: membership, `R̄`, per-member LOO cosine, the overlap readout |
| `bridge_rna/figures.py` | a pooled query node, and a two-query network for the compare case |
| `manifold/callbacks.py` | `_retrieval_overlay` draws every pooled member on the map, not one |
| `precompute/validate_cohorts.py` | **new.** The honesty gate. |

`bridge_rna/cohorts.py` is a separate module rather than more of `retrieval.py` because it depends on no embedding and no memmap at all.
It is pure metadata grouping plus 512-d arithmetic, it can be tested against the fixture corpus without either artifact, and keeping it apart is what stops `retrieval.py` from growing a second responsibility.

### The status banner

`_retrieval_phrase("cohort")` must name the path, as it must for every mode.
The invariant this repo already broke once, when every cached result was announced as demo-script output, is that the interface always says which path answered.
A pooled result must never be labelled with one member's name; the query node carries the cohort's name and its size.

## 6. How we know it works

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
The bucketed curve in section 3, which sets `LOW_N_THRESHOLD` and populates the confidence readout.

**6. Normalization.**
Over all 212 cohorts, `cos(spherical mean, raw mean)` has median **0.9999995** and worst case **0.99951**, and the two agree on the exact top-5 in 112 of 212 cohorts.
Exactly as predicted: within one cohort the L2-norm spread is about 1.09x, so the weighting has almost nothing to bite on.
Spherical is kept because it becomes the correct estimator the moment anyone unticks Tissue and the corpus-wide 3.9x spread is real.

## 7. Interface

The left rail gets a segmented control at the top: **Sample / Cohort / Upload**.
Only one query source is visible at a time, so the rail gets *shorter* than the current stacked "Query sample" plus "Or upload a sample" arrangement rather than longer.
The canvas, the inspector, the AI hypothesis, the top-k slider and the map hand-off are all shared and unchanged, because a cohort is a query like any other.

Cohort mode, top to bottom:

1. **OSDR study** dropdown, the same one Sample mode uses.
2. **Group by**, a row of facet chips. Study is pinned. A line under it reports how many cohorts the current definition produces and their size range.
3. **Cohort** dropdown, listing this study's cohorts with size and confidence state. Singletons are disabled with the reason, matching how the sample picker treats an unretrievable sample.
4. **The cohort card**: k pooled, `R̄`, the measured stability at this k, and the amber low-N flag when it applies.
5. **Members**, a collapsed disclosure listing every sample with its leave-one-out cosine, each with a checkbox. Any member flagged as an outlier is marked, never removed.
6. **Compare against**, an optional sibling-cohort picker, empty by default.
7. **Search cohort**.

The rail's existing rule holds: the fact that qualifies a control sits directly under that control.
The cohort-count line hangs under the facet chips, and the confidence card hangs under the cohort picker.

## 8. Testing

Three layers, each answering a question the others cannot.

**`tests/test_cohorts.py`, 34 tests, against the synthetic fixture corpus.**
Facet grouping, the estimator, `R̄`, leave-one-out cosines, low-N tiering, the pinned-study rule, and the sibling relation.
Two are worth calling out because they pin claims made in prose everywhere else.
`test_pooled_ranking_is_the_mean_of_the_members_own_cosines` checks the central algebraic claim directly: ranking by cosine to the spherical mean is identical to ranking by the unweighted average of the members' own cosines, which is what "ask every animal, then average the votes" means.
`test_one_animal_one_vote_regardless_of_transcriptome_concentration` scales one member's norm by 10 and asserts the pooled direction is unchanged, and then asserts that the *raw* mean would have been dragged, so the test cannot pass vacuously.

**`precompute/validate_cohorts.py`, 6 checks, against the real corpus.**
Section 6. This is the only layer that can speak to whether pooling works, as opposed to whether it computes what it says.

**`tests/e2e_cohort_check.py`, 60 browser checks, against the real app and the real cache.**
Define a cohort, retick facets, watch the count change, read the confidence card, open the member list, pool and search, open the inspector, exclude a member and watch every number restate, compare two arms, and follow the whole cohort to the map.
It asserts on what the page reports about itself, and two of its checks exist because this feature shipped those exact regressions and had them fixed: callbacks firing at page load so the canvas greeted a visitor with "Cohort retrieval failed", and the legend continuing to advertise a GSE column while a comparison that draws none was on screen.

**One trap, recorded because it cost a debugging cycle.**
The obvious wait predicates for a search - "the network has nodes", "the spinner is idle" - are both already true when a *second* search starts, so waiting on them returns instantly and the check then reads the previous result's banner.
The comparison step appeared to fail while the app was doing exactly the right thing.
`run_cohort_search` now waits for the status banner to change, and uses the spinner only as a secondary settle.
