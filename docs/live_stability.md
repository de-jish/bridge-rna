# Result stability, measured on the query that just ran

**Status: built, measured on the real corpus, and tested, 2026-08-06.**
This document replaces the confidence readout described in section 3 of `docs/cohort_retrieval.md`.
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

## 1. Why the precomputed number had to go

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

## 2. Why it moved to the right, and why after the search

The rail's standing rule is that the fact qualifying a control sits directly under that control.
A precomputed number obeyed that rule, because it was a function of the selection and nothing else: pick a cohort, read its size, look up the curve.

A measured number cannot obey it, because it does not exist until the query runs.
It is a property of the result, not of the selection, so it belongs with the result.
Putting a live measurement under the picker would mean either running the scan on every selection change, which turns a dropdown into a 1 s query, or showing a stale number from the previous cohort, which is worse than the curve was.

So the cohort card on the rail now states only what is true before the search: the role, the cohort's name, and how many samples are about to be pooled.
That is what the user asked for, and it is also the only honest thing the rail can say at that moment.

## 3. What is measured

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

## 4. Cost, and why it is affordable

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

## 5. What was deleted

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

## 6. Where the code goes

`bridge_rna/cohorts.py` keeps its promise to open no embedding and no memmap.
It gains the pure arithmetic only: `top_k_agreement`, `leave_one_out_vectors`, and `StabilityMeasurement`, which is built from rankings the caller supplies.
Every one of them is testable against the fixture corpus on a machine with neither artifact, which is what the module exists for.

`bridge_rna/retrieval.py` owns the scan and the wiring.
`run_cohort_retrieval` now returns `(hits, rows, stability)`, builds every query vector it needs before touching the memmap, and scores them in one pass.

`bridge_rna/panels.py` gains `build_stability_panel`, which renders one measurement or two.
The two-cohort case reuses the role dot and the cohort's own name, because `docs/map_key.md` established that cohort B's hex cannot agree across the views and the binding is therefore the name.

## 7. Making two measurements fit on screen at once

The first build of the panel labelled every block "RESULT STABILITY" and spelled out the full definition under each number, then added a three-line amber caution when the number was low.
On a single cohort that reads fine.
On a comparison it does not: measured in the browser, the two blocks wanted **644 px** inside a panel that had **389 px**, so cohort B's entire measurement sat below the fold.
A second measurement nobody can see is a second measurement nobody made, which is the whole thing this feature exists to prevent.

Three changes fixed it, and each one is a rule rather than a nudge.

**What is shared is said once.** The panel heading names the statistic and the subtitle states what was measured and at what depth, so a block carries only what differs between the two arms: the name, the number, the meter, the size, the baseline, and the member that moves it most. The per-block label was pure repetition and the definition was identical in both.

**The caution is one line.** It was a title over a three-line body, and the body said the same thing twice on a comparison. It now reads "Under 70%: read these as a neighbourhood, not a ranking", with the threshold interpolated from `STABILITY_FLOOR` so the sentence cannot drift from the rule that fires it.

**The details panel yields first.** `.details-panel` has `flex-shrink: 20` against the stability panel's 1. With equal shrink factors the overflow was split between them and the stability panel lost 165 px it needed; the details panel is the right one to give way, because it scrolls a reference list where the panel above it carries two numbers that only mean anything side by side. It stops at its 120 px floor, after which the stability panel shrinks and scrolls in turn, so a short viewport degrades instead of clipping the AI panel off the bottom of the column.

`tests/e2e_cohort_check.py` measures this rather than trusting it: after a two-arm search it reads both bounding boxes and asserts the second block ends inside the panel.
That check is what caught the overflow, and it caught it twice more while the fix was being tuned.

## 8. Alternatives considered and rejected

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

## 9. How it is tested

Against the fixture corpus, in pytest, with no real artifact:

- The fused `m`-query scan reproduces a naive per-query scan row for row, in index order and in score.
- A cohort whose members are identical vectors measures stability exactly 1.0, because every leave-one-out pool is the same vector.
- The statistic's bounds, its symmetry, and its behaviour on disjoint and identical lists.
- `run_cohort_retrieval` returns a measurement whose size, depth, and per-member length match the cohort and the slider.
- The rail card renders no stability, no meter, and no flag; the panel renders the measured value.
- The deleted names stay deleted.

Against the real corpus, in the browser:

- `tests/e2e_cohort_check.py` now gains a `--loops` flag and asserts that the panel is absent before a search, present after it with a number, carries both arms of a comparison, and reports a stability that is not the value the deleted curve would have produced for that size.
