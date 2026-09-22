# Ranked similarity profile

A fourth surface on the retrieval result: how cosine similarity decays with match rank.

The network answers "what are the top five" and the map answers "where do they sit".
Neither answers **"how far down is this still worth reading"**, because both draw a fixed slice and neither shows the decay.

Code: [`bridge_rna/profile.py`](../bridge_rna/profile.py) (the shape statistics, no Dash and no artifacts), `build_similarity_profile_figure` in [`bridge_rna/figures.py`](../bridge_rna/figures.py), `profile_panel` in [`bridge_rna/layout.py`](../bridge_rna/layout.py), and two callbacks in [`bridge_rna/callbacks.py`](../bridge_rna/callbacks.py).

## What the corpus actually looks like

Every design decision below was overturned or set by measuring the real thing first, and the measurements are the most valuable part of this document.

### There are no breakpoints in this corpus

The feature was specified around finding "a clear drop-off, elbow, or transition point".
**It is not there.**

120 random cached OSDR queries were scanned to top-1,000 and scored at top-100, top-250 and top-1,000:

| range | median prominence | max prominence | median step / span | breakpoints found |
| --- | --- | --- | --- | --- |
| top 100 | 1.32 | 2.39 | 0.060 | **0 / 120** |
| top 250 | 1.38 | 3.08 | 0.041 | **0 / 120** |
| top 1,000 | 1.41 | 3.08 | 0.022 | **0 / 120** |

The estimator is not blind. Against synthetic profiles built at the corpus's own scale:

| synthetic profile | prominence | step / span | reported |
| --- | --- | --- | --- |
| step of 30% of span at rank 60 | 7.69 | 0.19 | yes |
| step of 15% of span at rank 60 | 4.29 | 0.12 | yes |
| step of 5% of span at rank 60 | 1.98 | 0.06 | no |
| no step at all | 0.80 | 0.04 | no |

The real profiles sit with the no-step control, nowhere near a 15% step.
So the panel **does not draw a breakpoint marker**, and says outright that it looked and found nothing, with the number and with the corpus measurement that makes "nothing" the expected answer rather than a failed measurement.

Keeping the estimator is what makes that claim credible, and an uploaded sample or a future corpus could still clear the floors, in which case the marker draws.

### The estimator had to be fixed before it could be trusted

The first version took the largest windowed drop and compared it to the median drop in a surrounding window.
On real data it returned **rank 5 for 36 of 40 samples** - the first rank the window can evaluate.
A decay is steepest where it starts, so it was finding the leading edge and calling a slope a step.

The fix is two rules, both in `largest_local_step`:

1. **A candidate needs a full neighbourhood on both sides**, so the leading edge can never be reported.
2. **Prominence is measured against the busier side**, `delta / max(median(left), median(right))`, because a step is only a step if it stands out from what is on both sides of it.

With those, the leading edge is structurally unreachable and `test_the_leading_edge_can_never_be_reported_as_a_step` pins it.

### The similarity axis cannot be fixed

Span over the top 250 across those 120 queries:

| p5 | median | p95 | max |
| --- | --- | --- | --- |
| 0.00027 | 0.00194 | 0.01503 | 0.02867 |

A **55x spread**, and the whole corpus sits between about 0.98 and 0.9997.
A fixed window - the obvious way to keep samples comparable - would draw almost every sample as a flat line against the top of the panel.

So the axis is **fitted to the displayed slice**, and the span is published beside the curve as one of the three metrics.
A fitted axis magnifies; the span is the number that stops a magnified nothing from reading as a real separation.
The tick step is derived from the range rather than left to Plotly's default, which lands on 0.01 for a typical 0.024 span and gives two gridlines on the whole plot.

### What replaced the breakpoint

Two measured statistics that do discriminate:

- **Span over the displayed range**, 55x across the corpus. How tight or diffuse this query's neighbourhood is.
- **Rank at half that fall**, median 44 over 120 queries, spread 10 to 136. How quickly the curve gives up half its total decay.

Together they answer "are there 20 close analogs here or 200" without asserting a cutoff, which is the question the feature exists for.

## How it is wired

**One scan, three prefixes.** `search_hits_with_profile` scans to `max(topk, neighborhood_depth, profile_depth)` and slices the result three ways: the requested hits, the 250-deep evidence neighbourhood, and the 1,000-deep profile.
Going from 250 to 1,000 costs a larger top-k heap over the same 940,455-row pass, so the profile needs **no new artifact, no new index and no second memmap read**.
`search_hits_with_neighborhood` now delegates to it with `profile_depth=neighborhood_depth`, so nothing that does not draw a profile pays for one, and no existing caller changed.

**The profile travels as parallel lists**, not records. A thousand full hit records would be several hundred kilobytes of session store shipped on every search, almost all of it fields the profile never reads. It rides `hits-store` on the shell, so it survives the walk to the map and back exactly as the hits do.

**Only the cached path serves a full profile.** A cohort or an uploaded sample rides its own 250-deep pooled prefix rather than a 1,000-deep scan, so the range control disables what that result cannot honour. The demo and precomputed paths carry no profile at all, and the panel says why instead of drawing an empty chart.

**Placement.** Under the network in the centre column, because it describes the whole result rather than one selected hit, and the inspector's job is the one selected hit.

## Things that broke on the way, and what they taught

**The centre column clipped the panel.** `.workspace` was `overflow: hidden` with a single full-height canvas in it, which was right while there was one panel. With two, the profile's heading, its sentence and its metric values were cut off the top of the column and the panel read as though it began at its own controls. The column scrolls now, the way the inspector already does, and `.panel--canvas` gained a real `min-height` because a scrolling parent lets a `min-height: 0` flex child collapse to nothing.

**The range control came up on Top 100 and never moved.** The options callback ran against an empty store on page load and clamped the value to the only range a zero-depth profile could offer. Because 100 stays valid once a result arrives, nothing ever moved it back, so the declared default was silently never the default. It returns `no_update` for both props until there is a result.

**A stale server process cost half an hour.** Two `app.py` processes were bound to :8000 and the old one answered. Every fix looked like it had no effect. `pkill -f app.py` matched only one of them. Kill by pattern and then *verify the process count* before concluding a change did not work.

**Plotly hover cannot be driven from headless Playwright**, and this is not specific to this panel: the shipped retrieval network does not produce a tooltip under synthesised mouse events either. Hover here is verified by configuration and content rather than by rendering - `hovertemplate` present, 250 `hovertext` entries, correct values - and confirmed in a real browser.

## Tests

`tests/test_profile.py` is 20 tests on synthetic arrays built at the corpus's own scale, because a fixture at 0.9 to 0.5 would not exercise anything this module meets.
They pin the leading-edge rule, the two-sided prominence, the relative floor, the span and half-span statistics, and that the smooth headline never uses the words "cutoff", "threshold" or "recommended".

`tests/test_app.py` pins that the callback payload carries the profile as parallel lists rather than records, and that the demo path carries a reason instead.

Full suite: **528 passing**.

## Open questions

1. **Should the profile carry tissue composition?** The tissue mix changes across the ranked list and a thin band under the axis would show it, at the cost of a fourth encoding on one chart.
2. **Does a cohort comparison get one profile or two?** Two pooled query vectors means two curves, which reopens the axis and the metrics together.
3. **Is top-1,000 the right depth?** It is close to free on the scan the cached path already runs, but 1,000 is a choice and not yet a measurement.
4. **Should the panel be a tab on the canvas instead of a second panel?** It would keep the network at full height on a short viewport. The scrolling column is the cheaper answer and is what shipped.
