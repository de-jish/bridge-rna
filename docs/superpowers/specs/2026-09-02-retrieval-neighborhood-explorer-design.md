# Retrieval Neighborhood Explorer

**Date:** 2026-09-02
**Status:** Approved for implementation
**Applies to:** Bridge RNA map and retrieval data flow

## Purpose

When a user brings an OSDR retrieval onto the map, Bridge RNA currently marks
the query and its requested top hits and can frame those points. That answers
“where did my retrieval land?” but gives the user no structured way to
understand the biological and study context around the result.

The Retrieval Neighborhood Explorer adds a wider evidence set and a map-linked
drawer that answers:

- What tissues, species, and studies characterize this query's neighborhood?
- Which GEO studies contribute the most samples?
- Which exact GSM samples make up the neighborhood?
- Where do those samples appear on the projection?

The feature must preserve a strict distinction between similarity in the
model's original 512-dimensional embedding and proximity on a 2-D or 3-D
projection.

## User experience

### Entry points

The existing **Frame retrieval** action remains. In 2-D it keeps its current
viewport behavior and also opens the Neighborhood Explorer when neighborhood
data is available. A new **Explore neighborhood** action opens or reopens the
drawer without changing the viewport. It remains available in 3-D, where a
2-D axis frame cannot control the camera.

The drawer closes independently of the retrieval. Closing it removes the wider
neighborhood emphasis but leaves the query and requested hits on the map. A
new retrieval resets the drawer to its closed, Overview state so stale context
cannot be mistaken for the new result.

### Map marks

The three semantic layers are visually distinct:

1. The OSDR query retains its teal star.
2. The user's requested 3–30 retrieval hits retain their existing white open
   rings and existing retrieval rank behavior.
3. The wider 512-dimensional neighborhood appears only while the drawer is
   open, as 250 subtle teal open marks behind the requested hits.

Every wider-neighborhood mark has the same size, opacity, and color. No line,
hull, enclosing ring, distance gradient, or rank-size ramp is drawn. Such
encoding would imply geometric structure that the projection does not
preserve. Hovering a neighborhood mark shows its exact cosine rank and score,
GSM, GSE, tissue, and species. Requested hits stay visually dominant when a
sample belongs to both layers.

### Adaptive drawer

On wide screens, the map canvas narrows and a roughly 390 px drawer docks on
the right. Plotly's responsive resize preserves the map within the remaining
space. On narrow screens, the same drawer becomes a bottom sheet. Closing it
returns the full canvas.

The drawer header names the query, says that the evidence consists of nearest
neighbors in exact cosine space, shows the actual depth returned, and states
the requested-hit count. A comparison retrieval adds an arm selector; it does
not merge the two evidence sets into one synthetic neighborhood.

The drawer has three tabs.

#### Overview

Overview provides a deterministic, auditable answer rather than automatically
generated prose:

- a one- or two-sentence count summary with explicit numerators and
  denominators;
- tissue composition, with the top categories and an explicit remainder;
- species composition;
- number of represented GEO studies;
- concentration of the three largest studies;
- median, minimum, and maximum cosine score across the evidence set.

All percentages use the number of records for which the relevant metadata is
present, and the coverage denominator is printed beside them. Missing metadata
is not silently folded into “Other.” Example counts shown in the approved
mockup are illustrative; runtime values always come from the active result.

#### Studies

Studies groups every neighborhood sample by GSE. Rows are ordered by sample
count, then best exact rank, and show:

- GSE accession or an explicit “No GSE recorded” group;
- study title when available;
- dominant recorded tissue and its coverage;
- number of neighborhood samples;
- best exact cosine rank;
- a direct GEO link for real GSE accessions.

Selecting a study emphasizes all of its neighborhood samples on the map. The
selection is reversible and does not change the retrieval or evidence set.

#### Samples

Samples provides a searchable list of every record in the evidence set. Each
row shows exact rank, cosine score, GSM, GSE, tissue, and species. Selecting a
row emphasizes its map mark and reveals its locally cached metadata plus a GEO
link. The list is scrollable and remains usable by keyboard; 250 rows do not
require server-side pagination or a new grid dependency.

### Language

The interface calls the top 250 the **evidence neighborhood** or **512-D
neighborhood**, never the “visible neighborhood.” It calls the current top-k
the **requested hits**. A standing footer states that the evidence set is
nearest in 512-D and is not everything inside the visible map frame.

## Evidence definition

`NEIGHBORHOOD_DEPTH` is fixed at 250 in version one. The user continues to
choose the requested retrieval depth of 3–30 independently.

The same cosine scan must produce both sets:

- retain the best `max(requested_k, 250)` ARCHS4 rows;
- expose the first `requested_k` as the existing retrieval hits;
- expose the first 250 (or the corpus size, if smaller) as the evidence
  neighborhood.

This must not trigger a second pass over the 963 MB ARCHS4 memmap. The exact
hits are therefore always a prefix of the evidence neighborhood for the same
query.

For cohort retrieval, the pooled query's top 250 form the evidence set. The
leave-one-out and member-query lists used for stability remain interpreted at
the user's requested depth; expanding the pooled result must not silently
change the stability measurement. For a comparison, each arm keeps its own
top-250 evidence set and the drawer selects one arm at a time.

Cached OSDR, pooled cohort, and uploaded-query paths support the explorer. If a
legacy, demo, precomputed, or stale session result cannot supply locatable
ARCHS4 indices and a top-250 set, the map keeps its existing retrieval behavior
and the explorer gives a clear “run this retrieval again” or unavailable state.
It never fabricates a neighborhood from projected coordinates.

## Data shape and boundaries

The existing `hits-store` remains the cross-route source of truth. Existing
keys and the existing `hits` list are unchanged. A new JSON-safe
`neighborhood` object holds arm A's evidence rows; comparison data holds the
same shape under `comparison.neighborhood_b`.

Each neighborhood object contains:

- `depth_requested`: 250;
- `depth_returned`: actual number of rows;
- `metric`: `cosine`;
- `space`: `embedding-512d`;
- `hits`: ranked metadata records with `archs4_index`;
- optional query or arm label;
- an unavailable reason when the evidence set could not be produced.

Aggregation and presentation logic belongs in a focused
`bridge_rna/neighborhoods.py` module. It accepts ranked records and returns
plain, testable summary, study-group, and sample-row structures. It does not
read Dash state or Plotly coordinates. The map layout renders those structures;
the renderer only turns evidence indices and focus state into Plotly traces.

The retrieval layer should expose a result that lets callers obtain requested
hits and the wider ranked prefix from one scan while preserving existing public
entry points used by scripts and tests. No caller should need to infer the
requested set by re-scoring or by reading the map.

## Interaction state

Map-local stores own:

- whether the explorer is open;
- the active comparison arm;
- the active tab;
- sample-list search text;
- the selected study or sample focus.

Each piece of state has one callback writer. Opening through **Frame
retrieval** and opening through **Explore neighborhood** feed the same owner.
Focus changes only visual emphasis; it never mutates `hits-store`, the
viewport, or the underlying evidence rows.

## Empty, stale, and partial states

- No retrieval: both neighborhood actions and the drawer are hidden.
- Retrieval without a neighborhood payload: requested hits still work; the
  explorer asks the user to rerun the retrieval when that can recover it.
- Neighborhood rows without GEO metadata: ranks, scores, and accessions remain
  usable; coverage reads zero and no biological composition claim is made.
- Fewer than 250 available rows: the actual returned depth appears everywhere.
- Missing or out-of-range map indices: those rows remain in the evidence list,
  are counted honestly, and are omitted from map traces with a visible
  locatability count.
- No studies after grouping: Studies shows a meaningful empty state rather
  than a blank panel.
- Comparison: switching arms replaces the drawer evidence and teal marks; it
  does not hide or rewrite the existing requested-hit comparison overlay.

## Accessibility and responsive behavior

- Explorer tabs, close control, study rows, sample rows, links, and search are
  keyboard accessible.
- Opening the drawer announces its heading; closing it returns focus to the
  control that opened it.
- Color is never the only distinction: the key names every mark, and query,
  requested hits, wider neighbors, and focused rows use distinct shapes or
  outlines.
- Text and controls meet the project's existing WCAG AA token choices.
- At desktop widths the drawer is docked. At tablet and phone widths it becomes
  a bounded bottom sheet whose content scrolls without clipping the map or the
  page. The design is checked at 320, 768, 1024, and 1440 px.

## Performance constraints

- One retrieval performs one ARCHS4 memmap scan per query batch, as today.
- The stored neighborhood is capped at 250 rows per arm.
- Only the 250-point evidence trace carries hover metadata; the million-point
  ARCHS4 cloud remains free of per-point hover payload.
- Drawer aggregation is linear in at most 250 rows and may be recomputed from
  session data without disk or network access.
- Opening, closing, filtering, and focusing must not rerun the retrieval.

## Verification

Unit and integration tests must establish:

- requested hits are the exact ranked prefix of the 250-sample evidence set;
- cached, upload, cohort, and comparison paths populate the correct evidence
  shape without a second memmap scan;
- cohort stability still uses the user's requested depth;
- summary percentages and coverage denominators are correct with missing data;
- study grouping, ordering, unassigned rows, and sample filtering are stable;
- evidence traces have uniform styling, carry correct hover metadata, ignore
  invalid indices, draw behind exact hits, and never emit lines or hulls;
- each Dash output has one writer and stale payloads degrade safely;
- frame/open/close, tabs, arm switching, search, focus, 3-D availability, and
  responsive layouts work in a real browser;
- there are no browser console errors or accessibility warnings introduced by
  the feature.

The full existing pytest and browser suites remain green. Documentation and
screenshots are updated only where the shipped interface has materially
changed.

## Version-one exclusions

- statistics over the visible viewport or a lasso selection;
- automatic AI-authored summaries;
- user-configurable neighborhood depth;
- export/download;
- density contours, cluster hulls, or projection-distance thresholds;
- a new dedicated analysis route;
- changes to the model, embeddings, or offline projection build.

These exclusions keep the first release focused and prevent the map from
making claims its geometry cannot support.
