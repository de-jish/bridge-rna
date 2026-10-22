# Retrieval results: ranked table, filters, plots and export

Status: built on `feature/retrieval-table`, branched from `main` on 2026-10-21, and run against the real corpus in the browser.
This is the living implementation doc for the revamp. It records what was decided, why, and what was actually built.

## Objective

The retrieval view drew its result as a three-column network (query, GSM hits, GSE studies).
Rank was only implied by vertical order, the score was not drawn at all, tissue, species and title lived only in hover, and above 12 hits every GSM label was dropped.
At k=30 the network was thirty identical blue dots with crossing study edges.

The revamp replaces that canvas with a **ranked, filterable, exportable table of matches**, an inspector that opens whichever row is selected, and a **switchable plot** whose every view redraws from the rows the table currently shows.

Who: researchers at the Space Biosciences Research Branch choosing which GEO experiments resemble a spaceflight sample or cohort.

Confirmed with Joshua on 2026-10-21 through an interview.
The decisions below are his answers, not defaults.

| Decision | Answer |
| --- | --- |
| What filters act on | The rows actually retrieved (option "a"), not a deeper hidden pool. |
| Number of matches | Still starts at 3, now reaches **1,000**. |
| Modes covered | Sample, Upload and Cohort, including the cohort A/B comparison. |
| Compare tab (PR #2) and whitening (PR #3) | Out of the final app. This work branches from `main`; both PRs stay open and untouched. |
| What happens to a shortlist | It leaves the app as a CSV of exactly the filtered rows, with provenance. |
| Saved shortlists, a mini map, enrichment claims | Out of scope. |

## Success criteria

1. A user can retrieve 1,000 matches, filter to one GEO series, click a row and read its metadata in the right-hand card.
2. Every row shows its **true rank among all 940,455 ARCHS4 samples**, whatever filters or sort are active.
3. The default plot shows whether the top ranks are separated from the rest (the score-by-rank curve to depth 1,000).
4. Downloading produces exactly the rows on screen (all pages), in table order, with the query, mode, k and filters recorded in the file.
5. Retrieval, scoring, pooling, preprocessing, embeddings and projections are unchanged. A test pins that the top-k returned for a query is identical before and after.
6. All existing pytest and browser suites pass, updated where they asserted the old network as the result surface.

## Measurements that shaped the design

All on the real corpus, 2026-10-21.

**Scores are flat at the top.**
For `OSD-100|Mmus_C57-6J_EYE_FLT_Rep1_M23`, rank 1 scores 0.9970, rank 30 scores 0.9913, rank 250 scores 0.9731 and rank 1,000 scores 0.9634.
A score column alone cannot say whether a top 5 is separated from rank 50.
That is why the default plot is the score-by-rank curve to depth 1,000, with the retrieved k marked on it.

**One expression profile often carries two GSM accessions.**
12 of that query's top 30 hits were 6 profiles counted twice: GSE88819 and GSE124745 hold the same six eye samples (`wt_eye_0`, `ko_eye_1`, ...).
Over the top 1,000 hits of eight random cached queries, the nearest-neighbour cosine distance between hits splits cleanly:

| 1 - cosine to nearest other hit | hits |
| --- | ---: |
| below 1e-7 | 278 |
| 1e-7 to 1e-6 | 312 |
| 1e-6 to 1e-5 | 20 |
| 1e-5 to 1e-4 | 2,749 |
| 1e-4 and above | 4,473 |

Over the same eight plus the OSD-100 query above, all 305 pairs below 1e-6 are **in different GEO series**; all 14 pairs in the 1e-6 to 1e-5 gap are biological replicates **within one series** ("FVB rep1" and "FVB rep2").
The tight pairs are re-deposits ("Control_1" and "Control_1 [re-analysis]"), the same animals across two submissions, and at least one case of one profile deposited under two tissue labels (`DO226 ... adipose` and `DO226 ... skeletal muscle`, which cannot both be right).
Their vectors are bit-identical or within float32 rounding, and their norms match to three decimals, so these are not degenerate empty profiles collapsing together.
Connected groups are pairs.

So the rule is **cosine distance below 1e-6 means one profile with two accessions**, named `IDENTICAL_PROFILE_DISTANCE`.
The UI says "identical profile also deposited as GSM... (GSE...)" and never guesses why.
A 0.9999 cosine threshold would have been badly wrong: one query had 13,810 hit pairs above it.

**Scanning to depth 1,000 costs the same as scanning to 250.**
The memmap pass dominates: 0.4 to 0.5 s warm for k=1,000.
Reading 1,000 hit vectors back out of the memmap for the identical-profile check took about 1 ms.

**Payload.**
At k=1,000 the hits payload is 385 KB of JSON per cohort, 112 KB of which is the free-text `characteristics` field.
Production runs two gunicorn workers (`deploy/apply.py`), so a per-process server cache would not be shared between them and is not an option.

## Design

### Data flow

```
search callbacks ──► hits-store (shell, session, source of truth: map, inspector, AI)
                         │
                         ▼  one derivation per search, and on every view mount
                    results-store (view, lean rows + score profiles + identical groups)
                         │
        ┌────────────────┼──────────────────────┬──────────────────────┐
        ▼                ▼                      ▼                      ▼
  filter options    table + pager           results plot         CSV export
  (counts per       (filter, sort, page)    (score curve,        (filtered rows,
   GSE/tissue/                              series, network)      provenance)
   species)
```

`bridge_rna/results.py` is the only place that knows how a row is filtered, sorted, merged or paged.
It is pure standard library (no pandas, no Dash, no memmap; a test pins that), so it is tested against plain row lists.
The table, the plots, the export and the AI summary all call the same `apply_filters`, so the four can never disagree about which rows are in view.

`hits-store` keeps its shape and its job: the map draws from it and the inspector reads full GEO fields from it.
`results-store` is derived from it rather than written by the search callbacks, for two reasons.
First, the router rebuilds the retrieval view when you return from the map, and a derivation that runs on mount repaints the table from the session's `hits-store`.
Before this, returning from the map showed an empty canvas beside a populated inspector.
Second, the search callbacks stay about retrieval.

The derivation drops the long free-text fields the table never shows, so at k=1,000 each filter, sort or page request uploads about a third of the full payload.

### Retrieval changes (presentation support only)

- `SCORE_PROFILE_DEPTH = 1000`. Every cached, cohort and upload scan runs to `max(k, 250, 1000)` and returns the top-1,000 scores as a plain float list alongside the hits.
  The hits are still the first k of that one ranked scan, which is exactly how the 250-deep evidence neighborhood is already taken.
- Exact score ties are ordered by reference row index, with a stable sort.
  Identical profiles score exactly equal, so without this their relative order depended on sort internals.
- `reference_vectors(indices)` reads normalized hit vectors from the memmap for the identical-profile check.

No preprocessing, embedding, pooling or ranking arithmetic changes.

### The table

Columns, single query: Rank, GSM, Score, Series (GSE), Title, Tissue category, Species.
Cohort A/B comparison: one merged table with Rank A, Score A, Rank B and Score B in place of Rank and Score.
A blank rank means "not in that cohort's top k".

- Rank is always the position among all 940,455 ARCHS4 samples. Filtering and merging never renumber.
- Score shows four decimals. The inspector and the CSV carry full precision.
- "Tissue category" names the derived keyword bucket, so it is not mistaken for the source annotation (AGENTS.md).
- Every column header is a sort button with `aria-sort`. Default order is rank ascending.
- 50 rows per page, with a pager reading "Rows 51-100 of 412".
- The GSM cell is a real button, so rows are keyboard-reachable. A click anywhere on the row selects it through `assets/results-table.js`.
- A row with an identical profile carries a quiet mark naming the other accession.

The table is native HTML rendered by a server callback.
Dash DataTable is deprecated and will be removed in Dash 5.0, and Dash AG Grid would be a new dependency the NASA server would have to install, which only Joshua can do there.

### Filters

They sit in the Matches panel, because they act on results, not on the query:

- **Series**: a multi-select listing each GSE with its row count, plus an "Only these" or "Hide these" switch. This one control covers both "filter for a study" and "exclude a study".
- **Tissue category** and **Species**: multi-selects with counts.
- **Minimum score**.
- **Search**: case-insensitive substring over GSM, GSE, title and source name.
- **Merge identical profiles**: off by default, so the duplication stays visible until the user chooses to collapse it. It sits beside Download rather than among the filters, because it changes how rows are shown, not which qualify.
  The best-ranked accession is kept, and the summary says how many rows merging hid.
- **Retrieved by** (comparison only): either, both, A only, B only.

The count line always reads "N of M matches", and "Clear filters" is offered only while a filter is active.
A new search resets the filters, because a series chosen for one query rarely exists in the next one's results.

### Plots

The plot sits below the table, not beside or split with it: the centre column scrolls, the table takes the column's height less 96 px so the plot switch stays in view as the cue to scroll, and each figure asks for the height it needs (the network grows with its node count, to about 860 px at 30 matches).
Splitting the column evenly was built first and lost both ways: at 1,000 px tall a 50-row page showed four rows.

A segmented control picks one of three views. Each draws only the rows the filters keep.

1. **Score by rank** (default). The top-1,000 score profile as a line on a log rank axis, with the retrieved k as points: filtered rows emphasized and filtered-out rows muted.
   A comparison draws both cohorts' curves in their established cohort hues.
2. **Series**. Rows per GSE among the filtered rows, for the top 20 series by count, ordered by best rank.
   A comparison splits each bar by retrieved-by.
3. **Network**. The existing figure, drawn from the first 30 filtered rows, because a network beyond that is unreadable.

Clicking any mark selects it: a point or network node selects that GSM, and a series bar opens the series in the inspector.
Every figure uses the existing `[kind, node_id]` customdata convention, so one selection callback serves all three.

A fourth view, **Similarity**, was built on 2026-10-21 and removed on 2026-10-22 at Joshua's request after he reviewed it.
It drew the first 50 filtered rows as a pairwise cosine heatmap ordered by average-linkage clustering, with a dendrogram above.
It did not earn its place: every match is already close to the query, so the matrix spans a narrow cosine band, and the clustering it showed was not something a reader acted on.
Identical profiles, the one thing it made visible at a glance, are still marked on their table rows.
Its average-linkage code went with it; `bridge_rna/similarity.py` now holds only the identical-profile grouping.

A mini map of the hits was proposed and rejected.
Its 2-D neighbourhood would overlap the true cosine top-20 by a median of 0 (CLAUDE.md), so it would show closeness that the ranking does not support.
"View results on map" already covers it.

### Inspector

It is unchanged in content.
To keep a row click cheap at k=1,000, a clientside callback picks the selected row (or a series' rows) and both query records out of `hits-store` in the browser.
The server inspector callback then receives only that context, not the 385 KB payload.

### Export

"Download CSV" writes the filtered rows in table order across all pages: rank(s), score(s) at full precision, GSM, GSE, title, source name, characteristics, tissue category, species and identical-profile accessions.
Leading `#` lines record the query, mode, k, filters, sort, the reference size and the export time.
`pandas.read_csv(path, comment="#")` reads it back.

### AI summary

It summarizes the first 20 filtered rows in table order and says so ("Summarizing the first 20 of 412 filtered matches").
GEO enrichment runs only on those 20.
Before this, the prompt silently kept the first 20 hits while enrichment fetched every hit, which at k=1,000 would have been a thousand network round trips.

## Contracts

These are fixed before the parallel work starts, so the pure modules and the integration can be built at the same time.

### `results-store`

```python
{
  "mode": "cached" | "precomputed" | "demo" | "uploaded" | "cohort",
  "k": int,                      # matches requested per query
  "comparison": bool,            # a cohort A/B comparison
  "label_a": str, "label_b": str,  # query or cohort names ("" for b when single)
  "query_id": str,               # hits-store["sample_id"], for export provenance
  "reference_count": int,        # 940,455 on the real corpus
  "profile_a": [float, ...],     # top-1,000 scores, best first (shorter for demo)
  "profile_b": [float, ...],     # [] when not a comparison
  "rows": [Row, ...],            # best rank first: min(rank_a, rank_b), then archs4_index
}

Row = {
  "gsm": str, "gse": str, "title": str, "source_name": str,
  "tissue": str, "species": str,          # "" when unknown
  "archs4_index": int | None,             # None on the demo path
  "rank_a": int | None, "score_a": float | None,  # None: not in A's top k
  "rank_b": int | None, "score_b": float | None,  # always None when single
  "identical": [str, ...],                # other GSMs in these rows with an identical profile
}
```

For a single query every row has `rank_a` 1..k contiguous.
For a comparison the rows are the union of both top-k lists, joined on GSM.

### Filters, sort and page

```python
Filters = {
  "series": [str],                 # GSE accessions; empty means no series filter
  "series_mode": "only" | "hide",
  "tissue": [str], "species": [str],  # empty means all
  "min_score": float | None,       # a row passes if its best present score >= this
  "text": str,                     # case-insensitive substring of gsm, gse, title, source_name
  "merge_identical": bool,
  "retrieved_by": "either" | "both" | "a" | "b",  # "a" means A and not B
}
Sort = {"column": "rank_a" | "score_a" | "rank_b" | "score_b" | "gsm" | "gse"
                   | "title" | "tissue" | "species",
        "descending": bool}
```

Filters apply first and merging second, so a twin survives when its better-ranked copy was filtered out.
Sorting never renumbers ranks, and a row with no value in the sort column goes last in either direction.
Ties in any sort column fall back to best rank.

### Figures

Every result figure carries `customdata` rows of `[kind, node_id]`, where `kind` is `"gsm"` or `"gse"`, matching the network.
The existing `select_node` callback reads only that, so it serves every plot.

## Out of scope

The Compare tab and whitening (PRs #2 and #3), a mini map, saved shortlists, and any claim that a tissue or series is over- or under-represented.
The map keeps drawing every retrieved hit; table filters do not reach it.

## Commands

```bash
/Users/josh/Bridge-RNA/.venv/bin/python -m pytest tests/ -q
/Users/josh/Bridge-RNA/.venv/bin/python tests/e2e_check.py
/Users/josh/Bridge-RNA/.venv/bin/python tests/e2e_cohort_check.py
/Users/josh/Bridge-RNA/.venv/bin/python tests/e2e_upload_check.py
/Users/josh/Bridge-RNA/.venv/bin/python app.py   # http://127.0.0.1:8000
```

## Plan

Each step ends green on pytest and is committed on its own.

1. Retrieval: score profile, stable tie order, `reference_vectors`, identical-profile groups. Pin that the top-k is unchanged.
2. `bridge_rna/results.py`: row derivation, filters, merge, sort, page and CSV, with unit tests.
3. `bridge_rna/similarity.py`: average-linkage ordering and dendrogram geometry, with unit tests (removed with the Similarity view on 2026-10-22).
4. Layout: the k slider (log scale plus a number box), the Matches panel, filters, table, pager, plot panel and download.
5. Callbacks: the derivation, filter options, table, plot, selection, clientside inspector context, export and the AI cap. Remove the network as the result canvas.
6. Figures: score curve, similarity heatmap with dendrogram (removed 2026-10-22), series bars. The network is reused.
7. Styles in `assets/retrieve.css` on existing tokens, at the 1180 px and 680 px breakpoints.
8. Tissue vocabulary: "eyecup" maps to Eye. This is its own commit.
9. Browser suites updated and extended. Docs: CLAUDE.md, README and this file.

## What changed while building

- **Number of matches.** The exact box has no `min` or `max`: Dash 4 reports an out-of-range number as `None`, so typing 5000 made the next search silently run with the default 5. The callback clamps and writes back 1,000, and a search falls back to the slider position, never a constant. Found by `tests/e2e_results_check.py`.
- **Returning from the map.** Two bugs, both older than this work in effect. The sample search answered its mount-time call with `None`, wiping the session's results whenever anyone came back from the map. And the derivation never ran after a client-side route change, because Dash reruns a callback then only when one of its inputs is newly mounted; it now also takes `results-panel.id`.
- **Failed searches clear their results** in all three modes, rather than leaving the previous query's matches under an error banner.
- **The AI request is assembled in the browser** (`assets/results-table.js`), so one server callback can still own both writing and clearing the summary without every filter change posting the full payload.
- **Tissue vocabulary.** The eyecup fix grew, with Joshua's approval: 396 choroid plexus samples (brain) had been filed under Eye by the bare "choroid" stem; eye anatomy terms (sclera, iris, conjunctiva, ciliary body, limbus, macula, vitreous, photoreceptor plurals and more) now reach Eye with guards for sclerosis, irisin, limbic and growth cones; and Eye ranks with the organs, so fetal retina, retinal organoids and iPSC-derived RPE are Eye rather than Embryo / stem cell. 2,934 ARCHS4 samples changed bucket, plus 750 the cached parquet had already been stale by. The local `cache/archs4_metadata.parquet` was regenerated offline with `fetch_archs4_meta.py --retissue`; the previous file is kept as `cache/archs4_metadata.pre-eye-fix-2026-10-21.parquet`. Placed ARCHS4 samples: 852,341 (90.6%).
- **Performance under load.** On a quiet machine a 1,000-deep search measured 0.7 to 1.0 s. During the browser runs the machine was shared with another project's browser tests and had under 100 MB free, and a click-to-table search took 8 to 11 s, almost all of it page-cache misses in the 963 MB memmap scan; the old 250-deep scan took 4 to 7 s under the same conditions.

## Change log

- 2026-10-21: spec written from the interview and the measurements above.
- 2026-10-21: built. Retrieval profile and tie rule; `results.py`, `result_plots.py` and the tissue fix built in parallel by subagents against the contracts above; the view, callbacks, styles and browser suite integrated and verified here.
- 2026-10-22: Similarity view removed at Joshua's request after review; the plot switch is Score by rank, Series and Network.
