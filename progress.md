# Bridge RNA - Progress

Living status log.
Update after each meaningful change so another session can resume without losing context.

This file used to track Bridge Manifold alone.
The two repositories were merged on 2026-07-22 and it now covers the whole product; entries before that date describe the map half.

## 2026-07-23 (file ingestion: embed an uploaded OSDR sample live)

The Retrieve view can now take an OSDR sample the corpus has never seen: upload its counts, embed it live, and get the identical output (network graph + inspector + optional LLM summary) the picker produces, scored against the same 940,455-sample ARCHS4 index.

**It is a fourth query-vector source, not a new pipeline.** `bridge_rna/retrieval.py` was already built around one fact - the cosine scan (`_topk_cosine_from_memmap`) is shared, and the cached/precomputed/demo paths differ only in where the 512-d query vector comes from. Uploading is that fourth source. Everything downstream - the scan, the offline annotation (`_annotate_from_cache`), the `archs4_index` map join - is reused unchanged, so an uploaded sample's hits carry the same schema (gse / title / tissue / species + a map position) as a cached OSDR sample's. `run_uploaded_retrieval` returns mode `"uploaded"`, and the status banner names it via the shared `_retrieval_phrase`.

**The embedding is a subprocess, by the same rule the demo path follows.** The serving app never imports torch (pinned by a test), so `precompute/embed_upload.py` loads the checkpoint, embeds one counts file, writes a 512-d npy, and exits; `bridge_rna.retrieval.embed_uploaded_counts` shells out to it. The preprocessing is not re-implemented - it reuses the exact symbols funnelled through `manifold/bridge_rna.py`, so an uploaded sample is embedded in the same gene order, ortholog mapping, TPM/log1p pipeline, and encode call as the corpus. **Invariant 1 (the gene-digest gate) is enforced before any vector is produced.**

**Validated end to end on the real model and corpus.** Embedding OSD-100's own counts file through the upload path reproduces its precomputed cached vector at **cosine 1.00000000, max abs diff 0.0** - the definitive check that scores are comparable. A full uploaded search of the eye sample `Mmus_C57-6J_EYE_FLT_Rep1_M23` against all 940,455 ARCHS4 samples returns eye-tissue analogs (GSM6204794, GSM4256053) in its top 5, annotated and locatable on the map. Input contract: mouse Ensembl-indexed counts CSV/TSV (OSDR is Mus musculus); a file that maps zero orthologs, or a digest mismatch, is refused with a clean one-line reason, never embedded into a meaningless vector.

UI: a `dcc.Upload` dropzone and a sample-column picker in the Retrieve rail, a separate Embed-&-search callback writing the shared outputs with `allow_duplicate=True`. Downstream callbacks (details, AI summary, See-on-map) resolve the query row from a `query` dict now carried in the hits-store payload, so they work for a sample that is not in `samples_df`. Flask `MAX_CONTENT_LENGTH` capped at 200 MB. Tests 219 (8 new in `test_upload_ingestion.py`, including the live-vs-cached parity gate and the gene-digest abort). Design doc: `docs/file_ingestion.md`.

## 2026-07-23 (spectral init restores the species separation)

Josh reported that even at `n_neighbors=30` the map looked less segmented than the version he remembered, specifically the human/mouse split. He was right, and it was not `n_neighbors`.

**The cause was the initialization, and it had been hiding in a commit from 2026-07-22.** The original build and the 07-21 retune both used UMAP's default `init="spectral"`. The full-corpus rewrite (43f3af1) switched to a PCA init in the same commit that removed the landmark transform, because spectral through UMAP's own path wants a 942,563 x 970 float64 Lanczos basis (7.31 GB) and drove the machine into swap. That switch was never separately measured against the thing it cost.

Measured now, on a 120,000-point sample with everything else held fixed:

| init | 25-NN species purity | species silhouette |
| --- | --- | --- |
| PCA (was shipped) | ~0.999 | 0.026-0.052 |
| spectral (now) | ~0.999 | 0.356-0.461 |

Species is ~100% pure locally under either - it was never *mixed* - but the global arrangement is completely different: PCA init scatters the two species as many small interleaved islands, spectral consolidates them into two territories. Local metrics cannot see that, which is why `--quality` scored the PCA build as fine and why the regression shipped. On the real full corpus the shipped 2-D map went from species silhouette 0.027 to **0.356, 13x**.

**The 7.31 GB was an artifact of one default, not of the mathematics.** `_spectral_layout` sizes its Lanczos basis as `max(2k+1, sqrt(n))`, and the `sqrt(n)` term is 970 at this corpus size when only 3-4 eigenvectors are wanted. Computing the eigenvectors directly with a small basis (`ncv=32`) and a shifted operator (largest eigenvalues of `2I - L` are the smallest of `L`, and Lanczos converges on largest far faster) costs **20-22 s and 241 MB** at full scale. `umap_init_from_spectral` in `build_projections.py`; `--umap-init pca` reproduces the old build.

The rail's init chip is now derived from the record (`spectral init` / `PCA init`) rather than hardcoded, so it cannot say PCA after a spectral build. OSDR spread ratio dropped 0.921 to 0.759, which is expected and not a regression: the all-mouse OSDR corpus now sits in the mouse territory rather than being spread across a map where the species were interleaved.

Tests 210, plus two in `test_projections.py` pinning that the spectral init is cheap/deterministic/scaled and that it separates a graph-community the PCA init interleaves. 40 browser checks pass. UMAP rebuilt in ~14 min; t-SNE preserved via the stats merge, not re-run.

## 2026-07-23 (t-SNE as a third projection, UMAP back to n_neighbors=30, parameter readout on the rail)

Three changes, opened by an observation: on the real map the species split looked visibly less separated than it had before, and the question was why.
The answer turned out to be a tuning decision that had never been scored on the real corpus.

**1. UMAP's `n_neighbors` went back to 30, reversing the 15 shipped on 2026-07-21.**
Prompted by a visual observation - the species split looked less separated than it used to - but settled by measurement, and the measurement said something better than the observation did.
Scored on the real corpus with `validate_artifacts.py --quality --compare`, **30 beats 15 on both metrics in both dimensionalities**: umap2 recall 0.3955 to 0.4140 (+4.7%) and purity 0.5838 to 0.6014 (+3.0%), umap3 recall 0.4596 to 0.4746 and purity 0.6169 to 0.6212. The OSDR spread ratio went 0.850 to 0.921.
So there is no local-for-global trade to weigh. 15 was simply worse on the full corpus, including on the two local metrics that were used to pick it.

**The flaw was the subsample, not the metrics.** The 2026-07-21 experiment fitted every candidate on 60,000 points. `n_neighbors` is a density parameter: fifteen neighbours out of 60,000 is roughly sixteen times as large a share of the manifold as fifteen out of 942,563, so the same integer cannot mean the same thing in both corpora.
The transferable lesson is that **a hyperparameter scaling with corpus density cannot be tuned on a subsample of that corpus**, whatever it is scored on.
A first draft of this entry blamed the metrics for being too local; the `--compare` run disproved that and the entry was corrected rather than left standing.
The metric half of the 2026-07-21 decision - cosine on raw 512-d instead of euclidean on PCA-50 - is untouched and permanent.
Full write-up in `REFERENCE.md`, "n_neighbors back to 30: the subsample tuning did not transfer".

**2. t-SNE joined PCA and UMAP as a third projection.**
Not a new idea: the 2026-07-21 evaluation of ten candidate methods concluded that if a third were ever added it should be openTSNE at perplexity 30 with PCA initialization, and that is what shipped, fit directly on all 942,563 points rather than through the landmark transform that entry anticipated.

- `openTSNE`, not `sklearn.manifold.TSNE`. sklearn has no interpolation accelerator, so a corpus this size is impractical rather than merely slow, and it cannot take a precomputed neighbour graph.
- t-SNE builds its **own** k=90 graph (3 x perplexity) through the same `build_knn` call UMAP uses, rather than sharing a padded one. A k=90 NN-descent graph sliced to k=30 is not the graph NN-descent would have built at k=30.
- The self-column slice is load-bearing: pynndescent returns self in column 0 and openTSNE's own index strips it, so the graph is built at 91 and sliced `[:, 1:]`. Leaving it in would give every point a zero-distance neighbour.
- The 2-D and 3-D fits share one affinity matrix (~2 GB, the build's largest allocation). Safe because exaggeration is applied as `P *= e` and restored with `P /= e` in a finally block; measured round-trip error 1.2e-16, float64 epsilon.
- **2-D and 3-D are different algorithms.** openTSNE's FIt-SNE interpolation refuses more than two output dimensions ("currently unsupported (and generally a bad idea)"), so 3-D is Barnes-Hut, which is `n log n` with a much larger constant and dominates the build's wall clock.

**3. The control rail now states how the active projection was fit.**
`n_neighbors=30 · min_dist=0.1 · cosine · PCA init · fit on all 942,563 points`, sitting directly under the Projection pills the way the coverage readout sits under the color-by dropdown.
It reads `projection_stats.json` through a new `data.projection_stats()` loader and never constants in the serving code, so it cannot stay confident while the cache goes stale.
It takes the dimensionality as a real input because t-SNE's gradient method genuinely differs: 2-D says FIt-SNE, 3-D says Barnes-Hut.
A key the record does not carry drops its chip rather than rendering blank, so an older cache shows fewer parameters instead of empty slots.

Supporting changes:
- `projection_stats.json` is now **merged** rather than rewritten, so rebuilding one method does not erase what the others recorded. The merge is abandoned if the corpus row counts changed.
- `layout.METHOD_LABELS` drives the pills, their disabled state, and the default, so a fourth projection is one line rather than four edits that can disagree. An unbuilt method is disabled and visible, not hidden.
- `validate_artifacts.py` walks one `_COORD_PATHS` list (six entries now) instead of two that could drift, gates t-SNE on quality the way it gates UMAP, and prints `SKIP` rather than failing for a stage the build record shows was never run.
- `.bm-hint` moved from `--text-muted` to `--text-secondary`: `#8a99ac` at 11.5px measures 2.90:1 on the white panel and fails WCAG AA. Unrelated to this work, found while adding the adjacent rule, fixed anyway.

Measured on the real corpus after the build (`--quality`, 60,000-point sample, null purity 0.0710, 512-d ceiling 0.6267):

| coords | kNN recall @15 | 25-NN tissue purity | share of recoverable |
| --- | --- | --- | --- |
| pca2 / pca3 | 0.0374 / 0.1199 | 0.1654 / 0.2758 | 17.0% / 36.8% |
| umap2 / umap3 | 0.4140 / 0.4746 | 0.6014 / 0.6212 | 95.4% / 99.0% |
| tsne2 / tsne3 | **0.5124 / 0.5179** | 0.6182 / **0.6364** | 98.5% / **101.7%** |

t-SNE beats UMAP on both metrics in both dimensionalities, which is what the subsample evaluation predicted and is the one prediction from it that did transfer.
`tsne3` scoring above the 512-d ceiling is real rather than an error: collapsing 512 dimensions onto 3 averages away variation that is not tissue-related, so neighbourhoods get purer than they were in the original space. Read the share column as "how much survived", not as a score out of 100.

Build cost, one uninterrupted run: PCA 4.8 s, k=30 graph 130 s, UMAP 326 s + 347 s, k=91 graph 636 s, affinities 18 s, t-SNE-2d 407 s, **t-SNE-3d 8,128 s**. Total ~2.8 hours, of which the 3-D t-SNE is 81%.
That one stage is expensive for a library reason, not a tuning one, and it was measured rather than assumed: openTSNE parallelizes through OpenMP and the PyPI macOS wheels are built without it (`nm` finds zero `omp` symbols, `otool -L` no `libomp` in `_tsne`, `kl_divergence`, `quad_tree`), so `--tsne-jobs` is a no-op and both fits ran on one core. Its help text now says so. Building from source against `libomp` would recover roughly the core count and is deliberately not done, because threaded float summation makes the gradient order-dependent and this is the artifact every coordinate derives from.

Five defects were found by an adversarial review of the diff, each reproduced before being fixed, and all fixed. `--umap-neighbors` still defaulted to 15, so the documented rebuild command would have silently undone the change while every doc claimed 30; and `run_tsne` logged the module perplexity constant rather than the value in force, so an hours-long stage could misreport its own parameter in the log it is read back from.

Three more came from a second pass, and two of them share a root cause worth stating: **the build record is written before the fit it describes finishes.** Every stage saves its stats and its parquet before the next one starts, and the 3-D t-SNE fit is 81% of the wall clock, so a run interrupted there leaves a *complete* `tsne_*` record next to a missing `coords_tsne3.parquet`. The rail then asserted "Barnes-Hut, fit on all 942,563 points" beside a plot reading "coordinates not built yet" - precisely the failure `projection_params` was written to prevent, arriving by a route reading the record instead of constants does not close. `data.coords_available(method, dims)` now gates the readout, while `method_available` stays 2-D-only so a genuinely-built 2-D map is still offered. The third: `validate_artifacts.py` section 4 read `coords_umap2.parquet` unconditionally, so a `--skip-umap` build crashed with `FileNotFoundError` one section after section 2 had declared that same build a legitimate SKIP, killing the run before the pass/fail summary. Both are pinned by tests.

Tests: 198 to 208. The method loops in `test_data.py` and `test_render.py` now iterate `data.METHODS` rather than a literal pair, so a projection cannot ship without having been drawn. The fixture writes t-SNE coordinates and realistic `umap_*`/`tsne_*` stats, without which the parameter formatter's real path was never exercised. Browser checks 29 to 40.

## 2026-07-23 (doc consolidation into the README)

Folded the standalone explainer docs into `README.md` and removed them, so a new reader meets fewer scattered files.

What changed:
- Added a "How it works" section to the README covering the model and the shared space, retrieval, the map build, the shared tissue vocabulary, hover and inspect, the AI reading, and a "what the results mean" note on interpretation. It is condensed to what a new reader needs, not the exhaustive version.
- Deleted `docs/how-it-works.md` (a longer FAQ built earlier this session) and `docs/manifold.md`. Their necessary content is now in the README; the map-specific build notes and the synthetic dev-corpus command moved into the README's build and tests sections.
- Kept `IMPLEMENTATION.md`, `REFERENCE.md`, `CLAUDE.md`, and `progress.md`. The README still points to `IMPLEMENTATION.md` and `REFERENCE.md` for the design and the verified facts.
- The two screenshots (`docs/bridge-rna-interface.png`, `docs/bridge-rna-map.png`) stay in `docs/`.

Note on the entry below: it references `docs/manifold.md` as a live pointer, which no longer holds. That file's content is in the README as of this entry; the deeper facts it summarized were already duplicated in `IMPLEMENTATION.md` and `REFERENCE.md`.

## 2026-07-22 (README rewrite for new users)

Rewrote `README.md` to be readable for a first-time user: 400 lines down to 197.
The old README carried the full body of caveats, exact measurements, and honesty disclaimers inline, which buried the "what is this / how do I run it" a newcomer actually needs.

What changed:
- Kept: the one-line pitch, the two-view explanation with both screenshots, the quickstart (clone / install / run), optional AI setup, optional map build, tests, a trimmed project-layout table, and licensing/citing.
- Cut from the body and replaced with a short "Learn more" section pointing at `docs/manifold.md`, `IMPLEMENTATION.md`, and `REFERENCE.md`: the deep canonical-gene-list section, the "Implementation notes" (species mapping, normalization, index facts), the full "Reading the map honestly" section, the detailed "Known limitations" list, and the `demo_osdr_top5.py` CLI usage.
- The three map-reading caveats (non-quantitative distance, cross-corpus batch effect, coverage-aware colouring) are now three compressed clauses in "Learn more" rather than three full subsections; the full versions still live in the docs and the interface still discloses them.
- Simplified vocabulary throughout ("Earth studies" / "Earth corpus" instead of "ARCHS4/GEO", fewer precise counts inline).

No prose facts were lost from the repo; everything cut is still in `IMPLEMENTATION.md` / `REFERENCE.md` / `docs/manifold.md`.

## Current status: 2026-07-22 (map UI refinements)

Five changes to the map view, driven by user feedback that the interface over-explained and that some readouts were misleading below full budget.

**1. The point budget now depends on the dimensionality.**
3-D caps the ARCHS4 cloud at 40,000 for smooth rotation, but the control still offered 100k / 250k / 500k / All and silently redrew any of them as 40,000 - a control that lied about what it did.
In 3-D the tiers are now 10k / 20k / 30k / 40k with no "All"; `layout.budget_options(dims)` builds them and `callbacks.sync_budget_to_dims` swaps them (and clamps the value) when the dimensionality changes.
Switching back to 2-D restores the 100k / 250k / 500k / All tiers.

**2. Legend counts now report what is actually plotted, not the whole corpus.**
The old legend showed whole-corpus counts, which are meaningless below a full budget (a "40k" 3-D view was labelling categories in the hundreds of thousands).
`render._legend_with_drawn_counts` now recomputes each row's count per figure from the drawn ARCHS4 sample plus the OSDR overlay, so the numbers track the budget and the zoom, and a category with nothing on screen drops out of the key.
Colour and legend order are still fixed by the whole-corpus ranking, so a category keeps its colour whether or not it is currently drawn.

**3. Removed UI microcopy that read as AI-generated over-explaining.**
Gone: the projection hint ("UMAP preserves local neighborhoods…") under the UMAP/PCA toggle, and the standing "Reading across corpora" caution at the bottom of the rail.
The budget hint was trimmed to one line.
Both facts are preserved in the docs (README, `IMPLEMENTATION.md`, `REFERENCE.md`); the `.bm-caution` CSS was removed too.

**4. The README map screenshot is now the 3-D UMAP.**
`docs/bridge-rna-map.png` was a 2-D map framed on a retrieval; it is now the 3-D UMAP of the joint corpus coloured by tissue (3200x1960, captured with `scratchpad/shoot_3d_umap.py`).
The surrounding README prose was rewritten to match.

**5. Tests and the browser check were updated, not just left green.**
198 pytest tests pass (was 194): the two tests that pinned whole-corpus legend counts were flipped to the drawn-count contract, four new tests cover the dims-dependent budget tiers, the drawn-count legend, zero-count drop-out, and the 3-D cap, and the test asserting the batch-effect caution lived on the rail was replaced with one pinning its removal.
`tests/e2e_check.py` gained assertions that 3-D drops the "All" tier and caps near 40k and that 2-D restores it; all live browser checks pass.

## Current status: 2026-07-22 (later) - one app, and a retrieval 44x faster

Bridge Manifold and Bridge RNA are one repository and one application.
The merge kept all 19 of the manifold's commits rather than squashing them.

**The map made the retrieval fast.**
This was not the goal of the merge and is the most valuable thing to come out of it.
The manifold precompute had already embedded all 2,108 eligible OSDR samples with a preprocessing path checked bit-for-bit against the retrieval's own, and had already joined GEO metadata for all 940,455 ARCHS4 samples.
So the query vector never needed recomputing by a subprocess, and the hits never needed annotating over the network.

Measured on OSD-100 `Mmus_C57-6J_EYE_FLT_Rep1_M23`, top-5:

| path | wall clock | gse / title / tissue |
| --- | --- | --- |
| subprocess (`demo_osdr_top5.py`) | 22.1 s | all empty |
| cached (manifold artifacts) | ~0.5 s (warm) | populated, offline |

Identical accessions and identical scores to six decimal places.
The cached figure is the warm end-to-end `search_hits` time, measured at 0.44-0.57 s across runs; the first call after startup is nearer 0.8 s while the memmap pages in. 22.1 / 0.5 is the 44x in the heading.
`search_hits` returns which path ran so the interface can say so.

**Correction, made the same day.** This entry first said "the subprocess path stays for the 788 samples the manifold never embedded", and that is wrong for 71 of them.
Checked against each study's own counts matrix, the picker's 2,896 samples fall into three tiers, not two:

| tier | count | behaviour |
| --- | --- | --- |
| cached | **2,108** | precomputed vector, ~0.5 s, and on the map |
| subprocess | **0** | nothing reaches it while the cache exists |
| **unavailable** | **788** | no path can serve it |

**Corrected twice.** The first attempt said 788 fall back to the subprocess. The second said 717 do, having checked only whether a sample's name is a column in its counts matrix. Both were wrong, and an adversarial review caught the second.

`demo_osdr_top5.py` filters its metadata to rows *with a recorded spaceflight value* before it looks the requested name up, so 733 of the 788 raise "not found after filtering" - a different error from the counts-column one, which is why checking only for the column looked convincing. The other 55 pass the filter and match no column.

Both reproduced end to end: `OSD-141|Mmus_C57-6J_SPL_cells_Rep1_SP1` in 4 s, `OSD-462|RR10_KDN_WT_BSL_B11` in 2.3 s.

The lesson is the one this file keeps relearning: a plausible mechanism that explains the failures you looked at is not the mechanism. The second version was checked against one failing sample and it happened to be one of the 55.

- `app.py` is the single entry point: `/` retrieves, `/map` draws the manifold, one header and one port.
- `app_osdr_dash.py` (2,470 lines) is now the `bridge_rna/` package; 49 definitions were moved by exact line range and a checker asserts each appears once with a byte-identical body.
- Stylesheets are layered by load order: `00-tokens.css`, `01-shell.css`, `retrieve.css`, `map.css`.
- **194 tests pass**, up from 160, the 29 browser checks pass against the merged app, and `validate_artifacts.py` is clean.

### The two views are linked in both directions

**Retrieval → map.** A search offers "See N hits on the map", and the map draws the query as its teal star and each hit as a numbered white ring with the corpus receded to 0.35. "Frame the retrieval" zooms to a window containing all of them, which is necessary because at full-corpus scale the hits are a few pixels apart.

The translation is three lines, because there is nothing to translate: a hit's `archs4_index` is its row in the memmap, ARCHS4 occupies rows 0..940,454 of the map's point order, so the row *is* the point.

Three decisions there are about honesty rather than looks, and should survive future edits:

- **No line is ever drawn between the query and a hit.** It is the obvious and most striking choice and it would assert something false: the ranking is cosine distance in 512 dimensions and the map is a 2-D projection that does not preserve it. The hover states both orderings instead. For the OSD-100 eye query, 512-d rank 1 is only map rank 33, while 512-d rank 2 is map rank 2.
- **Every hit ring is identical** - no size, opacity or colour ramp across rank. The top five span 0.0016 cosine (the top twenty span 0.0041); any ramp would assert a difference the index does not contain.
- **Hits are white open rings, not the network graph's blue.** Measured: `#2b7fff` is 1.03:1 against `CATEGORICAL[0]`, which is Blood / immune, the largest bucket at 155,761 points, so a hit landing in 16.6% of the corpus would have been invisible. White is 3.64:1 there. Open, so the point underneath keeps its tissue colour and one glyph shows both that the model retrieved it and what GEO's free text calls it.

**Map → retrieval.** Clicking an OSDR point offers "Retrieve its Earth analogs", linking to `/?q=<sample_id>`. A URL parameter rather than a store mutation, so it is a real link that can be opened in a new tab, bookmarked, or pasted to a colleague.

### Search is 18x faster in the interface

GEO/PubMed enrichment was on by default and cost a network round trip per hit. The cached path already delivers series, title, source name, characteristics and tissue locally, so what enrichment still adds is study abstracts and publications - text most searches never open.

It is off by default now, and the two places that need the text fetch it themselves: the inspector for the one hit you open, and the AI panel for all of them before it writes. Measured in a browser, same query: **10.9 s → 0.6 s**.

### Defects found and fixed this session

1. **The status banner announced cached results as "real demo script output".**
   `run_search` special-cased only `mode == "precomputed"`, so the new path fell through to the else branch.
   The interface was asserting something untrue about how the answer was made.
2. **Five Dash component tokens were defined in both stylesheets with different values**, so whichever file sorted later silently decided how the *other* view's controls rendered on hover.
   One token layer now, with a test that no token is defined twice.
3. **`.app-header-chip` had no CSS rule anywhere** and the "Beta" tag was rendering as plain body text.
   Found by widening the classname check to cover the retrieval view; it had only ever checked the map's.
4. **The retrieval view carried two `hits-store` components.**
   Dash only validates ids in the *initial* layout, so this stayed invisible until the shell began serving views there.
   A test now checks each view for duplicate ids directly.
5. **`.app-root` declared `height: 100vh` under a header**, and `#page-content` was not a flex container, so the view collapsed to content height and left a band of bare canvas.
6. **The picker offered 71 samples that cannot be retrieved at all** - see the correction above. Now disabled with the reason, and the picker never defaults to a disabled option.
7. **Clicking a hit faded every other node in the retrieval network.** `build_network_figure` set `clickmode="event+select"`, so Plotly applied selection styling on each click and inspecting one result made the rest look dismissed. There is no selection feature in that graph; `clickmode="event"` fires `clickData` just as well. Found by looking hard at a screenshot taken for the README.
8. **The inspector's on-demand enrichment could never fire on a cached hit.** It asked whether any of `gse`/`title`/`geo_summary`/`pubmed_ids` had content, a fair proxy when a hit arrived either fully enriched or entirely bare. The cached path always fills `gse` and `title`, so the test passed for every hit and the abstract was never fetched. It now tests the study-context fields specifically.

### Found by an adversarial review of the day's work

A judged, verified review of the whole merge (5 dimensions, every finding refuted or confirmed by a second agent) caught defects the tests and my own passes had missed:

9. **The retrieval-tier classifier mislabelled 717 dead samples as slow.** `sample_tier` checked only whether a sample's name is a column in its counts matrix, but `demo_osdr_top5.py` first filters to rows with a recorded spaceflight value. 733 of the 788 unavailable samples fail that filter, so the true tiers are 2,108 cached / 0 subprocess / 788 unavailable. The number was wrong three times before it was right; the correction blocks above record all three.
10. **The inspector dropped 10 of the fields it fetched.** An on-demand NCBI fetch returns platform, entry type, release date, FTP link and the whole Publication section as `_biopython` columns the panel renders, but the merge back kept only columns the cached schema already had, which is none of those ten. It now adds a missing column before writing it.
11. **The retrieval network's edge width encoded rank, not similarity.** A min-max rescale drew the thinnest hit at 1.5 px and the thickest at 8 px whatever the scores were, so a 0.0016 spread looked as dramatic as a 0.4 one while the legend said "similarity score". Now mapped onto a fixed [0.90, 1.0] domain.
12. **The 3-D overlay crashed the figure callback.** `Scatter3d` rejects `star` and `cliponaxis` outright, so opening 3-D with a retrieval showing returned a 500 and left the stale 2-D figure up. The overlay had no test and the browser check never opened that state. Six tests cover it now.
13. **The header overstated retrievability by 55.** "Eligible OSDR samples: 2,163" counted samples the picker disables; it now counts the 2,108 that are actually retrievable, which is also the OSDR points on the map. Relabelled "Retrievable".
14. **A pasted `/?q=` deep link did nothing on cold load**, working only when followed from a live map. Handled at layout-build time now via `layout._initial_study`.
15. **The fix for 14 shipped a regression** that emptied both dropdowns on any load, live on `main` for three commits. Caught only in the final end-to-end pass, because the callback graph stayed valid - a working callback graph is not a working app. Live navigation is a callback (with `prevent_initial_call`); cold load is handled at layout-build time.

Plus a sweep of stale documentation numbers across all six docs: build time (~50 min to 10.5), test totals, browser checks (27 to 29), the top-5 cosine span (0.0041 was the top-20 span; the top-5 is 0.0016), tissue Unknown (839 unresolved against 882 total on the map), the memmap-never-opened claim (true of the map view, false of the retrieval view since the cached path opens it), and the two design docs that still described two separate apps.

The lesson worth keeping: the interface-honesty standard is easy to violate by accident.
Three of these - the tier count, the edge width, the header count - were the interface quietly asserting something the data did not support, and each looked fine until it was measured against the data it claimed to describe.

## densMAP: measured at full corpus scale, and rejected (2026-07-22)

Next-step item 11 is answered.
densMAP's rejection in the 2026-07-21 evaluation rested on `umap-learn` refusing to `.transform()` into a densMAP embedding, which was fatal under the landmark pattern and irrelevant once every point is fit directly.
So it was rebuilt at full scale: `build_projections.py --densmap`, `dens_lambda 0.5`, the same k-NN graph settings as the shipped build, 716 s for the 2-D fit and 812 s for the 3-D.

Scored by `validate_artifacts.py --quality --compare` on the same 60,000-point sample and the same nulls as the shipped coordinates:

| coords | 15-NN recall | 25-NN tissue purity | share of recoverable structure |
| --- | --- | --- | --- |
| umap2 **shipped** | **0.3955** | **0.5838** | **92.3%** |
| umap2 densMAP | 0.2321 | 0.5347 | 83.4% |
| umap3 **shipped** | **0.4596** | **0.6169** | **98.2%** |
| umap3 densMAP | 0.3389 | 0.5996 | 95.1% |

**densMAP loses on both metrics in both dimensionalities**: local fidelity -41.3% in 2-D and -26.3% in 3-D, tissue purity -8.4% and -2.8%.

The result worth recording is not the verdict but the size of the error in the estimate.
The 60,000-point evaluation predicted a local-fidelity cost of about 9% (0.377 to 0.344) and a tissue cost of about 4%.
At 942,563 points the local cost is **4.6x larger** than that prediction.
A method comparison run on a subsample is evidence about the subsample; the ranking it produces does not transfer to a corpus fifteen times the size, and this one did not.

densMAP's one advantage - density fidelity 0.441 to 0.739, measured at 60k - is real and is not enough.
Local fidelity is the property the map exists for: it is what makes "these points are near each other" mean anything.
Trading 41% of it for an honest impression of cluster density is the wrong trade for this instrument.

`--densmap` stays in `build_projections.py` so the measurement is repeatable, and nothing in `cache/` changed.

## Current status: 2026-07-22 - every point drawn, every reduction fit on every point

Two changes this session, both of which came down to the same thing: a cost that had been estimated rather than measured, and was wrong.

**1. The density underlay is gone. The map draws all 942,563 points.**
The raster existed because 940k live WebGL glyphs was assumed to be out of reach, so ~100k were drawn live and a precomputed PNG carried the rest.
Measured: building the figure costs the same at every budget, because the dominant cost is resolving one label array over the full corpus rather than the size of the sample drawn from it, and serializing all 942,563 points takes 0.15 s and 11.3 MB against 0.03 s and 1.3 MB at 100,000.
The default budget is now the whole corpus; 100k / 250k / 500k remain for a lighter view.
Verified in a browser: first interactive frame in **1.3 s** with **942,563 glyphs**, no console errors, budget switches re-rendering in 0.1 to 0.3 s.

**2. PCA and UMAP are both fit on all 942,563 points, in 2-D and 3-D.**
PCA was fit on a 60,000-point subsample; UMAP was a 122,563-point landmark fit with the remaining 819,999 pushed through `.transform()`, which does not lay those points out at all - it places each one by averaging where its landmark neighbours already sit.
The full build takes **10.5 minutes**, which is *faster* than the 15.8-minute landmark build it replaces, because the two `.transform()` passes (404 s and 467 s) are gone.
The "a direct 940k fit is hours" claim that shaped the entire first design was never measured.

Measured effect, by `validate_artifacts.py --quality --compare` against the saved landmark coordinates on one 60,000-point sample: **15-NN recall +8.1% in 2-D and +7.1% in 3-D**, tissue purity -1.5% and +1.5%.
So the full fit buys local fidelity and leaves biological fidelity where it was.
PCA barely moved at all (PC1 correlates 0.999998 with the subsampled fit), and that is recorded as a negative result: the exact fit is kept because it costs 4.5 s and removes an approximation, not because it changed the picture.

- `precompute/build_projections.py` ran end to end in ~10.5 min, rc=0: exact PCA 4.5 s, k-NN graph 59 s, UMAP-2d 251 s, UMAP-3d 251 s.
- `validate_artifacts.py --mixing --quality` passes, with the one documented cross-corpus batch-effect warning (54x, unchanged - it is a property of the 512-d space, not of the projection).
- **160 tests pass in about 1.1 s**, up from 144.
- The live cache is 217.8 MB, of which the app opens 80.8 MB.

### Three things this session found that were not the task

- **UMAP's spectral init cannot run on this corpus at all.**
  `_spectral_layout` sizes its Lanczos basis as `max(2k+1, sqrt(n))`, which at n = 942,563 is 970, so `eigsh` allocates a 942,563 x 970 float64 basis: **7.31 GB**.
  The first full-corpus attempt drove the machine into 7.6 GB of swap and made no progress in 25 minutes before it was killed.
  Passing the exact PCA coordinates as `init` instead took the 2-D fit to 251 s.
  This is the single change that made the whole thing viable.
- **The memoized colour plan was going to cost 1.4 GB.**
  Caching the per-point category array is what keeps a zoom or a budget change cheap now that they redraw the whole corpus, but under pandas 3.0 `.to_numpy()` on a string Series materializes a *fresh* Python `str` per element: 942,563 distinct objects to express 13 distinct values, measured at 127.5 MB per colour-by.
  Storing `int16` legend slots instead is 1.9 MB, and a warm full-corpus figure went from 1.33 s to 0.06 s because category selection became a vectorized integer compare rather than 942,563 string comparisons.
- **UMAP writes into the k-NN arrays it is given.**
  `fit()` assigns them through without copying and then writes into them in place to disconnect far neighbours (`umap_.py:2647-2654`), so the 2-D and 3-D fits sharing one graph would have let the first quietly edit the second one's input.
  Each fit now gets its own copy.

## Session 2026-07-21 - built, colored by real biology on both corpora, and tested

The full offline pipeline has run to completion on real data, and the app has been redesigned around one question the first build got wrong: what should the map show for the corpus the selected color-by does not describe.

`cache/` holds the real 942,563-point manifold: 940,455 ARCHS4 (510,709 human, 429,746 mouse) plus 2,108 OSDR.

- `embed_osdr.py` finished 2026-07-21 08:45:51 after ~11.3 h, all 2,108 samples, gene-digest gate passed.
  Realized rate was ~10 s/sample in fast stretches, degrading to ~49 s/sample between 05:44 and 08:26 under machine contention, so the original ~6.5 s/sample estimate was optimistic.
- `build_projections.py` ran 08:45:57 to 08:51:44, **5 min 47 s**, rc=0.
  The 30-90 min estimate in `REFERENCE.md` was wrong by an order of magnitude; measured per-stage timings are now recorded there.
  Two of those stages have since been deleted, so the same build is now a **291 s** job.
- `fetch_archs4_meta.py` ran in **33.7 s** over 39 requests and about 216 MB, resolving **99.911%** of all 940,455 accessions.
- `precompute/validate_artifacts.py --mixing` passes every structural and invariant check, with one substantive warning: the cross-corpus batch effect (see Notes and risks).
- **144 tests pass in about 0.55 s** against a hermetic synthetic corpus.

### What is done

- **Phase 0 scaffold**: package skeleton, path configuration with `BRIDGE_RNA_ROOT` / `MANIFOLD_CACHE_DIR` overrides, LFS-pointer preflight.
- **Phase 1 OSDR embeddings**: `embed_osdr.py`, gene-digest gated, resumable, with a cached expression stage. **Complete.**
  Preprocessing proven bit-for-bit identical to Bridge RNA's single-sample path.
- **Phase 2/4 projections**: `build_projections.py` writes PCA-2/3, landmark UMAP-2/3, the identity table, the ARCHS4 accession sidecar, and the density rasters. **Complete.**
  (Superseded 2026-07-22: both reductions are now fit on the full corpus and there are no density rasters.)
- **Phase 3 interactive plot**: layered renderer (density underlay, stratified ARCHS4 cloud, OSDR overlay), layer toggles, point budget, viewport level-of-detail.
  (Superseded 2026-07-22: no underlay, and the budget defaults to every point.)
- **Phase 5 coloring both corpora**: the ARCHS4 GEO metadata join, the shared tissue vocabulary, and the coverage-aware color-by registry that replaced the renderer's per-key branching.
- **Phase 6 polish**: searchable legend, theme-matched Dash 4 controls, hover cards, 3D, honest empty and degraded states.
- **Tests**: `tests/` with a synthetic corpus built from known latent clusters plus a synthetic `archs4_metadata.parquet` written in ARCHS4's free-text register and mapped through the real canonicalizer, so the tissue vocabulary is tested against GEO-shaped strings rather than against its own rules.

### Removed in this session, and not to be reintroduced

**The lasso selection tool and its 512-d statistical readout are gone in their entirety**, at Josh's explicit request.
Deleted: `manifold/coherence.py` (450 lines), `tests/test_coherence.py` (431 lines), the right-hand readout column in `layout.py`, the `selectedData` callback and every helper behind it in `callbacks.py`, the vector/moment/index loaders in `data.py`, and the readout and lasso-marquee sections of `assets/manifold.css`.
`dragmode` is now `pan`, and the graph config removes **both** `select2d` and `lasso2d` (the old config removed only `select2d`, so the lasso button was in fact still on the modebar).

Consequences, all of them verified:

- `build_projections.py` no longer builds `cache/joint_cosine.hnsw` (2.07 GB) or `cache/population_moments.npz` (4.2 MB), and its `--skip-hnsw` flag is gone.
- `requirements.txt` dropped `hnswlib` and `scipy` and added `requests`. The serving app's dependency surface is now `dash`, `plotly`, `numpy`, `pandas`, `pyarrow`, and nothing scientific.
- The live cache fell from about 2.3 GB to a measured **219.2 MB**, of which the app opens **82.3 MB**. (Both dead files have since been deleted, along with the density rasters; the cache now measures 217.8 MB with 80.8 MB opened.)
- The serving app no longer opens the 963 MB ARCHS4 memmap at all, so `BRIDGE_RNA_ROOT` is needed to *build* the cache and not to *run* the app.
- `validate_artifacts.py --mixing` used to load the ANN index.
  It now computes the **exact** top-51 neighbours of each of the 2,108 OSDR samples by streaming the memmap in 50,000-row blocks and merging a running top-k (`_osdr_neighbours`), which costs 10.3 s warm.
  That is why the index could be deleted.
  The mixing check itself is unchanged and is **not** a lasso feature: it is the honesty check behind the app's premise and must keep working.
- `manifold/preflight.APP_REQUIRED` was wrong in both directions and is fixed. It demanded the ARCHS4 memmap, `sample_locations.parquet` and the OSDR embeddings, none of which the app opens, while omitting `cache/points_meta.parquet`, which `layout.control_rail()` reads *first* through `data.counts()` - so a missing identity table passed preflight and then crashed during startup.

The test suite went 103 -> 144 tests and 4.54 s -> 0.55 s; the fixture no longer builds an ANN index, which was 43% of the old wall clock.

### The headline: ARCHS4 can now be colored by real biology

The problem: the app had about ten color-bys for the 2,108 OSDR samples and exactly one (species) for the 940,455 ARCHS4 samples.
Choosing any OSDR field painted 99.8% of the map one flat grey, which on a scientific plot reads as "ARCHS4 was measured and has no structure here".
A "Tissue (ARCHS4)" option existed but required the ARCHS4 gene HDF5 files, 62.3 GB human plus 50.7 GB mouse, which were never downloaded, so it had never once worked.

Three pieces fixed it.

1. **`precompute/fetch_archs4_meta.py`, rewritten around the Maayan Lab sigpy JSON API.**
   `POST https://maayanlab.cloud/sigpy/meta/samplemeta` with `{"species": ..., "samples": [...]}` returns per-GSM `{series, title, source, characteristics}` in bulk.
   Measured by running it: 33.7 s, 39 requests, ~216 MB, 99.911% of all 940,455 accessions (human 99.851%, mouse 99.982%).
   Output is `cache/archs4_metadata.parquet`, 940,455 rows, 32.5 MB, 51,284 distinct GEO series.
   Reading the same fields out of the remote gene HDF5 over range requests works but costs ~5 min and ~272 MB **per field**; downloading the files is 113 GB.
   The 839 unresolved samples are not GEO withdrawals - they are present in the release-matched v2.5 metadata and absent from the newer v2.latest the API serves, which disproves the "ARCHS4 releases are append-only" assumption. They get tissue `Unknown` rather than being dropped or guessed at.
2. **`manifold/tissue.py`, one tissue vocabulary shared by both corpora.**
   40 ordered keyword rules, first match wins, producing 37 distinct buckets plus `Other` and `Unknown`.
   All 48 OSDR raw values land in a named bucket, and 851,881 of 940,455 ARCHS4 samples (**90.6%**) do too, so the Tissue color-by covers **942,563 of 942,563 points**.
3. **`manifold/colorby.py`, the coverage-aware registry.**
   Coverage is a declared, first-class property: each `ColorBy` states its scope, its resolver, an optional hint, and an optional `(predicate, fix-hint)` pair for an artifact it needs, and `covers()` reports what it can color right now on this machine.
   That one fact drives the menu order, the disabled state, the coverage readout under the control, and what the renderer does.

The interface consequences are the point of the exercise: the menu lists whole-map fields first with their scope attached, a field with no data is shown *disabled* with the command that enables it rather than hidden, a coverage bar and an exact point count sit under the control, and **the renderer never paints a uniform grey glyph cloud** - a corpus a field does not describe is carried by the density raster, or by a deliberately faint context cloud at 0.35 opacity when there is no raster (3D, or the underlay switched off).

Tissue was then validated as biology rather than as batch, to the same standard every rejected candidate was held to: 25-NN label purity **0.8142** against a permuted null of **0.0501**, surviving both a batch control and a depth control at **0.7058**.

### Corrections to earlier assumptions

- **Corpus size.**
  The OSDR corpus is **2,163 eligible / 2,108 embedded**, not 2,896. 2,896 is the unfiltered TSV row count; 733 rows have no spaceflight factor and are excluded by the Bridge RNA filter Josh chose to match, and 55 more name a counts column that does not exist.
  All docs corrected.
- **Environment.** The versions in `REFERENCE.md` had drifted: pandas 3.0, dash 4.4, plotly 6.8, numpy 2.4, torch 2.12. Three of those releases changed behaviour the code depends on (see `REFERENCE.md` section 5).
- **datashader is not used.**
  The density raster is a numpy 2D histogram plus Pillow.
  Fewer fragile dependencies, and it is trivially fast at this scale.
- **MPS is not viable** for this model, and chunking does not help. Measured and documented.
- **The ARCHS4 HDF5 download was never necessary.**
  It had been the plan for the entire first build and had never been executed once.
  The API route returns the same fields three orders of magnitude cheaper.

## Decisions log

- **No on-demand statistics.**
  The map is read, not queried.
  A statistic computed from a screen region would be a number read off distorted UMAP pixels, and the selection readout was the app's largest source of complexity in service of a question the map answers qualitatively.
- **Coverage is a declared property, not a per-branch decision.**
  Every field states which corpora it can color right now; the menu, the coverage readout, and the renderer all read that one declaration.
  The alternative is what the first build did, and 99.8% of the map turned flat grey with nothing in the interface admitting it.
- **One shared tissue vocabulary rather than two tissue fields.** Two separate "Tissue" color-bys would each leave the other corpus grey, which is the grey-map failure by another route.
- **The tissue mapping is auditable keyword rules, not learned.** It fails towards "Other" rather than towards a confident guess; on a plot people read biology off, an honestly empty label beats a wrong one.
- **`Unknown` and `Other` stay distinct, and weak results are ranked.** Nothing recorded is not the same fact as recorded-but-unplaceable, and without the ranking an early unplaceable field pinned the answer to "Other" and blocked a later field that did identify the sample.
- **One palette across both corpora.**
  Categories are ranked once over the whole covered population, so a liver in GEO and a liver in OSDR share a color; ranking per layer silently gave one category two colors.
  Legend counts are whole-corpus counts, so they do not move with the point budget or the zoom.
- **The availability predicate is `data.archs4_metadata_available` itself**, never a path re-derived inside the registry. A second source of truth for the same file was a real bug; a test now pins it.
- The seven OSDR control arms stay distinct; the binary Flight-vs-Ground contrast is a separate derived field. Rationale: basal and vivarium controls are different experiments, and merging them erases real structure.
- L2-normalize before any reduction. Rationale: raw vectors carry a 4x magnitude spread that dominates PC1 (57.8% before normalization, 40.9% after).
- UMAP is offline only, via landmark fit then transform. Rationale: a direct 940k fit is hours and risks memory blowup.
- Standalone app importing Bridge RNA functions, not edits to the 2,470-line retrieval app. Rationale: isolation without losing the shared instrument feel.
- Batch structure is made visible, not corrected. The measured 54x tissue-controlled cross-corpus effect is stated on the control rail, always, rather than inside anything the user has to trigger.
- OSDR embedded in fp32 on CPU. Rationale: fidelity baseline, and measurement showed no faster option exists on this machine.
- Dash components are themed by remapping Dash 4's own `--Dash-*` design tokens rather than by overriding each component's rules. Rationale: one mapping themes every current and future Dash component; per-component overrides are a specificity war that silently rots on upgrade.

## Decisions from Josh

1. **2026-07-21: remove the lasso tool completely**, from the implementation and from every document. Done; the removal is recorded as history only, in `IMPLEMENTATION.md` section 1 and section 8.
2. **2026-07-21: never end up with a grey map.** This is what the coverage-aware registry, the shared tissue vocabulary, and the density fallback exist for.
3. 2026-07-20, ARCHS4 tissue coloring: FETCH NOW for v1. Delivered, and better than scoped - the API route removed the HDF5 blocker entirely rather than shipping behind a graceful degrade.
4. 2026-07-20, OSDR scope: MATCH the Bridge RNA filter (mouse + spaceflight factor). Honored - this is what yields 2,163 rather than 2,896.
5. 2026-07-20, batch handling: EXPOSE AND GUARD only. No correction, not even as a toggle.
6. 2026-07-20, environment: SHARE the Bridge RNA venv. Done; `requirements.txt` records the verified versions and splits serving from precompute.

## Color-by candidates that were built or tested and then rejected

Recorded with their evidence so nobody re-proposes them.
Full write-ups in `IMPLEMENTATION.md` section 7.5 and `REFERENCE.md` section 11.

- **Cosine similarity to an OSDR reference** (mean / flight / ground centroid, and a flight-minus-ground "spaceflight-likeness" axis). One field wearing four names, pairwise r 0.996-1.000. The interesting axis correlates r = -0.990 with PC1 and r = -0.779 with the raw L2 norm, and PC1 is a transcriptome-concentration axis, so the candidate measured concentration and called it resemblance to spaceflight. 1 in 10 random flight/ground relabelings beat it on spatial structure, 46.5% under a within-study permutation.
- **kNN tissue-label transfer from OSDR to ARCHS4.** Median best-match cosine 0.964 with 100% of points above 0.7, so no confidence threshold discriminates anything, and the winner beats the runner-up by a median of 0.00089 cosine. 54% of the targets are human samples that would have received mouse labels.
- **Unsupervised k-means cluster id (k=24).**
  Built, run on the real corpus, measured, then deleted along with its precompute stage. 81.9% of the label is recoverable from the 2-D UMAP coordinates alone (15-NN over a 120k sample, against a 12.4% majority-class baseline); a structure-free 24-cell Voronoi null reproduced its spatial coherence to within 1.5 points; seed-to-seed ARI ~0.45; 81% species-pure; explains 80.7% of the raw-L2-norm variance.
  A comment in `manifold/colorby.py` records the decision where someone would add it back.
- **Local UMAP density.** Redundant with the raster already drawn underneath.
  (That raster was removed on 2026-07-22, so this rejection no longer holds either. With every point now drawn live, glyph crowding is itself the density readout, which is a different argument and a weaker one.)
- **PC1-3.** Free but redundant with the axes on screen.
- **GEO series (GSE).** 51,284 distinct values, so a Top-11 legend would color ~3% of the map and dump the rest in "Other".
  Also a pure batch label (333x lift).
  Kept in the parquet for provenance, not offered as a color.

**Methodological note.**
Spatial eta-squared is not evidence: 30 arbitrary random directions in 512-d score 0.874 +/- 0.025 on this UMAP, because the UMAP was fit on those same vectors.
Judge a candidate against a structure-free null of the same *form*, and check whether it is recoverable from the coordinates or from depth.

## Defects found and fixed

Found by adversarial audit, browser-driven testing, and by running against the real corpus. Each was verified before being fixed.

**Still-current code:**

1. The plot occupied 450 px of an ~890 px pane - `dcc.Loading`'s wrapper divs broke the `height: 100%` chain.
2. Segmented controls had no selected state and the dropdown was entirely unstyled: Dash 4 rewrote both components' DOM and class names.
3. The legend search box was inert - no callback read it.
4. The ARCHS4 background cloud showed a hover label despite `hoverinfo="skip"`, because a `hovertemplate` overrides it.
5. pandas 3.0 leaves NA through `astype(str)`, so a phantom NA category reached the legend. The same trap appears in `fetch_archs4_meta.first_series`, where unresolved accessions arrive as float NaN and `value or ""` does not catch them because NaN is truthy; without the explicit isna guard they became the literal string `"nan"`, read as a real GSE, and overstated metadata coverage to a clean 100%.
6. 55 samples were dropped silently during preprocessing; now reported.
7. The expression cache could never hit (it compared eligible keys against kept keys), and resume was keyed on row count alone.
8. The gene-digest gate was skipped on the expression-cache path; the digest is now part of the cache key.
9. **The density ramp used 0.78% of its range.**
   `render_density` normalized `log1p(counts)` by the global max.
   Real occupancy is heavy-tailed - median occupied bin holds 2 points, max 638 - so dividing by the max crushed everything into the bottom of the scale.
   Only 0.78% of occupied bins cleared the 0.5 threshold where the navy-to-teal ramp turns teal, and alpha saturated at 0.4545, *before* that turn, so the densest cores were indistinguishable from merely-busy ones.
   Measured on the raster: 8 pixels total in the teal half.
   Fixed by normalizing against the 99.5th percentile of occupied bins (`DENSITY_CLIP_PCT`) and ramping alpha across the same span with a visibility floor (`DENSITY_ALPHA_FLOOR`).
   Teal-half pixels went 8 -> 2,377; mean occupied colour lifted RGB (21,50,82) -> (26,71,108).
   Filament structure that was a flat wash now reads.
   Added `build_projections.py --density-only` so ramp changes re-render from cached coordinates in seconds instead of repeating the projection build.
10. **`APP_REQUIRED` omitted `points_meta.parquet`**, which `layout.control_rail()` reads first, so a bare cache passed preflight and crashed during startup. It also demanded three artifacts the app never opens.
11. **The renderer ranked categories per layer**, so a category could take two different colors in one figure. Categories are now ranked once over the whole covered population.
12. **Residual traces were emitted last**, so ~308,000 grey glyphs painted over every colored category and the map read as grey even where it was not. Residual categories are now emitted first, receded to 0.26 opacity and 0.82 size.
13. **`lasso2d` was still on the modebar** after the selection feature was designed away, because the config removed only `select2d`.

**In code that has since been deleted** (kept because the reasoning generalizes):

14. The coherence null was a with-replacement bootstrap over a frozen 20k pool, giving a systematic z-bias growing with selection size. Rewritten analytically, then removed with the feature.
15. The lasso never produced a selection: plotly 6 serializes numpy `customdata` as base64 and Dash's event filter indexes the user data, so `customdata` arrived at the server as nothing. The plain-list convention that fixed it still stands for the OSDR hover.
16. The verdict could print "Coherent" next to a z and p saying the selection was looser than a matched random draw. The general lesson survives the feature: a summary sentence must be derived from the same statistics it sits beside, not chosen independently of them.

## Next steps

0. **From the design panel, not yet built.** A judged panel of four independent UI designs was run before this session's interface work; its spec is larger than what was built. The parts deliberately left for later, each with the reason:
   - **The retrieval view showing the manifold neighbourhood around the query**, instead of (or beside) the abstract network graph. This is the strongest idea in the spec and the largest change; it needs a second `dcc.Graph` and careful thought about what the network graph is still for.
   - **A full-corpus score histogram.** `_topk_cosine_from_memmap` already materializes all 940,455 scores and discards them one line later, so the data is free. It would show that the top-k sit in a long thin tail: for the OSD-100 query, the corpus median is 0.788 and the top 20 span 0.997 to 0.993.
   - **The agreement readout**, stating per query how much the map's own ordering agrees with the retrieval's. Measured over 40 random queries: overlap between the map's 20 nearest points and the true cosine top-20 is **mean 2.7 of 20, median 0**. The hover already states both ranks per hit; this would state the summary.
   - **An ARCHS4 point probe** on the map, so any point can be inspected rather than only OSDR points and retrieved hits.
   - **Demoting "AI hypothesis" to "AI reading"** and giving the panel dashed, inset chrome so it does not read as an instrument surface.



1. ~~Wait for `embed_osdr.py`~~ **DONE** 2026-07-21 08:45:51, all 2,108 samples.
2. ~~Run `precompute/build_projections.py`~~ **DONE** 2026-07-21 08:51:44, rc=0 in 5 min 47 s.
3. ~~Validate against the Phase 2/4 criteria~~ **DONE**.
   PC1 = 40.9% against the 57.8% pre-normalization figure, so invariant 2 holds.
   Codified as `precompute/validate_artifacts.py`, which exits nonzero on failure.
4. ~~Launch the app on the real corpus~~ **DONE** 2026-07-21, driven headless end to end against the real 942,563-point cache. Zero console errors.
5. ~~Give ARCHS4 a real biological color-by~~ **DONE** 2026-07-21: the sigpy metadata join, the shared tissue vocabulary, and the coverage-aware registry.
6. ~~Remove the lasso tool and its readout~~ **DONE** 2026-07-21, including the artifacts and dependencies that existed only to serve it.
7. ~~Delete the two dead cache files~~ **DONE** 2026-07-21. `cache/` went from 2.1 GB to 214 MB; the suite and `validate_artifacts.py` both pass without them.
   Neither was source data, so nothing was lost: both were derived from embeddings that are still intact, and `build_hnsw` / `build_population_moments` are recoverable from commit `3840ab3` if fast approximate kNN is ever wanted for an experiment.
8. ~~Strip the stale `cluster_*` keys from `cache/projection_stats.json`~~ **DONE** 2026-07-21, left behind by the cut k-means build.
9. ~~Re-run the browser checks at the other point budgets and measure frame rate~~ **DONE** 2026-07-22.
   All four tiers are driven headless against the real cache and assert the exact glyph count each produces (102,108 / 252,108 / 502,108 / 942,563), re-rendering in 0.1 to 0.3 s.
   The 3-D cap was re-measured rather than inherited: first paint barely moves with glyph count (1.1 s at 42k, 1.9 s at 402k) but a twelve-step camera drag scales linearly (5.6 s, 10.4 s, 18.5 s, 31.4 s at 42k / 102k / 202k / 402k), so 40,000 stays.
10. ~~Review the *visual* quality of the real UMAP map~~ **DONE** 2026-07-22, at the full 942,563 points rather than a 100k sample. Screenshots taken at the default view, an OSDR-only field, 3-D, PCA, and zoomed.
11. ~~Try densMAP now that it is possible~~ **DONE** 2026-07-22, and **rejected on the evidence**.
    Built at full scale (716 s in 2-D, 812 s in 3-D) and scored against the shipped coordinates.
    It loses on both metrics in both dimensionalities: 15-NN recall -41.3% in 2-D and -26.3% in 3-D, tissue purity -8.4% and -2.8%.
    Full numbers and the methodological lesson - the 60,000-point evaluation underestimated the local-fidelity cost by 4.6x - are in the densMAP section above.
12. Optional: switch the metadata fetch to the versioned metadata-only HDF5 files (`human_meta_v2.5.h5` 311.8 MB, `mouse_meta_v2.5.h5` 350.9 MB) if tissue ever needs to be a build **gate** rather than a color.
    That buys exactly 100.000% release-matched coverage for 663 MB and ~8.5 min, against 216 MB and 35 s.
    Not worth 15x the build time for 0.089% of points on a color.
13. Optional: `precompute/embed_osdr.py --metadata-only` if the metadata harmonization ever changes; it rewrites the parquet without re-embedding.

## How the projection build was chained (2026-07-21, completed)

`build_projections.py` was queued behind the in-flight embed run rather than waited on by hand.
A detached watcher polled the embed PID, verified the run actually succeeded, and only then started the projection build with default parameters.
It fired correctly: embed exited 08:45:51, the gates passed, and the build launched 08:45:57 - a 6-second unattended handoff.

The watcher refused to launch unless all of these held, because `build_projections.py` joins OSDR metadata to embeddings *positionally* and a truncated embedding would silently mislabel every OSDR point rather than fail:

- the embed log contains a `[done] wrote` line (clean completion, not a crash or a kill),
- `osdr_sample_embeddings.float32.npy` and `osdr_metadata.parquet` both exist and are non-empty,
- their row counts agree, the embedding dim is 512, and the values are finite.

That gate is worth rebuilding if this is ever re-run, since the failure it guards against is silent rather than loud.
`build_projections.py` now asserts the same row-count agreement itself before doing any work.

**Preflight verified before queuing** (2026-07-21):

- All inputs resolve and are real data, not Git LFS stubs: the ARCHS4 memmap is 963,025,920 bytes, exactly 940,455 x 512 x 2, matching `embedding_manifest.json`.
- Every import `build_projections.py` needs is installed: numpy 2.4.6, pandas 3.0.3, sklearn 1.9.0, umap 0.5.12, pyarrow 20.0.0, PIL 12.2.0.
- Disk 44 GiB free; RAM 17 GB against a measured peak of roughly 2.5 GB.
- **Full-path smoke test passed** end to end in an isolated `MANIFOLD_CACHE_DIR` with synthetic OSDR embeddings and `--archs4-limit 4000`, exercising IncrementalPCA, PCA transform, density rasters, UMAP 2-d and 3-d, and every parquet write. Exit 0.
  Running the real pipeline against a throwaway cache first is cheap insurance and is worth repeating before any long rebuild.

### Built artifacts in `cache/` (measured on disk 2026-07-21)

Full inventory, with which of them the app opens, is `REFERENCE.md` section 12.

| file | size | contents |
| --- | --- | --- |
| `archs4_metadata.parquet` | 32.51 MB | per-GSM GEO metadata + the canonical tissue bucket |
| `coords_pca3.parquet` / `coords_umap3.parquet` | 13.17 MB each | 3-d coordinates |
| `coords_pca2.parquet` / `coords_umap2.parquet` | 8.78 MB each | 2-d coordinates |
| `archs4_geo.parquet` | 4.63 MB | GEO accessions; the join key for the metadata fetch |
| `points_meta.parquet` | 4.36 MB | dataset / src_index / species_id identity table |
| `osdr_sample_embeddings.float32.npy` | 4.32 MB | the 2,108 x 512 OSDR embeddings |
| `osdr_expression.float32.npy` | 127.87 MB | resume intermediate for the multi-hour embed job |
| `osdr_expression_meta.parquet` | 0.097 MB | its metadata sidecar |
| `osdr_metadata.parquet` | 0.027 MB | OSDR labels, joined positionally |
| `density/pca2.png` / `density/umap2.png` | 0.86 MB / 0.61 MB | density underlays (deleted 2026-07-22) |
| `projection_stats.json` | 2.3 KB | variance profile and raster extents (now 14.4 KB: the full 512-component spectrum, no extents) |

Total live cache **219.2 MB**, of which the serving app opens **82.3 MB**.
`embed_osdr.py` cleaned up its own partial memmap and progress JSON on success, as designed.

`joint_cosine.hnsw` (2,070.4 MB) and `population_moments.npz` (4.2 MB) were deleted on 2026-07-21 once nothing produced or read them, which is what took `cache/` from 2.1 GB to 214 MB.

## Dimensionality-reduction evaluation (2026-07-21)

Question asked: is any method beyond PCA and UMAP worth adding?
Ten methods were fitted on an identical deterministic 60,000-point subsample (57,892 ARCHS4 + all 2,108 OSDR), each fed the PCA-50 the pipeline already builds, and scored against the **original 512-d normalized space** rather than against their own input.
The scoring code and every embedding are in the session scratchpad; the metrics are kNN recall at k=15 (local), Spearman rho of pairwise distances (global), Spearman rho of local density (density honesty), 25-NN purity of the shared `tissue` label (biological fidelity, permuted null 0.073), and the percentage of ARCHS4 points sharing a 100x100 grid bin with any OSDR point.

| method | local | global | density | **tissue** | mix % | fit 60k | out-of-sample |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PCA-2 | 0.037 | **0.849** | 0.419 | 0.179 | 42.9 | 0.2 s | trivial |
| UMAP, shipped settings | 0.377 | 0.113 | 0.441 | 0.636 | 8.9 | 46 s | works |
| UMAP, n_neighbors 15 + cosine on raw 512-d | 0.426 | - | - | 0.646 | - | 26 s | works |
| densMAP, dens_lambda 0.5 | 0.344 | 0.136 | **0.739** | 0.609 | 8.6 | 117 s | **none** |
| PaCMAP | 0.348 | 0.284 | 0.457 | 0.611 | 7.8 | - | works |
| LocalMAP | 0.419 | 0.316 | 0.081 | 0.644 | 5.7 | - | works |
| openTSNE, perplexity 30 | **0.581** | 0.290 | 0.444 | **0.668** | 2.2 | 108 s | works |
| openTSNE, perplexity 200 | 0.484 | 0.281 | 0.596 | 0.650 | 6.8 | 12,756 s | works |
| TriMap | 0.001 | 0.609 | 0.144 | 0.095 | 100.0 | - | - |
| PHATE | 0.237 | 0.225 | -0.050 | 0.435 | 14.7 | - | - |

**The cheapest win is not a new method.** Retuning the existing UMAP to `n_neighbors=15` with `metric="cosine"` on the raw 512-d vectors, instead of `n_neighbors=30` with euclidean on PCA-50, raises local fidelity from 0.380 to 0.426 and tissue purity from 0.630 to 0.646, and runs slightly faster.
The two changes compose and both are far larger than seed noise: three seeds per configuration gave a standard deviation of 0.001 to 0.002 on both metrics, so these are 8 to 37 standard deviations, not luck.
This was the control run deliberately, because shipping a "new method" that is really a parameter change would be embarrassing.

**Applied 2026-07-21.** The retune is shipped and the full corpus was rebuilt with it.
On the real 942,563-point map, 25-NN tissue purity over the 853,989 points with a real tissue bucket went **0.6448 to 0.6756 (+4.8%)** against a 0.0761 permuted null, and the OSDR spread ratio went 0.827 to 0.850.
The build cost roughly tripled, from 347 s to 950 s, entirely in `.transform()`; the landmark fit got slightly faster.
`validate_artifacts.py` passes, 144 tests pass, and the 27 browser checks pass against the rebuilt cache.

**If a third method is added, it is openTSNE at perplexity 30 with PCA initialization.**
**Executed 2026-07-23** - see the entry at the top of this file. It shipped exactly as specified here (openTSNE, perplexity 30, PCA init), but fit directly on all 942,563 points rather than through the landmark transform this paragraph anticipated, since the landmark pattern had already been removed from the pipeline by then.
It is the only candidate that beats UMAP on local fidelity (0.581 against 0.426 for the best UMAP) and on biological fidelity (0.668 against 0.646) at the same time, and its global fidelity is 2.6x UMAP's.
Its out-of-sample transform works and preserves *more* structure than UMAP's on the same test (recall 0.534 against 0.472), so the landmark fit-and-transform pattern the pipeline already uses would carry it to all 942,563 points.
Two honest caveats: t-SNE fills the plane as a disc, so whitespace carries no meaning where UMAP's islands at least suggest separation; and it separates the corpora *more* (2.2% shared bins against 8.9%), which cuts against this tool's premise of showing where spaceflight sits relative to Earth biology.
Perplexity 200 is not an option at 3.5 hours for 60,000 points.

**Rejected, with the number that kills each:**

- **densMAP** raises density fidelity from 0.441 to 0.739, fixing a known UMAP lie, and is free - a flag on a dependency already present.
  It was rejected because `umap-learn` raises `NotImplementedError: Transforming data into an existing embedding not supported for densMAP`, so it could not use the landmark pattern and would have needed a direct 942,563-point fit.
  **That rejection expired on 2026-07-22, and densMAP was then run and rejected again on fresh evidence** - see the "densMAP: measured at full corpus scale, and rejected" section near the top of this file.
  Rebuilt at full scale it loses to the shipped UMAP on both local fidelity (-41% in 2-D) and tissue purity, so it is still not shipped, now for a measured reason rather than an untested one.
- **PaCMAP** and **LocalMAP** buy real global fidelity (0.284 and 0.316 against UMAP's 0.113) but PaCMAP is worse than UMAP on both local structure and tissue purity, and LocalMAP destroys density fidelity (0.081). Neither earns a menu slot.
- **TriMap** collapses. Local fidelity 0.001, tissue purity 0.095 against a 0.073 null, and every point sharing a bin with OSDR - the rendered plot is empty.
- **PHATE** has *negative* density fidelity and produces the crescent it produces when there is no trajectory. It is a tool for developmental data being pointed at a heterogeneous grab-bag of GEO.
- **Keeping PCA is validated.** Its global fidelity of 0.849 is 3x the best neighbour embedding, and no nonlinear method comes close. The two shipped methods really do occupy the two ends. (Three since 2026-07-23; PCA still holds the global end.)

**The open question, now answered (2026-07-21).** The premise was wrong.
The pre-normalization L2 norm is not sequencing depth: the encoder's input is log1p-TPM, which is depth-normalized by construction, and measured against the exact OSDR expression matrix the norm correlates r = +0.987 with the share of expression held by a sample's top 100 genes and r = -0.930 with Shannon entropy.
It is a transcriptome-*concentration* axis, and the tissue ordering is the textbook one: liver 13.57, skeletal muscle 12.92 and heart 12.62 at the top, brain 8.31 and skin 7.84 at the bottom.
So the axis should not be projected out - it is biology, and 26.2% of its variance is explained by tissue identity alone.
Projecting it out does not work in any case: removing the single best-fitting direction moves a probe for it from held-out R^2 0.977 to only 0.975, because the signal is spread across many directions.
L2 normalization stays, for the reason that it removes a redundant encoding of something the direction already carries rather than because it removes an artifact.
Full measurement in `REFERENCE.md` section 4.

## Notes and risks

- **The cross-corpus batch effect is real and measured exactly (2026-07-21).**
  Controlling for both study and tissue, OSDR samples that share neither still neighbour each other **54x above chance** (11.491% observed against 0.21101% expected).
  Tissue is the dominant axis of bulk expression, so biology cannot explain it - this is the fp32/CPU versus bf16/CUDA precision and preprocessing difference.
  This remains load-bearing: cross-corpus distances are not trustworthy at face value.
  The standing caution that used to sit on the control rail was removed from the UI on 2026-07-22 at the user's request (over-explaining microcopy), so the fact now lives in the docs (README, `REFERENCE.md` section 4, `CLAUDE.md`) rather than on the rail.
  Full numbers in `REFERENCE.md` section 4.
  Re-check with `/Users/josh/Bridge-RNA/.venv/bin/python precompute/validate_artifacts.py --mixing`, which warns above 50x.
- The mixing check is the only thing left that opens the memmap, and it is opt-in. Everything else in the serving path reads `cache/` and nothing else.
- 839 ARCHS4 samples (0.089%) carry tissue `Unknown` because the newer release the API serves dropped them.
  They are not guessed at.
  If tissue ever becomes a build gate, switch to the versioned metadata-only HDF5 files and assert 100%.
- UMAP quality at 940k via landmark fit-and-transform ran clean, but *visual* quality on the real map is still unreviewed.
- `tests/` never touches the real data, so the suite stays fast and runs on a machine with neither the memmap nor the checkpoint.
