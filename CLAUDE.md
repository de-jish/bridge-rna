# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

For interface work, also follow [AGENTS.md](AGENTS.md). Every agent must load both
**taste** (`design-taste-frontend`) and **impeccable** before UI/UX planning or
implementation and apply both during final review. Preserve the scientific
workflows and use contextual disclosure to simplify secondary information.

## What this repo is

**Bridge RNA is one application with two views**, developed by the Space Biosciences Research Branch at NASA Ames Research Center.

- **Retrieve** (`/`) takes one NASA OSDR spaceflight RNA-seq sample and returns its closest Earth analogs out of the 940,455-sample ARCHS4/GEO index, as a network graph with an inspector and an optional LLM summary. Code in `bridge_rna/`.
- **Map** (`/map`) zooms out to the whole space: the 512-d ExpressionPerformer embeddings of both corpora - OSDR (2,108 NASA GeneLab spaceflight samples) and ARCHS4 (940,455 GEO samples) - reduced into one shared 2D/3D space, every one of the 942,563 points drawn as a live WebGL glyph and colored by biology. Code in `manifold/` and `precompute/`.

`app.py` is the single entry point and owns the header and the router.
**The router must decline to repaint a route that is already on screen.** `serve_layout` paints the requested view server-side, but `dcc.Location` publishes `pathname` once it mounts and Dash reads that as a change, so `prevent_initial_call` alone let the router rebuild the view on every load. Rebuilding the retrieval view reads the OSDR catalog, so its response landed a few hundred milliseconds later, on top of whatever the user had done meanwhile - clicking Cohort on arrival opened the cohort panel and then closed it again, 6 times out of 6, while the click's own callback had returned the right answer. `route-store` records what was painted and `app.navigation_for` is the decision, split out so it is testable without Dash plumbing. There is no `app_osdr_dash.py` and no `app_manifold.py`; both were deleted when the two repositories merged on 2026-07-22, and the map's 19 commits are in this history.

**Current state: built, run on the real corpus, and tested.** 399 tests pass in about twenty-seven seconds, plus 363 browser checks: 129 in `tests/e2e_check.py`, 70 in `tests/e2e_upload_check.py`, and 164 in `tests/e2e_cohort_check.py`.
Each browser suite counts and prints what it actually ran, because the documented totals were hand-written and had drifted.
The ARCHS4 GEO metadata join is built (`cache/archs4_metadata.parquet`, 940,455 rows, 51,284 distinct GEO series), so the map colors by tissue across both corpora rather than by species alone.

### The join between the halves, and why retrieval is fast

The two halves address the same things by the same keys, and nothing translates between them:

- An OSDR sample is `"<accession>|<sample name>"` on both sides - `bridge_rna.osdr.load_osdr_samples` builds it as `sample_id`, `precompute/embed_osdr.py` writes it as `sample_key`. Pinned by a test.
- An ARCHS4 hit's memmap row **is** its point index on the map, because ARCHS4 occupies rows `0..940,454` of the global point order. Also pinned by a test.

That is what makes `bridge_rna/retrieval.py`'s **cached path** possible, and it is the most valuable thing the merge produced.
The manifold precompute had already embedded all 2,108 eligible OSDR samples (gene-digest gated, preprocessing checked bit-for-bit against the retrieval's own single-sample path) and already joined GEO metadata for all 940,455 ARCHS4 samples.
So a query needs no subprocess and no network: **0.8 s against 22.1 s**, with `gse`/`title`/`tissue` populated where the subprocess path returned empty strings, and identical scores to six decimal places.

`search_hits` returns `(hits, mode)` where mode is `cached`, `precomputed`, or `demo`, and **the interface must always say which ran**.
It did not, once: the status banner special-cased only `precomputed`, so every cached result was announced as "real demo script output".
The banner label now comes from one shared `_retrieval_phrase(mode)` in `bridge_rna/callbacks.py`, so a new path names itself there rather than in a callback body.

**A fourth path exists for a sample the corpus never embedded: file ingestion (mode `"uploaded"`).**
A user can upload an OSDR counts file, have it embedded live, and get the identical output scored against the same ARCHS4 index.
It is a fourth *query-vector source*, not a new pipeline - the cosine scan, `_annotate_from_cache`, and the `archs4_index` map join are the cached path's, reused unchanged, so an uploaded hit carries the same schema as a cached one.
`bridge_rna.retrieval.run_uploaded_retrieval` is the entry point; it embeds via `bridge_rna.retrieval.embed_uploaded_counts`, which shells out to `precompute/embed_upload.py` because **the serving app never imports torch** - the same reason the `demo` path is a subprocess.
`embed_upload.py` reuses the exact preprocessing symbols from `manifold/bridge_rna.py` and enforces **invariant 1 (the gene-digest gate)** before producing any vector, so an uploaded sample is embedded in the same gene order as the corpus or the embed is refused.
Verified: embedding an OSDR sample's own counts through this path reproduces its precomputed cached vector at cosine 1.0, max abs diff 0.0 (`tests/test_upload_ingestion.py`).
Input is mouse Ensembl-indexed counts (OSDR is Mus musculus); a file mapping zero orthologs is rejected, never embedded into a meaningless vector.
**No metadata is required or accepted** on this path, and none would change anything: the query vector is a pure function of one counts column, so tissue, flight-vs-ground, and accession would only fill the inspector and the summary prompt, never a hit or a score.
Design doc: [`docs/design-notes.md`](docs/design-notes.md#file-ingestion). Format contract and a real working input: `examples/README.md` and `examples/osdr_upload_example.csv`.

**A fifth path pools a whole experimental group into one query: cohort retrieval (mode `"cohort"`).**
A spaceflight study does not have one sample, it has a group, and a single-sample top-5 is not a stable measurement: two replicates from the same cage share only **0.161** of their top-5 hits, because the entire top-500 of a 940,455-sample index spans a cosine range comparable to the gap between two animals in the same cage.
Pooling raises leave-one-out top-5 agreement to **0.738**, a **4.6x** gain, and that - not outlier protection - is the case for the feature.
`bridge_rna/cohorts.py` is the only file that knows what a cohort is, and it deliberately touches no embedding and no memmap: it is metadata grouping plus 512-d arithmetic, testable against the fixture corpus on a machine with neither artifact.
`bridge_rna.retrieval.run_cohort_retrieval` is the seam, and it reuses the cached path's scan, `_annotate_from_cache` and `archs4_index` join unchanged, so a cohort hit carries the same schema as a single-sample one and costs one memmap pass at any k - a pass that also measures the result's stability, by scoring `2k+1` query vectors instead of one.
Design and every measurement: [`docs/design-notes.md`](docs/design-notes.md#cohort-retrieval); the prior measurement that specified it: [`docs/design-notes.md`](docs/design-notes.md#cohort-pooling).

Four things about it are load-bearing.

**A cohort is `(study, tissue, spaceflight arm)` by default, and study is pinned.** That is OSDR's own curated ISA-Tab factor grouping: 212 cohorts with two or more members across 70 studies, median size 10, max 38, covering 2,105 of 2,108 embedded samples. Tissue or arm can be removed to merge cohorts, but **study can never be unticked**, because random samples from one study already reach 0.9805 mean pairwise cosine against 0.9933 for a real cohort - pooling across studies would average across the corpus's strongest batch boundary. Six further facets (sex, strain, genotype, habitat, duration, diet) were offered and **were removed on 2026-08-05, and must not be reintroduced without an argument that answers this**: every one of them could only *split* a cohort, and stability rises steeply with size (measured over all 212 cohorts: 0.34 at k=2 against 0.86 at k>=15), so they traded away the exact quantity the feature exists to buy in return for a contrast the two-arm comparison answers attributably. The columns remain in the parquet; re-adding one is a single line in `FACETS` and nothing else hard-codes a facet. They are no longer map color-bys either - those nine went on 2026-08-12, for a different reason recorded below. The arm is the raw OSDR value rather than the binary Flight/Ground collapse, because a basal animal was sacrificed at experiment start and a vivarium animal never entered flight hardware.

**Each member is L2-normalized before averaging.** This is invariant 2 applied one level up: the raw norm is a transcriptome-concentration axis spanning 3.9x, not a nuisance scale, so averaging raw vectors would let the most concentrated members cast the loudest votes. It is also the maximum-likelihood vMF mean direction, which makes ranking by the pooled vector exactly ranking by the unweighted mean of the members' own cosines - one animal, one vote. Measured, it changes almost nothing *here* (median cos 0.9999995 against the raw mean, since within-cohort norm spread is 1.09x) and it is kept because it becomes correct the moment anyone unticks Tissue.

**Result stability is measured on the query that just ran, and it lives on the right, after the search.** This changed on 2026-08-06 and the previous design must not come back. It used to be `STABILITY_BY_K`, a bucketed curve of leave-one-out top-5 agreement against cohort size measured over all 212 cohorts, quoted on the **rail** the moment a cohort was selected. The number was real and the label was accurate, and it was still misleading where it sat: a population average printed beside one cohort's name is read as a property of *that* cohort, and the spread inside a size bucket is most of the range. Measured live, a cohort of 7 scores **0.316** and a cohort of 6 scores **0.849**, and the curve told both of them 0.72; OSD-137's two 6-animal liver arms measure 0.59 and 0.64 against the same 0.72. So `retrieval.run_cohort_retrieval` now scores every leave-one-out pool **and** every member alone in the same memmap pass, and `cohorts.StabilityMeasurement` reports the mean Jaccard overlap between the list on screen and the list each member's absence would have produced, **at the depth the slider is on** rather than at a fixed 5. The single-sample baseline beside it is measured the same way, on the same cohort, in the same pass, which is what replaced the fixed `SINGLE_SAMPLE_STABILITY = 0.16`. Design, measurements and rejected alternatives: [`docs/design-notes.md`](docs/design-notes.md#live-stability).

Four things about it are load-bearing. **The rail states only what is true before a search** - the role, the name, and how many samples are about to be pooled - because a measured number does not exist until the query runs, and putting one under the picker would mean either a memmap pass per dropdown change or the previous cohort's figure sitting under the current cohort's name. **The panel is above the inspector, not inside it**, so clicking a hit does not scroll away the number describing the whole result. **A comparison measures both arms separately**, since an overlap of 0.25 between two arms at 0.86 is a different finding from 0.25 between one at 0.86 and one at 0.31. **It costs one memmap pass**: 2k+1 query vectors are built before the index is touched and scored together, measured at 0.44 s for 1 query and 1.00 s for the 77 a 38-animal cohort needs.

**A comparison draws its two arms as even columns with their rows aligned, and it was stacked until 2026-08-06.** Stacking gave the two arms visibly unequal treatment: measured on the real app, the panel held 456 px of content in 445 px, so cohort A rendered complete and cohort B's last row - the one naming the animal whose absence moves the result furthest - was clipped by its own fold at every viewport tested, 7.8 px at 1680x1050 and 65.6 px at 1280x800. Side by side the panel holds 354 px, nothing is clipped, the two arms are equal by construction rather than by coincidence, and **0.89 sits on the same baseline as 0.94** rather than 160 px below it, which is the comparison the panel exists to support. Four things hold it up. The alignment is `grid-template-rows: subgrid` with **rows addressed by class, never by child order**, because either of the last two rows can be missing from either arm and counting children would let cohort B's flag land in the row holding cohort A's member name. The tracks are `minmax(0, 1fr)`, since plain `1fr` floors at min-content and a 39-character mono sample key would push one column wider than its twin. **"differs by *facet*" moved to the panel header**, because it describes the pair rather than either arm and hanging it under B's letter started B's name a line below A's. And the "moves it most" row is **always drawn**, saying "every member equally" when `weakest_member` is None, because an absent row and a clipped row look identical and a clipped row on that exact line is what this fixed. The low-stability flag is deliberately *not* equalized: a counterpart badge for a healthy arm is the grade `R̄` was deleted for being. Measurements, the rejected alternatives, and the `flex: 1 1 0` fix that removed the last 3-11 px of clipping: [`docs/design-notes.md`](docs/design-notes.md#stability-panel-even-split).

`LOW_N_THRESHOLD` is 5 and survives, because that question - how large should a cohort be - is the one the curve can answer honestly, and the picker is the one place where size is all that is known. `precompute/validate_cohorts.py` check 5 still measures the curve and now **fails** if the knee moves away from 5, but only on a full sweep: on a `--cohorts 40` sample the knee lands at 10, from buckets standing on one cohort each, which is exactly the noise the bucketing exists to survive. `cohorts.STABILITY_FLOOR` is that same 0.70, applied to the measurement rather than to size standing in for it.

`R̄`, the vMF resultant length labelled "Group tightness", was **removed on 2026-08-05** for a related reason: across all 212 real cohorts its median is 0.9991 and it is no lower at k=2 than at k=30, so it never separated a trustworthy group from an untrustworthy one, while a number pinned within a thousandth of its maximum reads as a grade. `resultant_length` is deleted rather than left unused; the measurement survives in [`docs/design-notes.md`](docs/design-notes.md#cohort-pooling). The **per-member** leave-one-out cosine stays in the member list, because that one varies and names an animal - and the panel names the animal whose absence moves the result furthest for the same reason.

**The two-arm comparison runs two independent pooled queries and reports their overlap. It is never a difference vector.** `centroid(flight) - centroid(ground)` is not a transcriptome, so cosine-ranking an index of profiles against it is a category error, and the corpus-level version was already built and rejected (r = -0.990 with PC1, beaten by one in ten random relabelings). Only siblings differing in **exactly one facet** are offered, so the reported Jaccard overlap is attributable to that facet.

The map draws **every pooled member**, not one point: a cohort's query vector is a mean, no projection was fit on it, and inventing a coordinate for it would be a lie. `manifold/callbacks._retrieval_overlay` reads `member_ids` from the payload for that.

**A comparison draws both cohorts, and the two channels it uses are not interchangeable.** Until 2026-08-05 the map drew cohort A alone and said nothing about the other arm, which is an omission rather than a limitation - B's hits already carried `archs4_index`. `_retrieval_overlay` now returns a `cohorts` list with the flat `hit_points`/`query_points`/`query_point`/`query_label` keys as the **union** across it, which is what lets `_frame_for` and the retrieval summary handle a comparison without either learning what one is. **Hue tells you which cohort a member belongs to; ring shape tells you which cohort retrieved a hit, and the rings stay white.** That is measured: the comparison network's blue, warm and teal score 1.03, 1.00 and 1.07 to 1 against the worst categorical tissue hue on `PLOT_BG`, which is the same finding `theme.py` records as the reason a hit ring is white at all. Both hit symbols (`circle-open`, `square-open`) are in `Scatter3d`'s vocabulary, so 2-D and 3-D encode a hit identically. **A shared hit is emergent, never computed** - it is in both hit lists so it receives both traces and draws as a ring inside a square, which is why the map's shared count cannot drift from the banner's. Only the *hover* reads across the arms, and it has to: two traces sit at one coordinate and Plotly resolves one tooltip per position, so building each cohort's rows from its own hit list alone left one arm's rank and cosine unreachable on exactly the points a comparison exists to show - while `render.py` justified dropping the rank numerals on the grounds that the hover said more. It does now. The `show-retrieval` tick becomes one per cohort, and framing follows the ticks. Section 9 of [`docs/design-notes.md`](docs/design-notes.md#cohort-retrieval) carries the measurements and the rejected alternatives (a convex-hull footprint, connecting polylines, a dedicated shared hue).

**The two views agree that teal is cohort A and warm is cohort B.** `GRAPH_THEME` gained `cohort_a`/`cohort_b`/`cohort_shared`, and cohort A was swapped to teal with "retrieved by both" taking blue. That fixed a live inconsistency: teal is the query star in the single-query network and the query mark on the map, so running a comparison used to silently recolor the star the previous search had drawn teal.

**Cohort B's hex cannot agree across the views, so the binding is its name.** It is `#d9791b` in the retrieval network and `#ffc233` on the map, and neither can take the other's place: `#ffc233` measures about 1.8:1 on white, and `#d9791b` sits 0.3 dE from `CATEGORICAL[3]` under deuteranopia, unusable beside eleven tissue hues on navy. Both measurements are in `manifold/theme.py`. So every surface prints the cohort's own **name** beside its own mark - the two rail cards, the network's band labels, the map's per-cohort tick, and the map's key rows - and nothing asks the reader to carry a hue across the two views. Cohort A survives the trip unchanged at `#0bab9f`, which is what made the difference read as a different arm rather than a different surface.

**A comparison runs two pooled queries, so both surfaces describe two.** Arming one used to leave the second query's size stated nowhere while both the network figure and the map gave it a color and drew it. That is not symmetry: an overlap of 0.25 between a 12-animal cohort measuring 0.86 and a 2-animal cohort measuring 0.31 means something different from the same 0.25 between two cohorts of twelve, and the number that decides how much of it to believe was not on screen. Each pooled query now carries a card under its own picker - the selected cohort under the Cohort dropdown, the sibling under "Compare against" - which is the rail's standing rule and keeps the member ticks, which belong to cohort A alone, beside cohort A's card. The role line (`● COHORT A`, `● COHORT B · differs by spaceflight arm`) appears **only when there are two**; a lone cohort gets no letter, because there is nothing to distinguish it from. Its dot and left rule are `--accent-teal` and `--accent-warm`, which *are* `GRAPH_THEME`'s `cohort_a` and `cohort_b`.

`precompute/validate_cohorts.py` is the honesty gate and must keep passing: identity, leave-one-out stability, a structure-free null, a within-study null, the stability curve, and spherical-versus-raw normalization. It scores 9,270 query vectors in one 73-second memmap pass. Its identity check is worth reading before touching the estimator: pooling one sample perturbs the query vector by one float32 ulp, which is enough to permute ranks at an **exactly 0.0** score gap, so the gate is float32 score agreement plus an identical top-20 rather than an identical top-100.

An uploaded file is **staged**, not merely written: `bridge_rna.callbacks._upload_dir` owns one directory per process, a session's previous file is unlinked when its next upload arrives, and a directory abandoned by a killed process is reaped by the next run. The reaping is PID-tagged rather than signal-based because `atexit` does not run on SIGTERM or SIGKILL and a signal handler is not available either - uploads arrive on Dash's request threads, and `signal.signal` may only be called from the main one. Before this, every upload leaked its counts matrix into the system temp directory forever.

### Two retrieval tiers, not three

`bridge_rna.retrieval.sample_tier` classifies every OSDR sample the picker lists. Measured on the shipped metadata with the cache built:

| tier | count | behaviour |
| --- | --- | --- |
| `cached` | **2,108** | precomputed vector, ~0.5 s, and has a position on the map |
| `subprocess` | **0** | nothing reaches it while the cache exists |
| `unavailable` | **788** | no path can serve it |

**The zero is the point, and it took two attempts to get right.** An earlier version of `sample_tier` checked only whether a sample's name appears as a column in its counts matrix, and reported 717 samples as retrievable-but-slow. They are not. `demo_osdr_top5.py` filters its metadata to rows *with a recorded spaceflight value* before it looks the name up (`demo_osdr_top5.py:337-357`), so 733 of the 788 raise "not found after filtering", and the remaining 55 pass that filter but match no column.

Both failures were reproduced end to end: `OSD-141|Mmus_C57-6J_SPL_cells_Rep1_SP1` fails in 4 s, `OSD-462|RR10_KDN_WT_BSL_B11` in 2.3 s.

So with the cache present, every sample that can be retrieved at all is retrieved in half a second. `TIER_SUBPROCESS` is not dead code: without the cache those same 2,108 fall to it at about 22 s each.

Unavailable samples are shown **disabled with the reason** rather than hidden, and the picker never defaults to a disabled option.

Three features that appear in older prose are **gone and must not be reintroduced as current behavior**: the lasso selection tool and its 512-d statistical readout (with `manifold/coherence.py` and the right-hand readout panel), the hnswlib ANN index and population-moment artifacts that existed only to serve it, and the precomputed density raster underlay (with its PNGs, its `--density-only` flag, and the Pillow dependency).

**The nine OSDR-only color-bys are gone too, as of 2026-08-12** - Flight vs Ground, Spaceflight arm, Strain, Sex, Genotype, Study, Habitat, Mission duration, Diet - along with `data._flight_status`, the derived column it wrote, and the `_osdr_field` resolver factory.
None of them separated anything: they colored 2,108 of 942,563 points, scattered through a corpus whose structure is set by the 940,455 they sit among.
`colorby.REGISTRY` is Tissue and Species, and both color the whole map.
**The coverage machinery is not vestigial and must not be simplified away with them.** Tissue *becomes* an OSDR-only field on a machine that never fetched `cache/archs4_metadata.parquet`, which is the state a fresh clone starts in - and on that path the scope note, the amber partial bar, `GROUP_OSDR`, and the renderer's faint context cloud are all still live. That is invariant 5, and degraded Tissue is now the only route to it, so the tests assert it there rather than through a deleted field. Measurements and the fixture consequence: [`docs/design-notes.md`](docs/design-notes.md#osdr-only-color-bys).
Where that history is instructive it is recorded as history, clearly marked.

Two approximations that appear in older prose are also gone: PCA is no longer fit on a 60,000-point subsample and UMAP is no longer a landmark fit plus `.transform()`.
**All three reductions are now fit on every one of the 942,563 points.**
The cost that justified the approximations turned out not to exist, and both measurements are in `REFERENCE.md` section 4.

**There are three projections, not two.** t-SNE joined PCA and UMAP on 2026-07-23, built with `openTSNE`, and prose saying "both reductions" or "the two shipped methods" predates it.
It is not a new idea: `progress.md`'s 2026-07-21 evaluation scored ten candidates and concluded that if a third method were ever added it should be openTSNE at perplexity 30 with PCA initialization, which is exactly what shipped.

**UMAP's `n_neighbors` went back to 30 on 2026-07-23**, reversing the 15 that section "UMAP settings, chosen by measurement" argued for.
The reason is recorded rather than quietly overwritten, and it is not the one it first looked like.
Scored on the real corpus with `--quality --compare`, **30 beats 15 on both metrics in both dimensionalities** (umap2 recall 0.3955 to 0.4140, purity 0.5838 to 0.6014), so there was no local-for-global trade to weigh; 15 was simply worse.
The flaw was that 15 was chosen by fitting candidates on a **60,000-point subsample**. `n_neighbors` is a density parameter, so fifteen neighbours out of 60,000 covers roughly sixteen times as much of the manifold as fifteen out of 942,563, and the number could not mean the same thing in both places.
**A hyperparameter that scales with corpus density cannot be tuned on a subsample of that corpus.** That is the transferable lesson, and it is why `--compare` against the previous cache is now the thing to run before accepting a projection change.

## Read these first

- `IMPLEMENTATION.md` - the master plan: architecture, design decisions, tradeoffs, build order.
- `REFERENCE.md` - verified ground-truth facts (model config, gene digest, embedding stats, measured timings, reusable Bridge RNA interfaces with line numbers, color-by columns, theme tokens). Every fact was checked directly against the checkpoint/memmap/data, not from docs. Trust it over inference.
- `progress.md` - living status log, decisions made, open questions. Keep it updated after every meaningful change (per Josh's global convention).

When plans change, update these docs so they reflect what was actually built, not just the initial intent.

## Architecture: the offline/online split is the load-bearing decision

Everything expensive is precomputed once and cached; the app only ever loads artifacts.
UMAP over 942,563 points is a job measured in tens of minutes and gigabytes, and t-SNE over the same points is measured in hours, which is not something to do inside a callback, and it is what keeps the app responsive.
Do not move model inference, UMAP, or t-SNE into the serving app.

```
OFFLINE (precompute/, run once -> cache/)        ONLINE (app.py /map, loads artifacts only)
embed_osdr.py        -> osdr embeddings npy       coord parquets + points_meta + osdr_metadata
build_projections.py -> pca/umap/tsne coord pqs   + archs4_metadata + projection_stats
                     -> points_meta identity      renders every point as go.Scattergl
                     -> archs4_geo sidecar         colorby.py decides coverage; render.py draws it
fetch_archs4_meta.py -> archs4_metadata parquet
   (HTTP JSON API, ~35 s, no HDF5 download)
validate_artifacts.py -> exit code, gates a build
validate_cohorts.py  -> exit code, gates cohort pooling
   (6 checks over all 212 cohorts, one memmap pass)
```

The **map view** opens no embeddings, computes no statistics, and never touches a Git LFS object.
The retrieval view does: its cached path scans the 963 MB ARCHS4 memmap, an LFS object, on every search. That is the one place in the serving process that touches one.
`BRIDGE_RNA_ROOT` is needed to *build* the cache, not to run the app.
The whole live cache measures 217.8 MB, of which the app opens 80.8 MB; the rest is the embedding intermediates that make a re-embed cheap plus the accession sidecar the metadata fetch joins onto (`REFERENCE.md` section 12).

### Package layout

```
app.py                   the only entry point: header, router, both views on :8050
bridge_rna/cohorts.py    what a cohort is: three facets, grouping, the vMF
                         estimator, leave-one-out cosines, low-N tiering. Opens
                         no embedding and no memmap.
bridge_rna/              the retrieval half (config, util, preflight, osdr, ai, geo,
                         figures, retrieval, panels, layout, callbacks)
manifold/paths.py        every artifact path, one place; env-overridable
manifold/preflight.py    PRECOMPUTE_REQUIRED vs APP_REQUIRED, LFS pointer guard
manifold/data.py         cached loaders: coords, points_meta, osdr_metadata, archs4
                         tissue, and the projection_stats build record the rail reads
manifold/tissue.py       the shared tissue vocabulary (canonical_tissue, coalesce_tissue)
manifold/colorby.py      the coverage-aware color-by registry; the map's honesty layer
manifold/find.py         an identifier -> the points it names, and the
                         completions of one being typed. Four grammars, no
                         free text, no Dash, no embedding
manifold/render.py       layered figure build (ARCHS4 cloud, OSDR overlay)
manifold/sampling.py     stratified + viewport-aware ARCHS4 subsampling
manifold/layout.py       two-column shell, control rail, floating legend, the
                         METHOD_LABELS registry and the projection-parameter readout
manifold/callbacks.py    controls -> figure, zoom -> level of detail, coverage readout
assets/find-suggest.js   the find box's combobox behaviour: keyboard, open/closed,
                         scroll-into-view. The map's only JavaScript, because a
                         keystroke is the one thing Dash cannot express
manifold/theme.py        chrome tokens, dark plot canvas, validated categorical palette
manifold/bridge_rna.py   thin import shim for the reusable Bridge RNA functions
```

`umap`, `openTSNE`, `pynndescent`, and `sklearn` are imported only by `precompute/`, and nothing in the serving path touches them.
`tests/test_app.py::test_the_serving_app_does_not_import_the_scientific_stack` pins that by parsing module-scope imports, and `openTSNE` is in its list.
`PIL` is no longer a dependency of anything, because the density raster was the only thing that used it.

The map's dependency surface is still `dash`, `plotly`, `numpy`, `pandas`, `pyarrow` and nothing scientific - it draws precomputed coordinates and opens no embeddings.
The retrieval half adds `requests` (`bridge_rna/ai.py`, `bridge_rna/geo.py`) and reaches `torch` **only** on the `demo` path, through a subprocess, plus a few lazy `import torch` calls inside `bridge_rna/preflight.py` that read the checkpoint's config.
Nothing imports `torch` at module scope, so the app starts without it and the map works on a machine that has no model at all.

## Non-negotiable invariants

These are correctness gates, not style preferences.
Violating them produces output that looks fine but is scientifically wrong.

1. **Gene-digest gate.** `embed_osdr.py` must compute `canonical_gene_order_digest(genes)` and assert it equals `CANONICAL_GENES_SHA256` (`3f887ac8d329dce3c54d26448964904c07a345940cd3d9ebab18dd1f603194c5`). Abort the build on mismatch. An embedding built with the wrong gene order is silently invalid.
2. **L2-normalize before any reduction.** Raw ARCHS4 vectors are NOT normalized (norms 6.7-26.4, a 3.9x spread); unnormalized, PC1 captures 57.8% of variance and is a magnitude axis. The real normalized build lands at PC1 = 41.3% (exact, over the whole corpus; the 60,000-point subsample the earlier build used reported 40.9%), and `validate_artifacts.py` fails if it drifts back above 50%. **Do not describe that magnitude as sequencing depth** - the docs said so for a long time and it is wrong. The encoder's input is log1p-TPM, which is depth-normalized by construction, and the norm was measured on OSDR against the exact matrix that produced each embedding: it correlates **r = +0.987** with the share of expression held by the top 100 genes and **r = -0.930** with Shannon entropy. It is a transcriptome-*concentration* axis, and it is biology: liver 13.6, skeletal muscle 12.9 and heart 12.6 sit at the top, brain 8.3 and skin 7.8 at the bottom, which is the textbook ordering. Normalizing is still right, because a 3.9x magnitude spread would otherwise dominate the map, but it removes a redundant encoding rather than an artifact - the norm stays recoverable from the normalized direction at held-out R^2 = 0.977. See `REFERENCE.md` section 4.
3. **Read model hyperparameters from `ckpt['config']`, not the demo's fallback constants.** The demo defaults differ from the true trained config.
4. **Verify Git LFS pointers resolve before any run that opens one.** The checkpoint and memmap are LFS objects and can arrive as stub pointers. `manifold/preflight.py` guards `precompute/`; the **map view** opens no LFS object at all, which is why `PRECOMPUTE_REQUIRED` and `APP_REQUIRED` there have no overlap. The **retrieval view** does open the memmap on every cached search, and is guarded separately by `bridge_rna.preflight.preflight_retrieval_requirements`, whose LFS-pointer check runs at layout time and raises the setup banner. Keep `APP_REQUIRED` to what the app genuinely opens, in the order it opens it - `points_meta.parquet` is first because `layout.control_rail()` reads it through `data.counts()` while the layout is still being built.
5. **A color-by must never render a corpus it does not describe as though it were a category.** Coverage is declared in `manifold/colorby.py`, stated in the UI, and enforced in `manifold/render.py`. A field that does not describe ARCHS4 must draw those 940,455 points as a deliberately faint context cloud in a single color that is not in the categorical palette (`theme.ARCHS4_CONTEXT`), at 0.35 opacity, with no legend row. The failure this prevents is a uniform grey glyph cloud over 99.8% of the map, which reads as "ARCHS4 was measured and has no structure here" rather than "this field says nothing about ARCHS4". This used to have a second branch, in which the ARCHS4 layer drew nothing at all and a precomputed density raster carried the shape; the raster is gone, so the context cloud is now the only answer and it applies in 2-D and 3-D alike.
6. **Both corpora share one tissue vocabulary.** `manifold/tissue.canonical_tissue` is the only entry point, used by `precompute/fetch_archs4_meta.py` for ARCHS4 and by `manifold/data.osdr_tissue` for OSDR. Two tissue color-bys wearing one name would each leave the other corpus grey, which is invariant 5 by another route.
7. **The parameter readout must describe the coordinates on screen, never the code's defaults and never a record that outlived its artifact.** `manifold/layout.projection_params` reads `cache/projection_stats.json` through `data.projection_stats()` and nothing else, because duplicating the constants into the serving app would let the rail stay confident while the cache went stale - the same failure as the retrieval banner that announced every cached result as subprocess output. A key the record does not carry drops its chip; it never renders a blank slot or a guess. **Reading the record is necessary and not sufficient**, because every build stage saves its stats *before* the next stage runs: an interrupt inside the 3-D t-SNE fit, which is 81% of the wall clock, leaves a complete `tsne_*` record beside a missing `coords_tsne3.parquet`. So the readout is also gated on `data.coords_available(method, dims)` and says nothing when that exact coordinate set is absent. Note the deliberate asymmetry with `data.method_available`, which stays 2-D-only: a build that produced a 2-D map should still offer its pill.

## The three projections

`manifold/data.METHODS` is the registry and `manifold/layout.METHOD_LABELS` is the order the rail offers them in.
Adding a fourth is one line in each, plus a stage in `build_projections.py`; the pills, the disabled state, the default, the validator's coordinate walk, and two test loops are all derived rather than hand-listed, which is the point.

| method | what it is | fit cost on 942,563 points |
| --- | --- | --- |
| PCA | exact eigendecomposition, linear, honest about global magnitude | ~6 s |
| UMAP | `n_neighbors=30`, `min_dist=0.1`, cosine, spectral init | ~14 min |
| t-SNE | openTSNE, perplexity 30, cosine, PCA init rescaled to std 1e-4 | hours, almost all of it 3-D |

**UMAP starts from a spectral layout, and this is load-bearing for the species split.** A PCA start (shipped 2026-07-22 to 07-23) scatters the two species into many interleaved islands; a spectral start consolidates them into two territories, worth 13x the species silhouette on the real corpus at no cost in local fidelity. The reason spectral was ever abandoned - a 7.31 GB Lanczos basis - was an artifact of UMAP's `sqrt(n)` default `ncv`, not of the eigenproblem: `umap_init_from_spectral` solves for the 3-4 eigenvectors actually wanted with `ncv=32` and a shifted operator in 20 s and 241 MB. `--umap-init pca` reproduces the old build. This is the kind of global-structure choice `validate_artifacts.py --quality` cannot see, since both its metrics are local, so it is judged by the species silhouette in `REFERENCE.md` instead.

Three things about t-SNE are worth knowing before touching that stage.

**Its neighbour graph is not UMAP's, deliberately.** t-SNE wants `3 * perplexity` neighbours where UMAP wants `n_neighbors`, so each builds its own through the same `build_knn` call rather than sharing a padded one. Slicing a k=90 graph down to k=30 is not the graph NN-descent would have built at k=30, and the graph is the artifact every coordinate derives from.

**The self-column slice is load-bearing.** pynndescent returns each point's own index in column 0; openTSNE's own index strips it before returning, and `PrecomputedNeighbors` passes through whatever it is handed. So the graph is built at `3 * perplexity + 1` and sliced with `[:, 1:]`. Leaving self in would hand every point a zero-distance neighbour, which is not what a perplexity of 30 means.

**2-D and 3-D are built by different algorithms, and the UI says so.** openTSNE's interpolation accelerator refuses more than two output dimensions outright, so 2-D is FIt-SNE and 3-D is Barnes-Hut. That is why `projection_params` takes `dims`, why `projection_stats.json` records `tsne2_negative_gradient` and `tsne3_negative_gradient` separately, and why it is the one chip that differs between the two.

The 2-D and 3-D fits **do** share one affinity matrix, which is the largest allocation in the build at roughly 2 GB. That is safe because `optimize` applies exaggeration as `P *= e` and restores it with `P /= e` in a finally block; the round trip was measured on this dtype at a relative error of 1.2e-16, which is float64 epsilon.

## The color-by system

This is the part of the app most worth understanding before changing anything.

`manifold/colorby.py` is a registry of `ColorBy` specs.
Each declares its `scope` (which corpora it could describe), a `resolver` returning one array over the full corpus, an optional `hint`, and an optional `(predicate, fix-hint)` pair for an artifact it needs.
`covers()` reports which corpora it can color *right now on this machine*, and that single fact drives the menu order, the disabled state, the coverage readout, and what the renderer does.
`labels(key)` returns one array over the full corpus with a `NOT_COVERED` sentinel; everything downstream is a plain categorical render.
The availability predicate is `data.archs4_metadata_available` itself, never a re-derived path: a second source of truth for the same file was a real bug, and a test now pins it.

Four consequences are deliberate and should survive future edits.
The menu lists whole-map fields first and labels each with its scope ("Tissue · whole map", "Flight vs Ground · OSDR only", "... · unavailable").
A field with no data at all is shown and disabled with the command that enables it, because hiding it makes the app look like it never had the feature.
A coverage bar sits directly under the control, amber rather than red, since an OSDR-only field is working correctly and not failing; the exact point count sits beside it **only when coverage is partial**, because the readout exists to answer "why is most of my map not colored?" and a whole-map field never raises that question - a full bar says it, and "Colors all 942,563 points." only said it again in words.
One palette is shared across both corpora: categories are ranked once over the whole covered population, so a liver in GEO and a liver in OSDR get the same color and each category keeps its color and its place in the legend across every budget and zoom.
The legend *count*, by contrast, is the number of points actually on screen: `render._legend_with_drawn_counts` recomputes it per figure from the drawn ARCHS4 sample plus the OSDR overlay, and a category with nothing currently drawn drops out of the key, because a legend count is read as "how many of these am I looking at", not "how many exist".

ARCHS4 traces carry no `customdata`; it existed only to feed the lasso and cost about 600 KB of dead payload per figure.
The OSDR overlay keeps `[sample_key, category]` for hover.

### Finding a study among 942,563 points

`manifold/find.py` resolves a study identifier to the points it names: a GEO series (`GSE…`) or an OSDR study (`OSD-###`).
Matches are marked with a white X and a button offers to frame them.

**It took four grammars until 2026-08-13, and the two that went were the sample ones** - `GSM…` for one GEO sample, and an OSDR sample by its full `<study>|<name>` key or its bare name.
The question a reader brings to a million-point map is "where did this experiment land", not "where is replicate 3": one glyph among 942,563 is a dot, and its siblings are a neighbourhood with a shape.
Do not reintroduce them without an argument that answers that.
Clicking an OSDR diamond still selects that one sample, through `find.osdr_sample` rather than through the typed grammar - a mark on screen is an unambiguous reference to one point, and the box is not.
It opens no embedding and imports no Dash, so it is testable against the fixture corpus.
Design, measurements and rejected alternatives: [`docs/design-notes.md`](docs/design-notes.md#finding-a-study-on-the-map).

Five things about it are load-bearing.

**Free text is refused, and the refusal has to say so.** A substring scan of 940,455 titles costs about 200 ms, so this is a decision: it would return hundreds of rows for "liver" and read as a biological query when it is a string match. "Where is liver" is the Tissue color-by's question. So a miss carries a *reason* - `shape`, `absent`, `no_geo_metadata`, `empty` - and each gets its own sentence, because answering "liver" and "GSE999999" identically tells the first user their search is broken and the second nothing about whether this machine could have looked it up.

**`cache/archs4_metadata.parquet` is optional, and without it 99.8% of the map is unfindable.** `find.searchable()` reads `data.archs4_metadata_available` itself - never a re-derived path - and the `no_geo_metadata` sentence is `colorby.ARCHS4_META_HINT`, so the two controls that depend on that artifact cannot name two different fixes. Reporting a GEO id as "absent" there would be invariant 5 by another route.

**The 839 rows with an empty `series_id` are filtered before parsing.** `int(s[3:])` raises on them and takes the index build with it; coercing to 0 would file them under a series that does not exist. The fixture corpus carries the same shape of row. The index is int32 keys and rows: measured at 15.0 MB and 460 ms once while it still carried the per-sample GSM half, against 96.8 MB for a `pd.Index` on an 80.8 MB working set.

**Framing is a button and never automatic**, and the second reason is the stronger one. A found set can span the map - OSD-457's 192 samples framed to 1.22x the corpus before `_clamped_to_corpus` went in, so the "frame" zoomed *out* - and a 2-D neighbourhood is not a similarity neighbourhood here, its 20 nearest points overlapping the true cosine top-20 by a median of 0. Marking answers "where is it"; zooming asserts more. `test_finding_something_never_moves_the_viewport_by_itself` pins it structurally.

**The box completes an identifier as it is typed, and the suggestions are prefixes of the same two grammars - never a free-text search.** That refusal is the module's central decision and a dropdown was the obvious way to reintroduce it by accident; "liver" still produces no list and still points at the Tissue color-by. Three things about the wiring are load-bearing and each fixed a defect found in the browser. `dcc.Input`'s `debounce` is **off**, so `value` publishes per keystroke and the commit is named outright - `n_submit`, `n_blur`, and a chosen row - which is what keeps a 942,563-point figure from being rebuilt on every letter. **Nothing that changes per keystroke may be an Input of the callback that writes `find-store`**, because a pattern-matching `ALL` input fires whenever its family is *re-rendered* and Dash discards the response to a superseded request: with the suggestion family on that callback, a keystroke's no-op overtook an Enter the server had already answered, and the first find of every session was silently lost. So the family's input lives on `choose_suggestion`, which publishes a `find-chosen` store, and `test_nothing_that_changes_per_keystroke_can_reach_the_search` asserts the property rather than the symptom. Finally, **whether the list is open is the browser's business and what is in it is the server's** - two facts, one owner each, composed by CSS from `:empty` and one `is-closed` class, so no callback writes a `style` and the two cannot race. `assets/find-suggest.js` is the map's only JavaScript and exists for the one thing Dash cannot express: a keystroke. Design, measurements and the two-line row: [`docs/design-notes.md`](docs/design-notes.md#finding-a-study-on-the-map).

**One control undoes every framing, and that is a finding rather than a convenience.** Framing a find, framing a retrieval and a plain scroll-zoom all write the same `viewport-store`, so by the time it is written they are indistinguishable and there is exactly one thing to undo. "Reset view" sits **on the rail**, in a `View` group at its foot beside the two buttons that frame, and is shown only while the map is framed. It sat on the plot until 2026-08-13, on the argument that it qualifies the viewport and the viewport belongs to the plot; that argument is true and it still split one feature across two surfaces, with the button that framed the map on the rail and the button that unframed it most of a screen away on the canvas. Undoing an action belongs beside the action. **`callbacks.viewport_axes` must keep saying `autorange` outright when there is no viewport**: `uirevision="keep"` preserves the reader's zoom unless the incoming figure *changes* the attribute, and an absent range key is not a changed value, so a reset that merely omits it leaves the map where it was. Verified on the real corpus across all three projections and all four retrieval paths. Design: [`docs/design-notes.md`](docs/design-notes.md#finding-a-study-on-the-map).

**Marks are capped at `theme.FIND_MAX_MARKS` (500) and the cap states what it dropped**, in the status, the badge and the key row - the rule `RETRIEVAL_MAX_NUMERALS` already follows. The key row counts what is *drawn*, so a series of 8,764 shows 500.

**The click probe was cut, and a test guards the reason.** Inspecting an arbitrary cloud point is not buildable the obvious way: `hoverinfo="skip"` suppresses click picking as well as hover, and enabling hover to recover it costs a measured median **240 ms per mouse move** at 942,563 points. A DOM-listener route works and is cheap, but a click at full-corpus zoom has **758 drawn points within 3 px**, so it cannot mean "this sample". `test_archs4_cloud_carries_no_hover_or_customdata` fails if anyone re-enables hover to "fix" picking, and now carries that measurement. The *record* half shipped anyway through the find box: looking a GSM up renders exactly the GEO record a click would have shown.

### The key on the plot keys every mark, not just hue

Design doc, with the full audit and the rejected alternatives: [`docs/design-notes.md`](docs/design-notes.md#map-key).

With a comparison drawn the map carries **four** encodings at once - corpus glyph hue, member fill hue, hit ring shape, and corpus glyph shape - and until 2026-08-06 it explained one.
Ring shape, the channel that says which cohort retrieved a hit, had no key anywhere; neither did the diamond that has meant "one of the 2,108 spaceflight samples" since the map was built.
The floating panel is now the whole key: a retrieval section, the color list, and a corpus shape footer, in that order because that is how transient each is, and only the color list scrolls.

Four things about it are load-bearing.

**A comparison is grouped by role, not by cohort.** The encoding has two factors and they do not share a channel, so the two member rows sit adjacent and differ only in hue, and the two hit rows sit adjacent and differ only in shape. The reader sees each channel vary with the other held fixed. That layout *is* the explanation, which is what lets the sentence explaining it be deleted rather than reworded. Grouping by cohort was built first and buries exactly the thing worth showing.

**Shapes are CSS; hues come from `theme` through an inline style.** The rail's old swatches mirrored `RETRIEVAL_QUERY` and `RETRIEVAL_QUERY_B` into `map.css` by hand, because Plotly cannot read a CSS variable. There is now no second copy to drift, and `test_the_map_key_reads_its_hues_from_the_theme` asserts both halves: the key carries the theme's values, and neither hex appears in `map.css` again. A new glyph shape needs a `.bm-key-glyph.is-<shape>` rule, pinned by `test_the_key_glyph_shapes_all_have_a_stylesheet_rule` - the class is interpolated from the shape name, so it never appears in a `className` literal and the general classname test cannot see it.

**The key must follow the plot into 3-D.** `Scatter3d` rejects `star`, so a member draws as a diamond there and so does the key. A key asserting a mark 3-D does not draw is worse than the silence it replaced - which is also why the 3-D OSDR fix below was a prerequisite rather than a bonus.

**A single query does not pay for the comparison.** A plain search gets two rows, no headings and no names. An uploaded sample gets one, because it was never embedded into this map and draws no member mark.

The plot badges stay. They answer *what is drawn right now* and change on every zoom, where the key answers *what does this mark mean*; folding them together would make the key change on zoom.

**3-D used to discard the OSDR overlay's symbol and its white ring.** `_scatter`'s `Scatter3d` branch passed no `symbol` and hard-coded `line=dict(width=0)`, so 2,108 spaceflight samples arrived as plain circles in the same palette hue as the 940,455 beneath them, contradicting `render.py`'s own docstring. `test_osdr_markers_are_visually_distinct_from_the_cloud` missed it because it only ever ran `("pca", "2d")`. Both channels survive the trip: `diamond` is in `Scatter3d`'s vocabulary and gl-scatter3d honors `marker.line` as a border.

### The shared tissue vocabulary

OSDR is curated but hyper-specific ("Right extensor digitorum longus", "Left Lobe of the Liver"), 48 distinct values.
ARCHS4 has no curated tissue column at all; the signal is in GEO's free-text `characteristics_ch1` and `source_name_ch1`, which yields 42,754 distinct lowercased strings.
`manifold/tissue.py` folds both onto 37 buckets plus "Other" and "Unknown" (39 in `tissue.BUCKETS`, produced by 40 ordered keyword rules where the first match wins).
Most buckets are organs; four are last resorts ("Tumor / cancer", "Reference RNA", "Cell line", "Cultured cells") that are not tissues but name large and genuinely distinct slices of GEO.
It is deliberately auditable rather than learned, and it fails towards "Other" rather than towards a confident guess.

Two subtleties are load-bearing, and both were real bugs caught by tests.
Ordering and word boundaries matter: "bone marrow" must beat "bone", `\brenal\b` must not fire inside "adrenal", "cortex" is a kidney and adrenal word as well as a brain word so those rules run first, and smooth muscle is vascular and must not be claimed by the "muscle" stem; a `~` prefix opts a pattern out of the leading word boundary for morphemes GEO glues onto a stem, so "sarcoma" matches inside "osteosarcoma".
"Unknown" (nothing recorded) and "Other" (recorded but unplaceable) are kept distinct, and weak results are ranked, so an early unplaceable field cannot pin the answer to "Other" and block a later field that did identify the sample.

Measured on the real corpus: all 48 OSDR raw values map to a named bucket, none falling to "Other" or "Unknown", and 851,881 of 940,455 ARCHS4 samples (90.6%) do the same.
The Tissue color-by covers 942,563 of 942,563 points.
It was independently validated as biology rather than batch: 25-NN label purity 0.8142 against a 0.0501 permuted null, surviving both a batch control and a depth control at 0.7058.

### Why ARCHS4 metadata does not need the HDF5 files

`precompute/fetch_archs4_meta.py` posts to the Maayan Lab sigpy JSON API (`https://maayanlab.cloud/sigpy/meta/samplemeta`) and gets per-GSM GEO metadata in bulk.
Measured by actually running it: 33.7 seconds, 39 requests, about 216 MB, resolving 99.911% of all 940,455 accessions (human 99.851%, mouse 99.982%).
The alternatives were measured, not assumed: reading the same fields out of the remote gene HDF5 over range requests works but costs about 5 minutes and 272 MB *per field*, and downloading the files outright is 113 GB (62.3 GB human plus 50.7 GB mouse).
That download is what kept the ARCHS4 cloud grey for the entire earlier design, and it is no longer needed at all.

The 839 unresolved samples are not GEO withdrawals.
They are present in the release-matched v2.5 metadata and absent from the newer v2.latest the API serves, which disproves the assumption that ARCHS4 releases are append-only.
They get tissue "Unknown" rather than being dropped or guessed at.
There is a documented upgrade path that was deliberately not taken: ARCHS4 publishes metadata-only HDF5 files under *versioned* names (`human_meta_v2.5.h5` at 311.8 MB, `mouse_meta_v2.5.h5` at 350.9 MB; the unversioned "latest" spellings 403, which is why they are easy to miss), giving exactly 100.000% release-matched coverage for 663 MB and about 8.5 minutes.
Fifteen times the build time to recover 0.089% of points is not worth it for a color; switch if tissue ever needs to be a build gate rather than a color.

## Candidates that were built or tested and then rejected

Record these rather than rediscovering them.
Each was measured on the real corpus and cut on the evidence.

**Cosine similarity to an OSDR reference** (mean centroid, flight centroid, ground centroid, and a flight-minus-ground "spaceflight-likeness" axis).
The four scores are one field wearing four names, pairwise r 0.996 to 1.000.
The "spaceflight-likeness" axis correlates r = -0.990 with PC1 and r = -0.779 with the raw L2 norm.
PC1 is neither spaceflight nor sequencing depth: it is a transcriptome-concentration axis (`REFERENCE.md` section 4), so the candidate measured how concentrated a sample's transcriptome is and called it resemblance to spaceflight.
One in ten random flight/ground relabelings of the same sample sizes beat the real axis on spatial structure, and 46.5% did under a within-study permutation.

**kNN tissue-label transfer from OSDR to ARCHS4.**
Median best-match cosine is 0.964 with 100% of points above 0.7, so no confidence threshold discriminates anything, and the winning OSDR sample beats the runner-up by a median of 0.00089 cosine, meaning the winner is essentially arbitrary.
54% of the ARCHS4 targets are human samples that would have received mouse tissue labels.

**Unsupervised k-means cluster id (k=24).**
Built, run on the real corpus, measured, then deleted along with its precompute stage.
81.9% of the cluster label is recoverable from the 2-D UMAP coordinates alone (15-NN over a 120k sample, against a 12.4% majority-class baseline), so coloring by it mostly redraws the shape already on screen, and a structure-free 24-cell Voronoi null reproduced its spatial coherence to within 1.5 points of modal agreement.
It is arbitrary (seed-to-seed ARI about 0.45), 81% species-pure, and explains 80.7% of the raw-L2-norm variance.
Numbering an arbitrary partition "Cluster 1..24" on a scientific instrument invites exactly the over-reading the rest of the design prevents.

**Local UMAP density** was rejected as redundant with the density raster rendered underneath.
That raster is now gone, so the argument no longer holds and the candidate is genuinely open again.
If it is revisited, note that a density color-by encodes something the eye already reads off glyph crowding, and that it must be judged against the methodological note below rather than reinstated on the strength of this paragraph.
**PC1-3 as color-bys** are free (already in `coords_pca3.parquet`) but redundant with the axes on screen.
**GEO series (GSE)** has 51,284 distinct values, so a Top-11 legend would color about 3% of the map and dump the rest in "Other", a grey map by another route; it is also a pure batch label (333x lift). It stays in the parquet for provenance and is not offered as a color.

**Methodological note for whoever evaluates the next candidate.**
A between-bin/total variance ratio (spatial eta-squared) is not sufficient evidence that a color-by shows real structure.
Thirty arbitrary random directions in 512-d score eta-squared 0.874 +/- 0.025 on this UMAP, because the UMAP was fit on those same vectors.
Every candidate in the 0.89 to 0.94 band is indistinguishable from an arbitrary projection, and species (0.985) is the only one that clearly clears it.
Judge a candidate against a structure-free null of the same *form*, and check whether it is recoverable from the coordinates themselves or from transcriptome concentration.

## How the two halves relate

They share a model, an embedding index, a visual language, and the exact index join described at the top of this file.
The map never retrains the model and never re-embeds ARCHS4; those 940k embeddings already exist and are consumed as-is.
The reusable interfaces (OSDR preprocessing, `ExpressionPerformer`/`encode`, digest helpers, ortholog maps, memmap loaders, LFS guards) are catalogued with signatures and line numbers in `REFERENCE.md` section 6; `manifold/bridge_rna.py` is the one file that imports them, so the coupling stays visible in one place.

`hits-store` lives on the **shell** (`app.py`), not in the retrieval view, because the router destroys a view when you leave it and the map has to be able to draw a retrieval you ran before walking over to it.

The cross-corpus batch effect is a property of the shared space: OSDR pairs sharing neither study nor tissue still neighbour each other 54x above chance, because the two corpora were embedded on different hardware and in different precisions, so compare within a corpus, not across.
It was once a standing `.bm-caution` paragraph at the bottom of the control rail; that microcopy was removed from the UI as over-explaining, and the fact now lives in the docs (here, `README.md`, and `REFERENCE.md` section 4) rather than on the rail, while `precompute/validate_artifacts.py --mixing` still recomputes the number and warns above 50x.

## Environment and commands

Shares the Bridge RNA venv at `/Users/josh/Bridge-RNA/.venv`.
Pinned lower bounds and the reason for each are in `requirements.txt`; the bounds are not decorative, since pandas 3.0 stopped stringifying missing values in `astype(str)`, dash 4.0 replaced the Dropdown/RadioItems internals and their class names, and plotly 6.0 serializes numpy arrays as base64 typed arrays.
That last one is why shipping all 942,563 points to the browser is affordable at all: the coordinates travel as base64 typed arrays rather than as JSON number lists.

Run the pipeline in this order; `fetch_archs4_meta.py` joins onto the identity table `build_projections.py` writes.

```bash
/Users/josh/Bridge-RNA/.venv/bin/python precompute/embed_osdr.py         # OSDR embeddings, gene-digest gated. Hours; resumable.
/Users/josh/Bridge-RNA/.venv/bin/python precompute/build_projections.py  # full-corpus PCA + UMAP + t-SNE. See the timing note below.
/Users/josh/Bridge-RNA/.venv/bin/python precompute/fetch_archs4_meta.py  # ARCHS4 GEO metadata. ~35 s, needs network.
/Users/josh/Bridge-RNA/.venv/bin/python precompute/validate_artifacts.py --mixing --quality
/Users/josh/Bridge-RNA/.venv/bin/python tests/check_join.py              # the join, on the real corpus
/Users/josh/Bridge-RNA/.venv/bin/python app.py                          # http://127.0.0.1:8050

/Users/josh/Bridge-RNA/.venv/bin/python -m pytest tests/ -q              # 399 tests, about twenty-seven seconds
/Users/josh/Bridge-RNA/.venv/bin/python tests/e2e_check.py               # 129 browser checks, about five minutes
/Users/josh/Bridge-RNA/.venv/bin/python tests/e2e_upload_check.py        # 70 upload checks, about eight minutes
/Users/josh/Bridge-RNA/.venv/bin/python tests/e2e_cohort_check.py        # 164 cohort checks, about five minutes
/Users/josh/Bridge-RNA/.venv/bin/python tests/screenshot_readme.py       # rewrites README's two images, about three minutes
```

**A screenshot of either view must be sized to the content, never to a fixed viewport.**
Both views are fixed-height instruments that scroll internally, so a window shorter than the content does not produce a scrollable page - it produces a silently clipped one, which is how the shipped retrieval screenshot hid 410 px of the inspector from 2026-07-22 until it was recaptured.
`tests/screenshot_readme.py` grows the window until no scroll container overflows, and separately measures the *figure*, because a Plotly canvas is exactly as big as its container whether or not the drawing inside it fits: it re-renders through `Plotly.toImage` and fails if the outer 3 px of the image carries ink. The 3-D camera is set outright and its distance searched for the largest framing that leaves the band clean, rather than dollied with wheel events, which is what let the first recapture run the point cloud off the canvas while every DOM check passed.
Design, measurements and rejected alternatives: [`docs/design-notes.md`](docs/design-notes.md#readme-screenshots). `tests/screenshots.py` is the separate fourteen-frame gallery walkthrough and is unaffected.

**The build is no longer a ten-minute job.** PCA is seconds and UMAP is about fourteen minutes, but t-SNE dominates everything, and almost all of that is the 3-D fit: openTSNE's FIt-SNE interpolation refuses more than two output dimensions, so 3-D falls back to Barnes-Hut, which is `n log n` with a much larger constant.
`--skip-tsne` exists for exactly this reason, and skipping it is not a broken build: the t-SNE pill is shown disabled, the validator prints `SKIP` rather than failing, and everything else works.
Measured stage timings are in `REFERENCE.md` section 4.

The real flags are worth checking against `--help` before quoting them: `embed_osdr.py` takes `--device`, `--batch-size`, `--limit`, `--no-resume`, `--rebuild-expression`, `--metadata-only`; `build_projections.py` takes `--umap-neighbors`, `--tsne-perplexity`, `--tsne-jobs`, `--pca-report`, `--batch`, `--knn-jobs`, `--seed`, `--archs4-limit`, `--skip-umap`, `--skip-tsne`, `--densmap`, `--dens-lambda`; `fetch_archs4_meta.py` takes `--limit`; `validate_artifacts.py` takes `--mixing`, `--quality`, `--compare`.
Three flags that appear in older prose no longer exist: `--skip-hnsw` (there is no index to skip), `--density-only` (there is no raster to re-render), and `--pca-fit-sample` / `--umap-fit-sample` (no reduction is fit on a subsample any more).

Each method is an independent stage and `projection_stats.json` is **merged**, not rewritten, so rebuilding one does not erase what the others recorded.
The merge is abandoned and the record started fresh if the corpus row counts changed, because a half-stale record that still looks complete is worse than one that is obviously new.

There is no multi-gigabyte download anywhere in this pipeline.
The metadata step is optional: without `cache/archs4_metadata.parquet` the app still runs, the Tissue option is shown disabled with the command that enables it, and Species remains the whole-map default.

Every build stage ends with an **objective validation**, not a visual glance: digest match, an explained-variance profile checked against invariant 2, the stratified cross-corpus mixing check, and the projection-quality score.
`validate_artifacts.py --mixing` computes the *exact* top-51 neighbours of each OSDR sample by streaming the ARCHS4 memmap in 50k blocks and keeping a running top-k, which is what let the 2.07 GB ANN index be deleted.
That mixing check is not a lasso remnant; it is the honesty check behind the app's premise, and it must keep working.
`validate_artifacts.py --quality` scores every coordinate set on 15-NN recall against the exact 512-d neighbours and on 25-NN tissue purity, each against a null, and `--compare DIR` scores a second set of coordinate parquets alongside so a candidate build can be held against the shipped one.
Structural checks pass for any set of finite numbers, so they cannot tell a good projection from a scrambled one; that is what the quality check is for.

## Visual language

Light scientific-instrument chrome matching Bridge RNA exactly (canvas `#eef2f7`, panels `#fff`, accent `#2b7fff`, navy header `#14294a` with teal rule `#22c7bd`), with a dark navy *plot canvas* (`#0e1d34`) inside it for WebGL glyph contrast, the one deliberate departure.
The shell is two columns, a control rail and the plot; it was three before the readout panel was removed, and `.bm-body` is flexbox so `.bm-plot-wrap`'s `flex: 1` reclaimed the space with no layout math.
The eleven-hue categorical palette in `manifold/theme.py` was validated with the dataviz skill's checker against the navy plot surface (OKLCH L 0.48-0.67, worst adjacent-pair CVD deltaE 8.4, worst normal-vision deltaE 15.4, >= 3:1 on the surface); slot order is the CVD-safety mechanism, so do not shuffle it without re-validating.
Two greys sit at the neutral end because "Other" and "Unknown" are different answers, with Unknown the dimmer so absence recedes furthest.
The full token list is in `REFERENCE.md` section 9.
Selection tools are removed from the modebar (`select2d` and `lasso2d` both, and `dragmode` is `pan`) because no selection feature exists and a marquee that does nothing is worse than no marquee.

The rail has one rule about where a fact goes, and both readouts follow it: **the fact that qualifies a control sits directly under that control.**
`.bm-coverage` hangs under the color-by dropdown and `.bm-params` hangs under the Projection pills, at the same 9 px offset so the two line up.
Neither belongs in `.bm-plot-badges`, which reports what is drawn *right now* and changes on every zoom, while these describe how the coordinates were built.
Within `.bm-params` only the measured payload is set apart, in mono tabular figures at a half-step down, because "cosine" in a numeral font is noise while `30` and `942,563` want to sit on one grid.

`.bm-hint` was moved from `--text-muted` to `--text-secondary` at the same time: `#8a99ac` on the white panel measures 2.90:1, which fails WCAG AA at the 11.5 px every hint on the rail uses, and `--text-secondary` is 5.47:1 while still receding behind the controls.

### Contrast, keyboard, and the two breakpoints (2026-08-11)

**Every text token clears WCAG AA 4.5:1 on every surface the stylesheet defines, and a test enforces it.**
Both tiers below `--text-primary` used to fail. `--text-muted` was `#8a99ac`, 2.90:1 on white, and it carried every rail label, kicker, hint, slider mark and dropdown placeholder - the smallest type in the app was also the least legible - while `--accent` at 3.76:1 carried the primary button's white label and every blue link and tab.
The paragraph above records that finding for `.bm-hint` and fixed exactly that one class.
`--text-muted` is now `#616e80`, chosen so its *worst* ground clears the bar rather than only white, and there is a second blue: **`--accent-text` `#1663dd` runs wherever the accent carries text or is the ground white text sits on** - the primary button, the mode tabs, the facet chips, links, section titles, the selected segmented pill.
**`--accent` `#2b7fff` is untouched and is still the identity hue** for every border, fill, focus ring, panel dot and Plotly mark, because 3.76:1 clears the 3:1 a non-text mark needs.
The one exception is the brand tile, under WCAG 1.4.3's logotype clause, and `01-shell.css` says so where someone would otherwise "fix" it.
`test_every_text_token_clears_wcag_aa_on_every_surface` computes the ratios from the stylesheet, so a palette edit that reintroduces the failure fails there rather than in a browser; `test_theme_matches_the_bridge_rna_tokens` covers all 19 mirrored constants, because `theme.py`'s copy had already gone stale on `TEXT_MUTED`.

**A control hidden with `display: none` is a control nobody can reach.**
`.bm-seg` hid the radio input that carries the segmented control's state, which took Projection, Dimensions and the number of ARCHS4 points out of the tab order and out of the accessibility tree - three of the map's six controls, mouse-only.
It is clipped now, not removed. The `:focus-within` rule on the option label had been sitting in that file unable to fire the whole time, which is the tell to look for.

**Dash names a Dropdown by its own value and a Slider not at all**, so "OSDR study: OSD-100" reached a screen reader as "OSD-100, button".
`for` cannot fix either, because neither renders a labelable element.
Both views wrap each control in `role="group"` with `aria-labelledby` on the heading that already names it - `bridge_rna.layout._labelled` and `manifold.layout._group` - and `test_every_dropdown_and_slider_is_named_by_something` pins that the groups exist and that no `aria-labelledby` dangles.
`.visually-hidden` in `00-tokens.css` is the hook for the two places a name has to exist without being seen; it clips rather than using `display: none`, for the reason above.
The map's rail is an `<aside>` with a hidden `<h2>` and the plot is a `<main>`: before this the map was a `div` inside a `div` with one heading on the whole page, while the retrieval view already had nav, aside, main, aside.

**Class names in Dash 4 land on a wrapper, not on the control.**
`dcc.Input` renders `<div class="dash-input-container YOURCLASS"><input class="dash-input-element"></div>`, so `input.dash-input` matched nothing and never had; the field looked right only because the `.control input[type="email"]` selector beside it does reach the inner input.
Check which element a class actually lands on before writing a rule against it, and note that the box and the focus ring must go on the *same* element - putting the box on the wrapper while Dash still styles the input draws a ring inside a ring.

**Two breakpoints, one per view, and they answer the same question.**
The retrieval view stacks at 1180 px; the map stacks at 900 px and narrows its key at 620 px.
The map had none at all until 2026-08-11, and its two fixed widths are why that was broken rather than merely tight: a 268 px rail plus a 228 px key is 496 px of furniture, so at 393 px the plot was a 125 px strip with the key sitting on top of the rail.
Stacked, the plot takes `68vh` with a 420 px floor and the document scrolls, which is what the retrieval view already did.
**The stacked rail is CSS columns, not a grid**, because a grid row is as tall as its tallest cell and these groups are wildly unequal - that cost about 250 px of empty rail on an iPad.
**Rows inside a height-capped panel need `flex: none`**: a column flex item shrinks below its content height by default, so `.bm-legend-item` and `.bm-key-row` compressed into each other instead of the list scrolling, and `overflow-y: auto` had nothing to act on.
**A `dcc.Graph` whose container can change size needs `responsive: True`** - the map's did not have it, so Plotly kept the geometry it was first laid out with and drew a quadrant of the corpus into the whole canvas after any resize or breakpoint crossing.

**One quantity gets one channel, and that channel is in the key.**
The retrieval network sized each hit node by `16 + (score - min(score)) * 20`: a second encoding of what the edge width already carries, on a different scale, keyed nowhere, and the exact min-max rescale `_edge_width` exists to avoid. Over the 0.0016 spread these scores have it moved the diameter three hundredths of a pixel, so it looked constant while claiming to measure. It is constant.
The comparison network, by contrast, was missing a label it should have had: it named none of its hits, so the accessions existed only in a tooltip and the figure carried no identities on paper. They are drawn up to `COMPARISON_MAX_LABELS` (20) nodes and dropped above it, the same rule as the map's `RETRIEVAL_MAX_NUMERALS`.
