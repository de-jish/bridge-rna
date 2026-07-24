# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

**Bridge RNA is one application with two views**, developed by the Space Biosciences Research Branch at NASA Ames Research Center.

- **Retrieve** (`/`) takes one NASA OSDR spaceflight RNA-seq sample and returns its closest Earth analogs out of the 940,455-sample ARCHS4/GEO index, as a network graph with an inspector and an optional LLM summary. Code in `bridge_rna/`.
- **Map** (`/map`) zooms out to the whole space: the 512-d ExpressionPerformer embeddings of both corpora - OSDR (2,108 NASA GeneLab spaceflight samples) and ARCHS4 (940,455 GEO samples) - reduced into one shared 2D/3D space, every one of the 942,563 points drawn as a live WebGL glyph and coloured by biology. Code in `manifold/` and `precompute/`.

`app.py` is the single entry point and owns the header and the router. There is no `app_osdr_dash.py` and no `app_manifold.py`; both were deleted when the two repositories merged on 2026-07-22, and the map's 19 commits are in this history.

**Current state: built, run on the real corpus, and tested.** 207 tests pass in about three seconds, plus 40 browser checks in `tests/e2e_check.py`.
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
```

The **map view** opens no embeddings, computes no statistics, and never touches a Git LFS object.
The retrieval view does: its cached path scans the 963 MB ARCHS4 memmap, an LFS object, on every search. That is the one place in the serving process that touches one.
`BRIDGE_RNA_ROOT` is needed to *build* the cache, not to run the app.
The whole live cache measures 217.8 MB, of which the app opens 80.8 MB; the rest is the embedding intermediates that make a re-embed cheap plus the accession sidecar the metadata fetch joins onto (`REFERENCE.md` section 12).

### Package layout

```
app.py                   the only entry point: header, router, both views on :8050
bridge_rna/              the retrieval half (config, util, preflight, osdr, ai, geo,
                         figures, retrieval, panels, layout, callbacks)
manifold/paths.py        every artifact path, one place; env-overridable
manifold/preflight.py    PRECOMPUTE_REQUIRED vs APP_REQUIRED, LFS pointer guard
manifold/data.py         cached loaders: coords, points_meta, osdr_metadata, archs4
                         tissue, and the projection_stats build record the rail reads
manifold/tissue.py       the shared tissue vocabulary (canonical_tissue, coalesce_tissue)
manifold/colorby.py      the coverage-aware color-by registry; the map's honesty layer
manifold/render.py       layered figure build (ARCHS4 cloud, OSDR overlay)
manifold/sampling.py     stratified + viewport-aware ARCHS4 subsampling
manifold/layout.py       two-column shell, control rail, floating legend, the
                         METHOD_LABELS registry and the projection-parameter readout
manifold/callbacks.py    controls -> figure, zoom -> level of detail, coverage readout
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
7. **The parameter readout must describe the coordinates on screen, never the code's defaults.** `manifold/layout.projection_params` reads `cache/projection_stats.json` through `data.projection_stats()` and nothing else. Duplicating the constants into the serving app would let the rail stay confident while the cache went stale, which is the same failure as the retrieval banner that announced every cached result as subprocess output. A key the record does not carry drops its chip; it never renders a blank slot or a guess.

## The three projections

`manifold/data.METHODS` is the registry and `manifold/layout.METHOD_LABELS` is the order the rail offers them in.
Adding a fourth is one line in each, plus a stage in `build_projections.py`; the pills, the disabled state, the default, the validator's coordinate walk, and two test loops are all derived rather than hand-listed, which is the point.

| method | what it is | fit cost on 942,563 points |
| --- | --- | --- |
| PCA | exact eigendecomposition, linear, honest about global magnitude | ~6 s |
| UMAP | `n_neighbors=30`, `min_dist=0.1`, cosine, PCA init | ~14 min |
| t-SNE | openTSNE, perplexity 30, cosine, PCA init rescaled to std 1e-4 | hours, almost all of it 3-D |

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
A coverage bar and an exact point count sit directly under the control, amber rather than red, since an OSDR-only field is working correctly and not failing.
One palette is shared across both corpora: categories are ranked once over the whole covered population, so a liver in GEO and a liver in OSDR get the same color and each category keeps its color and its place in the legend across every budget and zoom.
The legend *count*, by contrast, is the number of points actually on screen: `render._legend_with_drawn_counts` recomputes it per figure from the drawn ARCHS4 sample plus the OSDR overlay, and a category with nothing currently drawn drops out of the key, because a legend count is read as "how many of these am I looking at", not "how many exist".

ARCHS4 traces carry no `customdata`; it existed only to feed the lasso and cost about 600 KB of dead payload per figure.
The OSDR overlay keeps `[sample_key, category]` for hover.

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
/Users/josh/Bridge-RNA/.venv/bin/python app.py                          # http://127.0.0.1:8050

/Users/josh/Bridge-RNA/.venv/bin/python -m pytest tests/ -q              # 207 tests, about three seconds
/Users/josh/Bridge-RNA/.venv/bin/python tests/e2e_check.py               # 40 browser checks, about two minutes
```

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
