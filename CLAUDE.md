# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Bridge Manifold is the exploratory map companion to **Bridge RNA** (a separate repo at `/Users/josh/Bridge-RNA`).
Bridge RNA retrieves the closest Earth analogs for one NASA spaceflight RNA-seq sample.
Bridge Manifold zooms out: it dimensionally reduces the 512-d ExpressionPerformer embeddings of both corpora - OSDR (2,108 NASA GeneLab spaceflight samples) and ARCHS4 (940,455 GEO samples) - into one shared 2D/3D space, renders about 100k live WebGL glyphs over a density raster of all 942,563 points, and colors them by biology.

**Current state: built, run on the real corpus, and tested.**
`manifold/` and `precompute/` are complete, the full offline pipeline has been executed against the real 942,563-point corpus, and 144 tests pass in about half a second.
The ARCHS4 GEO metadata join is built (`cache/archs4_metadata.parquet`, 940,455 rows, 51,284 distinct GEO series), so the map colors by tissue across both corpora rather than by species alone.

Two features that appear in older prose are **gone and must not be reintroduced as current behavior**: the lasso selection tool and its 512-d statistical readout (with `manifold/coherence.py` and the right-hand readout panel), and the hnswlib ANN index and population-moment artifacts that existed only to serve it.
Where that history is instructive it is recorded as history, clearly marked.

## Read these first

- `IMPLEMENTATION.md` - the master plan: architecture, design decisions, tradeoffs, build order.
- `REFERENCE.md` - verified ground-truth facts (model config, gene digest, embedding stats, measured timings, reusable Bridge RNA interfaces with line numbers, color-by columns, theme tokens). Every fact was checked directly against the checkpoint/memmap/data, not from docs. Trust it over inference.
- `progress.md` - living status log, decisions made, open questions. Keep it updated after every meaningful change (per Josh's global convention).

When plans change, update these docs so they reflect what was actually built, not just the initial intent.

## Architecture: the offline/online split is the load-bearing decision

Everything expensive is precomputed once and cached; the app only ever loads artifacts.
This is forced by measured cost (the landmark UMAP fit plus transform of 942,563 points is the bulk of a roughly five-minute projection build, and a direct 940k fit is hours) and is what keeps the app responsive.
Do not move model inference, UMAP, or density rasterization into the serving app.

```
OFFLINE (precompute/, run once -> cache/)        ONLINE (app_manifold.py, loads artifacts only)
embed_osdr.py        -> osdr embeddings npy       coord parquets + points_meta + osdr_metadata
build_projections.py -> pca/umap coord parquets   + archs4_metadata + density PNGs  (82.3 MB opened)
                     -> points_meta identity      renders go.Scattergl over the density underlay
                     -> density raster PNGs       colorby.py decides coverage; render.py draws it
fetch_archs4_meta.py -> archs4_metadata parquet
   (HTTP JSON API, ~35 s, no HDF5 download)
validate_artifacts.py -> exit code, gates a build
```

The serving app opens no embeddings, computes no statistics, and never touches a Git LFS object.
`BRIDGE_RNA_ROOT` is needed to *build* the cache, not to run the app.
The whole live cache measures 219.2 MB, of which the app opens 82.3 MB; the rest is the embedding intermediates that make a re-embed cheap plus the accession sidecar the metadata fetch joins onto (`REFERENCE.md` section 12).

### Package layout

```
app_manifold.py          entry point; preflight then Dash on :8051
manifold/paths.py        every artifact path, one place; env-overridable
manifold/preflight.py    PRECOMPUTE_REQUIRED vs APP_REQUIRED, LFS pointer guard
manifold/data.py         cached loaders: coords, points_meta, osdr_metadata, archs4 tissue
manifold/tissue.py       the shared tissue vocabulary (canonical_tissue, coalesce_tissue)
manifold/colorby.py      the coverage-aware color-by registry; the map's honesty layer
manifold/render.py       layered figure build (density underlay, ARCHS4 cloud, OSDR overlay)
manifold/sampling.py     stratified + viewport-aware ARCHS4 subsampling
manifold/layout.py       two-column shell, control rail, floating legend
manifold/callbacks.py    controls -> figure, zoom -> level of detail, coverage readout
manifold/theme.py        chrome tokens, dark plot canvas, validated categorical palette
manifold/bridge_rna.py   thin import shim for the reusable Bridge RNA functions
```

Only `precompute/` imports `torch`, `umap`, `sklearn`, `PIL`, or `requests`.
The serving app's dependency surface is `dash`, `plotly`, `numpy`, `pandas`, `pyarrow` and nothing scientific.

## Non-negotiable invariants

These are correctness gates, not style preferences.
Violating them produces output that looks fine but is scientifically wrong.

1. **Gene-digest gate.** `embed_osdr.py` must compute `canonical_gene_order_digest(genes)` and assert it equals `CANONICAL_GENES_SHA256` (`3f887ac8d329dce3c54d26448964904c07a345940cd3d9ebab18dd1f603194c5`). Abort the build on mismatch. An embedding built with the wrong gene order is silently invalid.
2. **L2-normalize before any reduction.** Raw ARCHS4 vectors are NOT normalized (norms 6.7-25.5); unnormalized, PC1 captures 57.8% of variance and is essentially a magnitude/depth axis. The real normalized build lands at PC1 = 40.9%, and `validate_artifacts.py` fails if it drifts back above 50%.
3. **Read model hyperparameters from `ckpt['config']`, not the demo's fallback constants.** The demo defaults differ from the true trained config.
4. **Verify Git LFS pointers resolve before any run that touches Bridge RNA.** The checkpoint and memmap live in Bridge RNA as LFS objects and can arrive as stub pointers. This now applies to `precompute/` only: the serving app reads its own cache and never opens an LFS object, which is why `PRECOMPUTE_REQUIRED` and `APP_REQUIRED` in `manifold/preflight.py` have no overlap. Keep `APP_REQUIRED` to what the app genuinely opens, in the order it opens it - `points_meta.parquet` is first because `layout.control_rail()` reads it through `data.counts()` while the layout is still being built.
5. **A color-by must never render a corpus it does not describe as though it were a category.** Coverage is declared in `manifold/colorby.py`, stated in the UI, and enforced in `manifold/render.py`. A field that does not describe ARCHS4 must let the density raster carry those 940,455 points, or, where there is no raster (3-D, or the underlay switched off), draw them as a deliberately faint context cloud in their own color at 0.35 opacity. The failure this prevents is a uniform grey glyph cloud over 99.8% of the map, which reads as "ARCHS4 was measured and has no structure here" rather than "this field says nothing about ARCHS4".
6. **Both corpora share one tissue vocabulary.** `manifold/tissue.canonical_tissue` is the only entry point, used by `precompute/fetch_archs4_meta.py` for ARCHS4 and by `manifold/data.osdr_tissue` for OSDR. Two tissue color-bys wearing one name would each leave the other corpus grey, which is invariant 5 by another route.

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
One palette is shared across both corpora: categories are ranked once over the whole covered population, so a liver in GEO and a liver in OSDR get the same color, and legend counts are whole-corpus counts that do not move when the budget or zoom changes.

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
The "spaceflight-likeness" axis correlates r = -0.990 with PC1 and r = -0.779 with the raw L2 norm: it is the sequencing-depth axis relabelled as biology.
One in ten random flight/ground relabelings of the same sample sizes beat the real axis on spatial structure, and 46.5% did under a within-study permutation.

**kNN tissue-label transfer from OSDR to ARCHS4.**
Median best-match cosine is 0.964 with 100% of points above 0.7, so no confidence threshold discriminates anything, and the winning OSDR sample beats the runner-up by a median of 0.00089 cosine, meaning the winner is essentially arbitrary.
54% of the ARCHS4 targets are human samples that would have received mouse tissue labels.

**Unsupervised k-means cluster id (k=24).**
Built, run on the real corpus, measured, then deleted along with its precompute stage.
81.9% of the cluster label is recoverable from the 2-D UMAP coordinates alone (15-NN over a 120k sample, against a 12.4% majority-class baseline), so coloring by it mostly redraws the shape already on screen, and a structure-free 24-cell Voronoi null reproduced its spatial coherence to within 1.5 points of modal agreement.
It is arbitrary (seed-to-seed ARI about 0.45), 81% species-pure, and explains 80.7% of the raw-L2-norm depth variance.
Numbering an arbitrary partition "Cluster 1..24" on a scientific instrument invites exactly the over-reading the rest of the design prevents.

**Local UMAP density** is redundant with the density raster already rendered underneath.
**PC1-3 as color-bys** are free (already in `coords_pca3.parquet`) but redundant with the axes on screen, and PC1 is the depth axis.
**GEO series (GSE)** has 51,284 distinct values, so a Top-11 legend would color about 3% of the map and dump the rest in "Other", a grey map by another route; it is also a pure batch label (333x lift). It stays in the parquet for provenance and is not offered as a color.

**Methodological note for whoever evaluates the next candidate.**
A between-bin/total variance ratio (spatial eta-squared) is not sufficient evidence that a color-by shows real structure.
Thirty arbitrary random directions in 512-d score eta-squared 0.874 +/- 0.025 on this UMAP, because the UMAP was fit on those same vectors.
Every candidate in the 0.89 to 0.94 band is indistinguishable from an arbitrary projection, and species (0.985) is the only one that clearly clears it.
Judge a candidate against a structure-free null of the same *form*, and check whether it is recoverable from the coordinates or from sequencing depth.

## Relationship to Bridge RNA

Bridge Manifold is a standalone Dash app that **imports/copies reusable functions from Bridge RNA rather than editing its 2,470-line retrieval app** - isolation without losing the shared-instrument feel.
It never retrains the model and never re-embeds ARCHS4; those 940k embeddings already exist and are consumed as-is from the Bridge RNA repo.
The reusable interfaces (OSDR preprocessing, `ExpressionPerformer`/`encode`, digest helpers, ortholog maps, memmap loaders, LFS guards) are catalogued with signatures and line numbers in `REFERENCE.md` section 6.

The cross-corpus batch effect is a property of the shared space and is disclosed on the control rail (`layout.control_rail`, `.bm-caution`): OSDR pairs sharing neither study nor tissue still neighbour each other 54x above chance, because the two corpora were embedded on different hardware and in different precisions.
Compare within a corpus, not across.

## Environment and commands

Shares the Bridge RNA venv at `/Users/josh/Bridge-RNA/.venv`.
Pinned lower bounds and the reason for each are in `requirements.txt`; the bounds are not decorative, since pandas 3.0 stopped stringifying missing values in `astype(str)`, dash 4.0 replaced the Dropdown/RadioItems internals and their class names, and plotly 6.0 serializes numpy arrays as base64 typed arrays.
Density rasters are a plain numpy 2D histogram rendered through PIL, deliberately not datashader, to keep the dependency surface small.

Run the pipeline in this order; `fetch_archs4_meta.py` joins onto the identity table `build_projections.py` writes.

```bash
/Users/josh/Bridge-RNA/.venv/bin/python precompute/embed_osdr.py         # OSDR embeddings, gene-digest gated. Hours; resumable.
/Users/josh/Bridge-RNA/.venv/bin/python precompute/build_projections.py  # PCA + UMAP coords, density rasters. ~5 min.
/Users/josh/Bridge-RNA/.venv/bin/python precompute/fetch_archs4_meta.py  # ARCHS4 GEO metadata. ~35 s, needs network.
/Users/josh/Bridge-RNA/.venv/bin/python precompute/validate_artifacts.py --mixing
/Users/josh/Bridge-RNA/.venv/bin/python app_manifold.py                  # http://127.0.0.1:8051

/Users/josh/Bridge-RNA/.venv/bin/python precompute/build_projections.py --density-only   # re-render rasters from cached coords
cd "/Users/josh/Bridge Manifold" && /Users/josh/Bridge-RNA/.venv/bin/python -m pytest tests/ -q   # 144 tests, ~0.5 s
```

The real flags are worth checking against `--help` before quoting them: `embed_osdr.py` takes `--device`, `--batch-size`, `--limit`, `--no-resume`, `--rebuild-expression`, `--metadata-only`; `build_projections.py` takes `--pca-fit-sample`, `--umap-fit-sample`, `--pca-components`, `--batch`, `--seed`, `--archs4-limit`, `--skip-umap`, `--density-only`; `fetch_archs4_meta.py` takes `--limit`; `validate_artifacts.py` takes `--mixing`.
The old `--skip-hnsw` flag no longer exists, because there is no index to skip.

There is no multi-gigabyte download anywhere in this pipeline.
The metadata step is optional: without `cache/archs4_metadata.parquet` the app still runs, the Tissue option is shown disabled with the command that enables it, and Species remains the whole-map default.

Every build stage ends with an **objective validation**, not a visual glance: digest match, an explained-variance profile checked against invariant 2, and the stratified cross-corpus mixing check.
`validate_artifacts.py --mixing` computes the *exact* top-51 neighbours of each OSDR sample by streaming the ARCHS4 memmap in 50k blocks and keeping a running top-k, which is what let the 2.07 GB ANN index be deleted.
That mixing check is not a lasso remnant; it is the honesty check behind the app's premise, and it must keep working.

## Visual language

Light scientific-instrument chrome matching Bridge RNA exactly (canvas `#eef2f7`, panels `#fff`, accent `#2b7fff`, navy header `#14294a` with teal rule `#22c7bd`), with a dark navy *plot canvas* (`#0e1d34`) inside it for WebGL glyph contrast, the one deliberate departure.
The shell is two columns, a control rail and the plot; it was three before the readout panel was removed, and `.bm-body` is flexbox so `.bm-plot-wrap`'s `flex: 1` reclaimed the space with no layout math.
The eleven-hue categorical palette in `manifold/theme.py` was validated with the dataviz skill's checker against the navy plot surface (OKLCH L 0.48-0.67, worst adjacent-pair CVD deltaE 8.4, worst normal-vision deltaE 15.4, >= 3:1 on the surface); slot order is the CVD-safety mechanism, so do not shuffle it without re-validating.
Two greys sit at the neutral end because "Other" and "Unknown" are different answers, with Unknown the dimmer so absence recedes furthest.
The full token list is in `REFERENCE.md` section 9.
Selection tools are removed from the modebar (`select2d` and `lasso2d` both, and `dragmode` is `pan`) because no selection feature exists and a marquee that does nothing is worse than no marquee.
