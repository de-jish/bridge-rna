# Bridge Manifold - Implementation Plan

Bridge Manifold is the exploratory companion to Bridge RNA.
Where Bridge RNA answers "what are the closest Earth analogs for one spaceflight sample," Bridge Manifold answers "what is the global shape of the whole embedding space, and where does any group of samples sit inside it."
It dimensionally reduces the 512-dimensional ExpressionPerformer embeddings of both corpora, ARCHS4 (940,455 human and mouse GEO samples) and OSDR (2,896 NASA GeneLab spaceflight samples), draws them together in one interactive WebGL scatter, colors them by biological and technical features, and lets a user lasso any region to get an honest statistical readout of whether the selected samples are meaningfully related.

This document is the master plan.
It records the architecture, the design decisions, the tradeoffs weighed, and the phased build order.
The verified ground-truth facts it relies on live in `REFERENCE.md`.
The living status log and the open questions that need Josh's input live in `progress.md`.

## 1. Goals and non-goals

### Goals

The tool must reduce and plot both corpora in a single shared 2D and 3D coordinate space so OSDR and ARCHS4 points are directly comparable.
It must offer both a fast linear method (PCA) and a structure-preserving nonlinear method (UMAP), and be explicit about when each one lies.
It must color points by feature, spanning OSDR biology (spaceflight status, tissue, strain, sex, genotype, study) and ARCHS4 identity (species, and tissue if we later join external metadata).
It must render ~940k points smoothly at interactive frame rates using Plotly WebGL scatter traces plus stratified sampling.
It must support lasso selection, and when a user selects a batch, report whether those samples are meaningfully related using statistics computed in the original 512-dimensional space, not in the distorted 2D projection.

### Non-goals

Bridge Manifold does not retrain or fine-tune the ExpressionPerformer model.
It does not re-embed ARCHS4; those 940,455 embeddings already exist and are consumed as-is.
It does not replace the Bridge RNA retrieval app; it is a separate app that reuses Bridge RNA's code and visual language.
It does not perform batch-effect correction by default; instead it makes batch structure visible and guards against misreading it (see the lasso analysis section and the open questions).

## 2. The system we are extending (verified)

Every fact in this section was verified directly against the checkpoint, the memmap, and the data files, not inferred from documentation.
The details, with line numbers, are in `REFERENCE.md`.

Bridge RNA's pipeline is:

```
OSDR counts -> human-ortholog TPM vector -> ExpressionPerformer embedding (512-d)
            -> cosine top-k over the ARCHS4 embedding index -> GEO metadata -> AI summary
```

The `ExpressionPerformer` is a 12-layer flash-attention transformer with hidden dim 512, 8 heads, ffn dim 2048, trained with `include_species_embedding=False`.
It ingests a length-15,165 log1p-TPM vector in a fixed canonical gene order and returns a 512-d embedding by mean-pooling the final hidden states over the gene axis.
The checkpoint is `checkpoints_performer/r7hnr92k/best_model.pt` (547 MB), and its true hyperparameters live in `ckpt['config']`, which is why any reimplementation must read config from the checkpoint rather than trusting the demo's fallback defaults.

The ARCHS4 index is a memory-mapped `940455 x 512` float16 array at `archs4_sample_embeddings_full/sample_embeddings.float16.mmap`, with a sidecar `sample_locations.parquet` carrying `global_index`, `geo_accession`, and `species_id` (0 = human, 510,709 samples; 1 = mouse, 429,746 samples).
The stored embeddings are NOT L2-normalized; measured L2 norms range from 6.7 to 25.5 (mean 10.6).
Retrieval normalizes at query time, so Bridge Manifold must L2-normalize before any reduction, or the dominant axis of variation will simply be vector magnitude.

OSDR embeddings do not exist on disk yet.
There is a hook in the Bridge RNA app for a precomputed OSDR query embedding file (`PRECOMPUTED_QUERY_EMBEDDING_CANDIDATES`), but no such file is present, so Bridge Manifold must generate them.

## 3. Architecture overview

Bridge Manifold splits cleanly into an offline precompute stage and an online serving stage.
This split is the single most important architectural decision, and it is forced by measured performance (see section 8).

```
OFFLINE (run once, cached artifacts)          ONLINE (Dash app, loads artifacts only)
-------------------------------------          ----------------------------------------
embed_osdr.py                                  app_manifold.py
  OSDR counts -> 2,896 x 512 embeddings          loads coord parquets  (mmap-cheap)
  cache/osdr_sample_embeddings.npy               loads ARCHS4 memmap   (for 512-d lasso stats)
                                                 loads hnswlib index   (for kNN-purity)
build_projections.py                             renders Scattergl + datashader underlay
  L2-normalize -> PCA-50 -> {pca2,pca3,           lasso -> coherence.py -> readout panel
    umap2, umap3} on the joint corpus
  cache/coords_*.parquet
  build hnswlib cosine index (512-d)
  render datashader density PNGs
```

The app never runs UMAP, never runs the model, and never holds all 940k glyphs live.
It reads precomputed coordinates, samples them, and draws them.
The only heavy thing it keeps resident is the 963 MB float16 memmap, which it touches only to pull the 512-d vectors for a lasso selection.

## 4. Data pipeline

### 4.1 OSDR embedding generation (`precompute/embed_osdr.py`)

This is the highest-risk piece, because a subtle preprocessing mismatch produces embeddings that look fine but are scientifically wrong.
The mitigation is to reproduce Bridge RNA's exact preprocessing rather than reinvent it, and to gate the run on the canonical gene digest.

The recipe, in exact order, reusing `demo_osdr_top5.py` logic:

1. Load the OSDR metadata TSV and filter to Mus musculus rows with a non-null counts path and a non-empty spaceflight factor.
2. For each sample, read its per-sample raw-count CSV, strip Ensembl version suffixes, select the sample's count column.
3. Map mouse Ensembl IDs to human gene symbols via `orthologs_one2one.txt` (one-to-one orthologs only), collapse duplicate human genes by summation.
4. Reindex to the 15,165 canonical genes in canonical order, filling missing genes with 0.
5. TPM-normalize using mouse exon lengths from `gencode_v49_mouse_gene_exon_lengths.csv`, then `log1p`.
6. Batch the resulting `[N, 15165]` float32 matrix through `ExpressionPerformer.encode(x, None, normalize=False)`.

The gate: compute `canonical_gene_order_digest(genes)` and assert it equals `CANONICAL_GENES_SHA256`.
If it does not match, abort the build; do not silently produce invalid embeddings.

Device: the existing code path is CPU fp32 (there is no MPS branch, and CUDA is absent on this machine).
For 2,896 samples through a 12-layer transformer this is minutes, which is acceptable for a one-time offline job.
We may add an optional MPS branch as a speedup, but CPU fp32 is the fidelity baseline and the default.

Output: `cache/osdr_sample_embeddings.float32.npy` (2,896 x 512, ~5.9 MB) plus a `cache/osdr_metadata.parquet` carrying `sample_key` and the color-by columns.

### 4.2 ARCHS4 consumption

ARCHS4 embeddings are consumed directly from the existing memmap; nothing is regenerated.
`sample_locations.parquet` provides `global_index`, `geo_accession`, and `species_id`.

### 4.3 Joint coordinate space

OSDR and ARCHS4 are reduced together so their coordinates are comparable.
The reducers are fit on a stratified subsample of the union and then used to transform the full corpus, which keeps both datasets in one space without paying the cost of fitting on all 943,351 points.

### 4.4 Note on comparability

OSDR is embedded here in fp32 on CPU, while ARCHS4 was embedded in bf16 on CUDA during index construction.
This introduces a small precision batch effect between the two corpora.
We measure this jitter (embed a handful of samples both ways, report the cosine delta) and surface it in the cross-dataset lasso warning, so a user never mistakes a precision artifact for biology.

### 4.5 ARCHS4 sample metadata (`precompute/fetch_archs4_meta.py`)

Decision (Josh, 2026-07-20): fetch ARCHS4 tissue metadata for v1, so ARCHS4 points color by tissue and not only species.
The local ARCHS4 artifacts carry only `geo_accession` and `species_id`, so tissue and source come from the ARCHS4 gene-level HDF5 files.
Bridge RNA already has the reader: `fetch_archs4_metadata(geo_accessions, human_h5, mouse_h5)` (demo_osdr_top5.py:463), which uses `archs4py` to pull per-GSM fields from `data/archs4/human_gene_v2.5.h5` and `data/archs4/mouse_gene_v2.5.h5` (downloaded from archs4.org/download, tens of GB each).

The step extracts tissue, source name, and series for all 940,455 accessions once and caches `cache/archs4_metadata.parquet` keyed by `geo_accession` (and joined to `global_index`).
The metadata group in the H5 is small relative to the expression matrix, so once the files are downloaded the extraction is fast; the download is the long pole and should be kicked off early.
Until the parquet exists, ARCHS4 colors by species only and the tissue color-by is disabled with a clear message, so the app degrades gracefully rather than failing.

## 5. Dimensionality reduction

### 5.1 The reduction spine: L2-normalize -> PCA-50 -> UMAP

Measured on a 25k ARCHS4 sample, PC1 alone captures 57.8% of variance and the first 50 PCs capture 96.4%.
That giant PC1 is almost certainly a magnitude or depth axis, which is exactly why we L2-normalize first.
PCA to 50 dimensions then denoises and compresses the input to UMAP, which is standard practice and makes UMAP both faster and cleaner.

### 5.2 PCA

PCA is cheap: fitting PCA-50 on 25k points takes ~12 s, and projecting all 940k to 2D via the fitted components takes ~7 s.
We precompute `pca2` and `pca3` coordinates for the full joint corpus.
Because PC1 dominates, a raw PCA-2D view is mostly one axis plus noise; we keep it because it is honest about global magnitude structure and it is a fast sanity layer, but UMAP is the primary exploratory view.

### 5.3 UMAP

UMAP is expensive and strictly offline.
Measured: a UMAP fit on 40k points (in PCA-50 space) takes 171 s, and `.transform()` runs at ~21 s per 60k.
A direct fit on all 940k would take hours and risk memory blowup, so we use the landmark pattern: fit on a stratified ~200k subsample, then `.transform()` the remaining ~740k in batches.
This lands the whole corpus in a single UMAP space in roughly 30 to 90 minutes offline, and the result is cached as `umap2` and `umap3` coordinate parquets.
The app loads these coordinates; it never invokes UMAP.

### 5.4 Honesty about UMAP distances

UMAP preserves local neighborhoods, not global distances, and cluster separation and cluster sizes in a UMAP plot are not quantitatively meaningful.
The app states this in the UI, and critically, every "meaningfully related" statistic is computed in the original 512-d space, never from the 2D UMAP coordinates.
The lasso defines which points; the pixels never define the statistic.

## 6. Rendering

The renderer is one `dcc.Graph` holding one set of Plotly `go.Scattergl` (WebGL) traces, because browsers cap WebGL contexts at roughly 8 to 16 and a single context shared across traces is the safe budget.

Layers, back to front:

1. Datashader density underlay.
   Offline, `ds.Canvas(2048x2048).points()` rasters all 940,455 coordinates to a PNG per reduction, placed as a `layout.images` underlay.
   This keeps the true global manifold shape always visible even though only a fraction of glyphs are live.
2. ARCHS4 background.
   A single Scattergl trace of 100,000 stratified points by default, 3 px, opacity ~0.45, hover disabled (hover hit-testing is a dominant cost at scale).
   For a categorical ARCHS4 color (species, or tissue via the fetched metadata) this splits into up to Top-11-plus-Other traces; for an OSDR-only color it renders as flat neutral grey.
3. OSDR overlay.
   All 2,896 OSDR points, always drawn, 7 to 8 px, distinct symbol with a white outline, full hover, so the spaceflight samples pop above the ARCHS4 cloud.

Point budgets (decisive):

| Layer | Default | Max |
| --- | --- | --- |
| OSDR | 2,896 (100%, never subsampled) | 2,896 |
| ARCHS4 | 100,000 (10.6%) | 250,000 (with a reduced-interactivity banner) |
| Total live glyphs | <=103k | <=253k |

These budgets sit comfortably under the ~150k to 300k range where Scattergl lasso and pan begin to degrade.

Level of detail: on zoom (`relayout`), the client sends the new x/y bounds and the server re-runs stratified sampling over the full 940k coordinates restricted to that window, returning a fresh 100k, so zooming reveals fine structure instead of just enlarging sparse dots.
3D uses `Scatter3d` capped near 60k points, since 3D WebGL is heavier.
Config: `displaylogo` off, `scrollZoom` on, and `uirevision` set so zoom survives a color-by change.

## 7. The lasso readout: "are these meaningfully related?"

This is the scientific heart of the tool, and it is deliberately conservative.
When a user lassoes a set S, the app gathers the selected `global_index` values from the trace `customdata`, pulls their L2-normalized 512-d vectors from the memmap, and computes a multi-part readout entirely in 512-d cosine geometry.

A. Geometric cohesion.
The headline is the mean cosine of S to its own centroid.
For L2-normalized vectors this statistic is exactly `||mean(S)||`, which is an identity rather than an approximation and is what makes an honest null affordable: a null draw needs only the *sum* of its sampled vectors, never the per-point dot products.

The null is the distribution of that statistic over uniform random selections of the same size and the same corpus composition.
It is computed analytically rather than by resampling.
Under a size-n draw without replacement from a population of N vectors with mean `mu` and covariance `Sigma`, the sample mean has mean `mu` and covariance `((N-n)/(n(N-1))) * Sigma`; 1,000 draws from that Gaussian give the null.
The finite-population correction is load-bearing, not a refinement: it is what makes the null degenerate when the selection is the whole corpus, where the only honest z-score is 0.

This replaced an earlier bootstrap that resampled with replacement from a fixed 20,000-vector background pool, which was measurably wrong.
Such a null converges to that *pool's* mean rather than the population's, while its own spread keeps shrinking like `1/sqrt(n)`, so the fixed gap between pool mean and population mean turns into a z-offset growing as `sqrt(n / pool_size)` with a sign fixed by the pool's random seed.
Measured at the real corpus ratio, genuinely random selections scored `|z| > 40`.
The population moments the analytic null needs are therefore computed exactly over the whole corpus in `build_projections.py`, never estimated from a subsample.
`tests/test_coherence.py` checks the analytic null against brute-force resampling at several selection sizes, and checks that random selections score `z ~ 0` from 1% to 80% of the corpus.

The companion robust statistic is kNN-purity via the prebuilt hnswlib cosine index: for each point in S, the fraction of its 15 nearest neighbors that are also in S, divided by the expected fraction |S|/N, reported as a fold-enrichment.
Effect size (z, fold) is reported alongside p, because with large |S| every p is ~0 and significance without effect size is false confidence.
For the same reason a category must clear 1.25x fold enrichment before the verdict names it as a *driver*; everything significant still appears in the enrichment table with its own fold, but a 1.05x deviation that reaches q < 1e-10 on sample size alone is not an explanation.

B. Metadata enrichment.
For every categorical field (spaceflight, tissue, strain, sex, study, species, source), each category is tested with a one-sided hypergeometric test against the full-population background, all p-values are pooled and Benjamini-Hochberg corrected, and the top categories are reported by fold-enrichment with their adjusted q-values.
This is what actually explains why a selection is related.

C. Batch-confound guard.
The app computes the fraction of S coming from the single most-represented study (`id.accession`).
If study is the top-enriched field, or if one study is more than half the selection, it warns that the coherence is likely study or batch driven rather than biological, and recomputes spaceflight enrichment within the dominant study, since Flight vs Ground is confounded with study batch but many OSD studies contain both arms.

D. Cross-dataset caution.
If S mixes OSDR and ARCHS4, the app warns that proximity is confounded by the fp32-vs-bf16 precision batch effect and the ortholog preprocessing provenance, and cites the measured precision jitter.

E. Honest negative.
If cohesion is not significant and nothing enriches at q < 0.05, the app says plainly that the selection resembles a random draw with no coherent structure.

The output is a single synthesized verdict string, for example:
"Coherent (z=8.3, empirical p<0.001, kNN-purity 5.1x); driven by tissue=Liver (fold 6.2x, q=3e-22) and spaceflight=Space Flight (fold 2.1x, q=1e-4); low batch confound (top study 18% of selection)."

## 8. Key design decisions and tradeoffs

Standalone app, not bolted into Bridge RNA.
Bridge Manifold is a separate Dash app in its own directory, importing reusable functions from Bridge RNA rather than editing the 2,470-line retrieval app.
This keeps the heavy exploratory tool from destabilizing the retrieval product, while a shared header and shared CSS make them feel like one instrument.
The tradeoff is a small amount of duplicated app scaffolding, which is worth it for isolation.

Offline precompute over interactive computation.
Every expensive step (model inference, UMAP, datashader rasters, the hnswlib index) is precomputed and cached, and the app only ever loads artifacts.
This is forced by the measured UMAP cost and is what makes the app responsive.
The tradeoff is a build step and cache management, which is the right trade for a 943k-point tool.

Statistics in 512-d, never in 2D.
The lasso selects points in the projection but every coherence and enrichment number is computed in the original 512-d space.
This is non-negotiable, because UMAP and PCA distances are distorted and a statistic read off the pixels would be a lie dressed as a measurement.

L2-normalize before reducing.
Because retrieval is cosine and the raw vectors carry a 4x magnitude spread that dominates PC1, we normalize first so the manifold reflects transcriptomic direction rather than sequencing depth.

Make batch visible, do not correct it (final).
Rather than applying Harmony or ComBat and risking erasure of real biology, the tool exposes study, species, and tissue as color-by dimensions and guards the lasso readout against batch-driven coherence.
Josh confirmed this is the final call: no correction algorithm, not even as a later toggle.

Bounded glyph budget with a density underlay.
Instead of trying to render 940k live points, we render ~100k live glyphs over a datashader raster of all 940k, so the global shape is always honest while interaction stays smooth.

## 9. Directory layout

This is the layout as built.

```
Bridge Manifold/
  app_manifold.py            # Dash entry: argparse host/port/debug, loopback guard
  manifold/
    paths.py                 # every artifact path; BRIDGE_RNA_ROOT / MANIFOLD_CACHE_DIR overrides
    preflight.py             # missing-artifact and Git-LFS-pointer guards
    bridge_rna.py            # the single seam that imports from the sibling repo
    data.py                  # npy/parquet/memmap loaders, population moments, module-level caches
    sampling.py              # stratified quota sampling, viewport re-stratification
    coherence.py             # the 512-d lasso statistics (analytic null, hypergeom, BH, kNN-purity)
    theme.py                 # plot theme (dark canvas) + categorical palette
    layout.py                # left control rail, main graph, legend, right readout panel
    callbacks.py             # color-by, layer toggles, method toggle, lasso, zoom LOD, legend filter
  precompute/
    embed_osdr.py            # OSDR counts -> 2,108 x 512 embeddings (loads torch), resumable
    fetch_archs4_meta.py     # ARCHS4 H5 -> per-GSM tissue/source/series parquet (optional)
    build_projections.py     # L2 -> PCA-50 -> {pca2,pca3,umap2,umap3}, hnswlib, density, moments
  tests/
    fixture_corpus.py        # synthetic corpus with known latent clusters
    conftest.py              # points the package at that corpus before import
    build_dev_corpus.py      # CLI to build a browsable corpus for the running app
    test_data.py             # global point order, vector gathers, color-by lookups
    test_coherence.py        # the null vs brute force, cluster vs random, BH, batch guard
    test_render.py           # customdata identity, budgets, viewport, legend
    test_app.py              # callback wiring, readout rendering, CSS class coverage
  assets/
    manifold.css             # Bridge RNA tokens, Dash 4 token remap, dark-canvas + legend rules
  cache/                     # generated (gitignored): embeddings, coords, index, density, moments
  requirements.txt
  REFERENCE.md
  IMPLEMENTATION.md
  progress.md
  README.md
```

`reduce.py` from the original sketch was never needed - reading coordinate parquets is three lines in `data.py`, and a module wrapping it would have been indirection for its own sake.

The files that import torch and umap are confined to `precompute/`, so the serving app has a light dependency surface: `dash`, `plotly`, `numpy`, `pandas`, `pyarrow`, `scipy`, and optionally `hnswlib`.

## 10. Reuse from Bridge RNA

The following are imported or copied rather than rewritten (signatures and line numbers in `REFERENCE.md`):
the OSDR preprocessing body from `load_random_osdr_sample_vector`, the model driver from `build_model_and_query_embedding`, `ExpressionPerformer` and its `encode`, `canonical_gene_order_digest` and `CANONICAL_GENES_SHA256`, `build_mouse_to_human_maps` and `normalize_counts_to_tpm_single`, `fetch_archs4_metadata` for the ARCHS4 tissue join, `_load_archs4_index` and `_topk_cosine_from_memmap`, the `preflight_retrieval_requirements` and `_is_lfs_pointer` LFS guards, and the theme tokens and header/panel/badge CSS classes.

Dependencies: Bridge Manifold shares the Bridge RNA venv and adds `dash`, `hnswlib`, and `archs4py` on top of the existing scientific stack (torch, umap-learn, scikit-learn, datashader, plotly).

## 11. Visual language

Bridge RNA is a light scientific-instrument theme, not a dark one.
Its tokens: canvas `#eef2f7`, panels `#ffffff`, primary text `#1a2432`, accent blue `#2b7fff`, teal `#0bab9f`, warm `#d9791b`, and a dark navy header `#14294a` with a teal rule `#22c7bd`.
Bridge Manifold matches this light chrome exactly, and uses a dark navy plot canvas inside it so the WebGL glyphs have contrast, which is the one deliberate departure.
The categorical palette for color-by will be finalized against the dataviz skill at build time, tuned for a dark scatter background and for colorblind safety.

## 12. Phased build plan

Each phase ends with an objective validation, not a visual glance.

Build status as of 2026-07-20 is tracked in `progress.md`; the plan below is the original phasing, kept for the validation criteria.

Phase 0. Scaffold.
Create the package skeleton, `manifold.css` importing the Bridge RNA tokens, and a preflight that fails clearly on missing LFS artifacts.
Validate: app boots and renders an empty themed shell.

Phase 1. OSDR embeddings.
Implement `embed_osdr.py` with the gene-digest gate; embed all 2,896 samples; cache the npy and metadata parquet.
Validate: digest matches, output shape is `2896 x 512`, and a spot check reproduces a Bridge RNA retrieval for a known sample.
In parallel, kick off the ARCHS4 H5 download and, once it lands, run `fetch_archs4_meta.py` to cache `archs4_metadata.parquet` (this is the long pole, so start it first).

Phase 2. PCA projections.
L2-normalize, fit IncrementalPCA-50 on a stratified subsample, project the joint corpus, write `pca2` and `pca3` parquets.
Validate: explained-variance matches the measured profile (PC1 ~58%), coordinates render.

Phase 3. First interactive plot.
Wire the Scattergl renderer, the datashader underlay, stratified sampling, the dataset and method toggles, and color-by for the OSDR fields plus ARCHS4 species and tissue (tissue from `archs4_metadata.parquet` once available).
Validate: 100k+2,896 points pan and zoom smoothly; OSDR overlay is legible; color-by switches without a full reload.

Phase 4. UMAP projections.
Implement the landmark fit-and-transform in `build_projections.py`; write `umap2` and `umap3`.
Validate: the offline job completes in the expected window and the OSDR points land in biologically sensible neighborhoods.

Phase 5. Lasso coherence.
Build the hnswlib index and `coherence.py`; wire the lasso to the readout panel with all five parts (cohesion, enrichment, batch guard, cross-dataset caution, honest negative).
Validate: a known-coherent selection (one tissue) scores high with the right enrichment; a random selection scores as incoherent.

Phase 6. Polish.
Custom searchable legend for high-cardinality color-bys, viewport re-stratification, 3D views, hover cards, and pixel-level theme matching.
Validate: high-cardinality color-bys are usable; the whole thing feels like one product with Bridge RNA.

The build order is deliberately offline-precompute-first, so that by the time the app is wired there is real data to render.
