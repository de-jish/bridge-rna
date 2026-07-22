# Bridge Manifold - Implementation Plan

> **This document predates the 2026-07-22 merge.**
> Bridge Manifold and Bridge RNA are now one repository and one application, served by `app.py`: the retrieval view at `/` and this map at `/map`.
> There is no `app_manifold.py` and no separate repository at `/Users/josh/Bridge Manifold`.
> The design decisions recorded below are still the ones the map is built on; the commands and the file layout have been updated where they would otherwise fail if followed.
> See `README.md` for the current product and `progress.md` for what changed.



Bridge Manifold is the exploratory companion to Bridge RNA.
Where Bridge RNA answers "what are the closest Earth analogs for one spaceflight sample," Bridge Manifold answers "what is the global shape of the whole embedding space, and where does spaceflight sit inside it."
It dimensionally reduces the 512-dimensional ExpressionPerformer embeddings of both corpora, ARCHS4 (940,455 human and mouse GEO samples) and OSDR (2,108 NASA GeneLab spaceflight samples), draws them together in one interactive WebGL scatter, and colors them by biology that is defined for both corpora rather than for one.

This document is the master plan.
It records the architecture, the design decisions, the tradeoffs weighed, the candidates that were built and then rejected, and the phased build order.
The verified ground-truth facts it relies on live in `REFERENCE.md`.
The living status log and the open questions live in `progress.md`.

## 1. Goals and non-goals

### Goals

The tool must reduce and plot both corpora in a single shared 2D and 3D coordinate space so OSDR and ARCHS4 points are directly comparable.
It must offer both a fast linear method (PCA) and a structure-preserving nonlinear method (UMAP), and be explicit about when each one lies.

It must color **both** corpora by real biology, not one corpus by biology and the other by nothing.
This is a hard requirement and it is the goal that shaped the current design.
A color-by that describes only the 2,108 OSDR samples paints 99.8% of the map a single flat color, and a flat color on a scientific plot reads as a measurement: "ARCHS4 was measured and has no structure here."
So the tool must carry at least one whole-map field that is genuine biology (tissue, folded onto one vocabulary shared by both corpora), it must state in the interface exactly how many points the selected field colors, and when a field genuinely does not describe a corpus it must render that corpus as *context* rather than as a category.
The rule, stated once: **the tool must never present an uncolored corpus as though the absence of a label were data.**

It must render all 942,563 points smoothly at interactive frame rates using Plotly WebGL scatter traces, with stratified sampling and viewport level-of-detail available for anyone who wants a lighter view.
It must make batch structure visible and disclose the measured cross-corpus technical effect where a user will actually read it.

### Non-goals

Bridge Manifold does not retrain or fine-tune the ExpressionPerformer model.
It does not re-embed ARCHS4; those 940,455 embeddings already exist and are consumed as-is.
It does not replace the Bridge RNA retrieval app; it is a separate app that reuses Bridge RNA's code and visual language.
It does not perform batch-effect correction by default; instead it makes batch structure visible as a color-by and discloses the measured effect on the control rail.

It does not compute statistics on demand.
An earlier version had a lasso selection tool with a 512-d statistical readout (cohesion against an analytic null, hypergeometric metadata enrichment, a batch-confound guard).
**That feature was removed in its entirety**, along with `manifold/coherence.py`, its tests, the hnswlib index, and the exact population moments it needed.
The map is read, not queried.
Everything below describes the tool as it now exists; the removal is recorded here only so nobody rebuilds it by accident.

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

OSDR embeddings did not exist on disk; Bridge Manifold generates them.
There is a hook in the Bridge RNA app for a precomputed OSDR query embedding file (`PRECOMPUTED_QUERY_EMBEDDING_CANDIDATES`), but no such file is present.

The memmap is read **only by the precompute scripts**.
The map view draws precomputed coordinates and never needs a 512-d vector, so it never opens it. (The retrieval view does, on every cached search.)

## 3. Architecture overview

Bridge Manifold splits cleanly into an offline precompute stage and an online serving stage.
This split is the single most important architectural decision, and it is forced by measured cost.

```
OFFLINE (precompute/, run once -> cache/)      ONLINE (app.py /map, loads artifacts only)
---------------------------------------        ----------------------------------------------
embed_osdr.py                                  app.py
  OSDR counts -> 2,108 x 512 embeddings          coords_{pca,umap}{2,3}.parquet
  cache/osdr_sample_embeddings.float32.npy       points_meta.parquet   (identity table)
  cache/osdr_metadata.parquet                    osdr_metadata.parquet (OSDR labels)
                                                 archs4_metadata.parquet (GEO join)
build_projections.py
  L2-normalize, then two full-corpus fits       renders every point as go.Scattergl
  -> {pca2, pca3, umap2, umap3} parquets        colorby registry -> one label array
  -> points_meta.parquet, archs4_geo.parquet    stratified sample + viewport LOD

fetch_archs4_meta.py
  sigpy JSON API -> archs4_metadata.parquet
  (series, title, source, characteristics,
   and the canonical tissue bucket)
```

The map never runs the model, never runs UMAP, and never opens the 963 MB memmap. (The retrieval half opens the memmap; it still never runs the model on the cached path.)
It reads small precomputed tables and draws them.
It *can* hold all 942,563 glyphs live, and by default it does; what it must never do is compute the coordinates they sit at.
The whole serving dependency surface is `dash`, `plotly`, `numpy`, `pandas`, and `pyarrow`: no scientific stack at all.

A practical consequence worth stating explicitly: `BRIDGE_RNA_ROOT` is needed to **build** the cache, not to **run** the app.
A machine holding only `cache/` can serve the map.
`manifold/preflight.py` encodes exactly that split, and its `APP_REQUIRED` list is deliberately short: the point identity table, the OSDR metadata, and the PCA 2D coordinates.
`points_meta.parquet` is listed first because it is read first, by `layout.control_rail()` through `data.counts()` while the layout is still being constructed; when it was missing from the list, a bare cache passed preflight and then crashed during startup.

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
The digest is part of the expression-cache key too, so the resume path cannot skip the gate.

Corpus size, corrected during the build: the TSV has 2,896 rows, 2,163 carry a non-empty spaceflight factor (the Bridge RNA filter Josh chose to match), and 2,108 of those produce an expression vector.
The other 55 name a sample column their counts matrix does not contain, and the script reports them rather than dropping them silently.
Earlier drafts of these documents used 2,896 as the corpus size, which was wrong.

Device: CPU fp32, which is both the fidelity baseline and, measured, the fastest available path on this machine.
MPS is not viable for this model: `F.scaled_dot_product_attention` has no fused kernel for a 15,165-token sequence on Metal and fails on a 6.85 GiB allocation, and chunking the attention makes it run but does not make it faster.
Measured throughput was ~6.5 s/sample in isolation; the real run took about 11.3 hours for 2,108 samples under machine contention.
The log1p-TPM stage is cached and the partial embedding is resumable, so an interrupted run never re-reads the counts CSVs.

Output: `cache/osdr_sample_embeddings.float32.npy` (2,108 x 512, 4.3 MB) plus `cache/osdr_metadata.parquet` carrying `sample_key` and the color-by columns.

### 4.2 ARCHS4 consumption

ARCHS4 embeddings are consumed directly from the existing memmap; nothing is regenerated.
`sample_locations.parquet` provides `global_index`, `geo_accession`, and `species_id`.
`build_projections.py` writes the accession list out as `cache/archs4_geo.parquet` in the fixed global order, which is what the metadata fetch later joins onto.

### 4.3 Joint coordinate space

OSDR and ARCHS4 are reduced together so their coordinates are comparable.
Global point order is fixed as `[all ARCHS4 in global_index order, then all OSDR in row order]`, and every artifact shares it, so a point index addresses the same sample in every table.
The reducers are fit on a stratified subsample of the union, with all OSDR points always included in the fit, and then used to transform the full corpus.

### 4.4 Comparability across the two corpora

OSDR is embedded here in fp32 on CPU, while ARCHS4 was embedded in bf16 on CUDA during index construction, through a different preprocessing path.
That is a precision and provenance batch effect between corpora, and it has now been measured rather than assumed.

Of the 50 nearest neighbours of each OSDR sample in 512-d cosine space, 34.3% are same-study OSDR, 22.9% are cross-study OSDR, and 42.9% are ARCHS4.
Against a per-sample chance model that is 5,233x for same-study (replicate structure, expected and biological) and 105x for cross-study.
Controlling for tissue, the dominant axis of variation in bulk expression: OSDR neighbour pairs sharing **neither study nor tissue** occur at 11.491% against 0.21101% expected, **54x over chance**.
Biology does not make liver cluster with brain, so that residue is technical.

Those are the *exact* figures. They were first obtained through an approximate index, which returned 34.2% and 5,227x; the index is gone and `validate_artifacts.py --mixing` now computes exact neighbours, so the two differ only in the third digit (`REFERENCE.md` section 4).

This is a property of the map itself, not of any particular reading of it, so it is disclosed on the control rail where every user sees it, in plain language, with the number attached.
`precompute/validate_artifacts.py --mixing` recomputes it and raises a warning above 50x, so the claim in the interface stays tied to a measurement that can be re-run.

### 4.5 ARCHS4 sample metadata (`precompute/fetch_archs4_meta.py`)

Every ARCHS4 point needs GEO metadata, because without it the corpus can only be colored by species and the whole map has exactly one non-OSDR field.
The local artifacts carry only `geo_accession` and `species_id`.

Three routes were compared, by measurement rather than by argument.

**Full HDF5 download (rejected).**
The obvious route, and the one the original plan specified, is the ARCHS4 gene-level HDF5 files, where per-sample metadata lives in small 1-D datasets under `meta/samples/`.
The current human build is 62.3 GB and mouse is 50.7 GB, for a few hundred MB of strings.
Those files were never downloaded, which is precisely why the "Tissue (ARCHS4)" color-by existed in the interface for the whole first build and had never once worked.

**Partial HDF5 read over HTTP range requests (rejected).**
This genuinely works: with `fsspec` and `h5py` the whole `meta/samples` group is enumerable in about 18 s without downloading the file.
But the fields are gzip-chunked variable-length strings, so a single field costs about 5 minutes and about 272 MB of transfer, and the six useful fields across both species run to hours.

**The Maayan Lab sigpy JSON API (chosen).**

```
POST https://maayanlab.cloud/sigpy/meta/samplemeta
{"species": "human"|"mouse", "samples": ["GSM...", ...]}
-> {GSM: {series, title, source, characteristics}}
```

Measured on the real corpus by actually running it: **33.7 seconds, 39 requests, about 216 MB of transfer, and 99.911% of all 940,455 accessions resolved** (human 99.851%, mouse 99.982%).
Batches of 25,000 accessions sustain 24k-42k samples/s with no rate limiting observed.
The output is `cache/archs4_metadata.parquet`: 940,455 rows, 32.5 MB, carrying `geo_accession`, `series_id`, `title`, `source_name`, `characteristics`, and the derived canonical `tissue` bucket, with 51,284 distinct GEO series.
Three orders of magnitude cheaper than either HDF5 route for exactly the same information.

Two guards are load-bearing rather than defensive.
The rows are reindexed onto the fixed global order by accession and never assembled positionally, because the response object silently omits misses and a positional assembly would shift every label after the first gap.
And a hit rate below 50% aborts the run without writing anything: the endpoint answers HTTP 200 with an empty object when the payload key is wrong (`gsm_ids` instead of `samples`), which would otherwise write a fully empty table and grey the map while reporting success.

**The 839 unresolved samples are not GEO withdrawals**, and it is worth being precise because the obvious explanation is wrong.
They are present with full metadata in the release-matched v2.5 metadata files and absent from the newer v2.latest that the API serves.
ARCHS4 releases are therefore not append-only: a rebuild can drop samples, which disproves the assumption that motivated using the newest release in the first place.
Those 839 points (0.089% of the corpus) get tissue `Unknown` and an empty series rather than being dropped or guessed at.
The coverage figure was cross-checked independently by a partial HDF5 read of `meta/samples/geo_accession` off the 62 GB remote file; both methods return exactly 509,949 human matches.

**The documented upgrade path, deliberately not taken.**
ARCHS4 publishes metadata-only HDF5 files under *versioned* names: `human_meta_v2.5.h5` (311.8 MB) and `mouse_meta_v2.5.h5` (350.9 MB).
The unversioned "latest" spellings return 403, which is why these files are easy to miss and why the first survey concluded that the gene files were the only route.
Reading the versioned pair gives exactly 100.000% coverage against this corpus and is release-matched to the embeddings, at a cost of 663 MB and roughly 8.5 minutes against 216 MB and 35 seconds, plus an `h5py` dependency nothing else needs.
For a color-by, 0.089% of points reading "Unknown" is not worth 15x the build time.
If this ever needs to be a build gate rather than a color, switch to the versioned files and assert 100%.

This step needs a network connection and nothing else: no `h5py`, no `archs4py`, no multi-GB download.
It remains optional.
Without it, ARCHS4 colors by species only, and the tissue field is offered but disabled with the exact command to run (section 7).

Ordering matters: `fetch_archs4_meta.py` joins onto `archs4_geo.parquet` and `points_meta.parquet`, so it runs **after** `build_projections.py`, and it aborts with that instruction if those files are missing.

## 5. Dimensionality reduction

### 5.1 The reduction spine: L2-normalize, then fit on everything

Measured on a 25k ARCHS4 sample before normalization, PC1 alone captures 57.8% of variance and the first 50 PCs capture 96.4%.
That giant PC1 is a magnitude axis, which is exactly why we L2-normalize first.
It is *not* a sequencing-depth axis, though this document said so for a long time; see `REFERENCE.md` section 4 for the measurement that corrected it.

Everything downstream of that normalization is fit on **all 942,563 points**.
This is a change from the first build, which fit PCA on a 60,000-point subsample and UMAP on a 122,563-point landmark set, and it is worth stating why the approximations were there and why they are not any more.
Both were adopted on estimates, not measurements: a direct UMAP fit was assumed to cost hours and risk memory blowup, and PCA was assumed to need the subsample because streaming the corpus through `IncrementalPCA` was the only full-corpus route considered.
Neither assumption survived being measured, and both are recorded in section 5.2 and 5.3 below.

An intermediate PCA-50 step also used to sit between normalization and UMAP.
It was removed earlier, on the measurement in `run_umap`: reducing to 50 components first discarded the 4.9% of variance those components do not carry, and cost 12% of local fidelity against feeding UMAP the raw 512-d vectors under cosine.
So the spine is now two independent reductions of the same normalized 512-d corpus rather than a chain.

### 5.2 PCA, exact over every point

PCA needs nothing from the data beyond its mean and its second moment, and both are sums.
One streaming pass accumulating `s = sum_i x_i` and `G = sum_i x_i x_i^T` in float64 determines the covariance exactly, `C = (G - n mu mu^T) / (n - 1)`, and `eigh(C)` then yields the same components and the same explained-variance ratios as `sklearn.decomposition.PCA` fit on the materialized 942,563 x 512 matrix, to float64 round-off.
`tests/test_projections.py` asserts exactly that equality rather than trusting the derivation.

Measured on the real corpus: the accumulation over all 942,563 points took **6 s** and the projection onto the leading 3 components took **2 s**.
The subsampled `IncrementalPCA` it replaced took about 1 s to fit and 1 s to transform, so exactness cost roughly 6 extra seconds, which is not a tradeoff so much as an oversight being corrected.

Two consequences follow from the fit being exact rather than truncated.
The full 512-eigenvalue spectrum is available for free, so `projection_stats.json` records all of it instead of the leading 50, and `validate_artifacts.py` checks that it sums to 1 as evidence the build did not silently fall back to a truncated fit.
Eigenvector signs are pinned with the same largest-absolute-entry rule sklearn's `svd_flip` uses, because an eigensolver is free to return `-v` for `v` and a rebuild that mirrored the map for no reason visible in the data would be a bad surprise.

The exact full-corpus figures are **PC1 = 41.3% and the cumulative over 50 PCs = 95.0%**, against the 40.9% / 95.1% the 60,000-point subsample reported.
The subsample was close, which is the honest thing to say about it; it was still an estimate of a number that costs 6 seconds to know.
That PC1 figure is the objective test for the normalization invariant: a build landing near 57.8% is evidence normalization was silently skipped, and `precompute/validate_artifacts.py` fails the build above a 50% ceiling.
Because PC1 still dominates, a raw PCA-2D view is mostly one axis plus noise; we keep it because it is honest about global magnitude structure and it is a fast sanity layer, but UMAP is the primary exploratory view.

### 5.3 UMAP, fit on the whole corpus

UMAP is expensive and strictly offline, but it is not as expensive as this document assumed for the whole first build.
The landmark pattern it used, fitting 122,563 points and pushing the remaining 819,999 through `.transform()`, is worth being precise about, because `.transform()` does not lay those points out at all.
It places each new point by a weighted average of where its landmark neighbours already sit, so 87% of the corpus could only ever land inside the region the landmarks had already staked out, and none of it exerted any force on the layout it appears in.

The current build fits all 942,563 points directly.
The k-nearest-neighbour graph is built once with `pynndescent` and handed to both the 2-d and the 3-d fit through UMAP's `precomputed_knn`, because the graph depends on the input space and `n_neighbors` and never on `n_components`.
That halves the neighbour search for the build and, more usefully, guarantees the two maps are layouts of the *same* graph rather than of two independent approximations of it.

Measured on the real corpus, on a 10-core M4 with 16 GB of RAM:

| stage | cost |
| --- | --- |
| materialize the normalized 942,563 x 512 corpus | 2 s, 1.93 GB resident |
| k=15 cosine neighbour graph, `n_jobs=1` | 115 s |
| UMAP 2-d layout | see `REFERENCE.md` section 4 |
| UMAP 3-d layout | see `REFERENCE.md` section 4 |

`n_jobs=1` on the neighbour graph is deliberate and is the default of the `--knn-jobs` flag.
NN-descent's heap updates race under threads, so the same seed gives a slightly different graph run to run at `n_jobs=-1`, and this graph is the artifact every downstream coordinate derives from.
Single-threaded it takes 115 s, which is a small price for a reproducible build; `--knn-jobs -1` is roughly 10x faster for anyone who wants it and does not care.
UMAP itself forces `n_jobs=1` whenever `random_state` is set (`umap_.py:1952`), so the layout is single-threaded regardless, and `random_state=42` stays for the same reason it always did.

The app loads these coordinates and never invokes UMAP.

### 5.4 Honesty about UMAP distances

UMAP preserves local neighborhoods, not global distances.
Cluster separation and cluster sizes in a UMAP plot are not quantitatively meaningful, and the gap between two blobs does not measure how different they are.
The control rail says so, in those words, directly under the projection toggle rather than in a footnote.

That honesty is now a constraint on what the tool is willing to *do*, not only on what it says.
The map is a picture, so the tool draws pictures: coordinates, colors, and density.
It deliberately does not compute a number from a region of the plot, because a statistic read off distorted pixels would be a lie dressed as a measurement, and the same reasoning rules out any future feature that turns a screen region into a score.
Where a candidate color-by claimed to encode a quantity, it was tested against a structure-free null before being offered, and most of them did not survive that test (section 7.5).

## 6. Rendering

The renderer is one `dcc.Graph` holding one set of Plotly `go.Scattergl` (WebGL) traces, because browsers cap WebGL contexts at roughly 8 to 16 and a single context shared across traces is the safe budget.

Layers, back to front:

1. **ARCHS4 background.**
   A WebGL sample of the 940,455-point corpus, 3.4 px, hover disabled, split into one trace per display category.
   Hover hit-testing is a dominant cost at this scale, and `hoverinfo="skip"` alone is not enough because a `hovertemplate` overrides it; the two are turned off together.
   ARCHS4 traces carry **no `customdata`**.
   It existed only to feed the removed selection tool, and it was roughly 600 KB of dead payload per figure.
2. **OSDR overlay.**
   All 2,108 OSDR points, always drawn, 8.5 px diamonds with a 1.1 px white ring and full hover, so the spaceflight samples stay findable in the cloud.
   Hover carries `[sample_key, category]`: which sample this is, and what it is under the current color-by.

There used to be a third layer underneath both, and its removal is the reason the budget table below looks different from the one this document used to carry.

**The density underlay, and why it is gone.**
Offline, all 942,563 coordinates went through a 2048x2048 numpy `histogram2d`, a `log1p`, and a navy-to-blue-to-teal ramp rendered to a PNG by Pillow, placed as a `layout.images` underlay at its recorded extent.
It existed to solve a problem: 940k live WebGL glyphs was assumed to be out of reach, so ~100k were drawn live and the raster carried the rest, keeping the global shape honest while interaction stayed smooth.

The premise turned out to be wrong, and measurably so.
Building the figure costs the same at every budget, because the dominant cost is resolving one label array over the full corpus rather than the size of the sample drawn from it.
Serializing all 942,563 points costs 0.15 s and produces an 11.3 MB payload, against 0.03 s and 1.3 MB at 100,000, and the coordinates travel as plotly 6 base64 typed arrays rather than as JSON number lists, which is what keeps that number small.
So the raster was buying a smoothness that did not need buying, at the cost of a second, unlabelled encoding of the same data sitting under the glyphs, a PNG artifact per projection, an extra build stage, a `--density-only` flag, and the Pillow dependency.

Removing it also collapses a branch that was quietly load-bearing.
The raster was doing double duty as the honest fallback for an uncovered corpus (section 7.4), which meant the fallback had two forms: draw nothing and let the raster carry the shape, or, where there was no raster (3-D, or the underlay toggled off), draw a faint context cloud.
Two paths to the same requirement is one more than the requirement needs, and the 3-D path was already the general one.
Now there is only the context cloud, in 2-D and 3-D alike.

Point budgets (decisive):

| Layer | Default | Range |
| --- | --- | --- |
| OSDR | 2,108 (100%, never subsampled) | 2,108 |
| ARCHS4, 2D | 940,455 (100%) | 100,000 / 250,000 / 500,000 / all 940,455 |
| ARCHS4, 3D | capped at 40,000 | 40,000 |
| Total live glyphs, 2D | 942,563 | 102k to 942,563 |

The default is now the whole corpus, which is the point of the change: the map's first frame shows every sample it has rather than a tenth of them over a picture of the rest.
The lower tiers stay because they are genuinely faster to pan and zoom on a slower machine, and because a viewport re-sample at a partial budget is how fine structure is revealed on zoom.
3-D keeps its own much lower cap because `Scatter3d` is not `Scattergl`: it has no WebGL fast path of the same kind, and 40,000 is where it stays interactive.

**One palette across both corpora.**
Categories are ranked once over the whole covered population and every layer draws from that single mapping, so a liver in GEO and a liver in OSDR get the same color.
Ranking per layer, which is what the first implementation did, silently gave one category two different colors whenever the two corpora ordered their categories differently, which is a legend that lies.
The top 11 categories take the validated categorical palette; residual categories ("Other", "Unknown") keep their own rows at the neutral end and always sort last, so they never outrank a category that carries information; everything past the palette folds into one grey overflow row.
Legend counts are whole-corpus counts, not counts of the drawn sample, so they do not move when the point budget or the zoom changes.
A legend number is read as "how many such samples exist", and it should answer that question.

**A corpus a field does not describe is drawn as context, not as data.**
See section 7.4 for the full argument.

Level of detail: on zoom (`relayout`), the new x/y bounds become a viewport and the server re-runs stratified sampling over the full 940k coordinates restricted to that window, so zooming reveals fine structure instead of enlarging sparse dots.
A relayout that is not a zoom (a hover, a legend click, a drag-mode switch) returns a sentinel that leaves the current sample alone rather than triggering a resample.
Config: `displaylogo` off, `scrollZoom` on, `dragmode="pan"`, and `uirevision` set so zoom survives a color-by change.
Both selection tools are removed from the modebar (`select2d` and `lasso2d`), because no selection feature exists and a marquee that does nothing is a promise the app does not keep.
The first version of this config removed `select2d` but not `lasso2d`, so the lasso button was in fact still on the modebar after the feature was gone.

## 7. Coloring both corpora honestly

This section is the intellectual core of the current design.
Everything in it exists to answer one question that the first build got wrong: what should a map show for the corpus that the selected field does not describe?

### 7.1 The problem

The first build had about ten color-bys for the 2,108 OSDR samples (spaceflight arm, flight status, tissue, strain, sex, genotype, study, habitat, duration, diet) and exactly one for the 940,455 ARCHS4 samples (species).
Choosing any OSDR field painted 940,455 of 942,563 points, 99.8% of the map, a single flat grey.

That is not a cosmetic problem.
On a scientific plot a uniform color is a statement, and the statement it makes is "these samples were measured and are all the same on this axis."
The truth was "this field says nothing about these samples", which is a completely different claim.
A "Tissue (ARCHS4)" option did exist, but it required the ARCHS4 gene HDF5 files (62.3 GB human, 50.7 GB mouse) which were never downloaded, so it had never once worked in the entire history of the app.

The redesign therefore had to do three things: give ARCHS4 a real biological label (section 4.5), make that label share a vocabulary with OSDR's so one field can paint the whole map (7.2), and make coverage a declared property so the interface and the renderer can both act on it (7.3, 7.4).

### 7.2 One tissue vocabulary, not two (`manifold/tissue.py`)

OSDR and ARCHS4 name tissues in disjoint registers.
OSDR is curated and anatomical but hyper-specific: "Right extensor digitorum longus", "Left Lobe of the Liver", 48 distinct values over 2,108 samples.
ARCHS4 has no curated tissue column at all; the signal lives in GEO's free-text `characteristics_ch1` and `source_name_ch1`, which yields 42,754 distinct lowercased strings over 940,455 samples.

Left alone, those are two vocabularies with no overlap, and "Tissue" has to be two separate color-bys that each leave the other corpus grey.
Folding both onto one canonical bucket list is what makes a single "Tissue" color-by paint the whole map, and it is the central idea of the redesign.
`canonical_tissue` is the only entry point, and it is called by `fetch_archs4_meta.py` for ARCHS4 and by `manifold/data.py` for OSDR, so the two corpora cannot drift apart.

The mapping is 40 ordered keyword rules, first match wins, into 37 distinct buckets plus "Other" and "Unknown" (39 entries in `tissue.BUCKETS`; a bucket may legitimately be named by more than one rule).
Most are organs; four are last resorts ("Tumor / cancer", "Reference RNA", "Cell line", "Cultured cells") that are not tissues but name large and genuinely distinct slices of GEO, and burying them in "Other" would hide real structure the map does show.
It is deliberately auditable rather than learned: every rule is readable, and it fails towards "Other" rather than towards a confident guess, because on a plot people read biology off, an honestly empty label beats a wrong one.

Two subtleties are worth recording, because both were real bugs that tests caught.

**Ordering and word boundaries are load-bearing.**
"bone marrow" must be tested before "bone", or every marrow sample becomes bone.
Patterns get a leading `\b` word boundary by default, so `renal` cannot fire inside "adrenal".
"cortex" is a kidney and adrenal word as well as a brain word, and GEO writes all three, so the Adrenal and Kidney cortex rules run ahead of Brain / CNS and the organ named in the string wins over the region word.
Smooth muscle is vascular and must not be claimed by the bare "muscle" stem, so it is placed above Skeletal muscle; a bucket may legitimately be named by more than one rule.
A `~` prefix opts a pattern out of the leading word boundary, which is needed for the morphemes GEO glues onto a stem: "sarcoma" has to match inside "osteosarcoma" and "carcinoma" inside "hepatocarcinoma".

**"Unknown" and "Other" are different facts and stay distinct.**
Nothing recorded is not the same as recorded but unplaceable, and collapsing them would overstate coverage.
Weak results are also *ranked* rather than first-wins: naming a cell line is more informative than naming nothing, and both are less informative than naming an organ.
Without that ranking an early unplaceable field pinned the answer to "Other" and blocked a later field that did identify the sample, which is how HeLa samples were reading as "Other".

Measured results on the real corpus:

- All 48 OSDR raw tissue values map to a named bucket; zero fall to "Other" or "Unknown". They land in 17 buckets, of which "Cultured cells" (18 samples) is the only non-anatomical one.
- 851,881 of 940,455 ARCHS4 samples, **90.6%**, land in a bucket other than "Other" or "Unknown".
- The "Tissue" color-by covers **942,563 of 942,563 points, 100%**.

Top buckets over ARCHS4: Blood / immune 155,761 (16.6%), Brain / CNS 103,182 (11.0%), Other 87,692 (9.3%), Embryo / stem cell 67,408 (7.2%), Liver 55,242 (5.9%), Tumor / cancer 53,117 (5.6%), Lung 40,710 (4.3%), Intestine 35,318 (3.8%), Bone marrow 34,409 (3.7%), Cultured cells 32,042 (3.4%), Breast / mammary 31,499 (3.3%).

The label was then validated as biology rather than as batch, because a keyword mapping over free text could easily be reproducing submitter conventions.
25-NN label purity on the map is **0.8142 against a 0.0501 permuted null**, and it survives both a batch control and a depth control at **0.7058**.
That is the standard every other candidate in section 7.5 was held to, and it is the only tissue-scale label that met it.

### 7.3 Coverage as a declared property (`manifold/colorby.py`)

Before this module the renderer branched on the color-by key with a chain of `if`/`elif`, and each branch decided for itself what to do about the corpus it did not describe.
That is why the grey-map failure existed at all: there was no single place that knew the answer, so there was no single place to fix it.

Now every color-by is a `ColorBy` declaring four things: its `scope` (the corpora it could describe if every artifact were present), a `resolver`, an optional `hint`, and an optional `(predicate, fix-hint)` pair for an artifact it needs.
`covers()` reports which corpora the field can color **right now, on this machine**, and that one fact drives everything downstream: the menu order, the disabled state, the coverage readout, and what the renderer draws.
`labels(key)` returns one array over the full corpus in the fixed global order with a `NOT_COVERED` sentinel marking points the field says nothing about; everything after that is a plain categorical render with no special cases.

The availability predicate is `data.archs4_metadata_available` itself, not a re-derived path.
That is deliberate: a registry that decided availability by rebuilding the artifact path would be a second source of truth for the same file, and the two could disagree about whether a field works.
This was a real bug, and it is now pinned by a test.

What the declaration buys, in the interface:

- **The menu lists whole-map fields first, each labelled with its scope**: "Tissue · whole map", "Flight vs Ground · OSDR only", "... · unavailable".
  Dash's dropdown has no option-group support, so the grouping is carried by ordering plus the suffix rather than by faked disabled header rows.
- **A field with no data at all is shown and disabled, with the command that enables it**, not hidden.
  Hiding it makes the app look like it never had the feature; showing it disabled next to `precompute/fetch_archs4_meta.py` says the feature exists and how to switch it on.
- **A coverage bar and an exact point count sit directly under the control**: "Colours all 942,563 points." against "Colours 2,108 of 942,563 points (0.2%). ARCHS4 is drawn as faint context."
  The partial bar is amber, not red, because an OSDR-only field is working correctly and is not failing.
  This is the control that answers "why is so much of my map not coloured?" before the user has to ask it.
- **The app opens on the best whole-map field that works**, and falls back to species rather than raising if a browser holds a stale key across a rebuild.

The degraded state, where `archs4_metadata.parquet` was never fetched, is a first-class path rather than an afterthought: Tissue stays available but drops out of the whole-map group and reports OSDR-only coverage with the fix attached, and Species still covers the whole map because it comes from the identity table.
That state is what a fresh clone starts in, so `tests/conftest.py` provides a `without_archs4_metadata` fixture and `tests/test_colorby.py` exercises it directly.

### 7.4 What the renderer does instead of a grey cloud

**The renderer never paints a uniform grey glyph cloud.**

When the selected field does not describe ARCHS4, those 940,455 points are drawn as a context cloud: 2.6 px at 0.35 opacity in a single color of their own (`#43597c`, deliberately close to the plot background), badged "ARCHS4: context only · Flight vs Ground is OSDR-only", and never given a legend swatch.
Points with no value under the current field are the absence of a value, not a value, and giving them a swatch is what made the map read as grey data in the first place.
The color matters as much as the opacity: `#43597c` is not in the categorical palette, so nothing about the cloud invites a reader to look for it in the legend.

This used to be the second of two paths.
The first was to draw nothing at all and let the precomputed density raster carry the shape, on the argument that a density field cannot be mistaken for a category, with the context cloud reserved for the cases where there was no raster (3-D, or the underlay toggled off).
The raster is gone (section 6), so the context cloud is now the only path, and it applies in 2-D and 3-D alike.
That is a simplification rather than a loss: the requirement was always "show the shape, do not impersonate a category", the context cloud satisfies it in every view, and having one path instead of two removes a branch where the two could have diverged.

The mirror case is handled too: under an ARCHS4-only field, OSDR keeps its distinct diamond in a single warm highlight, so the spaceflight corpus stays locatable without borrowing a color that means something else in the legend.

### 7.5 Candidates that were built or tested and then rejected

These are recorded with their evidence because they are the most reusable part of this document.
Each of them is an obvious thing to propose, and each of them is wrong for a reason that took measurement to find.

**(a) Cosine similarity to an OSDR reference. Rejected.**
Four variants were computed for every one of the 942,563 points: similarity to the OSDR mean centroid, to the flight centroid, to the ground centroid, and the flight-minus-ground difference, the last presented as a continuous "spaceflight-likeness" axis.
The four scores are one field wearing four names: pairwise correlations run from r = 0.996 to r = 1.000.
The interesting one, the flight-minus-ground axis, correlates r = -0.990 with PC1 and r = -0.779 with the raw L2 norm.
PC1 is neither spaceflight nor sequencing depth: it is a transcriptome-concentration axis (`REFERENCE.md` section 4).
The candidate therefore measured how concentrated a sample's transcriptome is and offered it as resemblance to spaceflight, which is a real biological quantity wearing the wrong name.
Against a null of random flight/ground relabelings of the same sample sizes, 1 in 10 random relabelings beat the real axis on spatial structure, and under a within-study permutation 46.5% did.
An axis a coin flip can reproduce half the time is not a measurement of spaceflight.

**(b) kNN tissue-label transfer from OSDR to ARCHS4. Rejected.**
The idea was to give every ARCHS4 point the tissue of its nearest OSDR sample, which would have colored the whole map without any metadata fetch.
The median best-match cosine is 0.964 and 100% of ARCHS4 points sit above 0.7, so no confidence threshold discriminates anything: everything looks like a confident match.
Worse, the winning OSDR sample beats the runner-up by a **median of 0.00089 cosine**, so which label a point receives is essentially arbitrary.
And 54% of the ARCHS4 targets are human samples that would have received mouse tissue labels.
This would have produced a beautifully colored map that was mostly noise, which is the single worst outcome available.

**(c) Unsupervised k-means cluster id (k = 24). Built, measured, then cut.**
This one got furthest: the precompute stage was written, run on the real corpus, and the artifact produced, before being deleted.
Direct measurement killed it.
**81.9% of the cluster label is recoverable from the 2-D UMAP coordinates alone** (15-NN over a 120k sample, against a 12.4% majority-class baseline), so coloring by it mostly redraws the shape already on screen.
A structure-free 24-cell Voronoi null reproduced its spatial coherence to within 1.5 points of modal agreement, so its apparent tidiness is a property of partitioning a 2-D-projectable space, not a discovery.
It is arbitrary (seed-to-seed ARI of about 0.45), 81% species-pure, and explains 80.7% of the raw L2 norm's depth variance.
Painting an arbitrary partition on a scientific instrument and numbering it "Cluster 1..24" invites exactly the over-reading the rest of this design prevents, and a numbered cluster is unusually good at inviting it because integers look like findings.
A comment in `manifold/colorby.py` records the decision at the point where someone would add it back.

**(d) Local UMAP density. Rejected as redundant, on an argument that has since expired.**
The rejection was that the density raster is already rendered underneath every 2-D view, so a density color-by would encode the same quantity twice, once as color and once as the picture it sits on.
That raster is gone (section 6), so this candidate is genuinely open again rather than settled.
Anyone revisiting it should note that with every point now drawn live, glyph crowding *is* the density readout, and should hold the candidate to section 7.6 rather than treating this paragraph as a standing verdict.

**(e) PC1 to PC3 as color-bys. Rejected.**
They are free, since they already sit in `coords_pca3.parquet`, but they are redundant with the axes on screen, and PC1 is a transcriptome-concentration axis, which is interesting in its own right but is not what a user reaches for a spaceflight map to see.

**(f) GEO series (GSE) as a color-by. Rejected, and kept as provenance.**
There are 51,284 distinct series, so a Top-11 legend would color about 3% of the map and dump the rest into "Other": a grey map reached by a different route.
It is also a pure batch label, with 333x lift, so a user who read it as biology would be reading it exactly backwards.
The column stays in `archs4_metadata.parquet` for provenance, and OSDR keeps `study` as an explicit batch color-by with a hint that says what it is for, but GSE is not offered as a color.

### 7.6 Methodological note: spatial variance is not evidence

This is worth recording prominently, because it is the trap every one of the candidates above walked into, and it will be the trap the next candidate walks into.

A between-bin over total variance ratio on the map, spatial eta-squared, is **not** sufficient evidence that a color-by shows real structure.
Measured here: **30 arbitrary random directions in 512-d score eta-squared 0.874 +/- 0.025 on this UMAP.**
They score that well because the UMAP was fit on those same vectors, so any linear functional of them is smooth on the map by construction.
Every candidate scoring in the 0.89 to 0.94 band is therefore indistinguishable from an arbitrary projection.
Species, at 0.985, is the only field that clearly clears the bar, which is why it is the reference for what a working color-by looks like.

The rule that follows: judge a candidate against a structure-free null **of the same form** (permuted labels for a categorical field, random directions for a continuous one, a Voronoi partition for a clustering), and then check whether the candidate is recoverable from the coordinates themselves or from transcriptome concentration.
A field that passes the eta-squared eye test and fails both of those checks is a picture of the projection, not a measurement of biology.

## 8. Key design decisions and tradeoffs

**Standalone app, not bolted into Bridge RNA.**
Bridge Manifold is a separate Dash app in its own directory, importing reusable functions from Bridge RNA rather than editing the 2,470-line retrieval app.
This keeps the heavy exploratory tool from destabilizing the retrieval product, while a shared header and shared CSS make them feel like one instrument.
The tradeoff is a small amount of duplicated app scaffolding, which is worth it for isolation.

**Offline precompute over interactive computation.**
Every expensive step (model inference, PCA, UMAP) is precomputed and cached, and the app only ever loads artifacts.
This is forced by the measured cost and is what makes the app responsive.
The tradeoff is a build step and cache management, which is the right trade for a 943k-point tool.

**Fit both reductions on the whole corpus, having checked what that actually costs.**
The first build subsampled both: `IncrementalPCA` on 60,000 points, UMAP on a 122,563-point landmark set with the remaining 819,999 pushed through `.transform()`.
Both approximations were adopted from estimates rather than measurements, and both estimates were wrong.
Exact full-corpus PCA is a single streaming pass that costs 6 s, and a direct UMAP fit on all 942,563 points is a job measured in tens of minutes, not the "hours" this document asserted for its entire first life.
The tradeoff is a longer build, paid once, offline, for coordinates in which every point participated in the layout it appears in.
The general lesson is the one worth keeping: an approximation adopted to dodge a cost nobody measured is not a tradeoff, it is a guess with a justification attached.

**Color-by coverage is a declared property, not a per-branch decision.**
Every field states which corpora it can color right now, and the menu, the coverage readout, and the renderer all read that one declaration.
The alternative, which is what the first build did, is that each rendering branch decides independently what to do about the corpus it does not describe, and the result was that 99.8% of the map turned flat grey with nothing in the interface admitting it.
The tradeoff is one more layer of indirection between a key and an array of labels, which buys a property that can be asserted in a test instead of eyeballed on a scatter.

**One shared tissue vocabulary rather than two tissue fields.**
OSDR's curated anatomy and ARCHS4's GEO free text are folded onto the same ~37 buckets by one ordered keyword mapper used by both pipelines.
Two separate "Tissue" fields would have been easier and would have been two color-bys wearing one name, each leaving the other corpus grey, with a legend implying a comparison the data did not support.
The tradeoff is a hand-maintained rule list that will need occasional additions, which is acceptable precisely because it is readable: an auditable mapping that fails to "Other" beats a learned one that fails to a confident wrong answer.

**Metadata over HTTP, not a 113 GB download.**
The sigpy JSON API returns the same per-GSM fields as the ARCHS4 gene HDF5 files in 33.7 seconds and 216 MB, against 113 GB of downloads or hours of range-request reads.
The cost is 0.089% of points reading "Unknown" because the newest release dropped them, and a documented upgrade path to the versioned metadata-only H5 files if 100% ever becomes a requirement.
This decision is why the ARCHS4 corpus has biology on it at all: the HDF5 route had been the plan for the entire first build and had never been executed once.

**Cut the k-means cluster color-by after building it.**
The artifact existed, the precompute stage ran on the real corpus, and it was deleted anyway, because 81.9% of its labels are recoverable from the 2-D coordinates and a structure-free Voronoi null reproduced its coherence.
Recording this as a decision rather than quietly dropping it matters: the work was not wasted, it produced the measurement that says the feature would have misled people, and that measurement is cheaper to read than to repeat.

**L2-normalize before reducing.**
Because retrieval is cosine and the raw vectors carry a 3.9x magnitude spread that dominates PC1 (57.8% before normalization, 41.3% after), we normalize first so the manifold reflects transcriptomic direction rather than magnitude.
The magnitude is redundant rather than technical: it is recoverable from the normalized direction at held-out R^2 = 0.977, and it measures transcriptome concentration.

**Make batch visible, do not correct it (final).**
Rather than applying Harmony or ComBat and risking erasure of real biology, the tool exposes study and species as color-by dimensions and discloses the measured 54x tissue-controlled cross-corpus effect on the control rail.
Josh confirmed this is the final call: no correction algorithm, not even as a later toggle.

**Draw every point, having checked what that actually costs too.**
The first build rendered ~100k live glyphs over a density raster of all 942,563, on the assumption that 940k live points was out of reach.
Measured, all 942,563 serialize in 0.15 s to an 11.3 MB payload, and figure build time does not vary with the budget at all.
So the raster was buying smoothness that did not need buying, and paying for it with a second unlabelled encoding of the same data, a per-projection PNG, a build stage, a flag, and a dependency.
It is gone, the budget now defaults to the whole corpus, and the lower tiers survive only as a comfort control (section 6).
This is the same lesson as the entry above, arriving from the rendering side rather than the compute side.

**No on-demand statistics.**
The selection readout was removed rather than repaired.
It was the app's largest source of complexity (a 450-line statistics module, a 2.07 GB ANN index, an exact covariance artifact, a third UI column) in service of a question the map itself answers qualitatively, and every number it produced had to be defended against a null that most readers would never see.
Removing it dropped the cache from about 2.3 GB to a measured 219.2 MB (since 217.8 MB, of which the app opens 80.8 MB, after the density rasters went too), took the serving app's dependency surface down to `dash`/`plotly`/`numpy`/`pandas`/`pyarrow`, and cut the test suite from 4.54 s to about 0.55 s.
Those two dead artifacts have since been deleted, along with the density rasters, so `cache/` now measures what `REFERENCE.md` section 12 records.

## 9. Directory layout

This is the layout as built.

```
Bridge Manifold/
  app.py                     # Dash entry (shared with retrieval): host/port/debug, loopback guard
  manifold/
    paths.py                 # every artifact path; BRIDGE_RNA_ROOT / MANIFOLD_CACHE_DIR overrides
    preflight.py             # missing-artifact and Git-LFS-pointer guards
    bridge_rna.py            # the single seam that imports from the sibling repo
    data.py                  # parquet loaders, the fixed global point order, module-level caches
    tissue.py                # the shared tissue vocabulary, used by both corpora
    colorby.py               # the coverage-aware color-by registry
    sampling.py              # stratified quota sampling, viewport re-stratification
    render.py                # layered figure: ARCHS4 cloud, OSDR overlay
    theme.py                 # plot theme (dark canvas) + validated categorical palette
    layout.py                # header, left control rail, plot (two columns)
    callbacks.py             # color-by, coverage readout, layers, method, zoom LOD, legend filter
  precompute/
    embed_osdr.py            # OSDR counts -> 2,108 x 512 embeddings (loads torch), resumable
    build_projections.py     # L2 -> exact full-corpus PCA + full-corpus UMAP -> 4 coord parquets
    fetch_archs4_meta.py     # sigpy API -> per-GSM series/title/source/characteristics + tissue
    validate_artifacts.py    # objective build gate: structure, invariant 2, mixing, projection quality
  tests/
    fixture_corpus.py        # synthetic corpus with known latent clusters and GEO-style metadata
    conftest.py              # points the package at that corpus before import
    build_dev_corpus.py      # CLI to build a browsable corpus; --no-archs4-meta gives the degraded state
    test_data.py             # global point order, label lookups, degraded loaders
    test_tissue.py           # the vocabulary: ordering collisions, boundary collisions, coalescing
    test_colorby.py          # declared coverage vs produced labels, availability, degraded state
    test_render.py           # shared palette, context-not-grey, budgets, viewport, legend
    test_projections.py      # the exact PCA against sklearn, sign stability, global point order
    test_app.py              # callback wiring, coverage readout, CSS class coverage
    e2e_check.py             # outside pytest: boots the real app + browser against the real cache
  assets/
    manifold.css             # Bridge RNA tokens, Dash 4 token remap, dark-canvas + legend rules
  cache/                     # generated (gitignored): embeddings, coords, metadata
  requirements.txt
  REFERENCE.md
  IMPLEMENTATION.md
  progress.md
  README.md
```

`reduce.py` from the original sketch was never needed; reading coordinate parquets is three lines in `data.py`, and a module wrapping it would have been indirection for its own sake.
`coherence.py` and `test_coherence.py` were deleted with the selection feature.

The files that import torch and umap are confined to `precompute/`, so the serving app has a light dependency surface: `dash`, `plotly`, `numpy`, `pandas`, `pyarrow`.
Nothing else.

## 10. Reuse from Bridge RNA

The following are imported or copied rather than rewritten (signatures and line numbers in `REFERENCE.md` section 6):
the OSDR preprocessing body from `load_random_osdr_sample_vector`, the model driver from `build_model_and_query_embedding`, `ExpressionPerformer` and its `encode`, `canonical_gene_order_digest` and `CANONICAL_GENES_SHA256`, `_strip_module_prefix`, `build_mouse_to_human_maps` and `normalize_counts_to_tpm_single`, the `preflight_retrieval_requirements` and `_is_lfs_pointer` LFS guard pattern, and the theme tokens and header/panel/badge CSS classes.

All of that is funnelled through `manifold/bridge_rna.py`, imported lazily inside a function so that merely importing the module does not pull in torch, and used only by `precompute/embed_osdr.py` and the preflight guards.

Two former reuses are gone.
`fetch_archs4_metadata` (the `archs4py` HDF5 reader at `demo_osdr_top5.py:463`) is no longer used, because the metadata now comes from the sigpy API (section 4.5).
`_load_archs4_index` and `_topk_cosine_from_memmap` are no longer used by the app; `validate_artifacts.py --mixing` streams the memmap itself in 50k blocks.

The result is worth stating plainly: **the serving app now shares no runtime code path with Bridge RNA's heavy stack.**
It imports no Bridge RNA module, opens no file in the Bridge RNA repository, and needs neither torch nor the checkpoint nor the memmap to run.
The coupling is entirely at build time.

Dependencies: Bridge Manifold shares the Bridge RNA venv and adds `dash` on top of the existing stack, plus `requests` for the metadata fetch.
`hnswlib` and `scipy` were dropped from `requirements.txt` with the selection feature (both are still installed in the shared venv and neither is imported), `archs4py` was never installed at all, and `h5py` is present only from the range-request experiment in section 4.5 and is imported by no shipped code.

## 11. Visual language

Bridge RNA is a light scientific-instrument theme, not a dark one.
Its tokens: canvas `#eef2f7`, panels `#ffffff`, primary text `#1a2432`, accent blue `#2b7fff`, teal `#0bab9f`, warm `#d9791b`, and a dark navy header `#14294a` with a teal rule `#22c7bd`.
Bridge Manifold matches this light chrome exactly, and uses a dark navy plot canvas (`#0e1d34`) inside it so the WebGL glyphs have contrast, which is the one deliberate departure.

The categorical palette was validated with the dataviz skill's checker against that navy plot surface: all eleven hues sit in the OKLCH L 0.48-0.67 band, clear the chroma floor, pass the adjacent-pair CVD floor (worst delta-E 8.4) and the normal-vision floor (worst delta-E 15.4), and hold at least 3:1 contrast against the surface.
Perfect all-pairs CVD separation is impossible past a few categories on a scatter, so high-cardinality color-bys lean on secondary encoding: a searchable legend, hover that names the exact category, and a distinct OSDR symbol.

Two greys sit at the neutral end of the palette rather than one, because "Other" (`#7f8ea3`) and "Unknown" (`#56657a`) are different answers and `manifold/tissue.py` goes to some trouble to keep them apart; throwing the distinction away at the last rendering step would waste it.
A third muted tone (`#43597c`) is reserved for the ARCHS4 context cloud, close enough to the background to read as scenery rather than as a category.

Dash components are themed by remapping Dash 4's own `--Dash-*` design tokens rather than by overriding each component's rules, because one mapping themes every current and future Dash component while per-component overrides are a specificity war that rots on upgrade.

## 12. Phased build plan

Each phase ends with an objective validation, not a visual glance.
Build status is tracked in `progress.md`; the plan below is the phasing with its validation criteria.

**Phase 0. Scaffold.**
Create the package skeleton, `manifold.css` importing the Bridge RNA tokens, and a preflight that fails clearly on missing or LFS-stub artifacts.
Validate: app boots and renders an empty themed shell; preflight names every missing artifact in one aggregated error.

**Phase 1. OSDR embeddings.**
Implement `embed_osdr.py` with the gene-digest gate; embed all eligible samples; cache the npy and the metadata parquet.
Validate: digest matches, output shape is `2108 x 512`, and the batched preprocessing path reproduces Bridge RNA's single-sample path bit-for-bit (measured: max abs diff 0.0 across three studies).

**Phase 2. PCA projections.**
L2-normalize, then take the exact PCA of the whole corpus from a streaming second-moment pass, project it, write `pca2` and `pca3`.
Validate: the components and explained-variance ratios equal a full `sklearn` fit to float64 round-off (`tests/test_projections.py`), the recorded 512-eigenvalue spectrum sums to 1, and PC1 lands well below the 57.8% pre-normalization figure, which is the objective test that normalization was applied (measured: 41.3%, cumulative over 50 PCs 95.0%).

**Phase 3. First interactive plot.**
Wire the Scattergl renderer, stratified sampling, the dataset and method toggles, and color-by.
Validate: the whole corpus pans and zooms smoothly; every drawn glyph sits on a real corpus coordinate; the OSDR overlay draws every OSDR point exactly once; color-by switches without a full reload.

**Phase 4. UMAP projections.**
Build the shared k-NN graph once and lay it out in 2-d and 3-d over all 942,563 points; write `umap2` and `umap3`.
Validate: `validate_artifacts.py` passes structurally (row counts agree across every artifact, all coordinates finite, every axis has real spread), OSDR occupies a real region of the map rather than collapsing to a blob, and `--quality` shows each coordinate set clearing its nulls on 15-NN recall and 25-NN tissue purity.

**Phase 5. Coloring both corpora.**
Fetch the ARCHS4 GEO metadata over the sigpy API, build the shared tissue vocabulary, and replace the renderer's per-key branching with the coverage-aware registry.
Validate, objectively and not by looking at the map:

- every enabled menu option produces a non-empty legend and a figure with at least one trace,
- the coverage a field **advertises** equals the number of labels its resolver actually **produces** (`covered == (labels != NOT_COVERED).sum()`, checked for every registered field),
- every whole-map field covers all 942,563 points and the app's default field is one of them,
- an OSDR-only field draws its ARCHS4 points as a faint context trace, in a color outside the categorical palette and with no legend swatch, in 2-D and 3-D alike,
- a category keeps one color across both corpora, and legend counts do not move with the point budget,
- with the metadata join removed, Tissue degrades to OSDR-only coverage and names the command that restores it, rather than vanishing or lying.

These are `tests/test_colorby.py`, `tests/test_tissue.py`, and the coverage tests in `tests/test_render.py` and `tests/test_app.py`.

**Phase 6. Polish.**
Searchable legend for high-cardinality color-bys, viewport re-stratification, 3D views, hover cards, and pixel-level theme matching.
Validate: high-cardinality color-bys are usable; every class name used in Python exists in the stylesheet; every callback output has exactly one writer; the whole thing feels like one product with Bridge RNA.

The build order is deliberately offline-precompute-first, so that by the time the app is wired there is real data to render.

## 13. Commands

Run everything from the Bridge RNA venv.
The order matters: the metadata fetch joins onto artifacts that `build_projections.py` writes.

```bash
/Users/josh/Bridge-RNA/.venv/bin/python precompute/embed_osdr.py         # OSDR embeddings, gene-digest gated. Hours.
/Users/josh/Bridge-RNA/.venv/bin/python precompute/build_projections.py  # full-corpus PCA + UMAP coords. ~10.5 min.
/Users/josh/Bridge-RNA/.venv/bin/python precompute/fetch_archs4_meta.py  # ARCHS4 GEO metadata. ~35 s, needs network.
/Users/josh/Bridge-RNA/.venv/bin/python precompute/validate_artifacts.py --mixing --quality
/Users/josh/Bridge-RNA/.venv/bin/python app.py                           # http://127.0.0.1:8050/map

# Score a candidate projection against the shipped one, on the same sample:
/Users/josh/Bridge-RNA/.venv/bin/python precompute/validate_artifacts.py --quality --compare /path/to/other/coords
```

`build_projections.py --skip-umap` stops after the PCA stage, which takes 8 s, for anyone who only needs coordinates to exist.
`--knn-jobs -1` trades reproducibility of the neighbour graph for roughly 10x on that one stage.

Tests:

```bash
/Users/josh/Bridge-RNA/.venv/bin/python -m pytest tests/ -q
```

185 tests in about two seconds, against a hermetic synthetic corpus (4,000 ARCHS4 + 300 OSDR points) that never touches the real memmap, the checkpoint, or the multi-hour artifacts.
The suite was 103 tests at 4.54 s before the redesign; removing the selection feature took the ANN index out of the fixture, which was 43% of the old runtime, and the color-by, tissue, and projection tests added back more coverage than was removed.
`test_projections.py` is the one file that imports from `precompute/`, because the exact-PCA claim is the kind that has to be checked against a reference implementation rather than asserted in a docstring.
The per-file split is in `REFERENCE.md` section 12.
