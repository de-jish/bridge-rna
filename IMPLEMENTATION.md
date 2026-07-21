# Bridge Manifold - Implementation Plan

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

It must render ~940k points smoothly at interactive frame rates using Plotly WebGL scatter traces over a precomputed density raster, with stratified sampling and viewport level-of-detail.
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
The serving app draws a precomputed map and never needs a 512-d vector, so it never opens it.

## 3. Architecture overview

Bridge Manifold splits cleanly into an offline precompute stage and an online serving stage.
This split is the single most important architectural decision, and it is forced by measured cost.

```
OFFLINE (precompute/, run once -> cache/)      ONLINE (app_manifold.py, loads artifacts only)
---------------------------------------        ----------------------------------------------
embed_osdr.py                                  app_manifold.py
  OSDR counts -> 2,108 x 512 embeddings          coords_{pca,umap}{2,3}.parquet
  cache/osdr_sample_embeddings.float32.npy       points_meta.parquet   (identity table)
  cache/osdr_metadata.parquet                    osdr_metadata.parquet (OSDR labels)
                                                 archs4_metadata.parquet (GEO join)
build_projections.py                             density/{pca2,umap2}.png
  L2-normalize -> IncrementalPCA-50
  -> {pca2, pca3, umap2, umap3} parquets         renders go.Scattergl over the raster
  -> density/{pca2,umap2}.png                    colorby registry -> one label array
  -> points_meta.parquet, archs4_geo.parquet     stratified sample + viewport LOD

fetch_archs4_meta.py
  sigpy JSON API -> archs4_metadata.parquet
  (series, title, source, characteristics,
   and the canonical tissue bucket)
```

The app never runs the model, never runs UMAP, never opens the 963 MB memmap, and never holds all 940k glyphs live.
It reads small precomputed tables, samples them, and draws them.
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

### 5.1 The reduction spine: L2-normalize -> PCA-50 -> UMAP

Measured on a 25k ARCHS4 sample before normalization, PC1 alone captures 57.8% of variance and the first 50 PCs capture 96.4%.
That giant PC1 is a magnitude and depth axis, which is exactly why we L2-normalize first.
PCA to 50 dimensions then denoises and compresses the input to UMAP, which makes UMAP both faster and cleaner.

### 5.2 PCA

PCA is cheap: `IncrementalPCA-50` is fit on a 60,000-point stratified ARCHS4 subsample plus all 2,108 OSDR points, then the full corpus is streamed through the fitted components in 50k blocks.
Measured on the real corpus, the fit took 1 s and the transform of all 942,563 points took 1 s.
We write `pca2` and `pca3` coordinates for the full joint corpus.

After normalization, **PC1 = 40.9% and the cumulative over 50 PCs = 95.1%**.
That pair is the objective test for the normalization invariant: a build landing near 57.8% is evidence that normalization was silently skipped, and `precompute/validate_artifacts.py` fails the build above a 50% ceiling.
Because PC1 still dominates, a raw PCA-2D view is mostly one axis plus noise; we keep it because it is honest about global magnitude structure and it is a fast sanity layer, but UMAP is the primary exploratory view.

### 5.3 UMAP

UMAP is expensive and strictly offline.
A direct fit on all 940k would take hours and risk memory blowup, so we use the landmark pattern: fit on a stratified 120,000-point ARCHS4 subsample plus all OSDR, then `.transform()` the remaining ~820k in 50k batches.
Measured on the real corpus: the 2-d fit took 71 s and its transform 71 s; the 3-d fit and transform together took 143 s.

The whole projection build ran end to end in **5 min 47 s** (347 s), against an original estimate of 30 to 90 minutes that was wrong by an order of magnitude.
Of that, 52 s built the ANN index and 4 s computed population moments, neither of which happens any more, so a current build is a **291 s** job, a little under 5 minutes.
Peak RSS was about 2.5 GB.
UMAP emits `n_jobs value 1 overridden to 1 by setting random_state`; at 71 s per fit, determinism is worth more than the threads, so `random_state=42` stays.

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

1. **Density underlay.**
   Offline, all 942,563 coordinates go through a 2048x2048 numpy `histogram2d`, `log1p`, and a navy-to-blue-to-teal ramp rendered to a PNG by Pillow, placed as a `layout.images` underlay at its recorded extent.
   This is deliberately not datashader: a plain histogram is trivially fast at this scale and removes a fragile dependency.
   The ramp is normalized against the 99.5th percentile of *occupied* bins rather than the global maximum, because occupancy is heavy-tailed (median occupied bin holds 2 points, maximum 638) and dividing by the maximum crushed the whole ramp into its bottom fraction: measured on the real raster, only 8 pixels reached the teal half.
   Alpha ramps across the same span with a 0.22 floor so sparse-but-occupied bins still read.
   `build_projections.py --density-only` re-renders the rasters from cached coordinates in seconds, so tuning the ramp never repeats the projection build.
2. **ARCHS4 background.**
   A single stratified WebGL sample, 3.4 px, hover disabled, split into one trace per display category.
   Hover hit-testing is a dominant cost at this scale, and `hoverinfo="skip"` alone is not enough because a `hovertemplate` overrides it; the two are turned off together.
   ARCHS4 traces carry **no `customdata`**.
   It existed only to feed the removed selection tool, and it was roughly 600 KB of dead payload per figure.
3. **OSDR overlay.**
   All 2,108 OSDR points, always drawn, 8.5 px diamonds with a 1.1 px white ring and full hover, so the spaceflight samples stay findable above a 100k-point cloud.
   Hover carries `[sample_key, category]`: which sample this is, and what it is under the current color-by.

**One palette across both corpora.**
Categories are ranked once over the whole covered population and every layer draws from that single mapping, so a liver in GEO and a liver in OSDR get the same color.
Ranking per layer, which is what the first implementation did, silently gave one category two different colors whenever the two corpora ordered their categories differently, which is a legend that lies.
The top 11 categories take the validated categorical palette; residual categories ("Other", "Unknown") keep their own rows at the neutral end and always sort last, so they never outrank a category that carries information; everything past the palette folds into one grey overflow row.
Legend counts are whole-corpus counts, not counts of the drawn sample, so they do not move when the point budget or the zoom changes.
A legend number is read as "how many such samples exist", and it should answer that question.

**A corpus a field does not describe is drawn as context, not as data.**
See section 7.4 for the full argument.

Point budgets (decisive):

| Layer | Default | Range |
| --- | --- | --- |
| OSDR | 2,108 (100%, never subsampled) | 2,108 |
| ARCHS4, 2D | 100,000 (10.6%) | 60,000 / 100,000 / 150,000 |
| ARCHS4, 3D | capped at 40,000 | 40,000 |
| Total live glyphs, 2D | ~102k | ~152k |

These budgets sit comfortably under the ~150k to 300k range where Scattergl pan and zoom begin to degrade, and 3D WebGL is heavier so it gets its own cap.

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
- **A coverage bar and an exact point count sit directly under the control**: "Colours all 942,563 points." against "Colours 2,108 of 942,563 points (0.2%). ARCHS4 is shown as density for context."
  The partial bar is amber, not red, because an OSDR-only field is working correctly and is not failing.
  This is the control that answers "why is so much of my map not coloured?" before the user has to ask it.
- **The app opens on the best whole-map field that works**, and falls back to species rather than raising if a browser holds a stale key across a rebuild.

The degraded state, where `archs4_metadata.parquet` was never fetched, is a first-class path rather than an afterthought: Tissue stays available but drops out of the whole-map group and reports OSDR-only coverage with the fix attached, and Species still covers the whole map because it comes from the identity table.
That state is what a fresh clone starts in, so `tests/conftest.py` provides a `without_archs4_metadata` fixture and `tests/test_colorby.py` exercises it directly.

### 7.4 What the renderer does instead of a grey cloud

**The renderer never paints a uniform grey glyph cloud.**

When the selected field does not describe ARCHS4, the ARCHS4 glyph layer steps aside and the precomputed density raster carries the manifold shape, with a badge on the plot saying exactly that: "ARCHS4: density only · Flight vs Ground is OSDR-only".
The raster already shows all 940,455 points, and it shows them more truthfully than a uniform glyph cloud would, because a density field cannot be mistaken for a category.
Drawing nothing in that layer is the honest option, not a degraded one.

Only when there is no raster to fall back on, which means 3D or the underlay toggled off, is a context cloud drawn: 2.6 px at 0.35 opacity in its own color (`#43597c`, close to the plot background), badged "context only", and deliberately never given a legend swatch.
Points with no value under the current field are the absence of a value, not a value, and giving them a swatch is what made the map read as grey data in the first place.

The mirror case is handled too: under an ARCHS4-only field, OSDR keeps its distinct diamond in a single warm highlight, so the spaceflight corpus stays locatable without borrowing a color that means something else in the legend.

### 7.5 Candidates that were built or tested and then rejected

These are recorded with their evidence because they are the most reusable part of this document.
Each of them is an obvious thing to propose, and each of them is wrong for a reason that took measurement to find.

**(a) Cosine similarity to an OSDR reference. Rejected.**
Four variants were computed for every one of the 942,563 points: similarity to the OSDR mean centroid, to the flight centroid, to the ground centroid, and the flight-minus-ground difference, the last presented as a continuous "spaceflight-likeness" axis.
The four scores are one field wearing four names: pairwise correlations run from r = 0.996 to r = 1.000.
The interesting one, the flight-minus-ground axis, correlates r = -0.990 with PC1 and r = -0.779 with the raw L2 norm.
It **is** the sequencing-depth axis, relabelled as biology, which is exactly the failure that L2 normalization exists to prevent, reintroduced one layer higher up.
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

**(d) Local UMAP density. Rejected as redundant.**
The density raster is already rendered underneath every 2-D view, so a density color-by would encode the same quantity twice, once as color and once as the picture it sits on.

**(e) PC1 to PC3 as color-bys. Rejected.**
They are free, since they already sit in `coords_pca3.parquet`, but they are redundant with the axes on screen, and PC1 is the depth axis, so offering it as a color would advertise sequencing depth as a feature.

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

The rule that follows: judge a candidate against a structure-free null **of the same form** (permuted labels for a categorical field, random directions for a continuous one, a Voronoi partition for a clustering), and then check whether the candidate is recoverable from the coordinates themselves or from sequencing depth.
A field that passes the eta-squared eye test and fails both of those checks is a picture of the projection, not a measurement of biology.

## 8. Key design decisions and tradeoffs

**Standalone app, not bolted into Bridge RNA.**
Bridge Manifold is a separate Dash app in its own directory, importing reusable functions from Bridge RNA rather than editing the 2,470-line retrieval app.
This keeps the heavy exploratory tool from destabilizing the retrieval product, while a shared header and shared CSS make them feel like one instrument.
The tradeoff is a small amount of duplicated app scaffolding, which is worth it for isolation.

**Offline precompute over interactive computation.**
Every expensive step (model inference, UMAP, density rasters) is precomputed and cached, and the app only ever loads artifacts.
This is forced by the measured cost and is what makes the app responsive.
The tradeoff is a build step and cache management, which is the right trade for a 943k-point tool.

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
Because retrieval is cosine and the raw vectors carry a 4x magnitude spread that dominates PC1 (57.8% before normalization, 40.9% after), we normalize first so the manifold reflects transcriptomic direction rather than sequencing depth.

**Make batch visible, do not correct it (final).**
Rather than applying Harmony or ComBat and risking erasure of real biology, the tool exposes study and species as color-by dimensions and discloses the measured 54x tissue-controlled cross-corpus effect on the control rail.
Josh confirmed this is the final call: no correction algorithm, not even as a later toggle.

**Bounded glyph budget with a density underlay.**
Instead of trying to render 940k live points, we render ~100k live glyphs over a raster of all 942,563, so the global shape is always honest while interaction stays smooth.
That underlay then does double duty as the honest fallback for an uncovered corpus (section 7.4), which is a large part of why the fallback is cheap.

**No on-demand statistics.**
The selection readout was removed rather than repaired.
It was the app's largest source of complexity (a 450-line statistics module, a 2.07 GB ANN index, an exact covariance artifact, a third UI column) in service of a question the map itself answers qualitatively, and every number it produced had to be defended against a null that most readers would never see.
Removing it dropped the cache from about 2.3 GB to a measured 219.2 MB (of which the app opens 82.3 MB), took the serving app's dependency surface down to `dash`/`plotly`/`numpy`/`pandas`/`pyarrow`, and cut the test suite from 4.54 s to about 0.55 s.
The two dead artifacts are still physically present on this machine as leftovers from the 2026-07-21 build, which is why `cache/` currently measures 2,293.8 MB; they can be deleted (`REFERENCE.md` section 12).

## 9. Directory layout

This is the layout as built.

```
Bridge Manifold/
  app_manifold.py            # Dash entry: argparse host/port/debug, loopback guard
  manifold/
    paths.py                 # every artifact path; BRIDGE_RNA_ROOT / MANIFOLD_CACHE_DIR overrides
    preflight.py             # missing-artifact and Git-LFS-pointer guards
    bridge_rna.py            # the single seam that imports from the sibling repo
    data.py                  # parquet loaders, the fixed global point order, module-level caches
    tissue.py                # the shared tissue vocabulary, used by both corpora
    colorby.py               # the coverage-aware color-by registry
    sampling.py              # stratified quota sampling, viewport re-stratification
    render.py                # layered figure: density underlay, ARCHS4 cloud, OSDR overlay
    theme.py                 # plot theme (dark canvas) + validated categorical palette
    layout.py                # header, left control rail, plot (two columns)
    callbacks.py             # color-by, coverage readout, layers, method, zoom LOD, legend filter
  precompute/
    embed_osdr.py            # OSDR counts -> 2,108 x 512 embeddings (loads torch), resumable
    build_projections.py     # L2 -> PCA-50 -> {pca2,pca3,umap2,umap3}, density rasters
    fetch_archs4_meta.py     # sigpy API -> per-GSM series/title/source/characteristics + tissue
    validate_artifacts.py    # objective build gate; exits nonzero on failure
  tests/
    fixture_corpus.py        # synthetic corpus with known latent clusters and GEO-style metadata
    conftest.py              # points the package at that corpus before import
    build_dev_corpus.py      # CLI to build a browsable corpus; --no-archs4-meta gives the degraded state
    test_data.py             # global point order, label lookups, degraded loaders
    test_tissue.py           # the vocabulary: ordering collisions, boundary collisions, coalescing
    test_colorby.py          # declared coverage vs produced labels, availability, degraded state
    test_render.py           # shared palette, context-not-grey, budgets, viewport, legend
    test_app.py              # callback wiring, coverage readout, CSS class coverage
  assets/
    manifold.css             # Bridge RNA tokens, Dash 4 token remap, dark-canvas + legend rules
  cache/                     # generated (gitignored): embeddings, coords, density, metadata
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
L2-normalize, fit `IncrementalPCA-50` on a stratified subsample plus all OSDR, project the joint corpus, write `pca2` and `pca3`.
Validate: PC1 lands well below the 57.8% pre-normalization figure, which is the objective test that normalization was applied (measured: 40.9%, cumulative 95.1%).

**Phase 3. First interactive plot.**
Wire the Scattergl renderer, the density underlay, stratified sampling, the dataset and method toggles, and color-by.
Validate: 100k + 2,108 points pan and zoom smoothly; every drawn glyph sits on a real corpus coordinate; the OSDR overlay draws every OSDR point exactly once; color-by switches without a full reload.

**Phase 4. UMAP projections.**
Implement the landmark fit-and-transform in `build_projections.py`; write `umap2` and `umap3`.
Validate: `validate_artifacts.py` passes structurally (row counts agree across every artifact, all coordinates finite, every axis has real spread) and OSDR occupies a real region of the map rather than collapsing to a blob.

**Phase 5. Coloring both corpora.**
Fetch the ARCHS4 GEO metadata over the sigpy API, build the shared tissue vocabulary, and replace the renderer's per-key branching with the coverage-aware registry.
Validate, objectively and not by looking at the map:

- every enabled menu option produces a non-empty legend and a figure with at least one trace,
- the coverage a field **advertises** equals the number of labels its resolver actually **produces** (`covered == (labels != NOT_COVERED).sum()`, checked for every registered field),
- every whole-map field covers all 942,563 points and the app's default field is one of them,
- an OSDR-only field draws zero ARCHS4 glyphs when the raster is on, and a faint context trace with no legend swatch when it is off,
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
/Users/josh/Bridge-RNA/.venv/bin/python precompute/build_projections.py  # PCA + UMAP coords, density rasters. ~5 min.
/Users/josh/Bridge-RNA/.venv/bin/python precompute/fetch_archs4_meta.py  # ARCHS4 GEO metadata. ~35 s, needs network.
/Users/josh/Bridge-RNA/.venv/bin/python precompute/validate_artifacts.py --mixing
/Users/josh/Bridge-RNA/.venv/bin/python app_manifold.py                  # http://127.0.0.1:8051

/Users/josh/Bridge-RNA/.venv/bin/python precompute/build_projections.py --density-only   # re-render rasters only
```

Tests:

```bash
cd "/Users/josh/Bridge Manifold" && /Users/josh/Bridge-RNA/.venv/bin/python -m pytest tests/ -q
```

144 tests in about 0.55 s, against a hermetic synthetic corpus (4,000 ARCHS4 + 300 OSDR points) that never touches the real memmap, the checkpoint, or the multi-hour artifacts.
The suite was 103 tests at 4.54 s before the redesign; removing the selection feature took the ANN index out of the fixture, which was 43% of the old runtime, and the color-by and tissue tests added back more coverage than was removed.
The per-file split is in `REFERENCE.md` section 12.
