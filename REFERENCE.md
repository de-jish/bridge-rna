# Bridge Manifold - Verified Reference

> **This document predates the 2026-07-22 merge.**
> Bridge Manifold and Bridge RNA are now one repository and one application, served by `app.py`: the retrieval view at `/` and this map at `/map`.
> There is no `app_manifold.py` and no separate repository at `/Users/josh/Bridge Manifold`.
> The design decisions recorded below are still the ones the map is built on; the commands and the file layout have been updated where they would otherwise fail if followed.
> See `README.md` for the current product and `progress.md` for what changed.



This is the ground-truth appendix for `IMPLEMENTATION.md`.
Every fact here was verified directly against the checkpoint, the memmap, the parquet, and the data files, not taken from documentation or from an agent's summary.
Sections 1 to 7 and 9 were verified on 2026-07-20 and re-checked against the shipped code on 2026-07-21; sections 8 and 10 to 12 were measured on 2026-07-21 against the real 942,563-point corpus.
Where a number could not be re-derived from an artifact that still exists, the measurement is labelled with when and how it was taken.
Paths are relative to the Bridge RNA repository at `/Users/josh/Bridge-RNA` unless noted.

## 1. The model (verified from `ckpt['config']` and state_dict)

The checkpoint is `checkpoints_performer/r7hnr92k/best_model.pt` (547 MB).
Top-level keys: `model_state_dict`, `optimizer_state_dict`, `scheduler_state_dict`, `scaler_state_dict`, `epoch`, `train_loss`, `val_loss`, `config`, `run_metadata`, `total_params`.

Config (the authoritative hyperparameters; the demo's default constants are only fallbacks and differ):

| Field | Value |
| --- | --- |
| hidden_dim | 512 |
| ffn_dim | 2048 |
| num_heads | 8 |
| num_layers | 12 |
| ree_base | 100 |
| feature_type | flash |
| compute_type | iter |
| normalization | log1p_tpm |
| mask_token | -10 |
| include_species_embedding | False |
| seed | 42 |

State dict: 196 tensors, transformer layer indices 0 through 11 (confirming 12 layers).
`gene_embedding.weight` shape is `(15165, 512)`, which fixes `num_genes = 15165`.
Because `include_species_embedding = False`, `species_ids` passed to `encode` is ignored in the forward pass, so OSDR embedding does not need it.

Loading recipe (from `demo_osdr_top5.py`):
`torch.load(path, map_location='cpu')`, read `cfg = dict(ckpt['config'])`, build `ExpressionPerformer` from cfg with `num_genes = query_vec.shape[0]`, pass `ckpt['model_state_dict']` through `_strip_module_prefix`, then `load_state_dict(..., strict=False)`, `model.to(device)`, `model.eval()`.
Device resolves to CPU on this machine: `torch.device(args.device if torch.cuda.is_available() and args.device.startswith('cuda') else 'cpu')`, and there is no MPS branch anywhere, so inference is fp32 on CPU.

## 2. The canonical gene list (verified digest match)

File: `data/archs4/train_orthologs/canonical_genes.csv`, column `gene_symbol`.
Count: 15,165 genes.
First five: `A1CF, A2M, A3GALT2, A4GALT, A4GNT`.
`sha256("\n".join(g.strip() for g in genes))` equals `CANONICAL_GENES_SHA256 = 3f887ac8d329dce3c54d26448964904c07a345940cd3d9ebab18dd1f603194c5`.
This match is a hard build gate for `embed_osdr.py`.

## 3. The ARCHS4 embedding index (verified)

Embeddings: `archs4_sample_embeddings_full/sample_embeddings.float16.mmap`, `940455 x 512` float16 (963,025,920 bytes, which is exactly 940455 * 512 * 2).
Metadata: `archs4_sample_embeddings_full/sample_locations.parquet`, 940,455 rows.
Columns: `global_index` (int64), `shard_idx` (int32), `shard_file` (string), `row_in_shard` (int32), `geo_accession` (string), `species_id` (int16).
Species split: `species_id` 0 (human) = 510,709; 1 (mouse) = 429,746; total 940,455.
Manifest (`embedding_manifest.json`): `total_samples=940455`, `embedding_dim=512`, `embedding_dtype=float16`, `normalization=log1p_tpm`, `feature_type=flash`, not L2-normalized.

This memmap is read by `precompute/build_projections.py` and by `precompute/validate_artifacts.py --mixing` only.
The map view never opens it, so `BRIDGE_RNA_ROOT` is required to *build* the cache and not to *draw* the map. The retrieval view does open it, on every cached search.

## 4. Measured vector statistics and timings (on Josh's Mac)

Vector L2 norms (25k random sample): mean 10.57, std 2.26, min 6.75, max 25.47; no NaN or Inf.
The 3.9x magnitude spread is why L2 normalization is required before reduction; see "What the embedding norm actually is" below for what that magnitude turned out to measure.

PCA variance (25k sample, before normalization): PC1 = 57.8% of variance; cumulative PC2 63.2%, PC5 71.3%, PC10 78.6%, PC20 87.2%, PC50 96.4%.
PCA-50 fit on 25k x 512 = 11.7 s; projecting 200k to 2D via fitted components = 1.44 s, extrapolating to ~6.7 s for all 940k.

UMAP (on PCA-50 input, L2-normalized): fit on 40k = 171 s; `.transform()` 60k = 21.5 s.
Landmark hybrid (fit ~200k subsample, transform the rest) was believed to be the only tractable path.
It was not; see the full-corpus build below.

### Landmark build, measured (2026-07-21, superseded)

The 30-to-90-minute estimate above was wrong by an order of magnitude, and the landmark design it justified has since been replaced.
Kept because the numbers are the baseline the full-corpus build is scored against.

| stage | original | after the n_neighbors/metric retune |
| --- | --- | --- |
| IncrementalPCA-50 fit (60k ARCHS4 + 2,108 OSDR) | 1 s | 1 s |
| PCA transform, all 942,563 points | 1 s | 2 s |
| UMAP-2d landmark fit, 122,108 points | 71 s | 68 s |
| UMAP-2d transform, remaining 820,455 | 71 s | 404 s |
| UMAP-3d fit + transform | 143 s | 467 s |
| **total** | **347 s** | **~950 s (15.8 min)** |

The retune roughly tripled the build, and all of the increase was in `.transform()`.

### Full-corpus build, actually measured (2026-07-22)

**Both reductions are now fit on all 942,563 points.** No subsample, no landmark set, no `.transform()`.
Measured end to end on a 10-core M4 with 16 GB of RAM:

| stage | cost | notes |
| --- | --- | --- |
| exact PCA over all 942,563 points | **4.5 s** | one streaming pass accumulating a 512x512 second moment in float64 |
| PCA transform to 3 components | 1 s | |
| materialize the normalized corpus | 1 s | 1.93 GB resident, the build's largest allocation |
| k=15 cosine neighbour graph, `n_jobs=1` | **59 s** | built once, reused by both fits via `precomputed_knn` |
| UMAP-2d layout, all 942,563 points | **251 s** | |
| UMAP-3d layout, all 942,563 points | **251 s** | |
| **total** | **~10.5 min** | against 15.8 min for the landmark build it replaced |

So fitting everything is not merely affordable, it is **faster than the landmark build was**, because the 404 s and 467 s `.transform()` passes are gone.
The "a direct 940k fit is hours" claim that shaped the entire first design was never measured.

Three settings make that possible, and two of them are non-obvious.

- **`precomputed_knn`, one graph for both fits.**
  The graph depends on the input space and `n_neighbors`, never on `n_components`, so it is built once with `pynndescent` and handed to both.
  That halves the neighbour search and guarantees the 2-d and 3-d maps are layouts of the *same* graph.
  UMAP assigns the arrays through without copying and then writes into them in place to disconnect far neighbours (`umap_.py:2647-2654`), so each fit gets its own copy.
- **`init` is the exact PCA, not spectral.**
  This one is not an optimization, it is the difference between finishing and not.
  `_spectral_layout` sizes its Lanczos basis as `max(2k+1, sqrt(n))` (`spectral.py:489`), which at n = 942,563 is 970 vectors, so `eigsh` allocates a **942,563 x 970 float64 basis: 7.31 GB**.
  Measured, that drove the build into 7.6 GB of swap and produced no progress in 25 minutes.
  `init="tswspectral"` uses the same formula and does not help.
  `init="pca"` is the documented alternative but would run a second PCA over the same matrix and copy it to centre it, so the exact PCA coordinates from the previous stage are passed directly, scaled the way `noisy_scale_coords` (`umap_.py:930`) scales UMAP's own.
- **`random_state=42`, and what it costs.**
  Seeding forces single-threaded execution three separate ways: `_validate_parameters` overwrites `n_jobs` to 1 (`umap_.py:1950-1954`, and its warning is printed after the assignment so it always reads the confusing "n_jobs value 1 overridden to 1"), `fit()` calls `numba.set_num_threads(1)` for the whole fit (`:2413-2415`, restored at `:2855`) which serializes the `njit(parallel=True)` kernels inside `fuzzy_simplicial_set`, and `_fit_embed_data` passes `random_state is None` as the `parallel` flag (`:2891`), selecting the serial layout kernel at `layouts.py:222-224`.
  Measured on this machine against a real UMAP graph, the serial layout costs **4.3x to 7.5x** the parallel one.
  Determinism is kept anyway: this is a once-per-corpus offline artifact, and the whole build is 10.5 minutes.

`--knn-jobs` defaults to 1 for the same reason and is a separate knob: NN-descent's heap updates race under threads, so the same seed gives a slightly different graph run to run at `n_jobs=-1`.
That is roughly 10x faster on that one 59-second stage and gives up graph reproducibility.
You cannot have both a parallel neighbour search and a bitwise-reproducible map.

### Was fitting everything worth it? (measured 2026-07-22)

Scored by `validate_artifacts.py --quality --compare`, on one 60,000-point sample, against the exact 15-NN of the same points in the original 512-d space:

| coords | 15-NN recall, landmark | 15-NN recall, full | 25-NN tissue purity, landmark | full |
| --- | --- | --- | --- | --- |
| pca2 | 0.0374 | 0.0374 | 0.1653 | 0.1654 |
| pca3 | 0.1211 | 0.1199 | 0.2758 | 0.2758 |
| **umap2** | 0.3660 | **0.3955 (+8.1%)** | 0.5926 | 0.5838 (-1.5%) |
| **umap3** | 0.4291 | **0.4596 (+7.1%)** | 0.6080 | **0.6169 (+1.5%)** |

The honest summary: **the full UMAP fit buys about 8% more local fidelity and leaves biological fidelity where it was**, moving 1.5% down in 2-d and 1.5% up in 3-d.
That is the shape of result a full fit should produce, since `.transform()` placed 87% of the corpus by averaging landmark positions rather than by letting those points act on the layout.
Against the anchors that make those numbers readable, umap2 keeps **92.3%** and umap3 **98.2%** of the tissue structure recoverable in 512-d (permuted-label null 0.0710, 512-d ceiling 0.6267).

**PCA barely moved, and that is worth recording as a negative result.**
The exact full-corpus fit and the 60,000-point `IncrementalPCA` fit agree to r = 0.999998 on PC1, 0.999941 on PC2 and 0.994802 on PC3, and PC1's share went 40.948% to 41.318%.
The subsample was an excellent approximation.
The exact fit is kept because it costs 4.5 seconds, removes an approximation from the pipeline, and yields the full 512-eigenvalue spectrum for free, not because it changed the picture.
Note the degradation with component index: by PC3 the agreement is already 0.9948, so a build wanting more than three components should not assume the subsample stays adequate.

### UMAP settings, chosen by measurement (2026-07-21)

The reducer runs `n_neighbors=15`, `metric="cosine"`, on the **raw 512-d L2-normalized vectors**, not `n_neighbors=30` with euclidean on PCA-50.
Scored on a 60,000-point sample against the original 512-d space, three seeds per configuration (seed sd was 0.001 to 0.002 on both metrics, so these gaps are 8 to 37 standard deviations):

| configuration | kNN recall @15 | 25-NN tissue purity |
| --- | --- | --- |
| n_neighbors 30, euclidean on PCA-50 (previous) | 0.380 | 0.630 |
| n_neighbors 15, euclidean on PCA-50 | 0.417 | 0.638 |
| n_neighbors 30, cosine on raw 512-d | 0.398 | 0.642 |
| **n_neighbors 15, cosine on raw 512-d (shipped)** | **0.426** | **0.646** |

The two changes compose. Reducing to PCA-50 first was discarding the 4.9% of variance those components do not carry, and `n_neighbors=30` was over-smoothing local structure.

Confirmed on the **real 942,563-point map**, not just the sample: 25-NN tissue purity over all 853,989 points carrying a real tissue bucket, 30,000 queries, went **0.6448 to 0.6756 (+4.8%)** against a permuted-label null of 0.0761, so the lift over null went 8.5x to 8.9x.
The OSDR spread ratio also improved, 0.827 to 0.850, meaning the spaceflight corpus occupies more of the map rather than less.

**PCA after L2 normalization, exact over the whole corpus: PC1 = 41.3%, cumulative over the first 50 of 512 components = 95.0%.**
(The 60,000-point subsample the earlier build used reported 40.9% and 95.1%.)
Contrast with the 57.8% above, which is the *pre-normalization* magnitude axis.
This pair is the objective test for invariant 2 - see `precompute/validate_artifacts.py`, which fails the build if PC1 is not well below 50%.
The full spectrum is now recorded in `projection_stats.json` and the validator checks that it has 512 entries summing to 1, which is the evidence the fit was exact rather than truncated.
Cumulative shares: PC2 49.0%, PC3 53.3%, PC10 70.4%, PC50 95.0%, PC100 99.0%, PC256 99.9%.

### What the embedding norm actually is (measured 2026-07-21, corrects an earlier claim)

Every earlier version of these documents called the pre-normalization L2 norm a **sequencing-depth axis**. That is wrong, and it was never measured - it was inferred from the fact that PC1 was large.

The theoretical objection comes first: the encoder's input is **log1p-TPM**, and TPM is depth-normalized by construction, so library size has already been divided out before the model sees anything. The norm cannot be library size.

Measured directly on OSDR, where `cache/osdr_expression.float32.npy` holds the exact log1p-TPM matrix that produced each of the 2,108 embeddings:

| the embedding norm vs | pearson | spearman |
| --- | --- | --- |
| share of expression held by the top 100 genes | **+0.987** | +0.986 |
| Gini concentration | +0.944 | +0.937 |
| Shannon entropy of the log1p-TPM vector | **-0.930** | -0.923 |
| number of genes detected | -0.654 | -0.698 |

The norm is a measure of how **concentrated** a sample's transcriptome is, and that is biology, not an artifact. The per-tissue ordering is the textbook one: liver 13.57, skeletal muscle 12.92 and heart 12.62 hold the top three, each dominated by a handful of genes (albumin and apolipoproteins; the sarcomeric proteins), while bone/cartilage 7.34, skin 7.84 and brain 8.31 sit at the bottom, brain being the most transcriptionally complex tissue there is. Across the ARCHS4 sample, **26.2%** of the norm's variance is explained by tissue identity alone.

Two consequences.

**L2 normalization does not remove this signal.** A ridge probe predicting the raw norm from the *normalized* 512-d direction reaches a held-out **R^2 = 0.977**, and PC1 of the normalized space has spearman **+0.957** with the norm. Normalizing removes a redundant encoding of something the direction already carries; it does not remove the thing.

**Do not try to project it out.** Removing the single best-fitting direction barely moves the probe (R^2 0.977 to 0.975), because the signal is spread across many directions - and it should not be removed anyway, since it is a real biological property rather than a technical covariate. Invariant 2's conclusion stands; only its stated reason was wrong. Normalize because a 3.9x magnitude spread would otherwise make the map about magnitude rather than direction.

Structural facts from the same artifacts, re-verified 2026-07-21 by re-running `validate_artifacts.py`: all four coordinate parquets carry exactly 942,563 finite rows, `points_meta.parquet` marks the first 940,455 as ARCHS4 and the trailing 2,108 as OSDR, and the OSDR block occupies a real region of the map rather than one blob (OSDR umap2 spread / corpus spread = 0.827).

### Cross-corpus separation, measured exactly (re-verified 2026-07-21)

These figures were first measured through the hnswlib approximate index.
That index has been deleted, and `validate_artifacts.py --mixing` now computes the **exact** top-51 neighbours of each of the 2,108 OSDR samples by streaming the memmap in 50,000-row blocks and merging a running top-k (`_osdr_neighbours`).
Exact costs 10.3 s with the memmap warm in the page cache, which is why a 2.07 GB approximate index was not worth keeping.
The exact numbers below differ from the approximate ones only in the third digit (same-study was 34.2%/5,227x, now 34.3%/5,233x).

Of the 50 nearest neighbours of each OSDR sample in the 512-d cosine space: 34.3% same-study OSDR, 22.9% cross-study OSDR, 42.9% ARCHS4.
Against a per-sample chance model that is 5,233x for same-study (replicate structure, expected and biological) and **105x for cross-study** (a corpus-level effect).

Controlling for tissue, the dominant axis of variation in bulk expression: OSDR neighbour pairs sharing **neither study nor tissue** occur at 11.491% against 0.21101% expected, **54x over chance**.
Biology does not make liver cluster with brain, so this is a technical batch effect from fp32/CPU versus bf16/CUDA inference and preprocessing differences, not spaceflight signal.

This is surfaced in the app as a standing caution at the bottom of the left control rail (`layout.control_rail`, `.bm-caution`), not inside any per-selection panel.
It is a property of the map itself rather than of anything a user does to it, so it is stated once, always, next to the controls.

Corroborating asymmetry, measured 2026-07-21 by a one-off analysis that is not part of the validation script: ARCHS4 queries find OSDR neighbours only 0.0300% of the time against 0.224% chance, a 7.5x *depletion*.
OSDR is a small dense island - dominant in its own neighbourhoods, nearly invisible from ARCHS4's.
Cosine gap (from the exact run): OSDR to its OSDR neighbours 0.9979, to its ARCHS4 neighbours 0.9966, a gap of 0.0014, against an ARCHS4 self-baseline of 0.9971.

## 5. Environment (re-verified 2026-07-21)

venv at `/Users/josh/Bridge-RNA/.venv`.
The versions recorded during design had drifted by the time the build ran; these are the ones the code was written and tested against.

| Package | Version | Notes |
| --- | --- | --- |
| numpy | 2.4.6 | |
| pandas | 3.0.3 | major release - see the behaviour note below |
| pyarrow | 20.0.0 | |
| scikit-learn | 1.9.0 | precompute only, and now only for the `--quality` scoring |
| umap-learn | 0.5.12 | precompute only |
| pynndescent | 0.6.0 | precompute only; the shared k-NN graph both UMAP fits use |
| plotly | 6.8.0 | serializes numpy arrays as base64 typed arrays |
| dash | 4.4.0 | rewritten Dropdown/RadioItems, new `--Dash-*` design tokens |
| torch | 2.12.1 | precompute only; MPS available, CUDA absent |
| requests | 2.34.2 | precompute only (ARCHS4 GEO metadata fetch) |
| pytest | 9.1.1 | dev |
| playwright | 1.61.0 | dev, browser checks against the running app |
| scipy | 1.16.0 | present in the shared venv, pulled in by umap-learn; not imported by Bridge Manifold |
| pillow | 12.2.0 | present in the shared venv, **no longer used** - it existed for the density raster PNGs |
| hnswlib | 0.8.0 | present in the shared venv, **no longer used** by Bridge Manifold |
| h5py | 3.16.0 | installed for the ARCHS4 HDF5 experiment in section 8; not imported by any shipped code |

`requirements.txt` splits the surface deliberately.
The map view (`manifold/`, served by `app.py`) needs only `dash`, `plotly`, `numpy`, `pandas`, `pyarrow`: it draws a precomputed map, opens no embeddings, and computes no statistics, so it carries no scientific stack at all.
`torch`, `scikit-learn`, `umap-learn`, `pynndescent`, and `requests` are precompute-only.
`scikit-learn` survives in that list only for `validate_artifacts.py --quality`; the build itself stopped importing it when `IncrementalPCA` was replaced by an exact streaming eigendecomposition.

Absent: `datashader`, `archs4py`, `cuml`, `pacmap`, `openTSNE`.

Four library behaviours the code actively depends on, each of which caused a real defect during the build:

- **pandas 3.0**: `Series.astype(str)` no longer stringifies missing values; they stay NA.
  A later `.replace({"nan": "Unknown"})` therefore never matches them, and the NA survives into legends as a phantom category.
  Every categorical path calls `fillna("Unknown")` explicitly.
  The same trap appears in `fetch_archs4_meta.first_series`, where unresolved accessions arrive as float NaN and `value or ""` does not catch them because NaN is truthy; without the explicit isna guard they became the literal string `"nan"`, which reads as a real GSE and overstates metadata coverage to a clean 100%.
- **plotly 6.x**: numpy arrays are serialized as base64 typed-array specs, so `customdata` is passed as a plain Python list.
  That convention was originally forced by a defect in the removed lasso path: Dash built its `selectedData` payload by indexing the *user* data, which yields `undefined` for an encoded array, so every selection read as empty.
  The selection feature is gone and `customdata` now feeds only the OSDR hover, but the plain-list convention stays.
- **dash 4.x**: `dcc.Dropdown` is a radix popover (not react-select), so all `.Select-*` CSS is dead; `labelClassName` on `RadioItems`/`Checklist` lands on an inner text span rather than the label.
  Components are styled through Dash's own structural classes and its `--Dash-*` tokens.
  Dash's dropdown also has no option-group support, which is why `colorby.menu_options()` carries grouping through ordering plus a scope suffix rather than faked header rows.
- **pandas 3.0, again, and this one is a memory trap rather than a correctness one**: `.to_numpy()` on a string-dtype Series materializes a *fresh* Python `str` object per element.
  `render._colour_plan` memoizes one per-point category array per color-by, and as strings that array held 942,563 distinct objects to express 13 distinct values: **127.5 MB measured, per color-by**, or about 1.4 GB across the 11-entry registry, against an app that otherwise opens 80.8 MB.
  It is stored as `int16` legend slots instead, which is 1.9 MB, and category selection becomes a vectorized integer compare rather than 942,563 Python string comparisons.
  Measured effect on the render path: a warm figure over the whole corpus went from 1.33 s to 0.06 s.

## 5b. Measured at build time

- OSDR encode, CPU fp32, 10 threads: **~6.5 s/sample** (B=1 and B=4 within noise of each other).
  2,108 samples is therefore a multi-hour job.
- MPS is not an option for this model: `F.scaled_dot_product_attention` has no fused kernel for a 15,165-token sequence on Metal, so it materializes the full attention matrix and fails on a 6.85 GiB allocation.
  Chunking the attention over the query axis makes it *run* and is numerically equivalent to CPU (cosine 0.99999988, max abs delta 4.2e-7), but it was not faster in measurement, so CPU fp32 remains both the fidelity baseline and the fastest available path.
- Preprocessing equivalence: the batched OSDR path reproduces Bridge RNA's `load_random_osdr_sample_vector` **bit-for-bit** (max abs diff 0.0) across samples from three different studies.

## 6. Reusable Bridge RNA interfaces

Line numbers are from the Bridge RNA repository as of 2026-07-20.
`manifold/bridge_rna.py` is the single import seam, and it currently returns exactly six symbols: `ExpressionPerformer`, `CANONICAL_GENES_SHA256`, `canonical_gene_order_digest`, `_strip_module_prefix`, `build_mouse_to_human_maps`, `normalize_counts_to_tpm_single`.
Everything else below is catalogued because it was verified and may be wanted later, with its current reuse status stated.

From `demo_osdr_top5.py`:

- `load_random_osdr_sample_vector(args) -> (np.ndarray[num_genes], sample_id, pd.Series)` at 327-406.
  The full OSDR preprocessing recipe (read counts, ortholog map, collapse, reindex to canonical, TPM, log1p).
  Not imported; `precompute/embed_osdr.py` reimplements it in batched form and is checked bit-for-bit against it (section 5b).
- `build_model_and_query_embedding(args, query_vec) -> np.ndarray[512]` at 409-436.
  Loads the checkpoint and runs `encode`; the single-sample driver that `embed_osdr.py` generalizes to batches.
- `build_mouse_to_human_maps(orthologs_path, mouse_exon_lengths_path) -> (dict, dict)` at 233-250. **Imported.**
- `normalize_counts_to_tpm_single(counts, exon_len_by_human_gene) -> pd.Series` at 253-267. **Imported.**
- `resolve_canonical_genes(explicit) -> Path` at 270-324, with candidates `data/archs4/train_orthologs/canonical_genes.csv` then `data/ensembl/canonical_genes.inferred.csv`.
- `fetch_archs4_metadata(geo_accessions, human_h5, mouse_h5) -> pd.DataFrame` at 463.
  **No longer reused.** It reads per-GSM fields out of the ARCHS4 gene-level HDF5 files through `archs4py`, which means a 62.3 GB plus 50.7 GB download that never happened, and `archs4py` is not installed.
  It is replaced by `precompute/fetch_archs4_meta.py`, which pulls the same fields from the Maayan Lab sigpy JSON API in 35 s over 216 MB with no HDF5 dependency at all (section 8).

From `generate_archs4_embeddings.py`:

- `class ExpressionPerformer.__init__(num_genes, hidden_dim, n_heads, n_layers, ffn_dim, ree_base, mask_token_id, feature_type, compute_type, include_species_embedding, num_species)` at 106-156. **Imported.**
- `ExpressionPerformer.encode(x, species_ids=None, normalize=False) -> Tensor[batch, 512]` at 187-193, mean-pools hidden states over the gene axis; accepts a full batch `x` of shape `[batch, num_genes]`.
- `ExpressionPerformer._encode_hidden(x, species_ids=None) -> Tensor[batch, num_genes, 512]` at 158-180; expression values enter only through the rotary expression embedding of `x`.
- `CANONICAL_GENES_SHA256` at 40 and `canonical_gene_order_digest(genes)` at 43-51. **Both imported**, and together they are the gene-digest build gate.
- `_strip_module_prefix(state_dict)` at 258, removes a leading `module.` from keys. **Imported.**

From `bridge_rna/` (the retrieval package, split from the now-deleted `app_osdr_dash.py` monolith):

- `_load_archs4_index() -> (np.memmap, pd.DataFrame, int)` at `bridge_rna/retrieval.py:293`, cached; reads the manifest, opens the float16 memmap `[n, d]`, loads `sample_locations.parquet`.
  **Reused by the retrieval view's cached path**, which calls it to open the 963 MB memmap on every cached search; the map view does not reuse it, since it opens no embeddings, and the two precompute scripts open the memmap directly with `np.memmap` from the manifest.
- `_topk_cosine_from_memmap(...)` at `bridge_rna/retrieval.py:372`, normalizes the query and does cosine top-k over the memmap, streaming it in 25,000-row chunks.
  **Reused by the retrieval view's cached path**, which pairs it with `_load_archs4_index` to score a query against the whole memmap. The map view does not use it; `validate_artifacts._osdr_neighbours` does its own exact blocked top-k for the mixing check.
- `_find_precomputed_query_embedding_file()` at `bridge_rna/retrieval.py:289`, and `PRECOMPUTED_QUERY_EMBEDDING_CANDIDATES` at `bridge_rna/config.py:41` (no such file currently exists on disk).
- `preflight_retrieval_requirements(...)` at `bridge_rna/preflight.py:151` and `_is_lfs_pointer(...)` at `bridge_rna/preflight.py:135`, the retrieval LFS guards.
  Reimplemented rather than imported, in `manifold/preflight.py`, so the map view can preflight without importing the retrieval package.
  The signature checked is the leading `version https://git-lfs.github.com/spec/v1` in a file under 4096 bytes.
- Ollama AI-summary integration, its constants `OLLAMA_BASE_URL=http://127.0.0.1:11434` and `OLLAMA_MODEL=gemma3:4b` at `bridge_rna/config.py:38-39` and the calls in `bridge_rna/ai.py`.
  Not used by the map view.

## 7. OSDR color-by columns

File: `data/osdr/metadata/selected_sample_metadata.tsv`, 2,896 rows x 44 columns.

**Corpus size (corrected).**
All 2,896 rows are *Mus musculus* and all have a counts path, but only **2,163** carry a non-empty spaceflight factor, which is the filter Bridge RNA's retrieval uses and the one Josh chose to match.
Of those, **2,108** produce an expression vector; the other 55 name a sample column their counts matrix does not contain, and `embed_osdr.py` reports them rather than dropping them silently.
So: 2,896 in the TSV -> 2,163 eligible -> 2,108 embedded.
Earlier drafts of these documents used 2,896 as the corpus size, which was wrong.

The spaceflight factor has nine raw spellings covering seven distinct arms, counted over the 2,163 eligible TSV rows:

| Value | n |
| --- | --- |
| Space Flight | 794 |
| Ground Control | 654 |
| Basal Control | 290 |
| Vivarium Control | 267 |
| Ground control | 96 |
| Ground Control Rerun | 35 |
| Vivarium control | 18 |
| Cohort Control #1 | 5 |
| Cohort Control #2 | 4 |

Casing variants are folded onto the most frequent spelling at precompute time.
The arms themselves are *not* merged: basal animals were sacrificed at experiment start and vivarium animals never entered flight hardware, so collapsing them into one "Ground Control" bucket would erase real experimental structure.

### As shipped, over the 2,108 embedded samples

Verified 2026-07-21 by reading `cache/osdr_metadata.parquet` directly.
The parquet carries exactly the columns the color-by registry resolves: `sample_key`, `spaceflight`, `tissue`, `strain`, `sex`, `genotype`, `study`, `habitat`, `duration`, `diet`.

| field | distinct values |
| --- | --- |
| spaceflight | 7 |
| tissue | 48 |
| strain | 8 |
| sex | 3 |
| genotype | 4 |
| study | 70 |
| habitat | 10 |
| duration | 28 |
| diet | 5 |

Spaceflight arms after case folding: Space Flight 777, Ground Control 738, Basal Control 278, Vivarium Control 271, Ground Control Rerun 35, Cohort Control #1 5, Cohort Control #2 4.
The derived `flight_status` (`data._flight_status`) collapses those to Ground 1,331 and Space Flight 777, with no Unknown.

### The color-by registry

Color-bys are no longer a chain of `if/elif` in the renderer.
`manifold/colorby.py` holds a registry of 11 `ColorBy` records, each declaring a `scope` (which corpora it *could* describe), a resolver, an optional hint, and an optional `(predicate, fix-hint)` pair for an artifact it needs.
`covers()` reports which corpora it can color *right now on this machine*, and that one fact drives the menu order, the disabled state, the coverage readout under the control, and what the renderer does with the corpus a field does not describe.

| key | label | scope | notes |
| --- | --- | --- | --- |
| tissue | Tissue | ARCHS4 + OSDR | needs `cache/archs4_metadata.parquet` for the ARCHS4 half; the default color-by |
| species | Species | ARCHS4 + OSDR | always available; the reference for what a working color-by looks like |
| flight_status | Flight vs Ground | OSDR | derived contrast |
| spaceflight | Spaceflight arm | OSDR | the seven raw arms kept distinct |
| strain | Strain | OSDR | |
| sex | Sex | OSDR | |
| genotype | Genotype | OSDR | |
| study | Study | OSDR | 70 studies; this is the batch axis |
| habitat | Habitat | OSDR | |
| duration | Mission duration | OSDR | |
| diet | Diet | OSDR | |

Two invariants of this table are worth stating because both were real defects that tests now pin:

- The availability predicate for `tissue` is `data.archs4_metadata_available` itself, not a path re-derived inside the registry.
  A second source of truth for the same file could disagree with the loader about whether the field works.
- `colorby.labels(key)` returns one array over the full corpus with a `NOT_COVERED` sentinel for points the field says nothing about.
  `NOT_COVERED` is never a category and never gets a legend swatch, because giving absence a swatch is exactly what made 99.8% of the map read as grey *data*.

High-cardinality fields (70 studies, 48 tissues) exceed Plotly's usable native legend (~12 rows), so the renderer keeps a Top-11-plus-residual trace model with a custom searchable legend.
Categories are ranked **once over the whole covered population** and every layer draws from that single mapping, so a liver in GEO and a liver in OSDR get the same color; ranking per layer silently gave one category two colors.
Legend counts are whole-corpus counts, so they do not move when the point budget or the zoom changes.

## 8. ARCHS4 per-sample GEO metadata (fetched and measured, 2026-07-21)

The local ARCHS4 artifacts carry only `geo_accession` and `species_id`; there is no local tissue or cell-type metadata.
Three routes to it were measured rather than assumed.

| route | cost | coverage |
| --- | --- | --- |
| Download the gene-level HDF5 files | 62.3 GB human + 50.7 GB mouse | 100%, release-matched |
| Partial HDF5 read over HTTP range requests (fsspec + h5py) | ~5 min and ~272 MB **per field**, hours for six fields across two species; the `meta/samples` group enumerates in ~18 s | 100%, release-matched |
| Maayan Lab sigpy JSON API (**chosen**) | **33.7 s, 39 requests, ~216 MB** | 99.911% |

### The endpoint

```
POST https://maayanlab.cloud/sigpy/meta/samplemeta
body: {"species": "human" | "mouse", "samples": ["GSM...", ...]}
->    {"GSM...": {"series": ..., "title": ..., "source": ..., "characteristics": ...}, ...}
```

Batch size 25,000 accessions per request sustains 24k-42k samples/s with no rate limiting observed, and the whole corpus fits in 39 requests.
The response object silently omits misses, so `fetch_archs4_meta.py` reindexes onto the fixed global order by accession rather than assembling positionally.
The endpoint answers HTTP 200 with an empty object when the payload key is wrong (`gsm_ids` instead of `samples`), which would write a fully-empty table and grey the map, so the script aborts below a 50% hit rate.

### The output artifact (verified by reading it)

`cache/archs4_metadata.parquet`: 940,455 rows, 32.5 MB, columns `global_index`, `geo_accession`, `series_id`, `title`, `source_name`, `characteristics`, `tissue`.
939,616 rows resolved to a GEO series, which is **99.911%**; per species, human 509,949/510,709 = **99.851%** and mouse 429,667/429,746 = **99.982%**.
The human figure was confirmed independently by a partial HDF5 read of `meta/samples/geo_accession` off the 62 GB remote file: both methods return exactly 509,949 human matches.
Distinct GEO series: **51,284**.

### The truth about the 839 misses

They are **not** GEO withdrawals, and the obvious explanation is wrong.
They are present with full metadata in the release-matched v2.5 metadata files and absent from the newer, larger v2.latest that this API serves.
ARCHS4 releases are therefore not append-only: a rebuild can drop samples.
Those 839 points (0.089% of the corpus) get tissue `Unknown` and an empty series rather than being dropped or guessed at.

### The 100%-coverage alternative, deliberately not taken

ARCHS4 publishes **metadata-only** HDF5 files under *versioned* names:

| file | size |
| --- | --- |
| `human_meta_v2.5.h5` | 311.8 MB |
| `mouse_meta_v2.5.h5` | 350.9 MB |

The unversioned "latest" spellings of those names return HTTP 403, which is why they are easy to miss.
Reading the versioned pair gives exactly 100.000% coverage against this corpus and is release-matched, at 663 MB and roughly 8.5 minutes against 216 MB and 35 seconds, plus an h5py dependency the serving app does not otherwise need.
For a color-by, 0.089% of points reading "Unknown" is not worth 15x the build time.
Switch to the versioned files if this ever needs to be a build **gate** rather than a color, and assert 100%.

### Ordering

`fetch_archs4_meta.py` joins onto `cache/archs4_geo.parquet` and `cache/points_meta.parquet`, so it must run **after** `build_projections.py`, and it aborts with that instruction if either file is missing.
It is still optional: without it the tissue color-by is shown disabled with the command that enables it attached, and ARCHS4 colors by species.

## 9. Visual theme tokens (from `assets/00-tokens.css` and `manifold/theme.py`)

Light scientific-instrument chrome, reused verbatim from Bridge RNA:
`--bg-canvas #eef2f7`, `--bg-panel #ffffff`, `--bg-panel-raised #f4f7fb`, `--bg-inset #f5f8fc`.
`--text-primary #1a2432`, `--text-secondary #5a6b7e`, `--text-muted #8a99ac`.
`--accent #2b7fff`, `--accent-hover #1f6ff0`, `--accent-teal #0bab9f`, `--accent-warm #d9791b`.
`--header-bg #14294a`, `--header-fg #f3f7fc`, `--header-line #22c7bd`.
`--status-good #1f9d57`, `--status-error #d64545`, `--status-warn #b7791f`.

The one deliberate departure is a dark navy plot canvas for WebGL glyph contrast: `PLOT_BG #0e1d34`, `PLOT_GRID #1c3252`, `PLOT_AXIS #2a456b`, `PLOT_TEXT #c7d6ea`.

Categorical palette, validated with the dataviz skill's checker against `#0e1d34`: all eleven hues sit in the OKLCH L 0.48-0.67 band, clear the chroma floor, pass the adjacent-pair CVD floor (worst ΔE 8.4) and the normal-vision floor (worst ΔE 15.4), and reach at least 3:1 contrast on the surface.
Slot order is the CVD-safety mechanism and must not be shuffled without re-validating: `#3987e5`, `#d95926`, `#199e70`, `#c98500`, `#d55181`, `#008300`, `#9085e9`, `#e66767`, `#1b95a3`, `#7d9a3c`, `#d84f96`.

Two greys sit at the neutral end, because "Other" and "Unknown" are different answers: `OTHER_COLOR #7f8ea3` and the dimmer `UNKNOWN_COLOR #56657a`, so absence recedes furthest.
`ARCHS4_CONTEXT #43597c` is deliberately close to the plot background, and is not in the categorical palette, so the faint context cloud drawn when the selected field does not describe ARCHS4 reads as scenery and offers a reader nothing to look up in the legend.
It used to apply only where there was no density raster to carry the shape; with the raster gone it applies in every view.
`OSDR_HIGHLIGHT #f2a03d` is the single warm color the OSDR overlay takes when the field describes ARCHS4 only.
The OSDR overlay is always a white-ringed `diamond`, so the 2,108 spaceflight samples stay findable among 940k circles.

Figure `dragmode` is **`pan`**, not `lasso` or `select`, and the graph config removes both `select2d` and `lasso2d` from the modebar.
There is no selection feature, and a drag that draws a marquee doing nothing would be a promise the app does not keep.
(For the record: an earlier config removed `select2d` only, so the lasso button was in fact still on the modebar after the rest of the feature had been designed away.)

## 10. The shared tissue vocabulary (`manifold/tissue.py`, measured 2026-07-21)

OSDR and ARCHS4 name tissues in disjoint registers, so "Tissue" would otherwise be two color-bys wearing one name, each leaving the other corpus grey.
Both are folded onto one canonical bucket list, which is what lets a single field paint the whole map.

The mapping is 40 ordered keyword rules producing 37 distinct buckets plus `Other` and `Unknown` (39 in `tissue.BUCKETS`).
First match wins.
Patterns are matched against a lowercased, whitespace-collapsed string with a leading `\b` word boundary; a `~` prefix drops that boundary for the morphemes GEO glues onto a stem.
`Unknown` (nothing was recorded) and `Other` (something was recorded that could not be placed) are kept distinct on purpose, and weak results are ranked (`Cell line`/`Reference RNA` 3, `Cultured cells` 2, `Other` 1, `Unknown` 0) so an early unplaceable field cannot pin the answer to `Other` and block a later field that did identify the sample.

Ordering and word boundaries are load-bearing, and each of these was a real defect a test now pins:

- `bone marrow` must be tested before `bone`.
- `\brenal` must not fire inside `adrenal`, which is what the leading word boundary is for.
- `cortex` is a kidney and adrenal word as well as a brain word, so `adrenal cortex` and `renal cortex` run ahead of Brain / CNS.
- `smooth muscle` is vasculature and must not be claimed by the bare `muscle` stem, so it is tested first.
- `~sarcoma` needs the boundary dropped or `osteosarcoma` misses.

### Measured against the real corpus

OSDR: 48 distinct raw values over 2,108 samples.
**Zero of the 48 fall to `Other` or `Unknown`.**
They land in 17 buckets: Liver 392, Kidney 264, Skeletal muscle 229, Skin 219, Brain / CNS 159, Thymus 149, Spleen 131, Intestine 93, Heart 89, Lung 86, Eye 72, Adrenal gland 56, Adipose 48, Breast / mammary 43, Bone / cartilage 36, Bone marrow 24, Cultured cells 18 (the only non-anatomical bucket used, and the counts sum to exactly 2,108).

ARCHS4 has no curated tissue column at all; the signal is in GEO free text.
Measured over `cache/archs4_metadata.parquet`: 90,449 distinct lowercased `source_name` values, and 433,961 samples carrying a `tissue:` key inside `characteristics` with 8,886 distinct lowercased values.
`derive_tissue` reads eleven characteristics keys best-evidence-first (`tissue`, `organ`, `organ part`, `source tissue`, `tissue type`, `tissue region`, `anatomical site`, `body site`, `cell type`, `celltype`, `cell line`) and falls back to `source_name`; `cell type` is deliberately near the end because it often carries a cell-line name where `tissue` would have carried an organ.

**851,881 of 940,455 ARCHS4 samples - 90.6% - land in a bucket other than `Other` or `Unknown`.**
Top buckets over ARCHS4:

| bucket | n | % of ARCHS4 |
| --- | --- | --- |
| Blood / immune | 155,761 | 16.6% |
| Brain / CNS | 103,182 | 11.0% |
| Other | 87,692 | 9.3% |
| Embryo / stem cell | 67,408 | 7.2% |
| Liver | 55,242 | 5.9% |
| Tumor / cancer | 53,117 | 5.6% |
| Lung | 40,710 | 4.3% |
| Intestine | 35,318 | 3.8% |
| Bone marrow | 34,409 | 3.7% |
| Cultured cells | 32,042 | 3.4% |
| Breast / mammary | 31,499 | 3.3% |

The `Tissue` color-by therefore covers **942,563 of 942,563 points, 100%**.

Independently validated as real biology rather than batch structure, measured 2026-07-21 during the redesign: 25-NN label purity **0.8142** against a permuted null of **0.0501**, and it survives both a batch control and a depth control at **0.7058**.

## 11. Rejected color-by candidates, with the measurements that rejected them

Each of these was built or tested against the real corpus and then cut.
The evaluation scripts were not kept, so this record is the record; it exists so the next person does not re-derive it.

**(a) Cosine similarity to an OSDR reference.**
Four variants were computed: mean centroid, flight centroid, ground centroid, and the flight-minus-ground "spaceflight-likeness" axis.
The four are one field wearing four names, with pairwise r between 0.996 and 1.000.
The spaceflight-likeness axis correlates **r = -0.990 with PC1** and **r = -0.779 with the raw L2 norm**: PC1 is a transcriptome-concentration axis (see above), so the candidate measured concentration and called it resemblance to spaceflight.
1 in 10 random flight/ground relabelings of the same sample sizes beat the real axis on spatial structure, and under a within-study permutation 46.5% did.

**(b) kNN tissue-label transfer from OSDR to ARCHS4.**
Median best-match cosine is 0.964 with 100% of points above 0.7, so no confidence threshold discriminates anything.
The winning OSDR sample beats the runner-up by a **median of 0.00089 cosine**, so the winner is essentially arbitrary.
54% of the ARCHS4 targets are human samples that would receive mouse tissue labels.

**(c) Unsupervised k-means cluster id (k=24).**
Built, run on the real corpus, measured, and then deleted along with its precompute stage and artifact.
**81.9% of the cluster label is recoverable from the 2-D UMAP coordinates alone** (15-NN over a 120k sample, against a 12.4% majority-class baseline), so coloring by it mostly redraws the shape already on screen.
A structure-free 24-cell Voronoi null reproduced its spatial coherence to within 1.5 points of modal agreement.
It is arbitrary (seed-to-seed ARI ~0.45), 81% species-pure, and explains 80.7% of the raw-L2-norm variance.
Numbering an arbitrary partition "Cluster 1..24" on a scientific instrument invites exactly the over-reading the rest of the design prevents.
(Those stale `cluster_*` keys are gone from `cache/projection_stats.json` too: the 2026-07-22 rebuild rewrote the file from scratch.)

**(d) Local UMAP density. Rejected on an argument that has since expired.**
The rejection was that the density raster is already rendered underneath every 2-D view, so a density color-by would encode the same quantity twice.
That raster no longer exists, so this candidate is open again rather than settled.
Anyone revisiting it should note that with every point now drawn live, glyph crowding *is* a density readout, and should hold the candidate to the methodological note below rather than treating this entry as a standing verdict.

**(e) PC1-3 as color-bys.** Free, since they are already in `coords_pca3.parquet`, but redundant with the axes on screen.

**(f) GEO series (GSE) as a color-by.** 51,284 distinct series, so a Top-11 legend would color about 3% of the map and dump the rest in "Other": a grey map by another route.
It is also a pure batch label (333x lift).
It is kept in the parquet for provenance and is not offered as a color.

### Methodological note for whoever evaluates the next candidate

A between-bin over total variance ratio (spatial eta-squared) is **not** sufficient evidence that a color-by shows real structure.
30 arbitrary random directions in 512-d score eta-squared **0.874 +/- 0.025** on this UMAP, because the UMAP was fit on those same vectors.
Every candidate in the 0.89-0.94 band is therefore indistinguishable from an arbitrary projection.
Species (0.985) is the only one that clearly clears the band.
Judge a candidate against a structure-free null **of the same form**, and check whether it is recoverable from the coordinates or from transcriptome concentration before believing it.

## 12. Artifact inventory and test suite

### What the build produces (sizes measured on disk 2026-07-21)

| artifact | size | written by | read by the app? |
| --- | --- | --- | --- |
| `points_meta.parquet` | 4.36 MB | build_projections | yes, first (`data.counts()` during layout) |
| `osdr_metadata.parquet` | 0.027 MB | embed_osdr | yes |
| `coords_pca2.parquet` | 8.78 MB | build_projections | yes |
| `coords_pca3.parquet` | 13.17 MB | build_projections | yes |
| `coords_umap2.parquet` | 8.79 MB | build_projections | yes |
| `coords_umap3.parquet` | 13.18 MB | build_projections | yes |
| `projection_stats.json` | 14.4 KB | build_projections | **no** - a build record the validator reads, nothing the app opens |
| `archs4_geo.parquet` | 4.63 MB | build_projections | no, join key for the metadata fetch |
| `archs4_metadata.parquet` | 32.51 MB | fetch_archs4_meta | yes, optional |
| `osdr_sample_embeddings.float32.npy` | 4.32 MB | embed_osdr | no, input to build_projections |
| `osdr_expression.float32.npy` | 127.87 MB | embed_osdr | no, resume intermediate for the multi-hour job |
| `osdr_expression_meta.parquet` | 0.097 MB | embed_osdr | no |

Total live cache: **217.8 MB**: 80.8 MB is what the serving app actually opens, 132.3 MB is embedding intermediates that exist so a re-embed never re-reads the counts CSVs, and 4.6 MB is the accession sidecar the metadata fetch joins onto.
`projection_stats.json` grew from 2.3 KB to 14.4 KB because it now records all 512 explained-variance ratios rather than the leading 50.

Three artifacts are **no longer produced**:

| removed artifact | size | why it is gone |
| --- | --- | --- |
| `joint_cosine.hnsw` | 2,070.4 MB | the ANN index existed only for the lasso's kNN-purity statistic and for the mixing check, which now computes exact neighbours in 10.3 s |
| `population_moments.npz` | 4.2 MB | the exact per-corpus mean and covariance existed only for the lasso's analytic null |
| `density/pca2.png`, `density/umap2.png` | 0.86 MB + 0.61 MB | the underlay they fed is gone (2026-07-22): every point is now drawn as a real glyph, measured at 0.15 s to serialize and 11.3 MB on the wire for all 942,563 |

`cache/` measured 2.1 GB before the deletion and 214 MB after it, and both the suite and `validate_artifacts.py` pass without them.
Neither was source data: both were derived from embeddings that are still intact, so nothing was lost that cannot be recomputed.
The two irreplaceable inputs are the ARCHS4 memmap in the Bridge RNA repo (918.4 MB, never written by this project) and `cache/osdr_sample_embeddings.float32.npy` (4.1 MB), which is the output of the 11.3-hour embedding job and is the one file here that is genuinely expensive to recreate.
The functions that built the two deleted artifacts, `build_hnsw` and `build_population_moments`, are recoverable from commit `3840ab3` if fast approximate kNN is ever wanted for an experiment; a rebuild of the index was a measured 52 s.

`manifold/preflight.APP_REQUIRED` lists exactly three artifacts, in the order they are read: `points_meta.parquet`, `osdr_metadata.parquet`, `coords_pca2.parquet`.
It previously demanded the ARCHS4 memmap, `sample_locations.parquet` and the OSDR embeddings, none of which the app opens, while omitting `points_meta.parquet`, which `layout.control_rail()` reads first - so a missing identity table passed preflight and crashed during startup.

### Tests

**160 tests, all passing, in about 1.0 s** (`/Users/josh/Bridge-RNA/.venv/bin/python -m pytest tests/ -q`), measured 2026-07-22 by running the suite.

| file | tests |
| --- | --- |
| `tests/test_tissue.py` | 46 |
| `tests/test_render.py` | 34 |
| `tests/test_app.py` | 27 |
| `tests/test_data.py` | 22 |
| `tests/test_colorby.py` | 18 |
| `tests/test_projections.py` | 13 |

The suite was 103 tests in 4.54 s two sessions ago, and 144 in 0.55 s before this one.
The runtime fell by 88% mostly because the fixture no longer builds an approximate-nearest-neighbour index, which was 43% of the old wall clock; it rose again to ~1.0 s because `test_projections.py` runs several 512-dimensional eigendecompositions.

`tests/test_projections.py` is new and is the only file that imports from `precompute/`.
It exists because the build's central claim, that a streaming second-moment pass reproduces a full-corpus PCA exactly, is not the kind of thing a docstring can establish: it is scored against `sklearn.decomposition.PCA` on planted low-rank data, to a max explained-variance-ratio difference below 1e-9 and component agreement below 1e-6.
It also pins the sign convention across block sizes, that the recorded spectrum is all 512 components and sums to 1, and that the corpus streams in the fixed global order every other artifact is positionally joined on.

The whole suite runs against a synthetic corpus built into a temp directory by `tests/fixture_corpus.py` (4,000 ARCHS4 + 300 OSDR points around known latent cluster centers), with `BRIDGE_RNA_ROOT` and `MANIFOLD_CACHE_DIR` set at conftest import time.
It never touches the 963 MB memmap or the multi-hour embedding artifacts and runs on a machine that has neither.
The fixture writes a synthetic `archs4_metadata.parquet` whose source strings are in ARCHS4's free-text register (`"liver"`, `"whole blood"`, `"Brain cortex"`, `"HeLa"`, `"left kidney"`, `"skeletal muscle biopsy"`) and maps them through the **real** canonicalizer, so the tests exercise the mapping rather than assuming it.
`tests/conftest.py` provides a `without_archs4_metadata` fixture that points `ARCHS4_METADATA_PARQUET` at a non-existent file and clears the loader caches on both sides, because the degraded state is what a fresh clone starts in and needs real coverage.
`render._colour_plan` is cleared there alongside the two data loaders: it memoizes a label array derived from the metadata, and a test proved that leaving it warm let Tissue keep colouring 940,455 ARCHS4 points from a join that no longer existed.

`tests/e2e_check.py` sits outside that suite and outside pytest's collection, because it needs the opposite of a hermetic fixture: it boots the real Dash app against the real `cache/` and drives a real browser.

```bash
/Users/josh/Bridge-RNA/.venv/bin/python tests/e2e_check.py          # about a minute
/Users/josh/Bridge-RNA/.venv/bin/python tests/e2e_check.py --headed # watch it
```

It exists because a green pytest run says nothing about whether 942,563 WebGL glyphs reach a browser.
Measured by it on 2026-07-22: first interactive frame **1.3 s** with all **942,563** glyphs, budget switches re-rendering in 0.1 to 0.3 s and drawing exactly the counts they advertise (102,108 / 502,108 / 942,563), zero console errors.
One trap it encodes: under plotly 6 the coordinates arrive as base64 typed-array specs, so `gd.data[i].x` has no `.length` and a naive glyph count silently yields `NaN`, which is indistinguishable from an empty plot; read `gd._fullData`, where the decoded `Float32Array` lives.

### Commands, in build order

```bash
/Users/josh/Bridge-RNA/.venv/bin/python precompute/embed_osdr.py          # OSDR embeddings, gene-digest gated. Hours.
/Users/josh/Bridge-RNA/.venv/bin/python precompute/build_projections.py   # full-corpus PCA + UMAP coords.
/Users/josh/Bridge-RNA/.venv/bin/python precompute/fetch_archs4_meta.py   # ARCHS4 GEO metadata. ~35 s, needs network.
/Users/josh/Bridge-RNA/.venv/bin/python precompute/validate_artifacts.py --mixing --quality
/Users/josh/Bridge-RNA/.venv/bin/python app.py                            # http://127.0.0.1:8050/map

# Score a candidate projection against the shipped one, on the same sample:
/Users/josh/Bridge-RNA/.venv/bin/python precompute/validate_artifacts.py --quality --compare DIR
```

The metadata fetch runs third because it joins onto artifacts `build_projections.py` writes.
`BRIDGE_RNA_ROOT` is needed for the first two steps and for `--mixing` and `--quality`, and not for running the app.

Flags that appear in older prose and no longer exist: `--skip-hnsw`, `--density-only`, `--pca-fit-sample`, `--umap-fit-sample`, `--pca-components`.
The current `build_projections.py` takes `--umap-neighbors`, `--pca-report`, `--batch`, `--knn-jobs`, `--seed`, `--archs4-limit`, `--skip-umap`, `--densmap`, and `--dens-lambda`.
