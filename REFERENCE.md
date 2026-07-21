# Bridge Manifold - Verified Reference

This is the ground-truth appendix for `IMPLEMENTATION.md`.
Every fact here was verified directly against the checkpoint, the memmap, the parquet, and the data files on 2026-07-20, not taken from documentation or from an agent's summary.
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

## 4. Measured vector statistics and timings (on Josh's Mac)

Vector L2 norms (25k random sample): mean 10.57, std 2.26, min 6.75, max 25.47; no NaN or Inf.
The 4x magnitude spread is why L2 normalization is required before reduction.

PCA variance (25k sample, before normalization): PC1 = 57.8% of variance; cumulative PC2 63.2%, PC5 71.3%, PC10 78.6%, PC20 87.2%, PC50 96.4%.
PCA-50 fit on 25k x 512 = 11.7 s; projecting 200k to 2D via fitted components = 1.44 s, extrapolating to ~6.7 s for all 940k.

UMAP (on PCA-50 input, L2-normalized): fit on 40k = 171 s; `.transform()` 60k = 21.5 s.
Landmark hybrid (fit ~200k subsample, transform the rest) is the tractable path.

### Full-corpus build, actually measured (2026-07-21)

The 30-to-90-minute estimate above was wrong by an order of magnitude.
The real `build_projections.py` run over 942,563 points (940,455 ARCHS4 + 2,108 OSDR) took **5 min 47 s end to end** on defaults.

| stage | measured |
| --- | --- |
| population moments, exact mean + covariance per corpus | 4 s |
| IncrementalPCA-50 fit (60k ARCHS4 + 2,108 OSDR) | 1 s |
| PCA transform, all 942,563 points | 1 s |
| UMAP-2d landmark fit, 122,108 points | 71 s |
| UMAP-2d transform, remaining 820,455 | 71 s |
| UMAP-3d fit + transform | 143 s |
| hnswlib cosine index, 942,563 points, M=16 ef=120 | 52 s |
| **total** | **347 s** |

Peak RSS ~2.5 GB against 17 GB. The index is 2,070,363,764 B on disk, matching the ~2 GB estimate.

UMAP emits `n_jobs value 1 overridden to 1 by setting random_state`: seeding forces a single-threaded layout.
At 71 s per fit that is not worth trading determinism for, so `random_state=42` stays.

**PCA after L2 normalization: PC1 = 40.9%, cumulative over 50 PCs = 95.1%.**
Contrast with the 57.8% above, which is the *pre-normalization* depth axis.
This pair is the objective test for invariant 2 - see `precompute/validate_artifacts.py`.

### Cross-corpus separation, measured (2026-07-21)

Of the 50 nearest neighbours of each OSDR sample in the 512-d cosine space: 34.2% same-study OSDR, 22.9% cross-study OSDR, 42.9% ARCHS4.
Against a per-sample chance model that is 5,227x for same-study (replicate structure, expected and biological) and **105x for cross-study** (a corpus-level effect).

Controlling for tissue, the dominant axis of variation in bulk expression: OSDR neighbour pairs sharing **neither study nor tissue** occur at 11.49% against 0.211% expected, **54x over chance**.
Biology does not make liver cluster with brain, so this is a technical batch effect from fp32/CPU versus bf16/CUDA inference and preprocessing differences, not spaceflight signal.

Corroborating asymmetry: ARCHS4 queries find OSDR neighbours only 0.0300% of the time against 0.224% chance, a 7.5x *depletion*.
OSDR is a small dense island - dominant in its own neighbourhoods, nearly invisible from ARCHS4's.
Cosine gap: OSDR to its OSDR neighbours 0.9979, to its ARCHS4 neighbours 0.9966, against an ARCHS4 self-baseline of 0.9971.

## 5. Environment (re-verified 2026-07-20 at build time)

venv at `/Users/josh/Bridge-RNA/.venv`.
The versions recorded during design had drifted by the time the build ran; these are the ones the code was written and tested against.

| Package | Version | Notes |
| --- | --- | --- |
| numpy | 2.4.6 | |
| pandas | 3.0.3 | major release - see the behaviour note below |
| pyarrow | 20.0.0 | |
| scikit-learn | 1.9.0 | |
| umap-learn | 0.5.12 | |
| plotly | 6.8.0 | serializes numpy arrays as base64 typed arrays |
| dash | 4.4.0 | rewritten Dropdown/RadioItems, new `--Dash-*` design tokens |
| torch | 2.12.1 | MPS available, CUDA absent |
| scipy | 1.16.0 | |
| pillow | 12.2.0 | |
| hnswlib | 0.8.0 | |
| pytest | 9.1.1 | dev |
| playwright | 1.61.0 | dev, browser checks against the running app |

Absent: `datashader` (not needed - the density raster is a numpy 2D histogram), `archs4py`, `h5py`, `cuml`, `pacmap`, `openTSNE`.

Three library behaviours the code actively depends on, each of which caused a real defect during the build:

- **pandas 3.0**: `Series.astype(str)` no longer stringifies missing values; they stay NA. A later `.replace({"nan": "Unknown"})` therefore never matches them, and the NA survives into legends and enrichment tests as a phantom category. Every categorical path calls `fillna("Unknown")` explicitly.
- **plotly 6.x**: numpy arrays are serialized as base64 typed-array specs. Dash builds its `selectedData` payload by indexing the *user* data, which yields `undefined` for an encoded array, so numpy `customdata` arrives at the server as nothing at all and every lasso reads as empty. `customdata` is passed as a plain Python list.
- **dash 4.x**: `dcc.Dropdown` is a radix popover (not react-select), so all `.Select-*` CSS is dead; `labelClassName` on `RadioItems`/`Checklist` lands on an inner text span rather than the label. Components are styled through Dash's own structural classes and its `--Dash-*` tokens.

## 5b. Measured at build time

- OSDR encode, CPU fp32, 10 threads: **~6.5 s/sample** (B=1 and B=4 within noise of each other). 2,108 samples is therefore a multi-hour job.
- MPS is not an option for this model: `F.scaled_dot_product_attention` has no fused kernel for a 15,165-token sequence on Metal, so it materializes the full attention matrix and fails on a 6.85 GiB allocation. Chunking the attention over the query axis makes it *run* and is numerically equivalent to CPU (cosine 0.99999988, max abs delta 4.2e-7), but it was not faster in measurement, so CPU fp32 remains both the fidelity baseline and the fastest available path.
- Preprocessing equivalence: the batched OSDR path reproduces Bridge RNA's `load_random_osdr_sample_vector` **bit-for-bit** (max abs diff 0.0) across samples from three different studies.

## 6. Reusable Bridge RNA interfaces

Line numbers are from the Bridge RNA repository as of 2026-07-20.

From `demo_osdr_top5.py`:

- `load_random_osdr_sample_vector(args) -> (np.ndarray[num_genes], sample_id, pd.Series)` at 327-406.
  The full OSDR preprocessing recipe (read counts, ortholog map, collapse, reindex to canonical, TPM, log1p).
- `build_model_and_query_embedding(args, query_vec) -> np.ndarray[512]` at 409-436.
  Loads the checkpoint and runs `encode`; the single-sample driver to generalize to batches.
- `build_mouse_to_human_maps(orthologs_path, mouse_exon_lengths_path) -> (dict, dict)` at 233-250.
- `normalize_counts_to_tpm_single(counts, exon_len_by_human_gene) -> pd.Series` at 253-267.
- `resolve_canonical_genes(explicit) -> Path` at 270-324, with candidates `data/archs4/train_orthologs/canonical_genes.csv` then `data/ensembl/canonical_genes.inferred.csv`.

From `generate_archs4_embeddings.py`:

- `class ExpressionPerformer.__init__(num_genes, hidden_dim, n_heads, n_layers, ffn_dim, ree_base, mask_token_id, feature_type, compute_type, include_species_embedding, num_species)` at 106-156.
- `ExpressionPerformer.encode(x, species_ids=None, normalize=False) -> Tensor[batch, 512]` at 187-193, mean-pools hidden states over the gene axis; accepts a full batch `x` of shape `[batch, num_genes]`.
- `ExpressionPerformer._encode_hidden(x, species_ids=None) -> Tensor[batch, num_genes, 512]` at 158-180; expression values enter only through the rotary expression embedding of `x`.
- `CANONICAL_GENES_SHA256` at 40 and `canonical_gene_order_digest(genes)` at 43-51.
- `_strip_module_prefix(state_dict)` at 258, removes a leading `module.` from keys.

From `app_osdr_dash.py`:

- `_load_archs4_index() -> (np.memmap, pd.DataFrame, int)` at ~696-723, cached; reads the manifest, opens the float16 memmap `[n, d]`, loads `sample_locations.parquet`.
- `_topk_cosine_from_memmap(...)` at ~775-797, normalizes and does cosine top-k over the memmap.
- `_find_precomputed_query_embedding_file()` at 692, and `PRECOMPUTED_QUERY_EMBEDDING_CANDIDATES` (no such file currently exists on disk).
- `preflight_retrieval_requirements(...)` and `_is_lfs_pointer(...)` LFS guards near 480-510.
- Ollama AI-summary integration near the top constants (`OLLAMA_BASE_URL=http://127.0.0.1:11434`, `OLLAMA_MODEL=gemma3:4b`), reusable if we later add narrative summaries of a selection.

## 7. OSDR color-by columns

File: `data/osdr/metadata/selected_sample_metadata.tsv`, 2,896 rows x 44 columns.

**Corpus size (corrected).** All 2,896 rows are *Mus musculus* and all have a counts path, but only **2,163** carry a non-empty spaceflight factor, which is the filter Bridge RNA's retrieval uses and the one Josh chose to match. Of those, **2,108** produce an expression vector; the other 55 name a sample column their counts matrix does not contain, and `embed_osdr.py` reports them rather than dropping them silently. So: 2,896 in the TSV -> 2,163 eligible -> 2,108 embedded. Earlier drafts of these documents used 2,896 as the corpus size, which was wrong.

The spaceflight factor has nine raw spellings covering seven distinct arms:

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

Casing variants are folded onto the most frequent spelling at precompute time. The arms themselves are *not* merged: basal animals were sacrificed at experiment start and vivarium animals never entered flight hardware, so collapsing them into one "Ground Control" bucket would erase real experimental structure. The app offers both `spaceflight` (the arm) and `flight_status` (the derived Flight-vs-Ground contrast).
The highest-value color-by fields (exact names may carry the `study.factor value.` / `study.characteristics.` / `study.parameter value.` prefixes seen in the raw header):

- spaceflight status (Flight vs Ground Control) - the primary scientific variable.
- material type / tissue (~61 values).
- strain (~9), sex (3), genotype (~3).
- habitat (~11), duration (~28), diet (~8).
- study accession `id.accession` (~94 studies) - also the batch axis for the confound guard.

High-cardinality fields (94 studies, 61 tissues) exceed Plotly's usable native legend (~12 rows) and need the custom searchable legend with a Top-11-plus-Other trace model.

## 8. ARCHS4 metadata (fetched for v1)

The local ARCHS4 artifacts carry only `geo_accession` and `species_id`; there is no local tissue or cell-type metadata.
Decision (Josh, 2026-07-20): fetch it for v1 so ARCHS4 colors by tissue.
The reader already exists: `fetch_archs4_metadata(geo_accessions, human_h5, mouse_h5) -> pd.DataFrame` at `demo_osdr_top5.py:463`, which uses `archs4py` to read per-GSM fields (tissue, source, title, series) from the ARCHS4 gene-level HDF5 files.
Required files: `data/archs4/human_gene_v2.5.h5` and `data/archs4/mouse_gene_v2.5.h5`, downloaded from https://archs4.org/download (tens of GB each), not currently present locally.
`precompute/fetch_archs4_meta.py` extracts metadata for all 940,455 accessions once into `cache/archs4_metadata.parquet` keyed by `geo_accession`.
Until that parquet exists, ARCHS4 colors by species only and renders as neutral grey under any OSDR-specific color-by.

## 9. Visual theme tokens (from `assets/style.css`)

Light scientific-instrument theme:
`--bg-canvas #eef2f7`, `--bg-panel #ffffff`, `--bg-panel-raised #f4f7fb`, `--bg-inset #f5f8fc`.
`--text-primary #1a2432`, `--text-secondary #5a6b7e`, `--text-muted #8a99ac`.
`--accent #2b7fff`, `--accent-hover #1f6ff0`, `--accent-teal #0bab9f`, `--accent-warm #d9791b`.
`--header-bg #14294a`, `--header-fg #f3f7fc`, `--header-line #22c7bd`.
`--status-good #1f9d57`, `--status-error #d64545`, `--status-warn #b7791f`.
Bridge Manifold reuses these tokens verbatim for its chrome and uses a dark navy plot canvas for glyph contrast.
