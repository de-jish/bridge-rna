# Bridge Manifold - Progress

Living status log.
Update after each meaningful change so another session can resume without losing context.

## Current status: 2026-07-21 - the real cache is built

**The full offline pipeline has run to completion on real data.** `cache/` now holds the real 942,563-point manifold: 940,455 ARCHS4 (510,709 human, 429,746 mouse) plus 2,108 OSDR.

- `embed_osdr.py` finished 2026-07-21 08:45:51 after ~11.3 h, all 2,108 samples, gene-digest gate passed.
  Realized rate was ~10 s/sample in fast stretches, degrading to ~49 s/sample between 05:44 and 08:26 under machine contention, so the original ~6.5 s/sample estimate was optimistic.
- `build_projections.py` ran 08:45:57 to 08:51:44, **5 min 47 s**, rc=0. The 30-90 min estimate in `REFERENCE.md` was wrong by an order of magnitude; measured per-stage timings are now recorded there.
- `precompute/validate_artifacts.py --mixing` passes all structural and invariant checks, with one substantive warning: the cross-corpus batch effect (see Notes and risks).

The application itself was already written, tested, and running: 101 tests pass against a hermetic synthetic corpus, and the app has been driven end to end in a real browser (controls, color-bys, 2D/3D, lasso -> readout).
It has **not** yet been launched against this real 940k cache - that is the next step.

### What is done

- **Phase 0 scaffold**: package skeleton, path configuration with `BRIDGE_RNA_ROOT` / `MANIFOLD_CACHE_DIR` overrides, LFS-pointer preflight.
- **Phase 1 OSDR embeddings**: `embed_osdr.py`, gene-digest gated, resumable, with a cached expression stage. Preprocessing proven bit-for-bit identical to Bridge RNA's single-sample path. **Currently running.**
- **Phase 2/4 projections**: `build_projections.py` writes PCA-2/3, landmark UMAP-2/3, the hnswlib index, density rasters, and exact population moments. Not yet run (waits on Phase 1).
- **Phase 3 interactive plot**: layered renderer (density underlay, stratified ARCHS4 cloud, OSDR overlay), all color-bys, layer toggles, point budget, viewport level-of-detail.
- **Phase 5 lasso coherence**: all five parts of the readout, with the null rebuilt analytically (see below).
- **Phase 6 polish**: searchable legend (now actually wired), theme-matched Dash 4 controls, hover cards, 3D, honest empty and degraded states.
- **Tests**: `tests/` with a synthetic corpus built from known latent clusters, so coherence has real ground truth to be checked against.
- **ARCHS4 tissue join**: `fetch_archs4_meta.py` written; cannot be run here (the gene HDF5 files are a tens-of-GB download and are absent, and neither `h5py` nor `archs4py` is installed). The app degrades to species-only coloring and says so on the plot.

### Corrections to earlier assumptions

- **Corpus size.** The OSDR corpus is **2,163 eligible / 2,108 embedded**, not 2,896. 2,896 is the unfiltered TSV row count; 733 rows have no spaceflight factor and are excluded by the Bridge RNA filter Josh chose to match, and 55 more name a counts column that does not exist. All docs corrected.
- **Environment.** The versions in `REFERENCE.md` had drifted: pandas 3.0, dash 4.4, plotly 6.8, numpy 2.4, torch 2.12. Three of those releases changed behaviour the code depends on (see `REFERENCE.md` section 5).
- **datashader is not used.** The density raster is a numpy 2D histogram plus Pillow. Fewer fragile dependencies, and it is trivially fast at this scale.
- **MPS is not viable** for this model, and chunking does not help. Measured and documented.

## Decisions log

- Statistics for the lasso readout are computed in the original 512-d cosine space, never from the 2D projection. Rationale: UMAP and PCA distances are distorted; a statistic off the pixels would be false. Enforced by a test that makes any coordinate access raise.
- **The coherence null is analytic, not a bootstrap.** The mean-cosine-to-centroid statistic equals `||mean(V)||` for unit vectors, so the null only needs the distribution of the sample mean, which is Gaussian with an exactly known mean and covariance under sampling without replacement. Rationale: the original with-replacement bootstrap over a fixed 20k background pool produced a z-offset growing as `sqrt(n / pool_size)`; at the real corpus ratio, genuinely random selections scored `|z| > 40` with a sign set by the pool's seed. Measured, not theorized.
- Population moments are computed exactly over the whole corpus, never from a subsample. Rationale: any sampling error in the mean becomes an unbounded z-bias, because the null's spread shrinks with selection size while that error does not.
- A category must clear 1.25x fold enrichment to be named a *driver* in the verdict. Rationale: at large |S| a 1.05x deviation reaches q < 1e-10 on sample size alone; reporting it as the explanation is exactly the false confidence the readout exists to prevent.
- The seven OSDR control arms stay distinct; the binary Flight-vs-Ground contrast is a separate derived field. Rationale: basal and vivarium controls are different experiments, and merging them erases real structure.
- L2-normalize before any reduction. Rationale: raw vectors carry a 4x magnitude spread that dominates PC1 (57.8%).
- UMAP is offline only, via landmark fit then transform. Rationale: a direct 940k fit is hours and risks memory blowup.
- Standalone app importing Bridge RNA functions, not edits to the 2,470-line retrieval app. Rationale: isolation without losing the shared instrument feel.
- Batch structure is made visible and guarded against, not silently corrected. The batch flag is only raised when there is coherence to attribute to it.
- OSDR embedded in fp32 on CPU. Rationale: fidelity baseline, and measurement showed no faster option exists on this machine.
- Dash components are themed by remapping Dash 4's own `--Dash-*` design tokens rather than by overriding each component's rules. Rationale: one mapping themes every current and future Dash component; per-component overrides are a specificity war that silently rots on upgrade.

## Decisions from Josh (2026-07-20)

1. ARCHS4 tissue coloring: FETCH NOW for v1. Script written; the HDF5 inputs are not present on this machine, so the feature ships behind a graceful degrade and a badge naming the script to run.
2. OSDR scope: MATCH the Bridge RNA filter (mouse + spaceflight factor). Honored - this is what yields 2,163 rather than 2,896.
3. Batch handling: EXPOSE AND GUARD only. No correction, not even as a toggle.
4. Environment: SHARE the Bridge RNA venv, adding `dash` and `hnswlib`. Done; `requirements.txt` records the verified versions.

## Defects found and fixed during the build

Found by an adversarial audit plus browser-driven testing; each was verified before fixing.

1. **(critical)** Coherence null was a with-replacement bootstrap over a frozen 20k pool - systematic z-bias growing with selection size. Rewritten analytically.
2. **(critical)** The lasso never produced a selection: plotly 6 serializes numpy `customdata` as base64, and Dash's event filter indexes the user data, so `customdata` arrived at the server as nothing. Now a plain list.
3. The plot occupied 450 px of an ~890 px pane - `dcc.Loading`'s wrapper divs broke the `height: 100%` chain.
4. Segmented controls had no selected state and the dropdown was entirely unstyled: Dash 4 rewrote both components' DOM and class names.
5. The readout kept showing statistics for a lasso that was no longer on screen after any control change.
6. The legend search box was inert - no callback read it.
7. The ARCHS4 background cloud showed a hover label despite `hoverinfo="skip"`, because a `hovertemplate` overrides it.
8. Cross-dataset and precision cautions were dropped whenever a selection read as incoherent.
9. The batch guard warned that coherence "may be batch-driven" for selections with no coherence at all.
10. pandas 3.0 leaves NA through `astype(str)`, so a phantom NA category reached legends and enrichment tests.
11. 55 samples were dropped silently during preprocessing; now reported.
12. The expression cache could never hit (it compared eligible keys against kept keys), and resume was keyed on row count alone.
13. The gene-digest gate was skipped on the expression-cache path; the digest is now part of the cache key.

## Next steps

1. ~~Wait for `embed_osdr.py`~~ **DONE** 2026-07-21 08:45:51, all 2,108 samples.
2. ~~Run `precompute/build_projections.py`~~ **DONE** 2026-07-21 08:51:44, rc=0 in 5 min 47 s.
3. ~~Validate against the Phase 2/4 criteria~~ **DONE**. PC1 = 40.9% against the 57.8% pre-normalization figure, so invariant 2 holds.
   Codified as `precompute/validate_artifacts.py`, which exits nonzero on failure so it can gate a rebuild instead of relying on eyeballing a scatter plot.
4. ~~Launch the app on the real corpus~~ **DONE** 2026-07-21. Driven headless end to end against the real 942,563-point cache: initial render, zoom with viewport LOD re-sampling, and a lasso to a populated readout. Zero console errors. Two defects found and fixed (below).
5. Decide how the measured batch effect should surface in the UI. The cross-dataset warning already exists and does fire correctly on mixed lassos, but it was written before the magnitude was known; 54x tissue-controlled enrichment may justify making it harder to ignore.
6. Re-run the browser checks at the 60k and 150k point budgets and measure frame rate; only the 100k default has been exercised so far.
7. Optional, if the ARCHS4 HDF5 files are ever downloaded: `pip install h5py`, run `fetch_archs4_meta.py`, and the tissue color-by appears by itself.
8. Optional: `precompute/embed_osdr.py --metadata-only` if the metadata harmonization ever changes; it rewrites the parquet without re-embedding.

## Defects found by running the real app (2026-07-21)

Both were invisible against the synthetic test corpus and only appeared at real scale and real distribution.

**1. The density ramp used 0.78% of its range.** `render_density` normalized `log1p(counts)` by the global max. Real occupancy is heavy-tailed - median occupied bin holds 2 points, max 638 - so dividing by the max crushed everything into the bottom of the scale. Only 0.78% of occupied bins cleared the 0.5 threshold where the navy-to-teal ramp turns teal, and alpha saturated at 0.4545, *before* that turn, so the densest cores were indistinguishable from merely-busy ones. Measured on the raster: 8 pixels total in the teal half.
Fixed by normalizing against the 99.5th percentile of occupied bins (`DENSITY_CLIP_PCT`) and ramping alpha across the same span with a visibility floor (`DENSITY_ALPHA_FLOOR`). Teal-half pixels went 8 -> 2,377; mean occupied colour lifted RGB (21,50,82) -> (26,71,108). Filament structure that was a flat wash now reads.
Added `build_projections.py --density-only` so ramp changes re-render from cached coordinates in seconds instead of repeating the 6-minute build.

**2. The verdict contradicted its own statistics.** `cohesive` was `(z >= 3 and p < 0.05) or knn_fold >= 2`, and the sentence printed a flat "Coherent" followed by whichever numbers happened to exist. A real 1,015-point skin selection produced **"Coherent (z=-1.8, p=0.963, kNN-purity 384.2x)"** - a coherence claim next to a z and p that say the selection is *looser* than a matched random draw.
The underlying logic is right: the two measures ask different questions, and a lasso over several tight but mutually distant groups is legitimately high on local purity and low on global tightness. The reporting was what failed. `cohesive_global` and `cohesive_local` are now tracked separately, and when only the local one fires the verdict says so and explains the shape: "Locally coherent (kNN-purity 384.2x), but looser overall than a matched random draw (z=-1.8, p=0.963): several close-knit groups sitting apart from each other rather than one cloud."
Pinned by `test_verdict_does_not_claim_coherence_next_to_contradicting_statistics`.

Checked and found **not** defective: the readout panel clips its cross-corpus warning at 1000px viewport height, but it is `overflow-y: auto` and genuinely scrollable, so nothing is unreachable.

## How the projection build was chained (2026-07-21, completed)

`build_projections.py` was queued behind the in-flight embed run rather than waited on by hand.
A detached watcher polled the embed PID, verified the run actually succeeded, and only then started the projection build with default parameters.
It fired correctly: embed exited 08:45:51, the gates passed, and the build launched 08:45:57 - a 6-second unattended handoff.

The watcher refused to launch unless all of these held, because `build_projections.py` joins OSDR metadata to embeddings *positionally* and a truncated embedding would silently mislabel every OSDR point rather than fail:

- the embed log contains a `[done] wrote` line (clean completion, not a crash or a kill),
- `osdr_sample_embeddings.float32.npy` and `osdr_metadata.parquet` both exist and are non-empty,
- their row counts agree, the embedding dim is 512, and the values are finite.

That gate is worth rebuilding if this is ever re-run, since the failure it guards against is silent rather than loud.

**Preflight verified before queuing** (2026-07-21):

- All inputs resolve and are real data, not Git LFS stubs: the ARCHS4 memmap is 963,025,920 bytes, exactly 940,455 x 512 x 2, matching `embedding_manifest.json`.
- Every import `build_projections.py` needs is installed: numpy 2.4.6, pandas 3.0.3, sklearn 1.9.0, umap 0.5.12, hnswlib, pyarrow 20.0.0, PIL 12.2.0.
- Disk 44 GiB free against a ~2 GB index; RAM 17 GB against a peak of roughly 2.5 GB.
- **Full-path smoke test passed** end to end in an isolated `MANIFOLD_CACHE_DIR` with synthetic OSDR embeddings and `--archs4-limit 4000`, exercising population moments, IncrementalPCA, PCA transform, density rasters, UMAP 2-d and 3-d, the hnswlib index, and every parquet write. Exit 0, all twelve artifacts produced.
  Running the real pipeline against a throwaway cache first is cheap insurance and is worth repeating before any long rebuild.

### Built artifacts in `cache/` (2026-07-21)

| file | size | contents |
| --- | --- | --- |
| `joint_cosine.hnsw` | 2.07 GB | cosine index over all 942,563 points |
| `coords_pca3.parquet` / `coords_umap3.parquet` | 13.2 MB each | 3-d coordinates |
| `coords_pca2.parquet` / `coords_umap2.parquet` | 8.8 MB each | 2-d coordinates |
| `archs4_geo.parquet` | 4.6 MB | GEO accessions for study context |
| `points_meta.parquet` | 4.4 MB | dataset / src_index / species_id identity table |
| `population_moments.npz` | 4.2 MB | exact per-corpus mean + covariance for the lasso null |
| `osdr_sample_embeddings.float32.npy` | 4.3 MB | the 2,108 x 512 OSDR embeddings |
| `osdr_metadata.parquet` | 27 KB | OSDR labels, joined positionally |
| `density/{pca2,umap2}.png` | 838 KB / 621 KB | density underlays |
| `projection_stats.json` | 1.8 KB | variance profile and raster extents |

`embed_osdr.py` cleaned up its own partial memmap and progress JSON on success, as designed.

## Notes and risks

- The hnswlib index holds raw float32 vectors: ~2 GB on disk and resident in the app. `--skip-hnsw` is supported and costs only the kNN-purity statistic.
- **The cross-corpus batch effect is real and now measured (2026-07-21).** Controlling for both study and tissue, OSDR samples that share neither still neighbour each other 54x above chance. Tissue is the dominant axis of bulk expression, so biology cannot explain it - this is the fp32/CPU versus bf16/CUDA precision and preprocessing difference. The app's cross-dataset warning is therefore load-bearing and must stay prominent; cross-corpus distances are not trustworthy at face value. Full numbers in `REFERENCE.md` section 4. Re-check with `python precompute/validate_artifacts.py --mixing`.
- UMAP quality at 940k via landmark fit-and-transform ran clean, but *visual* quality on the real map is still unreviewed.
- `tests/` never touches the real data, so the suite stays fast and runs on a machine with neither the memmap nor the checkpoint.
