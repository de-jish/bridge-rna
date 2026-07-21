# Bridge Manifold - Progress

Living status log.
Update after each meaningful change so another session can resume without losing context.

## Current status: 2026-07-20 (build session)

The application is written, tested, and running.
101 tests pass against a hermetic synthetic corpus, and the app has been driven end to end in a real browser (controls, color-bys, 2D/3D, lasso -> readout).

The one thing still outstanding is **data**: `precompute/embed_osdr.py` is mid-run.
It is a multi-hour CPU job (~6.5 s/sample x 2,108 samples). Until it finishes, the real `cache/` cannot be built.

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

1. **Wait for `embed_osdr.py`** to finish (~6 h from 21:20, monitor `cache/osdr_sample_embeddings.progress.json`). It resumes if interrupted.
2. Run `precompute/embed_osdr.py --metadata-only` afterwards if the metadata harmonization changed since the run started.
3. Run `precompute/build_projections.py`. Expect: PCA minutes, UMAP 30-90 min, hnswlib index ~2 GB.
4. Validate against the Phase 2/4 criteria: PC1 ~58% of variance, and OSDR points landing in sensible neighborhoods.
5. Launch the app on the real corpus and re-run the browser checks at 940k scale, watching frame rate at the 100k and 150k budgets.
6. Optional, if the ARCHS4 HDF5 files are ever downloaded: `pip install h5py`, run `fetch_archs4_meta.py`, and the tissue color-by appears by itself.

## Notes and risks

- The hnswlib index holds raw float32 vectors: ~2 GB on disk and resident in the app. `--skip-hnsw` is supported and costs only the kNN-purity statistic.
- OSDR (fp32/CPU) versus ARCHS4 (bf16/CUDA) introduces a precision batch effect between corpora. The cross-dataset warning is wired; the magnitude has not yet been measured on real embeddings, and should be once Phase 1 lands.
- UMAP quality at 940k via landmark fit-and-transform is unverified until the job runs.
- `tests/` never touches the real data, so the suite stays fast and runs on a machine with neither the memmap nor the checkpoint.
