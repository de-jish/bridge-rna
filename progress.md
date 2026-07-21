# Bridge Manifold - Progress

Living status log.
Update after each meaningful change so another session can resume without losing context.

## Current status: 2026-07-21 - built, colored by real biology on both corpora, and tested

The full offline pipeline has run to completion on real data, and the app has been redesigned around one question the first build got wrong: what should the map show for the corpus the selected color-by does not describe.

`cache/` holds the real 942,563-point manifold: 940,455 ARCHS4 (510,709 human, 429,746 mouse) plus 2,108 OSDR.

- `embed_osdr.py` finished 2026-07-21 08:45:51 after ~11.3 h, all 2,108 samples, gene-digest gate passed.
  Realized rate was ~10 s/sample in fast stretches, degrading to ~49 s/sample between 05:44 and 08:26 under machine contention, so the original ~6.5 s/sample estimate was optimistic.
- `build_projections.py` ran 08:45:57 to 08:51:44, **5 min 47 s**, rc=0.
  The 30-90 min estimate in `REFERENCE.md` was wrong by an order of magnitude; measured per-stage timings are now recorded there.
  Two of those stages have since been deleted, so the same build is now a **291 s** job.
- `fetch_archs4_meta.py` ran in **33.7 s** over 39 requests and about 216 MB, resolving **99.911%** of all 940,455 accessions.
- `precompute/validate_artifacts.py --mixing` passes every structural and invariant check, with one substantive warning: the cross-corpus batch effect (see Notes and risks).
- **144 tests pass in about 0.55 s** against a hermetic synthetic corpus.

### What is done

- **Phase 0 scaffold**: package skeleton, path configuration with `BRIDGE_RNA_ROOT` / `MANIFOLD_CACHE_DIR` overrides, LFS-pointer preflight.
- **Phase 1 OSDR embeddings**: `embed_osdr.py`, gene-digest gated, resumable, with a cached expression stage. **Complete.**
  Preprocessing proven bit-for-bit identical to Bridge RNA's single-sample path.
- **Phase 2/4 projections**: `build_projections.py` writes PCA-2/3, landmark UMAP-2/3, the identity table, the ARCHS4 accession sidecar, and the density rasters. **Complete.**
- **Phase 3 interactive plot**: layered renderer (density underlay, stratified ARCHS4 cloud, OSDR overlay), layer toggles, point budget, viewport level-of-detail.
- **Phase 5 coloring both corpora**: the ARCHS4 GEO metadata join, the shared tissue vocabulary, and the coverage-aware color-by registry that replaced the renderer's per-key branching.
- **Phase 6 polish**: searchable legend, theme-matched Dash 4 controls, hover cards, 3D, honest empty and degraded states.
- **Tests**: `tests/` with a synthetic corpus built from known latent clusters plus a synthetic `archs4_metadata.parquet` written in ARCHS4's free-text register and mapped through the real canonicalizer, so the tissue vocabulary is tested against GEO-shaped strings rather than against its own rules.

### Removed in this session, and not to be reintroduced

**The lasso selection tool and its 512-d statistical readout are gone in their entirety**, at Josh's explicit request.
Deleted: `manifold/coherence.py` (450 lines), `tests/test_coherence.py` (431 lines), the right-hand readout column in `layout.py`, the `selectedData` callback and every helper behind it in `callbacks.py`, the vector/moment/index loaders in `data.py`, and the readout and lasso-marquee sections of `assets/manifold.css`.
`dragmode` is now `pan`, and the graph config removes **both** `select2d` and `lasso2d` (the old config removed only `select2d`, so the lasso button was in fact still on the modebar).

Consequences, all of them verified:

- `build_projections.py` no longer builds `cache/joint_cosine.hnsw` (2.07 GB) or `cache/population_moments.npz` (4.2 MB), and its `--skip-hnsw` flag is gone.
- `requirements.txt` dropped `hnswlib` and `scipy` and added `requests`. The serving app's dependency surface is now `dash`, `plotly`, `numpy`, `pandas`, `pyarrow`, and nothing scientific.
- The live cache fell from about 2.3 GB to a measured **219.2 MB**, of which the app opens **82.3 MB**. The two dead files are still physically present on this machine as leftovers, which is why `cache/` measures 2,293.8 MB; deleting them is safe.
- The serving app no longer opens the 963 MB ARCHS4 memmap at all, so `BRIDGE_RNA_ROOT` is needed to *build* the cache and not to *run* the app.
- `validate_artifacts.py --mixing` used to load the ANN index.
  It now computes the **exact** top-51 neighbours of each of the 2,108 OSDR samples by streaming the memmap in 50,000-row blocks and merging a running top-k (`_osdr_neighbours`), which costs 10.3 s warm.
  That is why the index could be deleted.
  The mixing check itself is unchanged and is **not** a lasso feature: it is the honesty check behind the app's premise and must keep working.
- `manifold/preflight.APP_REQUIRED` was wrong in both directions and is fixed. It demanded the ARCHS4 memmap, `sample_locations.parquet` and the OSDR embeddings, none of which the app opens, while omitting `cache/points_meta.parquet`, which `layout.control_rail()` reads *first* through `data.counts()` - so a missing identity table passed preflight and then crashed during startup.

The test suite went 103 -> 144 tests and 4.54 s -> 0.55 s; the fixture no longer builds an ANN index, which was 43% of the old wall clock.

### The headline: ARCHS4 can now be colored by real biology

The problem: the app had about ten color-bys for the 2,108 OSDR samples and exactly one (species) for the 940,455 ARCHS4 samples.
Choosing any OSDR field painted 99.8% of the map one flat grey, which on a scientific plot reads as "ARCHS4 was measured and has no structure here".
A "Tissue (ARCHS4)" option existed but required the ARCHS4 gene HDF5 files, 62.3 GB human plus 50.7 GB mouse, which were never downloaded, so it had never once worked.

Three pieces fixed it.

1. **`precompute/fetch_archs4_meta.py`, rewritten around the Maayan Lab sigpy JSON API.**
   `POST https://maayanlab.cloud/sigpy/meta/samplemeta` with `{"species": ..., "samples": [...]}` returns per-GSM `{series, title, source, characteristics}` in bulk.
   Measured by running it: 33.7 s, 39 requests, ~216 MB, 99.911% of all 940,455 accessions (human 99.851%, mouse 99.982%).
   Output is `cache/archs4_metadata.parquet`, 940,455 rows, 32.5 MB, 51,284 distinct GEO series.
   Reading the same fields out of the remote gene HDF5 over range requests works but costs ~5 min and ~272 MB **per field**; downloading the files is 113 GB.
   The 839 unresolved samples are not GEO withdrawals - they are present in the release-matched v2.5 metadata and absent from the newer v2.latest the API serves, which disproves the "ARCHS4 releases are append-only" assumption. They get tissue `Unknown` rather than being dropped or guessed at.
2. **`manifold/tissue.py`, one tissue vocabulary shared by both corpora.**
   40 ordered keyword rules, first match wins, producing 37 distinct buckets plus `Other` and `Unknown`.
   All 48 OSDR raw values land in a named bucket, and 851,881 of 940,455 ARCHS4 samples (**90.6%**) do too, so the Tissue color-by covers **942,563 of 942,563 points**.
3. **`manifold/colorby.py`, the coverage-aware registry.**
   Coverage is a declared, first-class property: each `ColorBy` states its scope, its resolver, an optional hint, and an optional `(predicate, fix-hint)` pair for an artifact it needs, and `covers()` reports what it can color right now on this machine.
   That one fact drives the menu order, the disabled state, the coverage readout under the control, and what the renderer does.

The interface consequences are the point of the exercise: the menu lists whole-map fields first with their scope attached, a field with no data is shown *disabled* with the command that enables it rather than hidden, a coverage bar and an exact point count sit under the control, and **the renderer never paints a uniform grey glyph cloud** - a corpus a field does not describe is carried by the density raster, or by a deliberately faint context cloud at 0.35 opacity when there is no raster (3D, or the underlay switched off).

Tissue was then validated as biology rather than as batch, to the same standard every rejected candidate was held to: 25-NN label purity **0.8142** against a permuted null of **0.0501**, surviving both a batch control and a depth control at **0.7058**.

### Corrections to earlier assumptions

- **Corpus size.**
  The OSDR corpus is **2,163 eligible / 2,108 embedded**, not 2,896. 2,896 is the unfiltered TSV row count; 733 rows have no spaceflight factor and are excluded by the Bridge RNA filter Josh chose to match, and 55 more name a counts column that does not exist.
  All docs corrected.
- **Environment.** The versions in `REFERENCE.md` had drifted: pandas 3.0, dash 4.4, plotly 6.8, numpy 2.4, torch 2.12. Three of those releases changed behaviour the code depends on (see `REFERENCE.md` section 5).
- **datashader is not used.**
  The density raster is a numpy 2D histogram plus Pillow.
  Fewer fragile dependencies, and it is trivially fast at this scale.
- **MPS is not viable** for this model, and chunking does not help. Measured and documented.
- **The ARCHS4 HDF5 download was never necessary.**
  It had been the plan for the entire first build and had never been executed once.
  The API route returns the same fields three orders of magnitude cheaper.

## Decisions log

- **No on-demand statistics.**
  The map is read, not queried.
  A statistic computed from a screen region would be a number read off distorted UMAP pixels, and the selection readout was the app's largest source of complexity in service of a question the map answers qualitatively.
- **Coverage is a declared property, not a per-branch decision.**
  Every field states which corpora it can color right now; the menu, the coverage readout, and the renderer all read that one declaration.
  The alternative is what the first build did, and 99.8% of the map turned flat grey with nothing in the interface admitting it.
- **One shared tissue vocabulary rather than two tissue fields.** Two separate "Tissue" color-bys would each leave the other corpus grey, which is the grey-map failure by another route.
- **The tissue mapping is auditable keyword rules, not learned.** It fails towards "Other" rather than towards a confident guess; on a plot people read biology off, an honestly empty label beats a wrong one.
- **`Unknown` and `Other` stay distinct, and weak results are ranked.** Nothing recorded is not the same fact as recorded-but-unplaceable, and without the ranking an early unplaceable field pinned the answer to "Other" and blocked a later field that did identify the sample.
- **One palette across both corpora.**
  Categories are ranked once over the whole covered population, so a liver in GEO and a liver in OSDR share a color; ranking per layer silently gave one category two colors.
  Legend counts are whole-corpus counts, so they do not move with the point budget or the zoom.
- **The availability predicate is `data.archs4_metadata_available` itself**, never a path re-derived inside the registry. A second source of truth for the same file was a real bug; a test now pins it.
- The seven OSDR control arms stay distinct; the binary Flight-vs-Ground contrast is a separate derived field. Rationale: basal and vivarium controls are different experiments, and merging them erases real structure.
- L2-normalize before any reduction. Rationale: raw vectors carry a 4x magnitude spread that dominates PC1 (57.8% before normalization, 40.9% after).
- UMAP is offline only, via landmark fit then transform. Rationale: a direct 940k fit is hours and risks memory blowup.
- Standalone app importing Bridge RNA functions, not edits to the 2,470-line retrieval app. Rationale: isolation without losing the shared instrument feel.
- Batch structure is made visible, not corrected. The measured 54x tissue-controlled cross-corpus effect is stated on the control rail, always, rather than inside anything the user has to trigger.
- OSDR embedded in fp32 on CPU. Rationale: fidelity baseline, and measurement showed no faster option exists on this machine.
- Dash components are themed by remapping Dash 4's own `--Dash-*` design tokens rather than by overriding each component's rules. Rationale: one mapping themes every current and future Dash component; per-component overrides are a specificity war that silently rots on upgrade.

## Decisions from Josh

1. **2026-07-21: remove the lasso tool completely**, from the implementation and from every document. Done; the removal is recorded as history only, in `IMPLEMENTATION.md` section 1 and section 8.
2. **2026-07-21: never end up with a grey map.** This is what the coverage-aware registry, the shared tissue vocabulary, and the density fallback exist for.
3. 2026-07-20, ARCHS4 tissue coloring: FETCH NOW for v1. Delivered, and better than scoped - the API route removed the HDF5 blocker entirely rather than shipping behind a graceful degrade.
4. 2026-07-20, OSDR scope: MATCH the Bridge RNA filter (mouse + spaceflight factor). Honored - this is what yields 2,163 rather than 2,896.
5. 2026-07-20, batch handling: EXPOSE AND GUARD only. No correction, not even as a toggle.
6. 2026-07-20, environment: SHARE the Bridge RNA venv. Done; `requirements.txt` records the verified versions and splits serving from precompute.

## Color-by candidates that were built or tested and then rejected

Recorded with their evidence so nobody re-proposes them.
Full write-ups in `IMPLEMENTATION.md` section 7.5 and `REFERENCE.md` section 11.

- **Cosine similarity to an OSDR reference** (mean / flight / ground centroid, and a flight-minus-ground "spaceflight-likeness" axis). One field wearing four names, pairwise r 0.996-1.000. The interesting axis correlates r = -0.990 with PC1 and r = -0.779 with the raw L2 norm: it is the sequencing-depth axis relabelled as biology. 1 in 10 random flight/ground relabelings beat it on spatial structure, 46.5% under a within-study permutation.
- **kNN tissue-label transfer from OSDR to ARCHS4.** Median best-match cosine 0.964 with 100% of points above 0.7, so no confidence threshold discriminates anything, and the winner beats the runner-up by a median of 0.00089 cosine. 54% of the targets are human samples that would have received mouse labels.
- **Unsupervised k-means cluster id (k=24).**
  Built, run on the real corpus, measured, then deleted along with its precompute stage. 81.9% of the label is recoverable from the 2-D UMAP coordinates alone (15-NN over a 120k sample, against a 12.4% majority-class baseline); a structure-free 24-cell Voronoi null reproduced its spatial coherence to within 1.5 points; seed-to-seed ARI ~0.45; 81% species-pure; explains 80.7% of the raw-L2-norm depth variance.
  A comment in `manifold/colorby.py` records the decision where someone would add it back.
- **Local UMAP density.** Redundant with the raster already drawn underneath.
- **PC1-3.** Free but redundant with the axes on screen, and PC1 is the depth axis.
- **GEO series (GSE).** 51,284 distinct values, so a Top-11 legend would color ~3% of the map and dump the rest in "Other".
  Also a pure batch label (333x lift).
  Kept in the parquet for provenance, not offered as a color.

**Methodological note.**
Spatial eta-squared is not evidence: 30 arbitrary random directions in 512-d score 0.874 +/- 0.025 on this UMAP, because the UMAP was fit on those same vectors.
Judge a candidate against a structure-free null of the same *form*, and check whether it is recoverable from the coordinates or from depth.

## Defects found and fixed

Found by adversarial audit, browser-driven testing, and by running against the real corpus. Each was verified before being fixed.

**Still-current code:**

1. The plot occupied 450 px of an ~890 px pane - `dcc.Loading`'s wrapper divs broke the `height: 100%` chain.
2. Segmented controls had no selected state and the dropdown was entirely unstyled: Dash 4 rewrote both components' DOM and class names.
3. The legend search box was inert - no callback read it.
4. The ARCHS4 background cloud showed a hover label despite `hoverinfo="skip"`, because a `hovertemplate` overrides it.
5. pandas 3.0 leaves NA through `astype(str)`, so a phantom NA category reached the legend. The same trap appears in `fetch_archs4_meta.first_series`, where unresolved accessions arrive as float NaN and `value or ""` does not catch them because NaN is truthy; without the explicit isna guard they became the literal string `"nan"`, read as a real GSE, and overstated metadata coverage to a clean 100%.
6. 55 samples were dropped silently during preprocessing; now reported.
7. The expression cache could never hit (it compared eligible keys against kept keys), and resume was keyed on row count alone.
8. The gene-digest gate was skipped on the expression-cache path; the digest is now part of the cache key.
9. **The density ramp used 0.78% of its range.**
   `render_density` normalized `log1p(counts)` by the global max.
   Real occupancy is heavy-tailed - median occupied bin holds 2 points, max 638 - so dividing by the max crushed everything into the bottom of the scale.
   Only 0.78% of occupied bins cleared the 0.5 threshold where the navy-to-teal ramp turns teal, and alpha saturated at 0.4545, *before* that turn, so the densest cores were indistinguishable from merely-busy ones.
   Measured on the raster: 8 pixels total in the teal half.
   Fixed by normalizing against the 99.5th percentile of occupied bins (`DENSITY_CLIP_PCT`) and ramping alpha across the same span with a visibility floor (`DENSITY_ALPHA_FLOOR`).
   Teal-half pixels went 8 -> 2,377; mean occupied colour lifted RGB (21,50,82) -> (26,71,108).
   Filament structure that was a flat wash now reads.
   Added `build_projections.py --density-only` so ramp changes re-render from cached coordinates in seconds instead of repeating the projection build.
10. **`APP_REQUIRED` omitted `points_meta.parquet`**, which `layout.control_rail()` reads first, so a bare cache passed preflight and crashed during startup. It also demanded three artifacts the app never opens.
11. **The renderer ranked categories per layer**, so a category could take two different colors in one figure. Categories are now ranked once over the whole covered population.
12. **Residual traces were emitted last**, so ~308,000 grey glyphs painted over every colored category and the map read as grey even where it was not. Residual categories are now emitted first, receded to 0.26 opacity and 0.82 size.
13. **`lasso2d` was still on the modebar** after the selection feature was designed away, because the config removed only `select2d`.

**In code that has since been deleted** (kept because the reasoning generalizes):

14. The coherence null was a with-replacement bootstrap over a frozen 20k pool, giving a systematic z-bias growing with selection size. Rewritten analytically, then removed with the feature.
15. The lasso never produced a selection: plotly 6 serializes numpy `customdata` as base64 and Dash's event filter indexes the user data, so `customdata` arrived at the server as nothing. The plain-list convention that fixed it still stands for the OSDR hover.
16. The verdict could print "Coherent" next to a z and p saying the selection was looser than a matched random draw. The general lesson survives the feature: a summary sentence must be derived from the same statistics it sits beside, not chosen independently of them.

## Next steps

1. ~~Wait for `embed_osdr.py`~~ **DONE** 2026-07-21 08:45:51, all 2,108 samples.
2. ~~Run `precompute/build_projections.py`~~ **DONE** 2026-07-21 08:51:44, rc=0 in 5 min 47 s.
3. ~~Validate against the Phase 2/4 criteria~~ **DONE**.
   PC1 = 40.9% against the 57.8% pre-normalization figure, so invariant 2 holds.
   Codified as `precompute/validate_artifacts.py`, which exits nonzero on failure.
4. ~~Launch the app on the real corpus~~ **DONE** 2026-07-21, driven headless end to end against the real 942,563-point cache. Zero console errors.
5. ~~Give ARCHS4 a real biological color-by~~ **DONE** 2026-07-21: the sigpy metadata join, the shared tissue vocabulary, and the coverage-aware registry.
6. ~~Remove the lasso tool and its readout~~ **DONE** 2026-07-21, including the artifacts and dependencies that existed only to serve it.
7. Delete the two dead cache files, `cache/joint_cosine.hnsw` (2.07 GB) and `cache/population_moments.npz` (4.2 MB). Nothing reads them and nothing writes them.
8. `cache/projection_stats.json` still carries `cluster_k`, `cluster_sizes_archs4`, `cluster_sizes_osdr` and `cluster_osdr_span` from the cut k-means build. Nothing reads them and the current script does not write them; they will disappear on the next full build.
9. Re-run the browser checks at the 60k and 150k point budgets and measure frame rate; only the 100k default has been exercised so far.
10. Review the *visual* quality of the real UMAP map, which is still unreviewed at 940k.
11. Optional: switch the metadata fetch to the versioned metadata-only HDF5 files (`human_meta_v2.5.h5` 311.8 MB, `mouse_meta_v2.5.h5` 350.9 MB) if tissue ever needs to be a build **gate** rather than a color.
    That buys exactly 100.000% release-matched coverage for 663 MB and ~8.5 min, against 216 MB and 35 s.
    Not worth 15x the build time for 0.089% of points on a color.
12. Optional: `precompute/embed_osdr.py --metadata-only` if the metadata harmonization ever changes; it rewrites the parquet without re-embedding.

## How the projection build was chained (2026-07-21, completed)

`build_projections.py` was queued behind the in-flight embed run rather than waited on by hand.
A detached watcher polled the embed PID, verified the run actually succeeded, and only then started the projection build with default parameters.
It fired correctly: embed exited 08:45:51, the gates passed, and the build launched 08:45:57 - a 6-second unattended handoff.

The watcher refused to launch unless all of these held, because `build_projections.py` joins OSDR metadata to embeddings *positionally* and a truncated embedding would silently mislabel every OSDR point rather than fail:

- the embed log contains a `[done] wrote` line (clean completion, not a crash or a kill),
- `osdr_sample_embeddings.float32.npy` and `osdr_metadata.parquet` both exist and are non-empty,
- their row counts agree, the embedding dim is 512, and the values are finite.

That gate is worth rebuilding if this is ever re-run, since the failure it guards against is silent rather than loud.
`build_projections.py` now asserts the same row-count agreement itself before doing any work.

**Preflight verified before queuing** (2026-07-21):

- All inputs resolve and are real data, not Git LFS stubs: the ARCHS4 memmap is 963,025,920 bytes, exactly 940,455 x 512 x 2, matching `embedding_manifest.json`.
- Every import `build_projections.py` needs is installed: numpy 2.4.6, pandas 3.0.3, sklearn 1.9.0, umap 0.5.12, pyarrow 20.0.0, PIL 12.2.0.
- Disk 44 GiB free; RAM 17 GB against a measured peak of roughly 2.5 GB.
- **Full-path smoke test passed** end to end in an isolated `MANIFOLD_CACHE_DIR` with synthetic OSDR embeddings and `--archs4-limit 4000`, exercising IncrementalPCA, PCA transform, density rasters, UMAP 2-d and 3-d, and every parquet write. Exit 0.
  Running the real pipeline against a throwaway cache first is cheap insurance and is worth repeating before any long rebuild.

### Built artifacts in `cache/` (measured on disk 2026-07-21)

Full inventory, with which of them the app opens, is `REFERENCE.md` section 12.

| file | size | contents |
| --- | --- | --- |
| `archs4_metadata.parquet` | 32.51 MB | per-GSM GEO metadata + the canonical tissue bucket |
| `coords_pca3.parquet` / `coords_umap3.parquet` | 13.17 MB each | 3-d coordinates |
| `coords_pca2.parquet` / `coords_umap2.parquet` | 8.78 MB each | 2-d coordinates |
| `archs4_geo.parquet` | 4.63 MB | GEO accessions; the join key for the metadata fetch |
| `points_meta.parquet` | 4.36 MB | dataset / src_index / species_id identity table |
| `osdr_sample_embeddings.float32.npy` | 4.32 MB | the 2,108 x 512 OSDR embeddings |
| `osdr_expression.float32.npy` | 127.87 MB | resume intermediate for the multi-hour embed job |
| `osdr_expression_meta.parquet` | 0.097 MB | its metadata sidecar |
| `osdr_metadata.parquet` | 0.027 MB | OSDR labels, joined positionally |
| `density/pca2.png` / `density/umap2.png` | 0.86 MB / 0.61 MB | density underlays |
| `projection_stats.json` | 2.3 KB | variance profile and raster extents |

Total live cache **219.2 MB**, of which the serving app opens **82.3 MB**.
`embed_osdr.py` cleaned up its own partial memmap and progress JSON on success, as designed.

Two files are still on disk but are **no longer produced by anything**: `joint_cosine.hnsw` (2,070.4 MB) and `population_moments.npz` (4.2 MB), which is why `cache/` currently measures 2,293.8 MB.

## Notes and risks

- **The cross-corpus batch effect is real and measured exactly (2026-07-21).**
  Controlling for both study and tissue, OSDR samples that share neither still neighbour each other **54x above chance** (11.491% observed against 0.21101% expected).
  Tissue is the dominant axis of bulk expression, so biology cannot explain it - this is the fp32/CPU versus bf16/CUDA precision and preprocessing difference.
  The caution on the control rail is therefore load-bearing and must stay prominent; cross-corpus distances are not trustworthy at face value.
  Full numbers in `REFERENCE.md` section 4.
  Re-check with `/Users/josh/Bridge-RNA/.venv/bin/python precompute/validate_artifacts.py --mixing`, which warns above 50x.
- The mixing check is the only thing left that opens the memmap, and it is opt-in. Everything else in the serving path reads `cache/` and nothing else.
- 839 ARCHS4 samples (0.089%) carry tissue `Unknown` because the newer release the API serves dropped them.
  They are not guessed at.
  If tissue ever becomes a build gate, switch to the versioned metadata-only HDF5 files and assert 100%.
- UMAP quality at 940k via landmark fit-and-transform ran clean, but *visual* quality on the real map is still unreviewed.
- `tests/` never touches the real data, so the suite stays fast and runs on a machine with neither the memmap nor the checkpoint.
