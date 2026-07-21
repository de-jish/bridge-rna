# Bridge Manifold

Bridge Manifold is the exploratory map for Bridge RNA.

Bridge RNA takes one NASA spaceflight RNA-seq sample and retrieves its closest Earth analogs from a 940,455-sample ARCHS4/GEO index.
Bridge Manifold zooms out from a single query to the whole space.
It dimensionally reduces the 512-dimensional ExpressionPerformer embeddings of both corpora - OSDR (NASA GeneLab spaceflight samples) and ARCHS4 (940,455 human and mouse GEO samples) - draws them together in one interactive WebGL scatter, colors them by biological and technical features, and lets you lasso any region to ask whether the selected samples are meaningfully related.

## Status

Application complete: `manifold/` and `precompute/` are written and tested, 101 tests pass, and the app has been driven end to end in a browser.
The offline OSDR embedding job is the remaining long pole before the real map can be rendered.
See `progress.md` for live status.

## What it does

- Reduces both corpora into one shared 2D and 3D space, with PCA (fast, linear) and UMAP (structure-preserving, nonlinear).
- Renders ~100k live glyphs over a density raster of all 940k points, using Plotly WebGL scatter traces, so the global shape is always visible and interaction stays smooth. Zooming re-stratifies the sample inside the visible window rather than just enlarging sparse dots.
- Colors by feature: flight status, spaceflight arm, tissue, strain, sex, genotype, study, habitat, duration, and diet for OSDR; species for both corpora; tissue for ARCHS4 once the optional metadata join is built.
- Lasso a region and get an honest readout of coherence and enrichment, computed in the original 512-dimensional space and guarded against batch-driven false positives.

## The one thing worth reading the code for

Every statistic in the lasso readout is computed in the original 512-d cosine space.
The lasso decides *which* points; the pixels never define the number.
UMAP preserves local neighborhoods, not distances, so a statistic read off the projection would be a lie dressed as a measurement.

That principle is enforced, not merely intended: `tests/test_coherence.py` makes any access to projection coordinates raise while the readout runs.

The null model matters just as much as the space it lives in.
The cohesion statistic is compared against uniform random selections of the same size and corpus composition, computed analytically with an exact finite-population correction.
An earlier bootstrap that resampled a fixed background pool gave genuinely random selections `|z| > 40` at corpus scale - a bias that grows with selection size and whose sign depends on a random seed.
See `IMPLEMENTATION.md` section 7.

## How it relates to Bridge RNA

Bridge Manifold is a separate app that reuses Bridge RNA's model, embeddings, preprocessing, and visual language.
It lives in its own directory so the heavy exploratory tool never destabilizes the retrieval product, while a shared header and shared CSS make them feel like one instrument.
The ARCHS4 embeddings and the model checkpoint stay in the Bridge RNA repository and are consumed from there, read-only.
All imports from the sibling repo are funnelled through `manifold/bridge_rna.py`, so the coupling is visible in one file.

## Documents

- `IMPLEMENTATION.md` - the master plan: architecture, design decisions, tradeoffs, and the phased build order.
- `REFERENCE.md` - the verified ground-truth facts: model config, gene digest, embedding statistics, measured timings, library behaviours the code depends on, reusable Bridge RNA interfaces, color-by columns, and theme tokens.
- `progress.md` - the living status log, decisions, defects found and fixed, and next steps.

## Architecture in one picture

```
OFFLINE (run once, cached)                     ONLINE (Dash app, loads artifacts only)
embed_osdr.py        -> osdr embeddings         app_manifold.py
build_projections.py -> pca/umap coords          loads coord parquets + ARCHS4 memmap + hnswlib
                     -> hnswlib cosine index     renders Scattergl over a density underlay
                     -> density rasters          lasso -> 512-d coherence readout
                     -> population moments
fetch_archs4_meta.py -> tissue join (optional)
```

The app never runs UMAP or the model; it reads precomputed coordinates and draws them.

## Setup

Bridge Manifold shares the Bridge RNA virtualenv, because it consumes that repo's checkpoint and memmap directly.

```bash
/Users/josh/Bridge-RNA/.venv/bin/python -m pip install -r requirements.txt
```

Both repository locations are overridable by environment variable:
`BRIDGE_RNA_ROOT` (default `/Users/josh/Bridge-RNA`) and `MANIFOLD_CACHE_DIR` (default `./cache`).

## Build the cache, then run

```bash
PY=/Users/josh/Bridge-RNA/.venv/bin/python

$PY precompute/embed_osdr.py         # OSDR embeddings, gene-digest gated. Hours; resumable.
$PY precompute/build_projections.py  # PCA + UMAP coords, hnswlib index, density, moments
$PY precompute/fetch_archs4_meta.py  # optional ARCHS4 tissue join (needs the gene HDF5 files)
$PY app_manifold.py                  # http://127.0.0.1:8051
```

`embed_osdr.py` writes progress as it goes and resumes where it stopped, so an interrupted run does not restart from zero.
`build_projections.py` supports `--skip-umap` and `--skip-hnsw` for a faster first pass; skipping the index costs only the kNN-purity statistic.

## Run it before the data exists

The real cache takes hours to build. To exercise the interface immediately, build a synthetic corpus of the same shape:

```bash
PY=/Users/josh/Bridge-RNA/.venv/bin/python
$PY tests/build_dev_corpus.py --out /tmp/bm-dev --archs4 60000 --osdr 2000 --clean
BRIDGE_RNA_ROOT=/tmp/bm-dev/bridge_rna MANIFOLD_CACHE_DIR=/tmp/bm-dev/cache $PY app_manifold.py
```

The numbers are synthetic - shaped like the real corpus, with real cluster structure, but meaningless biologically.
It exists to test the instrument, not to be read.

## Tests

```bash
/Users/josh/Bridge-RNA/.venv/bin/python -m pytest tests/ -q
```

The suite builds its own synthetic corpus in a temp directory and never touches the 963 MB memmap or the checkpoint, so it runs in seconds on a machine that has neither.
The corpus is generated from known latent clusters with metadata derived from those clusters, which gives the coherence tests real ground truth: a cluster-shaped selection must read as coherent and enrich for that cluster's tissue, and a scattered one must not.
