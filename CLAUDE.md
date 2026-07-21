# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Bridge Manifold is the exploratory map companion to **Bridge RNA** (a separate repo at `/Users/josh/Bridge-RNA`).
Bridge RNA retrieves the closest Earth analogs for one NASA spaceflight RNA-seq sample.
Bridge Manifold zooms out: it dimensionally reduces the 512-d ExpressionPerformer embeddings of both corpora - OSDR (2,896 NASA GeneLab spaceflight samples) and ARCHS4 (940,455 GEO samples) - into one shared 2D/3D space, renders them in an interactive WebGL scatter, colors by feature, and lets a user lasso a region to get an honest "are these meaningfully related?" statistical readout.

**Current state: design complete, no application code written yet.** This directory currently holds only planning docs.
The build is a phased plan (Phase 0-6) laid out in `IMPLEMENTATION.md` section 12.

## Read these first

- `IMPLEMENTATION.md` - the master plan: architecture, design decisions, tradeoffs, phased build order. Start here.
- `REFERENCE.md` - verified ground-truth facts (model config, gene digest, embedding stats, measured timings, reusable Bridge RNA interfaces with line numbers, color-by columns, theme tokens). Every fact was checked directly against the checkpoint/memmap/data, not from docs. Trust it over inference.
- `progress.md` - living status log, decisions made, open questions. Keep it updated after every meaningful change (per Josh's global convention).

When plans change, update these docs so they reflect what was actually built, not just the initial intent.

## Architecture: the offline/online split is the load-bearing decision

Everything expensive is precomputed once and cached; the app only ever loads artifacts. This is forced by measured cost (UMAP fit is 30-90 min for the full corpus) and is what keeps the app responsive. Do not move model inference, UMAP, density rasterization, or index building into the serving app.

```
OFFLINE (precompute/, run once -> cache/)      ONLINE (app_manifold.py, loads artifacts only)
embed_osdr.py     -> osdr embeddings npy        loads coord parquets + ARCHS4 memmap + hnswlib
build_projections.py -> pca/umap coord parquets renders go.Scattergl + density PNG underlay
                     -> hnswlib cosine index     lasso -> coherence.py -> 512-d readout
                     -> density raster PNGs
fetch_archs4_meta.py -> archs4_metadata parquet
```

Planned package layout is in `IMPLEMENTATION.md` section 9. The two modules that import `torch`/`umap`/`archs4py` are confined to `precompute/` so the serving app keeps a light dependency surface (`manifold/` + `dash`).

## Non-negotiable invariants

These are correctness gates, not style preferences. Violating them produces output that looks fine but is scientifically wrong.

1. **Gene-digest gate.** `embed_osdr.py` must compute `canonical_gene_order_digest(genes)` and assert it equals `CANONICAL_GENES_SHA256` (`3f887ac8d329dce3c54d26448964904c07a345940cd3d9ebab18dd1f603194c5`). Abort the build on mismatch. An embedding built with the wrong gene order is silently invalid.
2. **L2-normalize before any reduction.** Raw ARCHS4 vectors are NOT normalized (norms 6.7-25.5); PC1 captures 57.8% of variance and is essentially a magnitude/depth axis. Normalize first or the manifold reflects sequencing depth, not biology.
3. **Lasso statistics are computed in the original 512-d cosine space, never from 2D projection coordinates.** The lasso only selects *which* points; UMAP/PCA pixel distances are distorted and must never define a statistic.
4. **Read model hyperparameters from `ckpt['config']`, not the demo's fallback constants.** The demo defaults differ from the true trained config.
5. **Verify Git LFS pointers resolve before any run.** The checkpoint and memmap live in Bridge RNA as LFS objects and can arrive as stub pointers; reuse Bridge RNA's `preflight_retrieval_requirements` / `_is_lfs_pointer` guards.

## Relationship to Bridge RNA

Bridge Manifold is a standalone Dash app that **imports/copies reusable functions from Bridge RNA rather than editing its 2,470-line retrieval app** - isolation without losing the shared-instrument feel. It never retrains the model and never re-embeds ARCHS4 (those 940k embeddings already exist and are consumed as-is from the Bridge RNA repo). The reusable interfaces (OSDR preprocessing, `ExpressionPerformer`/`encode`, digest helpers, ortholog maps, memmap loaders, `fetch_archs4_metadata`, LFS guards) are catalogued with signatures and line numbers in `REFERENCE.md` section 6.

## Environment and planned commands

Shares the Bridge RNA venv at `/Users/josh/Bridge-RNA/.venv` (has torch, umap-learn, sklearn, plotly). Bridge Manifold adds `dash`, `hnswlib`, and `archs4py` on top.
Density rasters are a plain numpy 2D histogram rendered through PIL, deliberately not datashader, to keep the dependency surface small - datashader is not installed and is not required. Record a `requirements.txt` here once dependencies are pinned.

```bash
# run from the Bridge RNA venv
python precompute/embed_osdr.py         # 2,896 OSDR embeddings, gene-digest gated
python precompute/fetch_archs4_meta.py  # ARCHS4 tissue metadata (needs the tens-of-GB gene H5 files)
python precompute/build_projections.py  # PCA + UMAP coords, hnswlib index, density rasters
python app_manifold.py                  # serves http://127.0.0.1:8051
```

The ARCHS4 H5 download is the long pole - kick it off early. Until `cache/archs4_metadata.parquet` exists, ARCHS4 colors by species only and the tissue color-by degrades gracefully rather than failing.

Each build phase ends with an **objective validation** (digest match, explained-variance profile, a coherent-vs-random lasso check), not a visual glance. See `IMPLEMENTATION.md` section 12 for the per-phase validation criteria.

## Visual language

Light scientific-instrument chrome matching Bridge RNA exactly (canvas `#eef2f7`, panels `#fff`, accent `#2b7fff`, navy header `#14294a` with teal rule `#22c7bd`), with a dark navy *plot canvas* inside it for WebGL glyph contrast - the one deliberate departure. Full token list is in `REFERENCE.md` section 9. Finalize the categorical color-by palette against the dataviz skill at build time (tuned for a dark scatter background and colorblind safety).
