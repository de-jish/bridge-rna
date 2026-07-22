# Bridge Manifold

Bridge Manifold is the exploratory map for Bridge RNA.

Bridge RNA takes one NASA spaceflight RNA-seq sample and retrieves its closest Earth analogs from a 940,455-sample ARCHS4/GEO index.
Bridge Manifold zooms out from a single query to the whole space.
It dimensionally reduces the 512-dimensional ExpressionPerformer embeddings of both corpora - OSDR (2,108 NASA GeneLab spaceflight samples) and ARCHS4 (940,455 human and mouse GEO samples) - draws them together in one interactive WebGL scatter, and colors them by biology that is defined on *both* sides of the map rather than on one.

## Status

Application complete and running on the real data.
The full pipeline has been run end to end on the 942,563-point corpus (940,455 ARCHS4 + 2,108 OSDR): OSDR embeddings, joint PCA and UMAP in 2D and 3D, density rasters, and the ARCHS4 GEO metadata join.
144 tests pass in about 0.55 s.
The live cache is 219.2 MB, of which the serving app opens 82.3 MB.
See `progress.md` for the live status log.

## What it does

- **Colors the whole map by one shared tissue vocabulary.**
  OSDR and ARCHS4 name tissues in completely disjoint registers: OSDR is curated but hyper-specific ("Right extensor digitorum longus", 48 distinct values), while ARCHS4 has no curated tissue column at all and the signal lives in 42,754 distinct free-text GEO strings.
  `manifold/tissue.py` folds both onto one canonical bucket list of 37 buckets plus "Other" and "Unknown", so a liver in GEO and a NASA liver get the same color from the same legend row.
  All 48 OSDR raw values land in a named bucket rather than in "Other" or "Unknown", and so do 851,881 of 940,455 ARCHS4 samples (90.6%).
  The "Tissue" color-by covers 942,563 of 942,563 points.
- **Reduces both corpora into one shared 2D and 3D space**, with PCA (fast, linear) and UMAP (structure-preserving, nonlinear).
  Vectors are L2-normalized first: raw ARCHS4 norms span 6.7 to 26.4, and without normalization PC1 is 57.8% of the variance and is a magnitude axis.
  That magnitude is not sequencing depth, as this file used to claim - it measures transcriptome concentration, at r = +0.987 with the share of expression held by a sample's top 100 genes.
  The built map has PC1 at 40.9%, with 95.1% cumulative over 50 components.
- **Renders ~100k live glyphs over a density raster of all 942,563 points**, using Plotly WebGL scatter traces, so the global shape is always visible and interaction stays smooth.
  Zooming re-stratifies the sample inside the visible window rather than just enlarging sparse dots.
- **States what each color-by actually covers, before you pick it.**
  The menu lists whole-map fields first and labels every option with its scope ("Tissue · whole map", "Flight vs Ground · OSDR only").
  A coverage bar and an exact point count sit directly under the control.
  A field whose data has not been built is shown *disabled* with the command that enables it, not hidden.
- **Offers OSDR spaceflight detail as spaceflight detail.**
  Flight vs Ground, spaceflight arm, strain, sex, genotype, study, habitat, mission duration, and diet are defined for the 2,108 OSDR samples and are labelled as such.
  Species is the second whole-map field and the reference for what a working color-by looks like.

## The one thing worth reading the code for

**The app never paints a corpus it cannot describe as though it were data.**

That sounds like a small rendering rule. It is the whole design.

Before this rule existed, picking any OSDR field painted 940,455 points - 99.8% of the map - a single flat grey.
A user reads that as "ARCHS4 was measured, and it has no structure here."
It actually meant "this field was never defined for ARCHS4."
Those are opposite claims, and the map made the false one look like a measurement.

`manifold/colorby.py` fixes it by making coverage a declared, first-class property.
Every color-by reports which corpora it can color *right now*, given which artifacts exist on this machine, and that one fact drives the menu order, the disabled state, the coverage readout, and what the renderer does.
When a field does not describe ARCHS4, the glyph layer steps aside and the precomputed density raster carries the manifold shape, with a badge saying so.
The raster shows the true distribution of all 940,455 points and cannot be mistaken for a category.
Drawing nothing there is the honest option, not the degraded one.

The same standard was applied to the color-bys themselves, and it eliminated most of the candidates.
Each of these was built or measured before it was cut, and the evidence is recorded so nobody has to rediscover it:

- **Cosine similarity to an OSDR reference**, including a "spaceflight-likeness" axis of flight-centroid minus ground-centroid.
  The four variants turned out to be one field wearing four names (pairwise r 0.996 to 1.000), and the spaceflight axis correlates r = -0.990 with PC1 and r = -0.779 with the raw L2 norm.
  PC1 is a transcriptome-concentration axis, so the candidate measured how concentrated a sample's transcriptome is and labelled it resemblance to spaceflight.
  One in ten random flight/ground relabelings of the same sample sizes beat it on spatial structure; under a within-study permutation, 46.5% did.
- **kNN tissue-label transfer from OSDR to ARCHS4.**
  The median best-match cosine is 0.964 and 100% of points sit above 0.7, so no confidence threshold discriminates anything.
  The winning OSDR sample beats the runner-up by a median of 0.00089 cosine, which makes the winner essentially arbitrary.
  And 54% of the targets are human samples that would have received mouse tissue labels.
- **Unsupervised k-means cluster id (k=24).**
  Built, run on the real corpus, measured, then deleted.
  81.9% of the cluster label is recoverable from the 2D UMAP coordinates alone, so coloring by it mostly redraws the shape already on screen.
  A structure-free 24-cell Voronoi null reproduced its spatial coherence to within 1.5 points.
  It is arbitrary (seed-to-seed ARI ~0.45), 81% species-pure, and explains 80.7% of the raw-L2-norm variance.
- **GEO series (GSE).**
  51,284 distinct series, so a Top-11 legend would color ~3% of the map and dump the rest in "Other": a grey map by another route.
  It is also a pure batch label, 333x enriched over chance.
  It stays in the parquet for provenance and is not offered as a color.

The methodological note is worth more than any individual verdict.
A between-bin variance ratio (spatial eta-squared) is **not** sufficient evidence that a color-by shows real structure.
30 arbitrary random directions in 512-d score eta-squared 0.874 +/- 0.025 on this UMAP, because the UMAP was fit on those same vectors.
Every candidate in the 0.89 to 0.94 band is therefore indistinguishable from an arbitrary projection, and species (0.985) is the only one that clearly clears it.
Judge a candidate against a structure-free null of the *same form*, and check whether it is recoverable from the coordinates or from transcriptome concentration before believing it.

Tissue survives that bar.
Its 25-NN label purity is 0.8142 against a permuted null of 0.0501, and it holds at 0.7058 under both a batch control and a depth control.

There is one more thing the map is honest about, and it is printed on the control rail rather than buried: OSDR and ARCHS4 were embedded on different hardware and in different precisions, and OSDR samples sharing neither study nor tissue still neighbour each other 54x above chance.
Biology cannot explain cross-tissue clustering, so some of the distance between the two corpora is technical.
Compare within a corpus, not across it.
`precompute/validate_artifacts.py --mixing` is the check that produces that number, and it recomputes it exactly rather than approximately, so the sentence in the interface stays tied to a measurement anyone can re-run.

## How it relates to Bridge RNA

Bridge Manifold is a separate app that reuses Bridge RNA's model, embeddings, preprocessing, and visual language.
It lives in its own directory so the heavy exploratory tool never destabilizes the retrieval product, while a shared header and shared CSS make them feel like one instrument.
The ARCHS4 embeddings and the model checkpoint stay in the Bridge RNA repository and are consumed from there, read-only, by the offline precompute scripts.
All imports from the sibling repo are funnelled through `manifold/bridge_rna.py`, so the coupling is visible in one file.

## Documents

- `IMPLEMENTATION.md` - the master plan: architecture, design decisions, tradeoffs, and the phased build order.
- `REFERENCE.md` - the verified ground-truth facts: model config, gene digest, embedding statistics, measured timings, library behaviours the code depends on, reusable Bridge RNA interfaces, color-by columns, and theme tokens.
- `progress.md` - the living status log, decisions, defects found and fixed, and next steps.

## Architecture in one picture

```
OFFLINE (run once, cached)                      ONLINE (Dash app, loads artifacts only)
embed_osdr.py         -> osdr embeddings         app_manifold.py
build_projections.py  -> pca/umap coords          loads coord parquets + label tables
                      -> density rasters          color-by registry declares coverage
                      -> point identity table     renders Scattergl over a density underlay
fetch_archs4_meta.py  -> ARCHS4 GEO metadata
                         + canonical tissue
validate_artifacts.py -> exit code; gates a build
```

The app never runs the model or UMAP; it reads precomputed coordinates and draws them.
It also never opens the 963 MB ARCHS4 memmap, because it draws a precomputed map and so never needs a 512-d vector at request time.
The serving dependency surface is therefore `dash`, `plotly`, `numpy`, `pandas`, and `pyarrow` - nothing scientific.
`BRIDGE_RNA_ROOT` is required to *build* the cache, not to run the app.

## Setup

Bridge Manifold shares the Bridge RNA virtualenv, because the precompute scripts consume that repo's checkpoint and memmap directly.

```bash
/Users/josh/Bridge-RNA/.venv/bin/python -m pip install -r requirements.txt
```

Both repository locations are overridable by environment variable:
`BRIDGE_RNA_ROOT` (default `/Users/josh/Bridge-RNA`) and `MANIFOLD_CACHE_DIR` (default `./cache`).

## Build the cache, then run

The order matters: the metadata fetch joins positionally onto the identity table that `build_projections.py` writes, and it aborts if that table is missing.

```bash
PY=/Users/josh/Bridge-RNA/.venv/bin/python

$PY precompute/embed_osdr.py                       # OSDR embeddings, gene-digest gated. Hours; resumable.
$PY precompute/build_projections.py                # PCA + UMAP coords, density rasters. ~5 min.
$PY precompute/fetch_archs4_meta.py                # ARCHS4 GEO metadata. ~35 s, needs network.
$PY precompute/validate_artifacts.py --mixing      # gates the build; exits nonzero on failure.
$PY app_manifold.py                                # http://127.0.0.1:8051
```

`embed_osdr.py` writes progress as it goes and resumes where it stopped, so an interrupted run does not restart from zero.
`build_projections.py` supports `--skip-umap` for a faster first pass, and `--density-only` to re-render just the rasters from cached coordinates when tuning the color ramp.

`fetch_archs4_meta.py` deserves a note, because the obvious route is a trap.
ARCHS4's per-sample metadata lives in gene-level HDF5 files that are 62.3 GB for human and 50.7 GB for mouse, which is why the ARCHS4 cloud stayed grey for so long.
Reading the same fields out of those files over HTTP range requests genuinely works, but costs roughly 5 minutes and 272 MB *per field*.
The Maayan Lab sigpy JSON API returns the same information in bulk: measured on the real corpus, 33.7 seconds, 39 requests, 216 MB, and 99.911% of all 940,455 accessions resolved.
The 839 that do not resolve are not GEO withdrawals - they are present in the release-matched v2.5 metadata and absent from the newer release the API serves, which disproves the assumption that ARCHS4 releases are append-only.
They get tissue "Unknown" rather than being dropped or guessed at.
The path to exactly 100% is documented in that script's docstring (the *versioned* metadata-only HDF5 files) and deliberately not taken, because 15x the build time to recover 0.089% of points is a bad trade for a color.

## Run it before the data exists

The real cache takes hours to build.
To exercise the interface immediately, build a synthetic corpus of the same shape:

```bash
PY=/Users/josh/Bridge-RNA/.venv/bin/python
$PY tests/build_dev_corpus.py --out /tmp/bm-dev --archs4 60000 --osdr 2000 --clean
BRIDGE_RNA_ROOT=/tmp/bm-dev/bridge_rna MANIFOLD_CACHE_DIR=/tmp/bm-dev/cache $PY app_manifold.py
```

The numbers are synthetic - shaped like the real corpus, with real cluster structure, but meaningless biologically.
It exists to test the instrument, not to be read.

Add `--no-archs4-meta` to build the same corpus *without* the ARCHS4 metadata join.
That is the degraded state a fresh clone starts in, and it is the fastest way to see what the coverage UI does about it: Tissue drops out of the whole-map group, reports OSDR-only coverage, and names the script that restores it.

## Tests

```bash
/Users/josh/Bridge-RNA/.venv/bin/python -m pytest tests/ -q
```

144 tests, about 0.55 s.
The suite builds its own synthetic corpus in a temp directory (4,000 ARCHS4 + 300 OSDR points) and never touches the 963 MB memmap or the checkpoint, so it runs on a machine that has neither.

The corpus is generated from known latent clusters with metadata derived from those clusters, which gives the render tests real category structure to assert against rather than noise.
Its synthetic `archs4_metadata.parquet` is deliberately written in ARCHS4's free-text register and mapped through the *real* canonicalizer, so `manifold/tissue.py` is tested against strings shaped like GEO's rather than against its own rules.
A `without_archs4_metadata` fixture runs the degraded path - the state a fresh clone starts in - so the coverage UI, the disabled menu entry, and the density fallback are covered rather than assumed.
