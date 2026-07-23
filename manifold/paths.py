"""Central path configuration for the manifold half of Bridge RNA.

The manifold consumes the model checkpoint, the ARCHS4 embedding memmap, and
the OSDR data read-only, and writes only into ``cache/``. Paths are resolved
once here so a relocation is a one-line change.

``BRIDGE_RNA_ROOT`` is still an independent knob rather than a synonym for the
repository root. The map used to live in a sibling repository and the two were
merged; the environment variable is what lets a build point at a *different*
checkout of the source artifacts - which is exactly what ``tests/`` and
``tests/build_dev_corpus.py`` do when they stand up a synthetic corpus.
"""

from __future__ import annotations

import os
from pathlib import Path

# This file is manifold/paths.py; the repository root is its parent's parent.
REPO_ROOT = Path(__file__).resolve().parent.parent

# Where the model checkpoint, the ARCHS4 embeddings, and the OSDR data live.
# Defaults to this repository; overridable so a build can read them elsewhere.
BRIDGE_RNA_ROOT = Path(os.environ.get("BRIDGE_RNA_ROOT", str(REPO_ROOT))).resolve()

# --- The manifold's own generated artifacts --------------------------------
# The cache location is overridable so a dev or test run can build a small
# throwaway corpus without touching the real multi-hour artifacts.
CACHE_DIR = Path(
    os.environ.get("MANIFOLD_CACHE_DIR", str(REPO_ROOT / "cache"))
).resolve()
ASSETS_DIR = REPO_ROOT / "assets"

OSDR_EMBEDDINGS_NPY = CACHE_DIR / "osdr_sample_embeddings.float32.npy"
OSDR_METADATA_PARQUET = CACHE_DIR / "osdr_metadata.parquet"

# Intermediates for the multi-hour embedding job: the log1p-TPM stage is cached
# so a re-embed never re-reads the counts CSVs, and the partial embedding plus
# its progress sidecar let an interrupted run resume instead of restarting.
OSDR_EXPRESSION_NPY = CACHE_DIR / "osdr_expression.float32.npy"
OSDR_EXPRESSION_META_PARQUET = CACHE_DIR / "osdr_expression_meta.parquet"
OSDR_EXPRESSION_KEY_JSON = CACHE_DIR / "osdr_expression_key.json"
OSDR_EMBEDDINGS_PARTIAL = CACHE_DIR / "osdr_sample_embeddings.partial.f32"
OSDR_EMBEDDINGS_PROGRESS = CACHE_DIR / "osdr_sample_embeddings.progress.json"

# Identity table and ARCHS4 accession sidecar, written by build_projections.py.
POINTS_META_PARQUET = CACHE_DIR / "points_meta.parquet"
ARCHS4_GEO_PARQUET = CACHE_DIR / "archs4_geo.parquet"

# Joint (OSDR + ARCHS4) projection coordinates, one parquet per method.
COORDS_PCA2 = CACHE_DIR / "coords_pca2.parquet"
COORDS_PCA3 = CACHE_DIR / "coords_pca3.parquet"
COORDS_UMAP2 = CACHE_DIR / "coords_umap2.parquet"
COORDS_UMAP3 = CACHE_DIR / "coords_umap3.parquet"
COORDS_TSNE2 = CACHE_DIR / "coords_tsne2.parquet"
COORDS_TSNE3 = CACHE_DIR / "coords_tsne3.parquet"

PROJECTION_STATS_JSON = CACHE_DIR / "projection_stats.json"


# --- Bridge RNA source artifacts (read-only) -------------------------------
CHECKPOINT = BRIDGE_RNA_ROOT / "checkpoints_performer" / "r7hnr92k" / "best_model.pt"

ARCHS4_DIR = BRIDGE_RNA_ROOT / "archs4_sample_embeddings_full"
ARCHS4_MMAP = ARCHS4_DIR / "sample_embeddings.float16.mmap"
ARCHS4_LOCATIONS = ARCHS4_DIR / "sample_locations.parquet"
ARCHS4_MANIFEST = ARCHS4_DIR / "embedding_manifest.json"

OSDR_DATA_DIR = BRIDGE_RNA_ROOT / "data" / "osdr"
OSDR_METADATA_TSV = OSDR_DATA_DIR / "metadata" / "selected_sample_metadata.tsv"
ORTHOLOGS_TXT = BRIDGE_RNA_ROOT / "data" / "ensembl" / "orthologs_one2one.txt"
MOUSE_EXON_LENGTHS_CSV = (
    BRIDGE_RNA_ROOT / "data" / "gencode" / "gencode_v49_mouse_gene_exon_lengths.csv"
)
CANONICAL_GENES_CSV = (
    BRIDGE_RNA_ROOT / "data" / "archs4" / "train_orthologs" / "canonical_genes.csv"
)

# Per-sample GEO metadata for the ARCHS4 corpus (series, title, source name,
# characteristics, and the derived tissue bucket), fetched over the network by
# precompute/fetch_archs4_meta.py. Optional: without it ARCHS4 colors by
# species only, and the app says so rather than showing a grey cloud.
ARCHS4_METADATA_PARQUET = CACHE_DIR / "archs4_metadata.parquet"


def ensure_cache_dirs() -> None:
    """Create the writable cache directory if it does not yet exist."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
