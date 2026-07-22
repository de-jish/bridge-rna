"""Central path configuration for Bridge Manifold.

Every load-bearing artifact lives in the Bridge RNA repository; Bridge Manifold
consumes them read-only and writes only into its own ``cache/`` directory. Paths
are resolved once here so a relocation of either repo is a one-line change.
"""

from __future__ import annotations

import os
from pathlib import Path

# This file is manifold/paths.py; the project root is its parent's parent.
MANIFOLD_ROOT = Path(__file__).resolve().parent.parent

# The Bridge RNA repository. Overridable via env for portability.
BRIDGE_RNA_ROOT = Path(
    os.environ.get("BRIDGE_RNA_ROOT", "/Users/josh/Bridge-RNA")
).resolve()

# --- Bridge Manifold's own generated artifacts -----------------------------
# The cache location is overridable so a dev or test run can build a small
# throwaway corpus without touching the real multi-hour artifacts.
CACHE_DIR = Path(
    os.environ.get("MANIFOLD_CACHE_DIR", str(MANIFOLD_ROOT / "cache"))
).resolve()
ASSETS_DIR = MANIFOLD_ROOT / "assets"

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
