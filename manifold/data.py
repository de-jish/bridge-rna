"""Artifact loaders with module-level caches.

The serving app loads only precomputed artifacts: coordinate parquets, the
identity table, OSDR metadata, the ARCHS4 memmap (touched only to pull 512-d
vectors for a lasso selection), and the hnswlib index. Nothing here imports
torch or umap.

Global point order is fixed as [ARCHS4 (0..N_ARCHS4-1), then OSDR
(N_ARCHS4..N-1)], matching build_projections.py. A point index `i` addresses
the same sample across every artifact.
"""

from __future__ import annotations

import json
from functools import lru_cache

import numpy as np
import pandas as pd

from . import paths

# OSDR-specific color-by fields (defined only for OSDR points).
OSDR_FIELDS = ["flight_status", "spaceflight", "tissue", "strain", "sex", "genotype",
               "study", "habitat", "duration", "diet"]
# Fields defined for both corpora.
SHARED_FIELDS = ["species"]

METHODS = {
    "pca": {"2d": paths.COORDS_PCA2, "3d": paths.COORDS_PCA3, "density": "pca2"},
    "umap": {"2d": paths.COORDS_UMAP2, "3d": paths.COORDS_UMAP3, "density": "umap2"},
}


@lru_cache(maxsize=1)
def stats() -> dict:
    if paths.PROJECTION_STATS_JSON.exists():
        return json.loads(paths.PROJECTION_STATS_JSON.read_text())
    return {}


@lru_cache(maxsize=1)
def points_meta() -> pd.DataFrame:
    """dataset (0=archs4,1=osdr), src_index, species_id - one row per point."""
    return pd.read_parquet(paths.CACHE_DIR / "points_meta.parquet")


@lru_cache(maxsize=1)
def counts() -> tuple[int, int, int]:
    m = points_meta()
    n_archs4 = int((m["dataset"] == 0).sum())
    n_osdr = int((m["dataset"] == 1).sum())
    return n_archs4, n_osdr, n_archs4 + n_osdr


@lru_cache(maxsize=1)
def osdr_metadata() -> pd.DataFrame:
    df = pd.read_parquet(paths.OSDR_METADATA_PARQUET)
    if "spaceflight" in df.columns:
        df["flight_status"] = df["spaceflight"].map(_flight_status)
    return df


def _flight_status(v: str) -> str:
    """Collapse the spaceflight arm onto the binary contrast Flight vs Ground.

    OSDR records seven distinct control arms (Ground Control, Basal Control,
    Vivarium Control, Cohort Control #1/#2, Ground Control Rerun, ...) and these
    are *not* interchangeable: a basal animal was sacrificed at experiment
    start, a vivarium animal never entered flight hardware. The raw arm is kept
    as its own color-by so that structure stays visible; this derived field
    exists only for the one question the corpus is built around - did the animal
    fly - and is deliberately the coarser of the two.
    """
    s = str(v).strip().lower()
    if not s or s in ("nan", "none", "na", "n/a", "unknown"):
        return "Unknown"
    if "flight" in s and "control" not in s:
        return "Space Flight"
    if any(k in s for k in ("control", "basal", "vivarium", "ground", "cohort")):
        return "Ground"
    return "Unknown"


@lru_cache(maxsize=1)
def archs4_geo() -> np.ndarray:
    return pd.read_parquet(paths.ARCHS4_GEO_PARQUET)["geo_accession"].to_numpy()


def archs4_tissue_available() -> bool:
    """True once precompute/fetch_archs4_meta.py has cached the tissue join."""
    return paths.ARCHS4_METADATA_PARQUET.exists()


@lru_cache(maxsize=1)
def archs4_tissue() -> np.ndarray | None:
    """Per-ARCHS4-point tissue labels, or None if the join was never built.

    The ARCHS4 gene HDF5 files are a tens-of-GB optional download, so this is
    the one color-by that can legitimately be missing. Returning None lets the
    caller fall back to a neutral cloud and say why, rather than failing.
    """
    if not archs4_tissue_available():
        return None
    n_archs4, _, _ = counts()
    df = pd.read_parquet(paths.ARCHS4_METADATA_PARQUET)
    if "global_index" in df.columns:
        df = df.sort_values("global_index")
    labels = df["tissue"].astype(str).to_numpy()
    if len(labels) < n_archs4:
        # A short join would silently shift every label; pad instead.
        labels = np.concatenate([labels, np.full(n_archs4 - len(labels), "Unknown")])
    return labels[:n_archs4]


@lru_cache(maxsize=8)
def coords(method: str, dims: str) -> np.ndarray:
    """(N, 2 or 3) float32 coordinate array for a method ('pca'|'umap')."""
    path = METHODS[method][dims]
    if not path.exists():
        return np.empty((0, int(dims[0])), dtype=np.float32)
    df = pd.read_parquet(path)
    return df.to_numpy(dtype=np.float32)


def method_available(method: str) -> bool:
    return METHODS[method]["2d"].exists()


@lru_cache(maxsize=1)
def _archs4_memmap() -> np.memmap:
    manifest = json.loads(paths.ARCHS4_MANIFEST.read_text())
    n, d = int(manifest["total_samples"]), int(manifest["embedding_dim"])
    return np.memmap(paths.ARCHS4_MMAP, dtype=np.float16, mode="r", shape=(n, d))


@lru_cache(maxsize=1)
def _osdr_embeddings() -> np.ndarray:
    return np.load(paths.OSDR_EMBEDDINGS_NPY).astype(np.float32)


def normalized_vectors(point_indices: np.ndarray) -> np.ndarray:
    """L2-normalized 512-d vectors for the given global point indices (mixed corpora)."""
    n_archs4, _, _ = counts()
    point_indices = np.asarray(point_indices, dtype=np.int64)
    out = np.empty((len(point_indices), 512), dtype=np.float32)

    is_osdr = point_indices >= n_archs4
    a_idx = point_indices[~is_osdr]
    o_idx = point_indices[is_osdr] - n_archs4

    if len(a_idx):
        mm = _archs4_memmap()
        # memmap fancy-indexing needs sorted access for speed; gather then place.
        order = np.argsort(a_idx)
        vecs = np.asarray(mm[a_idx[order]], dtype=np.float32)
        buf = np.empty((len(a_idx), 512), dtype=np.float32)
        buf[order] = vecs
        out[~is_osdr] = buf
    if len(o_idx):
        out[is_osdr] = _osdr_embeddings()[o_idx]

    norms = np.linalg.norm(out, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return out / norms


@lru_cache(maxsize=2)
def population_moments(corpus: str) -> tuple[np.ndarray, np.ndarray]:
    """Exact mean and covariance of a corpus's L2-normalized 512-d vectors.

    These are what the coherence null is built from, so they must describe the
    *whole* population, not a sample of it: an estimate from a subsample leaves
    a fixed offset between the estimated and true mean, and because the null's
    spread shrinks with selection size that offset turns into a z-score bias
    that grows without bound. See coherence._permutation_null.

    Computed by streaming the corpus once (a few seconds over the 963 MB ARCHS4
    memmap) and then cached to disk, so the cost is paid once per machine
    rather than once per lasso.
    """
    if corpus not in ("archs4", "osdr"):
        raise ValueError(f"unknown corpus {corpus!r}")

    path = paths.POPULATION_MOMENTS_NPZ
    if path.exists():
        with np.load(path) as z:
            key_mu, key_cov = f"{corpus}_mu", f"{corpus}_cov"
            if key_mu in z and key_cov in z:
                return z[key_mu], z[key_cov]

    moments = {}
    for name in ("archs4", "osdr"):
        moments[f"{name}_mu"], moments[f"{name}_cov"] = _stream_moments(name)
    try:
        np.savez(path, **moments)
    except OSError:
        pass  # a read-only cache is not a reason to fail a lasso
    return moments[f"{corpus}_mu"], moments[f"{corpus}_cov"]


def _stream_moments(corpus: str, chunk: int = 50000) -> tuple[np.ndarray, np.ndarray]:
    """One pass over a corpus accumulating sum and sum-of-outer-products."""
    n_archs4, n_osdr, _ = counts()
    if corpus == "archs4":
        lo, hi = 0, n_archs4
    else:
        lo, hi = n_archs4, n_archs4 + n_osdr

    total = hi - lo
    acc_sum = np.zeros(512, dtype=np.float64)
    acc_outer = np.zeros((512, 512), dtype=np.float64)
    for start in range(lo, hi, chunk):
        block = normalized_vectors(np.arange(start, min(start + chunk, hi)))
        block64 = block.astype(np.float64)
        acc_sum += block64.sum(axis=0)
        acc_outer += block64.T @ block64

    mu = acc_sum / total
    cov = acc_outer / total - np.outer(mu, mu)
    return mu, cov


@lru_cache(maxsize=1)
def hnsw_index():
    import hnswlib

    if not paths.HNSW_INDEX.exists():
        return None
    idx = hnswlib.Index(space="cosine", dim=512)
    idx.load_index(str(paths.HNSW_INDEX))
    idx.set_ef(80)
    return idx


# --- Color-by helpers -------------------------------------------------------

def species_labels() -> np.ndarray:
    """'human'/'mouse' per point over the full corpus."""
    sid = points_meta()["species_id"].to_numpy()
    return np.where(sid == 0, "human", "mouse")


def osdr_field_values(field: str) -> pd.Series:
    """The OSDR color-by values (length n_osdr), indexed 0..n_osdr-1.

    An unknown field yields all-"Unknown" rather than raising, so a color-by
    that was never populated by the precompute step degrades to a flat cloud
    instead of taking the app down.
    """
    df = osdr_metadata()
    if field not in df.columns:
        return pd.Series(["Unknown"] * len(df))
    # fillna after astype(str): pandas 3.0 keeps missing values as NA through a
    # string cast, and an NA leaking through here becomes a phantom category in
    # both the legend and the enrichment tests.
    return df[field].astype(str).fillna("Unknown").reset_index(drop=True)
