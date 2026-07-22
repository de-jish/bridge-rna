"""Artifact loaders with module-level caches.

The serving app loads only small precomputed artifacts: coordinate parquets, the
identity table, the OSDR label table, and the ARCHS4 GEO metadata join.
Nothing here imports torch or umap, and nothing here
opens the 963 MB ARCHS4 embedding memmap - the app draws a precomputed map, so
it never needs a 512-d vector at request time.

`projection_stats.json` is deliberately not loaded here. The app read it only to
place the density raster at its recorded extent, and with the raster gone the
file is a build record that `precompute/validate_artifacts.py` checks, not
something the app needs open.

Global point order is fixed as [ARCHS4 (0..N_ARCHS4-1), then OSDR
(N_ARCHS4..N-1)], matching build_projections.py. A point index `i` addresses
the same sample across every artifact.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from . import paths, tissue

METHODS = {
    "pca": {"2d": paths.COORDS_PCA2, "3d": paths.COORDS_PCA3},
    "umap": {"2d": paths.COORDS_UMAP2, "3d": paths.COORDS_UMAP3},
}


@lru_cache(maxsize=1)
def points_meta() -> pd.DataFrame:
    """dataset (0=archs4,1=osdr), src_index, species_id - one row per point."""
    return pd.read_parquet(paths.POINTS_META_PARQUET)


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


# --- ARCHS4 GEO metadata ----------------------------------------------------

def archs4_metadata_available() -> bool:
    """True once precompute/fetch_archs4_meta.py has cached the GEO join."""
    return paths.ARCHS4_METADATA_PARQUET.exists()


@lru_cache(maxsize=1)
def archs4_metadata() -> pd.DataFrame | None:
    if not archs4_metadata_available():
        return None
    df = pd.read_parquet(paths.ARCHS4_METADATA_PARQUET)
    if "global_index" in df.columns:
        df = df.sort_values("global_index").reset_index(drop=True)
    return df


@lru_cache(maxsize=1)
def archs4_tissue() -> np.ndarray | None:
    """Per-ARCHS4-point tissue bucket, or None if the join was never fetched.

    Already canonicalized by the precompute step, so it shares a vocabulary with
    `osdr_tissue` and the two can be coloured by one field.
    """
    df = archs4_metadata()
    if df is None or "tissue" not in df.columns:
        return None
    n_archs4, _, _ = counts()
    labels = df["tissue"].astype(str).to_numpy()
    if len(labels) < n_archs4:
        # A short join would silently shift every label; pad instead.
        labels = np.concatenate(
            [labels, np.full(n_archs4 - len(labels), tissue.UNKNOWN)])
    return labels[:n_archs4]


@lru_cache(maxsize=1)
def osdr_tissue() -> np.ndarray:
    """Per-OSDR-point tissue bucket, folded onto the shared vocabulary.

    OSDR's raw values are anatomically precise but hyper-specific ("Right
    extensor digitorum longus"). Canonicalizing here is what lets one "Tissue"
    color-by paint both corpora; all 48 raw values map to an anatomical bucket.
    """
    raw = osdr_field_values("tissue")
    return raw.map(tissue.canonical_tissue).to_numpy()


# --- Coordinates --------------------------------------------------------------

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
    # the legend.
    return df[field].astype(str).fillna("Unknown").reset_index(drop=True)
