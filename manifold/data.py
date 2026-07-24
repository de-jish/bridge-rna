"""Artifact loaders with module-level caches.

The serving app loads only small precomputed artifacts: coordinate parquets, the
identity table, the OSDR label table, the ARCHS4 GEO metadata join, and the
build record that says how the coordinates were fit.
Nothing here imports torch or umap, and nothing here opens the 963 MB ARCHS4
embedding memmap - the map draws precomputed coordinates, so it never needs a
512-d vector at request time. The retrieval half does open that memmap, on
every cached search; this module is not on that path.

`projection_stats.json` is read here, but only as a build *record* - never for
coordinates. It was dropped from this module once, when the density raster that
placed itself at the recorded extent was removed and nothing else needed the
file. The projection-parameter readout on the control rail brought it back: the
rail states the settings that produced the coordinates on screen, and that
answer has to come from what the build actually wrote rather than from constants
duplicated in the serving code, which is how a rail ends up confidently
describing a build it is not showing.

Global point order is fixed as [ARCHS4 (0..N_ARCHS4-1), then OSDR
(N_ARCHS4..N-1)], matching build_projections.py. A point index `i` addresses
the same sample across every artifact.
"""

from __future__ import annotations

import json
from functools import lru_cache

import numpy as np
import pandas as pd

from . import paths, tissue

# Every projection the app can draw, and the parquet each one lives in.
# `coords()` caches one array per (method, dims) pair, so this dict's length is
# what sizes that cache; the order the control rail offers them in is
# `layout.METHOD_LABELS`, not this.
METHODS = {
    "pca": {"2d": paths.COORDS_PCA2, "3d": paths.COORDS_PCA3},
    "umap": {"2d": paths.COORDS_UMAP2, "3d": paths.COORDS_UMAP3},
    "tsne": {"2d": paths.COORDS_TSNE2, "3d": paths.COORDS_TSNE3},
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
    """True if this projection can be drawn at all, i.e. its 2-D coords exist.

    Deliberately 2-D only, because that is what the Projection pill gates and a
    build that produced a 2-D map should offer it. Whether a *particular*
    dimensionality exists is `coords_available`.
    """
    return METHODS[method]["2d"].exists()


def coords_available(method: str, dims: str) -> bool:
    """True if this exact (method, dims) coordinate set is on disk.

    The two are not the same question, and the gap between them is reachable
    rather than theoretical: the 3-D t-SNE fit is 81% of the build's wall clock,
    and every stage writes its parquet and saves the stats record before the
    next one starts. An interrupt anywhere in those two hours leaves
    coords_tsne2.parquet and a *complete* tsne_* record on disk with no
    coords_tsne3.parquet. The pill should stay enabled, since the 2-D map is
    real; the parameter readout must not describe the 3-D fit that never
    finished.
    """
    path = METHODS.get(method, {}).get(dims)
    return bool(path and path.exists())


@lru_cache(maxsize=1)
def projection_stats() -> dict:
    """The build record: how each projection was fit, and with what parameters.

    Returns ``{}`` when the file is absent or unreadable rather than raising.
    This feeds a label on the control rail, not a correctness gate, so a cache
    built before a stage existed shows fewer parameters instead of taking the
    map down - the same degradation `osdr_field_values` chooses for a colour-by
    the precompute step never populated. `precompute/validate_artifacts.py` is
    where a missing or malformed record is an error.
    """
    if not paths.PROJECTION_STATS_JSON.exists():
        return {}
    try:
        loaded = json.loads(paths.PROJECTION_STATS_JSON.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


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
