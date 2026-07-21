#!/usr/bin/env python3
"""Extract per-GSM tissue metadata for the ARCHS4 corpus into a parquet.

The local ARCHS4 artifacts carry only ``geo_accession`` and ``species_id``, so
until this step runs the 940,455-point background cloud can only be colored by
species. Tissue lives in the ARCHS4 gene-level HDF5 files, which are tens of GB
each and are downloaded separately from https://archs4.org/download.

    python precompute/fetch_archs4_meta.py

Output: ``cache/archs4_metadata.parquet`` with one row per ARCHS4 point, in the
same order as ``sample_locations.parquet`` (and therefore the same order as the
first N rows of every other Bridge Manifold artifact).

Why this reads the HDF5 directly rather than calling Bridge RNA's
``fetch_archs4_metadata``: that helper is built for retrieval, where a handful
of accessions come back from a top-k query, and it looks samples up by
accession. Asking it for all 940,455 accessions turns a bulk column read into
close to a million lookups. The metadata group is a set of plain 1-D string
datasets a few hundred MB in total, so this reads the columns whole and joins in
pandas. ``archs4py`` remains the fallback when a direct read is not possible.

This step is optional. Nothing else in the pipeline depends on it, and the
serving app checks for the parquet and degrades to species-only coloring with a
visible note when it is absent.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from manifold import paths  # noqa: E402

# Candidate dataset names inside the ARCHS4 HDF5 metadata group. ARCHS4 has
# moved these between releases, so each field lists the spellings seen in the
# wild and the first one present wins.
FIELD_CANDIDATES = {
    "geo_accession": ["meta/samples/geo_accession", "meta/samples/Sample_geo_accession"],
    "series": ["meta/samples/series_id", "meta/samples/Sample_series_id"],
    "source": ["meta/samples/source_name_ch1", "meta/samples/Sample_source_name_ch1"],
    "characteristics": ["meta/samples/characteristics_ch1",
                        "meta/samples/Sample_characteristics_ch1"],
    "title": ["meta/samples/title", "meta/samples/Sample_title"],
}

# Keys inside the free-text characteristics field that name a tissue.
TISSUE_KEYS = ("tissue", "organ", "source tissue", "tissue type", "cell type", "cell line")

_TISSUE_RE = re.compile(
    r"(?:^|;|\|)\s*(?:" + "|".join(re.escape(k) for k in TISSUE_KEYS) + r")\s*:\s*([^;|]+)",
    re.IGNORECASE,
)


def _decode(arr) -> np.ndarray:
    """HDF5 string datasets come back as bytes; normalize to a str array."""
    out = np.asarray(arr)
    if out.dtype.kind in ("S", "O"):
        return np.array([v.decode("utf-8", "replace") if isinstance(v, bytes) else str(v)
                         for v in out], dtype=object)
    return out.astype(str)


def read_h5_metadata(h5_path: Path, species_label: str) -> pd.DataFrame:
    """Read the metadata columns out of one ARCHS4 gene HDF5 file."""
    import h5py

    with h5py.File(str(h5_path), "r") as f:
        cols: dict[str, np.ndarray] = {}
        for field, candidates in FIELD_CANDIDATES.items():
            for key in candidates:
                if key in f:
                    cols[field] = _decode(f[key][:])
                    break
        if "geo_accession" not in cols:
            raise KeyError(
                f"{h5_path.name} has no recognizable sample accession dataset; "
                f"tried {FIELD_CANDIDATES['geo_accession']}"
            )
    n = len(cols["geo_accession"])
    df = pd.DataFrame({k: v for k, v in cols.items() if len(v) == n})
    df["species"] = species_label
    print(f"[meta] {h5_path.name}: {len(df):,} samples, columns {sorted(df.columns)}",
          flush=True)
    return df


def read_via_archs4py(h5_path: Path, accessions: list[str], species_label: str) -> pd.DataFrame:
    """Fallback path when h5py is unavailable but archs4py is."""
    import archs4py as a4

    df = a4.meta.samples(str(h5_path), accessions)
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()
    if "geo_accession" not in df.columns:
        df = df.reset_index().rename(columns={"index": "geo_accession"})
    df["species"] = species_label
    return df


def derive_tissue(df: pd.DataFrame) -> pd.Series:
    """Best-effort tissue label from the characteristics, then the source name.

    ARCHS4 has no curated tissue column; the information is embedded in GEO's
    free-text ``characteristics_ch1`` as ``key: value`` pairs. Parsing it is
    inherently lossy, so anything unrecognized becomes "Unknown" rather than a
    guess - an honestly empty label beats a confidently wrong one on a plot
    people will read biology off.
    """
    n = len(df)
    tissue = pd.Series(["Unknown"] * n, index=df.index, dtype=object)

    if "characteristics" in df.columns:
        extracted = df["characteristics"].astype(str).str.extract(_TISSUE_RE, expand=False)
        tissue = extracted.fillna(tissue)

    if "source" in df.columns:
        src = df["source"].astype(str).str.strip()
        blank = tissue.isna() | tissue.astype(str).str.strip().isin(["", "Unknown", "nan"])
        tissue = tissue.where(~blank, src)

    # fillna before the replaces: pandas 3.0 keeps missing values as NA through
    # astype(str), so they would never match the literal "nan" key below.
    tissue = (
        tissue.astype(str)
        .fillna("Unknown")
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
        .replace({"": "Unknown", "nan": "Unknown", "None": "Unknown", "NA": "Unknown"})
        .fillna("Unknown")
    )
    # Fold casing variants onto the most common spelling, as for OSDR fields.
    counts = tissue.value_counts()
    canonical: dict[str, str] = {}
    for label in counts.index:
        canonical.setdefault(str(label).casefold(), str(label))
    return tissue.map(lambda v: canonical.get(str(v).casefold(), str(v)))


def main() -> None:
    ap = argparse.ArgumentParser(description="Cache ARCHS4 per-GSM tissue metadata.")
    ap.add_argument("--human-h5", type=Path, default=paths.ARCHS4_HUMAN_H5)
    ap.add_argument("--mouse-h5", type=Path, default=paths.ARCHS4_MOUSE_H5)
    args = ap.parse_args()

    paths.ensure_cache_dirs()

    present = [(p, label) for p, label in
               ((args.human_h5, "human"), (args.mouse_h5, "mouse")) if p.exists()]
    if not present:
        raise SystemExit(
            "No ARCHS4 gene HDF5 file found. Expected at least one of:\n"
            f"  {args.human_h5}\n  {args.mouse_h5}\n"
            "These are tens of GB and are not bundled with either repository.\n"
            "Download them from https://archs4.org/download, then re-run.\n"
            "This step is optional: without it the app colors ARCHS4 by species "
            "only and says so in the UI."
        )

    loc = pd.read_parquet(paths.ARCHS4_LOCATIONS).sort_values("global_index")
    loc = loc.reset_index(drop=True)
    accessions = loc["geo_accession"].astype(str).tolist()
    print(f"[meta] {len(accessions):,} ARCHS4 accessions to annotate", flush=True)

    try:
        import h5py  # noqa: F401
        frames = [read_h5_metadata(p, label) for p, label in present]
    except ImportError:
        print("[meta] h5py unavailable; falling back to archs4py lookups", flush=True)
        try:
            import archs4py  # noqa: F401
        except ImportError:
            raise SystemExit(
                "Neither h5py nor archs4py is installed. Install one of them:\n"
                "  python -m pip install h5py        # preferred, bulk column read\n"
                "  python -m pip install archs4py    # fallback, per-accession lookup"
            )
        frames = [read_via_archs4py(p, accessions, label) for p, label in present]

    frames = [f for f in frames if not f.empty]
    if not frames:
        raise SystemExit("No metadata could be read from the ARCHS4 HDF5 files.")

    meta = pd.concat(frames, ignore_index=True)
    meta["geo_accession"] = meta["geo_accession"].astype(str).str.strip()
    meta = meta.drop_duplicates(subset="geo_accession", keep="first")
    meta["tissue"] = derive_tissue(meta)

    keep = [c for c in ("geo_accession", "tissue", "source", "series", "title", "species")
            if c in meta.columns]
    joined = loc[["global_index", "geo_accession"]].merge(
        meta[keep], on="geo_accession", how="left")
    joined["tissue"] = joined["tissue"].fillna("Unknown")

    matched = int((joined["tissue"] != "Unknown").sum())
    print(f"[meta] matched tissue for {matched:,}/{len(joined):,} "
          f"({matched / len(joined) * 100:.1f}%)", flush=True)
    top = joined["tissue"].value_counts().head(10)
    for label, count in top.items():
        print(f"        {label[:48]:50s} {count:,}", flush=True)

    assert len(joined) == len(loc), "the join changed the row count; ordering is not safe"
    joined.to_parquet(paths.ARCHS4_METADATA_PARQUET, index=False)
    print(f"[done] wrote {paths.ARCHS4_METADATA_PARQUET.name} ({len(joined):,} rows)",
          flush=True)


if __name__ == "__main__":
    main()
