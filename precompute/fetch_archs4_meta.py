#!/usr/bin/env python3
"""Fetch per-sample GEO metadata for the 940,455 ARCHS4 points.

    python precompute/fetch_archs4_meta.py

Output: ``cache/archs4_metadata.parquet``, one row per ARCHS4 point in the fixed
global order, carrying ``geo_accession``, ``series_id``, ``title``,
``source_name``, ``characteristics`` and the derived ``tissue`` bucket.

Why this does not read the ARCHS4 HDF5 files
--------------------------------------------
The obvious route - and the one this script used to take - is the gene-level
HDF5 files from archs4.org, where per-sample metadata lives in small 1-D
datasets under ``meta/samples/``. That route works but is a bad trade, and the
alternatives were measured rather than assumed:

  * Full download: the current human build is 62.3 GB and mouse 50.7 GB. For a
    few hundred MB of strings. This is what kept the ARCHS4 cloud grey.
  * Partial read over HTTP range requests (fsspec + h5py): genuinely works, and
    the whole ``meta/samples`` group is enumerable in ~18 s. But the fields are
    gzip-chunked vlen strings, so one field costs ~5 min and ~272 MB of
    transfer; the six useful fields across both species run to hours.
  * The Maayan Lab sigpy JSON API, used here: **35 s, 39 requests, 216 MB, and
    99.9% of the local corpus.** Three orders of magnitude better than either
    HDF5 route for exactly the same information.

Version skew was the risk worth checking, since these embeddings came from the
older ARCHS4 v2.5 build. Measured coverage against the current release is
99.911% (human 99.851%, mouse 99.982%), confirmed independently by a partial
HDF5 read of ``meta/samples/geo_accession`` off the 62 GB remote file - both
methods return exactly 509,949 human matches.

The 839 unresolved samples are *not* GEO withdrawals, and it is worth being
precise about that because the obvious explanation is wrong: they are present
with full metadata in the release-matched v2.5 metadata files, and absent from
the newer, larger v2.latest that this API serves. ARCHS4 releases are therefore
not append-only - a rebuild can drop samples. Those 839 points get an
``Unknown`` tissue and an empty series rather than being dropped or guessed at,
which is 0.089% of the corpus.

Closing that last 0.089% is possible but not worth it. ARCHS4 publishes
metadata-only HDF5 files under *versioned* names - ``human_meta_v2.5.h5``
(311.8 MB) and ``mouse_meta_v2.5.h5`` (350.9 MB); the unversioned "latest"
spellings 403, which is why they are easy to miss. Reading those gives exactly
100.000% coverage against this corpus and is release-matched. The cost is 663 MB
and roughly 8.5 minutes against 216 MB and 35 seconds, plus an h5py dependency
the serving app does not otherwise need. For a colour-by, 0.089% of points
reading "Unknown" is not worth 15x the build time; if this ever needs to be a
gate rather than a colour, switch to the versioned files and assert 100%.

This step needs a network connection and nothing else - no h5py, no archs4py,
no multi-GB download. It is still optional: without it the app colors ARCHS4 by
species and by unsupervised embedding cluster, and the tissue color-by is
offered but disabled with the reason attached.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from manifold import paths, tissue as tissue_map  # noqa: E402

API_URL = "https://maayanlab.cloud/sigpy/meta/samplemeta"
# Measured: 25,000 accessions per request sustains ~24k-42k samples/s with no
# rate limiting observed, and the whole corpus fits in 39 requests.
BATCH = 25_000
RETRIES = 3
TIMEOUT = 600

# The API returns these four keys per accession.
API_FIELDS = ("series", "title", "source", "characteristics")

# A response that resolves almost nothing means the request was malformed, not
# that GEO lost the samples. The endpoint answers HTTP 200 with an empty object
# when the payload key is wrong (`gsm_ids` instead of `samples`), which would
# otherwise write a fully-empty metadata table and silently grey the map.
MIN_HIT_RATE = 0.5

# Keys inside GEO's free-text characteristics that name a tissue, best evidence
# first. `cell type` is deliberately last: it often carries a cell-line name
# where `tissue` would have carried an organ.
TISSUE_KEYS = ("tissue", "organ", "organ part", "source tissue", "tissue type",
               "tissue region", "anatomical site", "body site", "cell type",
               "celltype", "cell line")


def log(msg: str) -> None:
    print(f"[meta] {msg}", flush=True)


def fetch_species(session, species: str, accessions: np.ndarray) -> dict[str, dict]:
    """POST every accession for one species, in batches, with retries."""
    out: dict[str, dict] = {}
    total = len(accessions)
    if total == 0:
        return out
    t0 = time.time()
    for start in range(0, total, BATCH):
        chunk = [str(a) for a in accessions[start:start + BATCH]]
        payload = {"species": species, "samples": chunk}
        for attempt in range(1, RETRIES + 1):
            try:
                resp = session.post(API_URL, json=payload, timeout=TIMEOUT)
                resp.raise_for_status()
                body = resp.json()
                if not isinstance(body, dict):
                    raise ValueError(f"expected a JSON object, got {type(body).__name__}")
                out.update(body)
                break
            except Exception as exc:  # noqa: BLE001 - retried, then re-raised
                if attempt == RETRIES:
                    raise SystemExit(
                        f"ABORT: {species} batch at offset {start} failed after "
                        f"{RETRIES} attempts: {exc}\n"
                        "The ARCHS4 metadata step needs network access to "
                        f"{API_URL}. It is optional - the app runs without it."
                    ) from exc
                wait = 2 ** attempt
                log(f"  retry {attempt}/{RETRIES} for {species} offset {start} "
                    f"in {wait}s ({exc})")
                time.sleep(wait)
        log(f"  {species}: {min(start + BATCH, total):,}/{total:,} "
            f"({time.time() - t0:.1f}s)")
    return out


def first_series(value: object) -> str:
    """The primary GSE for a sample.

    210,217 samples belong to several series and the API returns them
    comma-separated. The first is the submitting series, which is the one that
    defines the batch.

    The isna guard is load-bearing, not defensive: the 839 accessions the API
    cannot resolve arrive here as float NaN, and `value or ""` does not catch
    them because NaN is truthy. Without it they became the literal string "nan",
    which reads as a real GSE, becomes a phantom series, and overstates coverage
    to a clean 100%.
    """
    if value is None or (isinstance(value, float) and value != value):
        return ""
    s = str(value).strip()
    if not s or s.lower() in ("nan", "none"):
        return ""
    return s.split(",")[0].strip()


def _characteristics_value(text: str, key: str) -> str:
    """Pull ``key: value`` out of GEO's comma/semicolon-delimited characteristics.

    The field is free text with no escaping, so this is deliberately simple:
    find the key at a delimiter boundary, take everything up to the next
    delimiter. Anything ambiguous falls through to the next key and ultimately
    to source_name.
    """
    low = text.lower()
    at = 0
    while True:
        at = low.find(key + ":", at)
        if at < 0:
            return ""
        # Must sit at the start or just after a delimiter, so "cell type:" does
        # not match inside "single cell type:".
        if at == 0 or low[at - 1] in ",;|":
            tail = text[at + len(key) + 1:]
            for delim in (",", ";", "|"):
                cut = tail.find(delim)
                if cut >= 0:
                    tail = tail[:cut]
            return tail.strip()
        at += 1


def derive_tissue(df: pd.DataFrame) -> pd.Series:
    """Canonical tissue bucket per sample, from characteristics then source_name.

    Both corpora go through ``manifold.tissue``, so an ARCHS4 liver and an OSDR
    "Left Lobe of the Liver" land in the same bucket and one color-by can paint
    the whole map.
    """
    chars = df["characteristics"].astype(str).fillna("")
    source = df["source_name"].astype(str).fillna("")

    # Extract each candidate key once across the whole column, then let
    # coalesce_tissue pick the best-evidence non-empty answer per row.
    columns = [chars.map(lambda t, k=key: _characteristics_value(t, k))
               for key in TISSUE_KEYS]
    columns.append(source)

    stacked = pd.concat(columns, axis=1)
    return pd.Series(
        [tissue_map.coalesce_tissue(*row) for row in stacked.itertuples(index=False)],
        index=df.index, dtype=object,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Cache ARCHS4 per-sample GEO metadata.")
    ap.add_argument("--limit", type=int, default=0,
                    help="Debug: only annotate the first N ARCHS4 points.")
    args = ap.parse_args()

    try:
        import requests
    except ImportError:
        raise SystemExit("ABORT: this step needs `requests` (python -m pip install requests).")

    paths.ensure_cache_dirs()
    if not paths.ARCHS4_GEO_PARQUET.exists() or not paths.POINTS_META_PARQUET.exists():
        raise SystemExit(
            "ABORT: run precompute/build_projections.py first - this step joins "
            f"onto {paths.ARCHS4_GEO_PARQUET.name} and {paths.POINTS_META_PARQUET.name}."
        )

    geo = pd.read_parquet(paths.ARCHS4_GEO_PARQUET)["geo_accession"].astype(str).to_numpy()
    pmeta = pd.read_parquet(paths.POINTS_META_PARQUET)
    species_id = pmeta.loc[pmeta["dataset"] == 0, "species_id"].to_numpy()
    if len(species_id) != len(geo):
        raise SystemExit(
            f"ABORT: {paths.ARCHS4_GEO_PARQUET.name} has {len(geo):,} rows but "
            f"points_meta marks {len(species_id):,} ARCHS4 points. These are "
            "joined positionally; rebuild both with build_projections.py."
        )
    if args.limit:
        geo, species_id = geo[:args.limit], species_id[:args.limit]

    log(f"{len(geo):,} ARCHS4 accessions to annotate via {API_URL}")

    t0 = time.time()
    session = requests.Session()
    records: dict[str, dict] = {}
    for sid, species in ((0, "human"), (1, "mouse")):
        subset = geo[species_id == sid]
        log(f"{species}: {len(subset):,} accessions")
        records.update(fetch_species(session, species, subset))

    hit_rate = len(records) / max(len(geo), 1)
    log(f"fetched {len(records):,} records in {time.time() - t0:.1f}s "
        f"({hit_rate * 100:.3f}% of requested)")
    if hit_rate < MIN_HIT_RATE:
        raise SystemExit(
            f"ABORT: the API resolved only {hit_rate * 100:.1f}% of the requested "
            "accessions. That is a malformed request, not missing data - the "
            "payload key must be `samples`. Nothing was written."
        )

    # Reindex onto the fixed global order. Never trust response order: the
    # returned object silently omits misses, so positional assembly would shift
    # every label after the first gap.
    frame = pd.DataFrame(
        [records.get(acc) or {} for acc in geo],
        index=pd.RangeIndex(len(geo)),
        columns=list(API_FIELDS),
    )
    out = pd.DataFrame({
        "global_index": np.arange(len(geo), dtype=np.int32),
        "geo_accession": geo,
        "series_id": frame["series"].map(first_series).astype(str),
        "title": frame["title"].fillna("").astype(str),
        "source_name": frame["source"].fillna("").astype(str),
        "characteristics": frame["characteristics"].fillna("").astype(str),
    })

    log("deriving canonical tissue buckets")
    out["tissue"] = derive_tissue(out)

    assert len(out) == len(geo), "the join changed the row count; ordering is not safe"

    matched = int((out["series_id"] != "").sum())
    placed = int((~out["tissue"].isin([tissue_map.UNKNOWN, tissue_map.OTHER])).sum())
    log(f"metadata resolved for {matched:,}/{len(out):,} ({matched / len(out) * 100:.3f}%)")
    log(f"tissue placed in an anatomical bucket for {placed:,}/{len(out):,} "
        f"({placed / len(out) * 100:.1f}%)")
    log(f"{out['series_id'].nunique():,} distinct GEO series")
    log("top tissue buckets:")
    for label, count in out["tissue"].value_counts().head(15).items():
        log(f"        {str(label)[:32]:34s} {count:>9,}  {count / len(out) * 100:5.1f}%")

    out.to_parquet(paths.ARCHS4_METADATA_PARQUET, index=False)
    log(f"[done] wrote {paths.ARCHS4_METADATA_PARQUET.name} ({len(out):,} rows, "
        f"{paths.ARCHS4_METADATA_PARQUET.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
