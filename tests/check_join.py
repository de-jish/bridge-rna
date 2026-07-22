#!/usr/bin/env python3
"""Prove, against the real corpus, that a retrieval marks the right points.

Deliberately not part of the pytest suite, and named so pytest does not collect
it. The suite runs against a synthetic fixture so it works on a machine with
neither the 963 MB memmap nor the multi-hour artifacts, and it already asserts
this contract there. This asserts it on the real 942,563-point corpus, where
the numbers are large enough that an off-by-one would be invisible by eye.

    /Users/josh/Bridge-RNA/.venv/bin/python tests/check_join.py [--sample KEY]

The claim under test is the one the whole merged app rests on, and it is
enforced by nothing but arithmetic:

  * an ARCHS4 hit's `archs4_index` is its row in the embedding memmap, and
    ARCHS4 occupies rows 0..n_archs4-1 of the map's global point order, so the
    row *is* the point;
  * an OSDR sample is the same `"<accession>|<sample name>"` key on both sides.

If either ever drifts, the map would keep drawing rings - just around the wrong
samples, confidently and silently. That is the failure this file exists to make
loud.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DEFAULT_SAMPLE = "OSD-100|Mmus_C57-6J_EYE_FLT_Rep1_M23"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sample", default=DEFAULT_SAMPLE,
                    help="OSDR sample key to retrieve.")
    ap.add_argument("--topk", type=int, default=10)
    args = ap.parse_args()

    from bridge_rna import config, osdr, retrieval
    from manifold import callbacks as mcb
    from manifold import data as mdata, paths as mpaths

    if not mpaths.POINTS_META_PARQUET.exists():
        print("No map cache on this machine; nothing to check against.\n"
              "Build it with precompute/build_projections.py.")
        return 0

    samples = osdr.load_osdr_samples(config.OSDR_METADATA_PATH)
    hits, mode = retrieval.search_hits(samples, args.sample, args.topk,
                                       enable_biopython_metadata=False)
    overlay = mcb._retrieval_overlay(
        {"sample_id": args.sample, "hits": hits.to_dict(orient="records")})
    if overlay is None:
        print(f"FAIL: no overlay produced for {args.sample} (mode={mode})")
        return 1

    n_archs4, n_osdr, total = mdata.counts()
    points_meta = mdata.points_meta()
    geo = pd.read_parquet(mpaths.ARCHS4_GEO_PARQUET)
    osdr_meta = mdata.osdr_metadata()

    print(f"corpus: {n_archs4:,} ARCHS4 + {n_osdr:,} OSDR = {total:,}")
    print(f"query : {args.sample}  (retrieval mode: {mode})\n")

    ok = True
    for rank, (point, gsm) in enumerate(
            zip(overlay["hit_points"], overlay["hit_labels"]), start=1):
        on_map = str(geo["geo_accession"].iloc[point])
        is_archs4 = int(points_meta["dataset"].iloc[point]) == 0
        good = on_map == gsm and is_archs4
        ok &= good
        print(f"  hit {rank:>2}: point {point:>7,}  map={on_map:<12} "
              f"retrieval={gsm:<12} {'OK' if good else 'MISMATCH'}")

    query = overlay["query_point"]
    if query is None:
        print("\nFAIL: the query has no point on the map")
        return 1
    q_is_osdr = int(points_meta["dataset"].iloc[query]) == 1
    q_key = str(osdr_meta["sample_key"].iloc[query - n_archs4])
    q_ok = q_is_osdr and q_key == args.sample
    ok &= q_ok
    print(f"\n  query : point {query:>7,}  map={q_key}")
    print(f"          {'OK' if q_ok else 'MISMATCH'}")

    print("\n" + "=" * 62)
    if ok:
        print("EVERY POINT ADDRESSES THE CORRECT SAMPLE")
        return 0
    print("JOIN IS BROKEN: the map would mark the wrong samples")
    return 1


if __name__ == "__main__":
    sys.exit(main())
