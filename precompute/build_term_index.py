#!/usr/bin/env python3
"""Count how many GEO series use each word, and write cache/archs4_terms.parquet.

    .venv/bin/python precompute/build_term_index.py

Reads `cache/archs4_metadata.parquet` (written by fetch_archs4_meta.py), splits
every sample's title and source name with `bridge_rna.digest.words` - the same
tokenizer the app uses on a result set - and counts each word once per series.
About a minute on the full 940,455-sample corpus, and no network.

Then it checks itself and exits non-zero on failure: the series total must
equal the number of distinct series (plus samples with none), every count must
be between 2 and that total, and a handful of words any GEO corpus uses must
be present. Design: docs/match-digest.md.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from bridge_rna import digest  # noqa: E402
from manifold import paths  # noqa: E402

# Words any real GEO metadata dump contains in more than a few series. A table
# missing one of them was built from the wrong file or the wrong fields.
SPOT_CHECK = ("liver", "brain", "heart", "kidney", "lung")


def check(frame: pd.DataFrame, metadata: pd.DataFrame) -> list[str]:
    """Every reason the table is wrong; empty when it is right."""
    problems = []
    table = digest.table_from_frame(frame)
    if table is None:
        return ["the __series__ total row is missing"]
    has_series = metadata["series_id"].fillna("").astype(str).str.strip() != ""
    expected_total = (metadata.loc[has_series, "series_id"].nunique()
                      + int((~has_series).sum()))
    if table.total_series != expected_total:
        problems.append(f"series total {table.total_series:,} != "
                        f"{expected_total:,} distinct series in the metadata")
    counts = list(table.counts.values())
    if counts and (min(counts) < digest.MIN_SERIES or max(counts) > table.total_series):
        problems.append("a word count falls outside [MIN_SERIES, total series]")
    if len(metadata) >= 100_000:  # the fixture corpus is too small for this
        missing = [w for w in SPOT_CHECK if w not in table.counts]
        if missing:
            problems.append(f"common words missing: {', '.join(missing)}")
    return problems


def main() -> int:
    source = paths.ARCHS4_METADATA_PARQUET
    if not source.exists():
        print(f"[terms] {source} is missing. Run precompute/fetch_archs4_meta.py first.",
              file=sys.stderr)
        return 1
    t0 = time.time()
    columns = ["geo_accession", "series_id", *digest.TEXT_FIELDS]
    metadata = pd.read_parquet(source, columns=columns)
    frame = digest.build_term_table(metadata)
    problems = check(frame, metadata)
    if problems:
        for p in problems:
            print(f"[terms] FAIL {p}", file=sys.stderr)
        return 1
    target = paths.ARCHS4_TERMS_PARQUET
    tmp = target.with_suffix(".parquet.tmp")
    frame.to_parquet(tmp, index=False)
    tmp.replace(target)
    table = digest.table_from_frame(frame)
    print(f"[terms] {len(table.counts):,} words over {table.total_series:,} series "
          f"from {len(metadata):,} samples in {time.time() - t0:.1f}s; "
          f"wrote {target} ({target.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
