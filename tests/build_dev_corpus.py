#!/usr/bin/env python3
"""Build a browsable synthetic corpus so the app can be run before real data exists.

The real cache takes hours to produce (a multi-hour embedding job, then UMAP over
940k points). This builds the same artifacts at whatever scale is asked for, in
seconds, so the interface can be exercised and reviewed end to end while the real
precompute runs.

    python tests/build_dev_corpus.py --out /tmp/bm-dev --archs4 60000 --osdr 2000
    MANIFOLD_CACHE_DIR=/tmp/bm-dev/cache BRIDGE_RNA_ROOT=/tmp/bm-dev/bridge_rna \\
        python app_manifold.py

The data is synthetic. It is shaped like the real corpus - same files, same
dtypes, same global point order, real cluster structure - but the numbers mean
nothing biologically. It exists to test the instrument, not to be read.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import fixture_corpus  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a synthetic Bridge Manifold corpus.")
    ap.add_argument("--out", type=Path, required=True, help="Directory to build into.")
    ap.add_argument("--archs4", type=int, default=60000)
    ap.add_argument("--osdr", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--clean", action="store_true", help="Remove --out first.")
    ap.add_argument("--no-archs4-meta", action="store_true",
                    help="Omit the ARCHS4 metadata join, to see the degraded UI "
                         "a fresh clone starts in: Tissue drops to OSDR-only and "
                         "the coverage readout names the script that fixes it.")
    args = ap.parse_args()

    if args.clean and args.out.exists():
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True, exist_ok=True)

    desc = fixture_corpus.build_all(args.out, n_archs4=args.archs4, n_osdr=args.osdr,
                                    seed=args.seed,
                                    with_archs4_meta=not args.no_archs4_meta)
    print(f"built {desc['total']:,} points "
          f"({desc['n_archs4']:,} ARCHS4 + {desc['n_osdr']:,} OSDR)")
    print(f"  BRIDGE_RNA_ROOT={desc['bridge_rna_root']}")
    print(f"  MANIFOLD_CACHE_DIR={desc['cache_dir']}")


if __name__ == "__main__":
    main()
