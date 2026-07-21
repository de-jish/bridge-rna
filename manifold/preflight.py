"""Preflight guards: fail clearly on missing or unresolved LFS artifacts.

The checkpoint and the 963 MB ARCHS4 memmap are Git LFS objects in the Bridge
RNA repository. On a fresh checkout they can arrive as ~130-byte text stubs
rather than the real binary, and every downstream failure would then be
mysterious. These guards turn that into one legible error at startup.

Only the precompute scripts touch those LFS objects. The serving app reads its
own cache and nothing else, which is why the two required-artifact lists have
no overlap.
"""

from __future__ import annotations

from pathlib import Path

from . import paths

_LFS_SIGNATURE = b"version https://git-lfs.github.com/spec/v1"


def is_lfs_pointer(path: Path) -> bool:
    """True if the file looks like an unresolved Git LFS pointer stub."""
    try:
        if path.stat().st_size > 4096:
            return False  # real binaries are far larger than a pointer stub
        with open(path, "rb") as fh:
            return fh.read(len(_LFS_SIGNATURE)) == _LFS_SIGNATURE
    except OSError:
        return False


def check_artifacts(required: list[tuple[str, Path]]) -> list[str]:
    """Return a list of human-readable problems for the given (label, path) pairs."""
    problems: list[str] = []
    for label, path in required:
        if not path.exists():
            problems.append(f"missing {label}: {path}")
        elif is_lfs_pointer(path):
            problems.append(
                f"{label} is an unresolved Git LFS pointer (run `git lfs pull` in "
                f"the Bridge RNA repo): {path}"
            )
    return problems


PRECOMPUTE_REQUIRED = [
    ("model checkpoint", paths.CHECKPOINT),
    ("ARCHS4 embedding memmap", paths.ARCHS4_MMAP),
    ("ARCHS4 sample locations", paths.ARCHS4_LOCATIONS),
    ("ARCHS4 manifest", paths.ARCHS4_MANIFEST),
    ("OSDR metadata TSV", paths.OSDR_METADATA_TSV),
    ("ortholog map", paths.ORTHOLOGS_TXT),
    ("mouse exon lengths", paths.MOUSE_EXON_LENGTHS_CSV),
    ("canonical gene list", paths.CANONICAL_GENES_CSV),
]

# What the serving app genuinely opens. It draws a precomputed map and never
# needs a 512-d vector, so neither the 963 MB memmap nor the OSDR embeddings
# appear here - listing them made a machine with only the cache fail preflight
# for artifacts it would never have read.
#
# points_meta is first because it is read first: layout.control_rail() calls
# data.counts() while the layout is still being built, so a missing identity
# table used to sail past preflight and crash during startup instead.
APP_REQUIRED = [
    ("point identity table", paths.POINTS_META_PARQUET),
    ("OSDR metadata", paths.OSDR_METADATA_PARQUET),
    ("PCA 2D coordinates", paths.COORDS_PCA2),
]


def require(required: list[tuple[str, Path]], context: str) -> None:
    """Raise a single aggregated error if any required artifact is unusable."""
    problems = check_artifacts(required)
    if problems:
        bullet = "\n  - ".join(problems)
        raise SystemExit(
            f"Bridge Manifold preflight failed for {context}:\n  - {bullet}"
        )
