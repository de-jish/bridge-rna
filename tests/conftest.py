"""Pytest bootstrap: point Bridge Manifold at a synthetic corpus before import.

``manifold.paths`` resolves its constants at import time from ``BRIDGE_RNA_ROOT``
and ``MANIFOLD_CACHE_DIR``. Both are therefore set here, at conftest import,
which runs before any test module imports the package. The corpus is built once
per session into a temp directory and torn down with it, so the suite never
touches the real 963 MB memmap or the multi-hour embedding artifacts and can run
on a machine that has neither.
"""

from __future__ import annotations

import atexit
import os
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS_DIR.parent))
sys.path.insert(0, str(TESTS_DIR))

import fixture_corpus  # noqa: E402

_FIXTURE_ROOT = Path(tempfile.mkdtemp(prefix="bridge-manifold-fixture-"))
atexit.register(shutil.rmtree, _FIXTURE_ROOT, True)

_DESC = fixture_corpus.build_all(_FIXTURE_ROOT, n_archs4=4000, n_osdr=300)

os.environ["BRIDGE_RNA_ROOT"] = str(_DESC["bridge_rna_root"])
os.environ["MANIFOLD_CACHE_DIR"] = str(_DESC["cache_dir"])


@pytest.fixture(scope="session")
def corpus() -> dict:
    """Descriptor for the synthetic corpus: sizes, coords, cluster ground truth."""
    return _DESC


@pytest.fixture
def without_archs4_metadata(monkeypatch):
    """Run a test as if the optional GEO metadata fetch had never been run.

    This is the degraded state the coverage UI exists for, and it is the state a
    fresh clone starts in, so it needs real coverage rather than being assumed.
    Pointing the path at a non-existent file is enough: every reader gates on
    ``.exists()``. The caches must be cleared on both sides of the patch, since
    they would otherwise carry the metadata in or out of the test.

    ``render._colour_plan`` belongs in this list for the same reason the two
    loaders do: it memoizes a label array derived from the metadata, so leaving
    it warm would let a test see Tissue colouring ARCHS4 in a state where the
    join does not exist. Every cache that stands downstream of an artifact this
    fixture hides has to be cleared here.
    """
    from manifold import colorby, data, paths, render

    def clear():
        data.archs4_metadata.cache_clear()
        data.archs4_tissue.cache_clear()
        render._colour_plan.cache_clear()

    clear()
    monkeypatch.setattr(paths, "ARCHS4_METADATA_PARQUET",
                        paths.CACHE_DIR / "does-not-exist.parquet")
    yield colorby
    clear()
