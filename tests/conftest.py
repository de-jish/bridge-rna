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


@pytest.fixture(scope="session")
def osdr_cluster(corpus):
    """Latent cluster id per OSDR point - the ground truth the tests assert against."""
    return corpus["osdr_cluster"]
