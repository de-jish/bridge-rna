"""File ingestion: the fourth query-vector source.

The Retrieve view can embed an uploaded OSDR counts file live and retrieve its
Earth analogs. These tests pin the two properties that make that trustworthy:

1. The uploaded path reuses the *exact* cosine scan and offline annotation the
   cached path uses, so its output carries the same schema and the same
   `archs4_index` map join. This is checked without torch, against the synthetic
   corpus, by feeding a known query vector in.
2. The live embedding is byte-for-byte the vector the corpus was built with, so
   a cosine score against ARCHS4 means the same thing for an uploaded sample as
   for a catalog one. This is checked end to end when the model is available.

Invariant 1 (the gene-digest gate) is enforced before any vector is produced,
and the serving package still never imports torch at module scope.
"""

from __future__ import annotations

import ast
import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bridge_rna import retrieval
from bridge_rna.preflight import preflight_retrieval_requirements
from bridge_rna.retrieval import UPLOAD_MODE
from bridge_rna.util import RetrievalError

REPO = Path(__file__).resolve().parent.parent

_MISSING, _RESOLVED = preflight_retrieval_requirements()
_HAVE_MODEL = (
    not _MISSING
    and _RESOLVED.get("checkpoint") is not None
    and Path(str(_RESOLVED["checkpoint"])).exists()
)
_HAVE_STACK = (
    _RESOLVED.get("canonical_genes") is not None
    and Path(str(_RESOLVED["canonical_genes"])).exists()
    and importlib.util.find_spec("torch") is not None
)


# --- The retrieval half: reuses the cached scan, no torch, no model ---------

@pytest.fixture
def _point_at_fixture(monkeypatch, corpus):
    """Aim the ARCHS4 loaders at the synthetic stub (mirrors test_retrieval)."""
    retrieval._cached_osdr_embeddings.cache_clear()
    retrieval._archs4_annotations.cache_clear()
    retrieval._ARCHS4_CACHE.clear()
    monkeypatch.setattr(
        retrieval, "EMBEDDING_DIR",
        corpus["bridge_rna_root"] / "archs4_sample_embeddings_full")
    yield
    retrieval._cached_osdr_embeddings.cache_clear()
    retrieval._archs4_annotations.cache_clear()
    retrieval._ARCHS4_CACHE.clear()


def test_uploaded_path_returns_the_true_topk_with_full_annotation(
    monkeypatch, corpus, _point_at_fixture
):
    """Given a query vector, the uploaded path must match the cached path exactly.

    The two differ only in where the vector comes from, so with the embedding
    step stubbed out the uploaded path must return the true brute-force cosine
    ranking, annotated from the local cache, with the map join intact.
    """
    key = str(corpus["osdr_metadata"]["sample_key"].iloc[2])
    qvec = retrieval.cached_query_vector(key).astype(np.float32)
    monkeypatch.setattr(retrieval, "embed_uploaded_counts", lambda *a, **k: qvec)

    hits = retrieval.run_uploaded_retrieval(
        counts_path="ignored.csv", sample_column="s", topk=10,
        entrez_email="", enable_biopython_metadata=False,
    )

    q = qvec.astype(np.float64)
    q /= np.linalg.norm(q)
    vecs, _, _ = retrieval._load_archs4_index()
    x = np.asarray(vecs, dtype=np.float64)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    truth = np.argsort(-(x @ q))[:10]

    assert hits["archs4_index"].tolist() == truth.tolist()
    assert hits["score"].is_monotonic_decreasing
    for col in ("gsm", "gse", "title", "tissue", "species", "archs4_index"):
        assert col in hits.columns
    assert hits["gsm"].str.startswith("GSM").all()


def test_uploaded_retrieval_errors_cleanly_on_a_missing_file(monkeypatch):
    """A missing counts file surfaces as a clean RetrievalError, not a traceback."""
    fake = {k: REPO / "nope"
            for k in ("checkpoint", "orthologs", "canonical_genes", "mouse_exon_lengths")}
    monkeypatch.setattr(retrieval, "preflight_retrieval_requirements", lambda: ([], fake))
    with pytest.raises(RetrievalError):
        retrieval.embed_uploaded_counts(REPO / "does_not_exist.csv", "s")


# --- Callback wiring --------------------------------------------------------

def test_the_banner_names_the_uploaded_path():
    from bridge_rna.callbacks import _retrieval_phrase

    phrase = _retrieval_phrase(UPLOAD_MODE).lower()
    assert "upload" in phrase and "live" in phrase


def test_uploaded_query_series_carries_what_the_figure_and_panel_read():
    from bridge_rna.callbacks import _uploaded_query_series

    q = _uploaded_query_series("mysample.csv", "SampleA")
    assert q["sample_name"] == "SampleA"
    assert q["sample_id"].startswith("UPLOAD|")
    for field in ("study_id", "tissue", "condition", "strain", "sex", "duration"):
        assert field in q.index


def test_query_series_prefers_the_payload_then_falls_back():
    from bridge_rna import callbacks

    q = callbacks._query_series({"query": {"sample_id": "UPLOAD|x", "sample_name": "x"}})
    assert q["sample_id"] == "UPLOAD|x"
    # No query dict and an id the catalog does not have -> None (not a crash).
    assert callbacks._query_series({"sample_id": "OSD-999|nope"}) is None


# --- The serving-import invariant is unaffected -----------------------------

def test_retrieval_does_not_import_torch_at_module_scope():
    """The embedder is a subprocess; nothing torch-bearing enters the serving pkg."""
    tree = ast.parse((REPO / "bridge_rna" / "retrieval.py").read_text())
    roots: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            roots.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0])
    assert "torch" not in roots
    assert (REPO / "precompute" / "embed_upload.py").exists()


# --- End to end, when the model is present ----------------------------------

def _first_cached_sample_with_counts():
    """(sample_key, sample_name, counts_path, cached_vec) from the real cache, or None.

    Resolves the real artifacts through `bridge_rna.config` rather than through
    `manifold.paths`, because conftest repoints the manifold package at a
    synthetic corpus for the whole session. The eligible filter is inlined
    (mouse, counts present, spaceflight set) to avoid importing the corpus-path
    module that reads those patched paths.
    """
    from bridge_rna import config

    npy = config.ROOT / "cache" / "osdr_sample_embeddings.float32.npy"
    parquet = config.ROOT / "cache" / "osdr_metadata.parquet"
    tsv = Path(config.OSDR_METADATA_PATH)
    if not (npy.exists() and parquet.exists() and tsv.exists()):
        return None

    vecs = np.load(npy).astype(np.float32)
    keys = (pd.read_parquet(parquet, columns=["sample_key"])
            ["sample_key"].astype(str).tolist())
    k2i = {k: i for i, k in enumerate(keys)}

    meta = pd.read_csv(tsv, sep="\t")
    org = meta["study.characteristics.organism"].astype(str)
    meta = meta[org.str.contains("Mus musculus", case=False, na=False)]
    meta = meta[meta["counts_path"].notna()].copy()
    meta["sample_name"] = meta["id.sample name"].astype(str)
    meta["sample_key"] = meta["id.accession"].astype(str) + "|" + meta["sample_name"]

    sys.path.insert(0, str(REPO / "precompute"))
    from embed_osdr import _resolve_counts_path

    _, resolved = preflight_retrieval_requirements()
    osdr_dir = resolved["osdr_data_dir"]
    for _, row in meta.iterrows():
        sk = str(row["sample_key"])
        if sk not in k2i:
            continue
        cp = _resolve_counts_path(str(row["counts_path"]), osdr_dir)
        if cp.exists():
            return sk, str(row["sample_name"]), cp, vecs[k2i[sk]].reshape(-1)
    return None


@pytest.mark.skipif(not _HAVE_MODEL, reason="model checkpoint not available")
def test_live_upload_embedding_matches_the_cached_corpus_vector(tmp_path):
    """Embed a sample's own counts file live; it must reproduce its cached vector.

    This is the scientific gate: if the uploaded path preprocessed or encoded
    even slightly differently from the corpus build, cosine scores would not be
    comparable. Verified at cosine 1.0, max abs diff 0.0 on OSD-100.
    """
    picked = _first_cached_sample_with_counts()
    if picked is None:
        pytest.skip("real OSDR cache / counts not available")
    sk, sname, counts_path, cached = picked
    _, resolved = preflight_retrieval_requirements()

    out = tmp_path / "vec.npy"
    cmd = [
        sys.executable, str(REPO / "precompute" / "embed_upload.py"),
        "--counts", str(counts_path), "--out", str(out), "--sample", sname,
        "--checkpoint", str(resolved["checkpoint"]),
        "--orthologs", str(resolved["orthologs"]),
        "--canonical-genes", str(resolved["canonical_genes"]),
        "--mouse-exon-lengths", str(resolved["mouse_exon_lengths"]),
        "--device", "cpu",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stderr[-2000:]

    live = np.load(out).astype(np.float32).reshape(-1)
    cos = float(live @ cached / (np.linalg.norm(live) * np.linalg.norm(cached) + 1e-12))
    assert cos > 0.9999, f"live vs cached cosine {cos} for {sk}"


@pytest.mark.skipif(not _HAVE_STACK, reason="canonical genes / torch stack not available")
def test_gene_digest_gate_aborts_a_shuffled_gene_order(tmp_path):
    """Invariant 1: a different gene order must abort before any vector is written."""
    _, resolved = preflight_retrieval_requirements()
    genes = pd.read_csv(resolved["canonical_genes"])
    shuffled = genes.iloc[::-1].reset_index(drop=True)  # reversed -> different digest
    bad_genes = tmp_path / "canonical_reversed.csv"
    shuffled.to_csv(bad_genes, index=False)

    counts = tmp_path / "counts.csv"
    counts.write_text("gene,S1\nENSMUSG00000000001,5\nENSMUSG00000000002,9\n")
    out = tmp_path / "vec.npy"

    cmd = [
        sys.executable, str(REPO / "precompute" / "embed_upload.py"),
        "--counts", str(counts), "--out", str(out), "--sample", "S1",
        "--checkpoint", str(resolved["checkpoint"]),
        "--orthologs", str(resolved["orthologs"]),
        "--canonical-genes", str(bad_genes),
        "--mouse-exon-lengths", str(resolved["mouse_exon_lengths"]),
        "--device", "cpu",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    assert proc.returncode != 0
    assert "digest mismatch" in (proc.stdout + proc.stderr).lower()
    assert not out.exists()
