"""Retrieval: one OSDR sample in, its nearest ARCHS4 analogs out.

There are three paths to a query embedding, tried in this order, and the app
always reports which one ran because they differ in speed and in how much
metadata comes back with the answer.

1. **cached** - the manifold precompute already embedded all 2,108 eligible
   OSDR samples (`cache/osdr_sample_embeddings.float32.npy`), using a
   preprocessing path checked bit-for-bit against the single-sample path below
   (max abs diff 0.0). Looking the vector up costs nothing, so the whole query
   is one cosine pass over the memmap: **0.5 s**, against 22 s for the
   subprocess. Hits are annotated from `cache/archs4_metadata.parquet`, which
   carries GEO series, title, source name, characteristics, and the canonical
   tissue bucket for all 940,455 ARCHS4 samples - offline, with no NCBI call.
   Verified on OSD-100 eye sample Rep1_M23: identical top-5 accessions and
   identical scores to six decimal places against the subprocess path.
2. **precomputed** - an `osdr_query_embeddings.parquet` supplied out of band.
3. **demo** - shell out to `demo_osdr_top5.py`, which loads the checkpoint and
   embeds the sample from its counts matrix. The only path that works for a
   sample the manifold never embedded, and the slowest by a factor of 44.

The cosine scan itself is the same code in every case, so the three paths
differ only in where the 512-d query vector came from.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import (
    DEMO_SCRIPT_PATH,
    EMBEDDING_DIR,
    OSDR_METADATA_PATH,
    PRECOMPUTED_QUERY_EMBEDDING_CANDIDATES,
    ROOT,
)
from .geo import _enrich_hits_from_ncbi_eutils
from .preflight import preflight_retrieval_requirements
from .util import (
    RetrievalError,
    _extract_gse,
    _find_first_existing,
    _first_non_empty,
    _last_nonempty_line,
    _safe_str,
)

_ARCHS4_CACHE: dict[str, Any] = {}
_OSDR_QUERY_CACHE: dict[str, Any] = {}


# --- The cached path: the manifold precompute answers the query -------------

@lru_cache(maxsize=1)
def _cached_osdr_embeddings() -> tuple[np.ndarray, dict[str, int]] | None:
    """The 2,108 precomputed OSDR query vectors, keyed by sample_id.

    Returns None when the manifold cache has not been built, which is the state
    a fresh clone starts in; the caller falls back to the subprocess path.

    The two tables are joined *positionally* by the precompute step, so a length
    disagreement means every vector would be attributed to the wrong sample.
    That is a silent failure rather than a loud one, so it is checked here and
    the whole path is refused rather than half-trusted.
    """
    from manifold import paths as mpaths

    npy, meta = mpaths.OSDR_EMBEDDINGS_NPY, mpaths.OSDR_METADATA_PARQUET
    if not npy.exists() or not meta.exists():
        return None
    vectors = np.load(npy).astype(np.float32)
    keys = pd.read_parquet(meta, columns=["sample_key"])["sample_key"].astype(str)
    if len(keys) != len(vectors):
        print(
            f"[retrieval] refusing the cached path: {len(vectors)} embeddings but "
            f"{len(keys)} sample keys. They are joined positionally, so every "
            "query would use another sample's vector. Re-run "
            "precompute/embed_osdr.py.",
            file=sys.stderr, flush=True,
        )
        return None
    return vectors, {k: i for i, k in enumerate(keys)}


def cached_query_vector(sample_id: str) -> np.ndarray | None:
    """The precomputed 512-d query vector for an OSDR sample, if there is one."""
    cached = _cached_osdr_embeddings()
    if cached is None:
        return None
    vectors, index = cached
    row = index.get(_safe_str(sample_id))
    return None if row is None else vectors[row]


def cached_query_coverage() -> tuple[int, bool]:
    """(how many OSDR samples the fast path can answer, whether it is usable)."""
    cached = _cached_osdr_embeddings()
    return (0, False) if cached is None else (len(cached[1]), True)


# --- What can actually be retrieved, and how ---------------------------------

TIER_CACHED = "cached"        # precomputed vector: ~0.5 s, and on the map
TIER_SUBPROCESS = "subprocess"  # embeddable from counts: ~22 s, not on the map
TIER_UNAVAILABLE = "unavailable"  # no usable counts column: retrieval raises


# One entry per study, not per sample: classifying all 2,896 samples would
# otherwise re-read the same 94 header rows thousands of times.
@lru_cache(maxsize=256)
def _counts_columns(path: str) -> frozenset:
    """The column names of a counts matrix, read from its header row only."""
    for candidate in (Path(path), ROOT / path):
        if candidate.exists():
            try:
                return frozenset(pd.read_csv(candidate, nrows=0).columns)
            except Exception:
                return frozenset()
    return frozenset()


def sample_tier(sample_id: str, sample_name: str, counts_path: str) -> str:
    """Which retrieval path, if any, can answer for this OSDR sample.

    The third tier is not hypothetical and was not obvious. 71 of the 2,896
    samples the picker lists name no column in their own study's counts matrix,
    so `demo_osdr_top5.py` raises "found but has no readable counts/columns
    after processing" and no path can serve them - measured, and reproduced end
    to end on OSD-462|RR10_KDN_WT_BSL_B11. They are `OSD-462` (54), `OSD-374`
    (16) and `OSD-612` (1).

    Offering them and failing after the click is the thing this exists to
    prevent. The picker disables them and says why, which is the same treatment
    the map's color-by menu gives a field whose artifact has not been built.
    """
    if cached_query_vector(sample_id) is not None:
        return TIER_CACHED
    path = _safe_str(counts_path)
    if not path:
        return TIER_UNAVAILABLE
    return (TIER_SUBPROCESS if _safe_str(sample_name) in _counts_columns(path)
            else TIER_UNAVAILABLE)



@lru_cache(maxsize=1)
def _archs4_annotations() -> pd.DataFrame | None:
    """GEO metadata for every ARCHS4 sample, positioned by memmap row index.

    This is the join that makes the cached path *better* annotated than the
    subprocess path rather than merely faster: `global_index` in this table is
    the memmap row a cosine scan returns, so annotating k hits is k lookups
    against a table already in memory, with no network call at all.
    """
    from manifold import data as mdata

    df = mdata.archs4_metadata()
    if df is None:
        return None
    return df.reset_index(drop=True)


def _annotate_from_cache(idx: np.ndarray, scores: np.ndarray) -> pd.DataFrame:
    """Build the app's hit schema for memmap rows `idx`, offline."""
    ann = _archs4_annotations()
    out = pd.DataFrame()
    if ann is None:
        # No metadata join built: return accessions and scores, and let the
        # caller's optional NCBI enrichment fill the rest. Blank is honest here;
        # inventing a title would not be.
        _, meta, _ = _load_archs4_index()
        out["gsm"] = meta.iloc[idx]["geo_accession"].astype(str).to_numpy()
        out["score"] = scores.astype(float)
        for col in ("gse", "title", "source_name", "characteristics",
                    "geo_summary", "geo_design", "pubmed_ids", "tissue", "species"):
            out[col] = ""
        return out

    rows = ann.iloc[idx]
    out["gsm"] = rows["geo_accession"].astype(str).to_numpy()
    out["score"] = scores.astype(float)
    out["gse"] = rows["series_id"].astype(str).fillna("").to_numpy()
    out["title"] = rows["title"].astype(str).fillna("").to_numpy()
    out["source_name"] = rows["source_name"].astype(str).fillna("").to_numpy()
    out["characteristics"] = rows["characteristics"].astype(str).fillna("").to_numpy()
    out["tissue"] = rows["tissue"].astype(str).fillna("").to_numpy()

    # Species comes from the identity table rather than from GEO free text.
    try:
        from manifold import data as mdata

        species_id = mdata.points_meta()["species_id"].to_numpy()[idx]
        out["species"] = np.where(species_id == 0, "Homo sapiens", "Mus musculus")
    except Exception:
        out["species"] = ""

    # These three are not in the cache. They stay blank unless the optional
    # NCBI enrichment is asked for, because a missing summary is a fact.
    for col in ("geo_summary", "geo_design", "pubmed_ids"):
        out[col] = ""

    # pandas 3.0 leaves NA through astype(str); a literal "nan" in a GSE field
    # reads as a real accession downstream.
    for col in ("gse", "title", "source_name", "characteristics", "tissue"):
        out[col] = out[col].replace({"nan": "", "None": "", "<NA>": ""})
    return out


def run_cached_query_retrieval(sample_id: str, topk: int) -> pd.DataFrame:
    """The fast path: precomputed query vector, one memmap pass, local metadata."""
    q_vec = cached_query_vector(sample_id)
    if q_vec is None:
        raise RuntimeError(f"No precomputed embedding for {sample_id}.")
    index_vecs, _, _ = _load_archs4_index()
    idx, score = _topk_cosine_from_memmap(index_vecs=index_vecs, q_vec=q_vec, k=topk)
    hits = _annotate_from_cache(idx, score)
    hits["archs4_index"] = idx.astype(int)
    return hits.sort_values("score", ascending=False).reset_index(drop=True)


# --- Shared machinery: the query vector sources and the cosine scan --------

def _extract_vector_from_value(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        arr = value.astype(np.float32).reshape(-1)
        return arr if arr.size > 0 else None
    if isinstance(value, (list, tuple)):
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
        return arr if arr.size > 0 else None
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        try:
            parsed = json.loads(s)
            arr = np.asarray(parsed, dtype=np.float32).reshape(-1)
            return arr if arr.size > 0 else None
        except Exception:
            return None
    return None


def _find_precomputed_query_embedding_file() -> Path | None:
    return _find_first_existing(PRECOMPUTED_QUERY_EMBEDDING_CANDIDATES)


def _load_archs4_index() -> tuple[np.memmap, pd.DataFrame, int]:
    key = str(EMBEDDING_DIR.resolve())
    cached = _ARCHS4_CACHE.get(key)
    if cached is not None:
        return cached["vecs"], cached["meta"], cached["dim"]

    manifest_path = EMBEDDING_DIR / "embedding_manifest.json"
    meta_path = EMBEDDING_DIR / "sample_locations.parquet"
    if not manifest_path.exists() or not meta_path.exists():
        raise RuntimeError(
            f"ARCHS4 embedding files missing under {EMBEDDING_DIR}; expected embedding_manifest.json and sample_locations.parquet"
        )

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    n = int(manifest["total_samples"])
    d = int(manifest["embedding_dim"])
    emb_dtype = manifest.get("embedding_dtype", "float16")
    dtype = np.float16 if emb_dtype == "float16" else np.float32
    mmap_path = EMBEDDING_DIR / f"sample_embeddings.{emb_dtype}.mmap"
    if not mmap_path.exists():
        raise RuntimeError(f"ARCHS4 memmap not found: {mmap_path}")

    vecs = np.memmap(mmap_path, dtype=dtype, mode="r", shape=(n, d))
    meta = pd.read_parquet(meta_path)

    _ARCHS4_CACHE[key] = {"vecs": vecs, "meta": meta, "dim": d}
    return vecs, meta, d


def _load_precomputed_osdr_queries(path: Path) -> pd.DataFrame:
    key = str(path.resolve())
    cached = _OSDR_QUERY_CACHE.get(key)
    if cached is not None:
        return cached

    raw = pd.read_parquet(path)
    sample_id_col = None
    for c in ["sample_id", "sample", "id.sample name", "sample_name"]:
        if c in raw.columns:
            sample_id_col = c
            break
    if sample_id_col is None:
        raise RuntimeError(
            f"Precomputed OSDR embedding file {path} is missing a sample id column (expected one of: sample_id, sample, id.sample name, sample_name)."
        )

    vector_col = None
    for c in ["embedding", "vector", "query_embedding", "emb"]:
        if c in raw.columns:
            vector_col = c
            break

    if vector_col is not None:
        out = pd.DataFrame()
        out["sample_key"] = raw[sample_id_col].astype(str)
        out["embedding"] = raw[vector_col].apply(_extract_vector_from_value)
        out = out[out["embedding"].notna()].copy()
        _OSDR_QUERY_CACHE[key] = out
        return out

    emb_cols = [c for c in raw.columns if re.match(r"^(emb|e|dim)_?\d+$", str(c), flags=re.IGNORECASE)]
    if emb_cols:
        emb_cols = sorted(
            emb_cols,
            key=lambda x: int(re.search(r"(\d+)$", str(x)).group(1)) if re.search(r"(\d+)$", str(x)) else 0,
        )
        out = pd.DataFrame()
        out["sample_key"] = raw[sample_id_col].astype(str)
        out["embedding"] = raw[emb_cols].to_numpy(dtype=np.float32).tolist()
        out["embedding"] = out["embedding"].apply(lambda v: np.asarray(v, dtype=np.float32))
        _OSDR_QUERY_CACHE[key] = out
        return out

    raise RuntimeError(
        f"Precomputed OSDR embedding file {path} has no recognized embedding column. Expected one of [embedding, vector, query_embedding, emb] or numbered columns like emb_0..emb_n."
    )


def _topk_cosine_from_memmap(index_vecs: np.memmap, q_vec: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    q = np.asarray(q_vec, dtype=np.float32).reshape(-1)
    if q.size == 0:
        raise RuntimeError("Query embedding is empty.")
    q = q / (float(np.linalg.norm(q)) + 1e-12)

    n = int(index_vecs.shape[0])
    d = int(index_vecs.shape[1])
    if q.shape[0] != d:
        raise RuntimeError(f"Embedding dimension mismatch: query dim={q.shape[0]} but ARCHS4 dim={d}")

    k = max(1, min(int(k), n))
    chunk = 25000
    scores = np.empty(n, dtype=np.float32)
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        x = np.asarray(index_vecs[start:end], dtype=np.float32)
        x /= (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)
        scores[start:end] = x @ q

    top_idx = np.argpartition(scores, -k)[-k:]
    top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
    return top_idx, scores[top_idx]


def run_precomputed_query_retrieval(sample_id: str, sample_name: str, topk: int) -> pd.DataFrame:
    q_path = _find_precomputed_query_embedding_file()
    if q_path is None:
        raise RuntimeError("No precomputed OSDR query embedding parquet found.")

    query_df = _load_precomputed_osdr_queries(q_path)
    row = query_df[query_df["sample_key"].astype(str) == str(sample_id)]
    if row.empty:
        row = query_df[query_df["sample_key"].astype(str) == str(sample_name)]
    if row.empty:
        raise RuntimeError(
            f"No precomputed embedding found for sample_id '{sample_id}' (or sample name '{sample_name}') in {q_path}"
        )

    q_vec = np.asarray(row.iloc[0]["embedding"], dtype=np.float32)
    index_vecs, meta, _ = _load_archs4_index()
    idx, score = _topk_cosine_from_memmap(index_vecs=index_vecs, q_vec=q_vec, k=topk)

    hits = meta.iloc[idx].copy().reset_index(drop=True)
    hits["score"] = score

    normalized = pd.DataFrame()
    normalized["gsm"] = hits.get("geo_accession", "").astype(str)
    normalized["score"] = pd.to_numeric(hits.get("score", 0), errors="coerce").fillna(0.0)
    normalized["gse"] = ""
    normalized["title"] = ""
    normalized["source_name"] = ""
    normalized["characteristics"] = ""
    normalized["geo_summary"] = ""
    normalized["geo_design"] = ""
    normalized["pubmed_ids"] = ""
    return normalized.sort_values("score", ascending=False).reset_index(drop=True)


def run_real_retrieval(
    sample_name: str,
    topk: int,
    entrez_email: str | None = None,
    enable_biopython_metadata: bool = True,
) -> pd.DataFrame:
    """Run existing demo script and normalize output into the app schema."""
    missing, resolved = preflight_retrieval_requirements()
    if missing:
        raise RuntimeError("Missing retrieval prerequisites: " + "; ".join(missing))

    with tempfile.TemporaryDirectory(prefix="osdr_dash_") as td:
        prefix = Path(td) / "retrieval"
        cmd = [
            sys.executable,
            str(DEMO_SCRIPT_PATH),
            "--embedding-dir",
            str(EMBEDDING_DIR),
            "--topk",
            str(int(topk)),
            "--osdr-sample-name",
            sample_name,
            "--osdr-data-dir",
            str(resolved["osdr_data_dir"]),
            "--osdr-metadata",
            str(OSDR_METADATA_PATH),
            "--checkpoint",
            str(resolved["checkpoint"]),
            "--orthologs",
            str(resolved["orthologs"]),
            "--canonical-genes",
            str(resolved["canonical_genes"]),
            "--mouse-exon-lengths",
            str(resolved["mouse_exon_lengths"]),
            "--save-report-prefix",
            str(prefix),
            "--device",
            "cuda",
        ]

        email = _safe_str(entrez_email)
        if enable_biopython_metadata and email:
            cmd.extend([
                "--biopython-metadata",
                "--biopython-pubmed",
                "--entrez-email",
                email,
            ])

        proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=600)

        hits_csv = Path(f"{prefix}.top_hits.csv")
        meta_csv = Path(f"{prefix}.archs4_metadata.csv")

        if proc.returncode != 0:
            msg = _safe_str(proc.stderr) or _safe_str(proc.stdout)
            # Log the full traceback server-side for debugging, but never surface it
            # to the UI - the primary viewport gets only a clean one-line message.
            print(
                "[run_real_retrieval] demo subprocess failed (returncode "
                f"{proc.returncode}). Full output below:\n{msg}",
                file=sys.stderr,
                flush=True,
            )
            missing_mod = re.search(r"ModuleNotFoundError: No module named '([^']+)'", msg)
            if missing_mod:
                module_name = missing_mod.group(1)
                raise RetrievalError(
                    f"Demo retrieval import failed: missing module '{module_name}'. "
                    "Install/provide this dependency in the app environment.",
                    detail=msg,
                )
            # Almost always the actual exception message, not the whole stack trace.
            raise RetrievalError(
                _last_nonempty_line(msg) or "Demo retrieval failed.", detail=msg
            )
        if not hits_csv.exists():
            raise RuntimeError("Demo retrieval did not produce top hits CSV.")

        hits = pd.read_csv(hits_csv)
        meta = pd.read_csv(meta_csv) if meta_csv.exists() else pd.DataFrame()

        if "geo_accession" not in hits.columns:
            raise RuntimeError("top hits CSV is missing geo_accession column.")

        merged = hits.rename(columns={"geo_accession": "gsm"}).copy()
        if not meta.empty and "geo_accession" in meta.columns:
            meta2 = meta.rename(columns={"geo_accession": "gsm"})
            merged = merged.merge(meta2, on="gsm", how="left")

        normalized = pd.DataFrame()
        normalized["gsm"] = merged["gsm"].astype(str)
        normalized["score"] = pd.to_numeric(merged.get("score", 0), errors="coerce").fillna(0.0)
        normalized["gse"] = merged.apply(
            lambda r: _extract_gse(
                _first_non_empty(r, ["series_id", "geo_gse_biopython", "gse", "GSE"])
            ),
            axis=1,
        )
        normalized["title"] = merged.apply(
            lambda r: _first_non_empty(r, ["title", "geo_title_biopython", "Title"]), axis=1
        )
        normalized["source_name"] = merged.apply(
            lambda r: _first_non_empty(r, ["source_name_ch1", "source_name", "source"]), axis=1
        )
        normalized["characteristics"] = merged.apply(
            lambda r: _first_non_empty(r, ["characteristics_ch1", "characteristics", "traits"]), axis=1
        )
        normalized["geo_summary"] = merged.apply(
            lambda r: _first_non_empty(r, ["geo_summary_biopython", "summary", "geo_summary"]), axis=1
        )
        normalized["geo_design"] = merged.apply(
            lambda r: _first_non_empty(r, ["geo_overall_design_biopython", "geo_design", "design"]), axis=1
        )
        normalized["pubmed_ids"] = merged.apply(
            lambda r: _first_non_empty(r, ["geo_pubmed_ids_biopython", "pubmed_id", "pubmed_ids"]), axis=1
        )

        # Preserve richer metadata fields so the details panel can mirror CLI output.
        extra_cols = [
            "species",
            "source_name_ch1",
            "characteristics_ch1",
            "series_id",
            "geo_gse_biopython",
            "geo_platform_biopython",
            "geo_taxon_biopython",
            "geo_entry_type_biopython",
            "geo_gds_type_biopython",
            "geo_pdat_biopython",
            "geo_n_samples_biopython",
            "geo_ftp_link_biopython",
            "geo_title_biopython",
            "geo_summary_biopython",
            "geo_overall_design_biopython",
            "geo_abstract_biopython",
            "geo_pubmed_ids_biopython",
            "pubmed_id",
            "pubmed_title_biopython",
            "pubmed_journal_biopython",
            "pubmed_pub_date_biopython",
            "pubmed_doi_biopython",
            "pubmed_authors_biopython",
        ]
        for col in extra_cols:
            if col in merged.columns:
                normalized[col] = merged[col]

        # Fallback metadata enrichment via NCBI E-utilities fills right-panel fields
        # when local H5/Biopython sources are sparse.
        if enable_biopython_metadata and _safe_str(entrez_email):
            normalized = _enrich_hits_from_ncbi_eutils(normalized, _safe_str(entrez_email))

        normalized = normalized.sort_values("score", ascending=False).reset_index(drop=True)
        return normalized


def search_hits(
    samples_df: pd.DataFrame,
    sample_id: str,
    topk: int,
    entrez_email: str | None = None,
    enable_biopython_metadata: bool = True,
) -> tuple[pd.DataFrame, str]:
    """Retrieve top-k ARCHS4 analogs for one OSDR sample.

    Returns (hits, mode) where mode is "cached", "precomputed", or "demo". The
    caller shows the mode, because it is the difference between a half-second
    answer annotated from the local cache and a 22-second subprocess.
    """
    row = samples_df.loc[samples_df["sample_id"] == sample_id]
    if row.empty:
        raise ValueError(f"Unknown sample_id: {sample_id}")
    sample_row = row.iloc[0]
    sample_name = _safe_str(sample_row["sample_name"])

    if cached_query_vector(sample_id) is not None:
        hits = run_cached_query_retrieval(sample_id=sample_id, topk=topk)
        # The cache has no study summaries or PubMed links. Fill them only when
        # asked, since it is a network round trip per accession.
        if enable_biopython_metadata and _safe_str(entrez_email):
            hits = _enrich_hits_from_ncbi_eutils(hits, _safe_str(entrez_email))
        return hits, "cached"

    q_file = _find_precomputed_query_embedding_file()
    if q_file is not None:
        return run_precomputed_query_retrieval(
            sample_id=sample_id, sample_name=sample_name, topk=topk), "precomputed"

    return (
        run_real_retrieval(
            sample_name=sample_name,
            topk=topk,
            entrez_email=entrez_email,
            enable_biopython_metadata=enable_biopython_metadata,
        ),
        "demo",
    )



def _archs4_sample_count() -> int | None:
    """Total ARCHS4 samples in the embedding index, read from the manifest."""
    try:
        manifest = json.loads((EMBEDDING_DIR / "embedding_manifest.json").read_text())
    except Exception:
        return None
    for key in ("total_samples", "num_samples", "n_samples"):
        value = manifest.get(key)
        if isinstance(value, int) and value >= 0:
            return value
    return None
