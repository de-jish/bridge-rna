"""Retrieval: an OSDR query in, its nearest ARCHS4 analogs out.

Five paths produce a query vector and the app always reports which one ran,
because they differ in speed and in how much metadata comes back. The three
below are the ones `search_hits` chooses between for a single catalog sample,
tried in this order; the other two are chosen by the user rather than by
fallback - `run_uploaded_retrieval` for a file the corpus has never seen, and
`run_cohort_retrieval` for a whole experimental group pooled into one vector.
All five end in the same cosine scan.

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
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pandas as pd

from .config import (
    DEMO_SCRIPT_PATH,
    EMBEDDING_DIR,
    OSDR_METADATA_PATH,
    PRECOMPUTED_QUERY_EMBEDDING_CANDIDATES,
    ROOT,
    UPLOAD_EMBED_SCRIPT_PATH,
)
from . import scoring as _scoring
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


def cached_query_vectors(sample_ids: list[str] | tuple[str, ...]
                         ) -> tuple[np.ndarray, list[str]]:
    """The precomputed vectors for several OSDR samples, and the ones missing.

    Returns `(rows, missing)` where `rows` is `(k, 512)` in the order the ids
    were given, minus any that have no cached vector. The missing list is
    returned rather than swallowed because a cohort silently pooling five of its
    six members would produce a different answer than the one the interface
    claims to have computed.
    """
    found: list[np.ndarray] = []
    missing: list[str] = []
    for sid in sample_ids:
        vec = cached_query_vector(sid)
        if vec is None:
            missing.append(_safe_str(sid))
        else:
            found.append(vec)
    rows = (np.stack(found).astype(np.float32) if found
            else np.zeros((0, 0), dtype=np.float32))
    return rows, missing


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


def _passes_demo_filter(condition: str) -> bool:
    """Whether `demo_osdr_top5.py` will even consider this sample.

    It filters its metadata to mouse rows with a counts path *and a recorded
    spaceflight value*, then looks the requested name up in what survives
    (`demo_osdr_top5.py:337-357`). A sample with no spaceflight value is not
    "slow to retrieve", it raises "Requested OSDR sample name not found after
    filtering" - a different failure from the counts-column one, and the reason
    the first version of this function was wrong.

    Every row in the shipped metadata is Mus musculus and every row has a
    counts path, so the spaceflight value is the only one of the three that
    discriminates; it is still checked here rather than assumed.
    """
    s = _safe_str(condition).lower()
    return bool(s) and s not in ("nan", "none", "na", "n/a")


def sample_tier(sample_id: str, sample_name: str, counts_path: str,
                condition: str = "") -> str:
    """Which retrieval path, if any, can answer for this OSDR sample.

    Measured on the shipped metadata, with the cache built:

        cached       2,108   precomputed vector, ~0.5 s, and on the map
        subprocess       0
        unavailable    788   no path can serve them

    The zero is the point, and it took two attempts to find. An earlier version
    of this function checked only whether the sample's name appears as a column
    in its counts matrix, and reported 717 samples as retrievable-but-slow.
    They are not: 733 of the 788 carry no spaceflight value, so the demo
    script's own filter drops them before it ever looks for the name, and the
    remaining 55 pass that filter but match no column. Reproduced end to end -
    `OSD-141|Mmus_C57-6J_SPL_cells_Rep1_SP1` fails in 4 s with "not found after
    filtering", and `OSD-462|RR10_KDN_WT_BSL_B11` in 2.3 s with "found but has
    no readable counts/columns after processing".

    So with the cache present, every sample that can be retrieved at all is
    retrieved in half a second. `TIER_SUBPROCESS` is not dead: without the
    cache those same 2,108 samples fall to it and take about 22 seconds each.

    Offering a sample and failing after the click is what this prevents. The
    picker disables the unavailable ones and says why, which is the treatment
    the map's color-by menu already gives a field whose artifact is not built.
    """
    if cached_query_vector(sample_id) is not None:
        return TIER_CACHED
    if not _passes_demo_filter(condition):
        return TIER_UNAVAILABLE
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


def run_cached_query_retrieval(sample_id: str, topk: int,
                               scoring: str = _scoring.COSINE) -> pd.DataFrame:
    """The fast path: precomputed query vector, one memmap pass, local metadata."""
    q_vec = cached_query_vector(sample_id)
    if q_vec is None:
        raise RuntimeError(f"No precomputed embedding for {sample_id}.")
    index_vecs, _, _ = _load_archs4_index()
    idx, score = _topk_cosine_from_memmap(index_vecs=index_vecs, q_vec=q_vec, k=topk,
                                          scoring=scoring)
    hits = _annotate_from_cache(idx, score)
    hits["archs4_index"] = idx.astype(int)
    return _with_plain_cosine(_ordered(hits), q_vec, scoring)


def _with_plain_cosine(ranked: pd.DataFrame, q_vec: np.ndarray,
                       scoring: str) -> pd.DataFrame:
    """Every match carries its plain cosine beside the score that ranked it.

    Under plain cosine the two are the same number and are copied. Under
    whitening the cosine is recomputed in float64 for the rows returned only
    (at most 1,000, about a millisecond), so the inspector and the CSV can show
    the familiar number while the ranking uses the whitened one.
    """
    if scoring != _scoring.WHITENED or ranked.empty:
        ranked["cosine"] = ranked["score"].astype(float)
        return ranked
    q = np.asarray(q_vec, dtype=np.float64).reshape(-1)
    q = q / max(float(np.linalg.norm(q)), 1e-12)
    ranked["cosine"] = reference_vectors(ranked["archs4_index"].to_numpy()) @ q
    return ranked


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


def _tie_exact_top(block: np.ndarray, take: int) -> np.ndarray:
    """Column positions of each row's `take` best scores, lower column on a tie.

    `argpartition` alone keeps an arbitrary member of a tie that straddles the
    cut. Exact ties are common in this index (a profile deposited under two
    accessions has a bit-identical vector; one query had 229 tied neighbours in
    its top 1,000), and on the real corpus a 30-deep scan kept the later of two
    tied rows at rank 30 while a 1,000-deep scan kept the earlier one. Only the
    rare rows with a tie at their cut pay for a full sort.
    """
    part = np.argpartition(block, -take, axis=1)[:, -take:]
    kept = np.take_along_axis(block, part, axis=1)
    cut = kept.min(axis=1, keepdims=True)
    straddles = (block == cut).sum(axis=1) > (kept == cut).sum(axis=1)
    for r in np.flatnonzero(straddles):
        cols = np.arange(block.shape[1])
        part[r] = np.lexsort((cols, -block[r]))[:take]
    return part


def _topk_cosine_matrix(index_vecs: np.memmap, q_mat: np.ndarray, k: int,
                        block_bytes: int = 200_000_000,
                        progress: Any = None, *,
                        preserve_matrix_product: bool = False,
                        scoring: str = "cosine",
                        ) -> tuple[np.ndarray, np.ndarray]:
    """Top-k for `m` query vectors in **one** pass over the index.

    The single implementation of the cosine scan. Every path in this module
    reaches the ARCHS4 index through here, as do offline validation batches.

    Returns `(idx, scores)`, both `(m, k)`, each row sorted best first.

    ``preserve_matrix_product`` keeps a formerly batched query on BLAS's
    matrix-matrix path. A single-column matrix can use matrix-vector arithmetic
    instead; its float32 accumulation can reorder near-tied cohort matches.
    One zero padding column preserves the matrix product and is discarded before
    ranking. It represents no sample and never reaches the returned results.

    The top-k is kept as a running merge per query rather than by partitioning a
    full `(n, m)` score matrix, which would allocate 290 MB at m=77 and grow
    without bound. Memory stays at one block. That is the same technique
    `precompute/validate_cohorts.py` and `validate_artifacts.py --mixing` use,
    and the validator now calls straight in here rather than keeping a second
    copy of it: `progress(rows_done, total)` is what lets a several-thousand
    query sweep still print where it has got to.

    ``scoring="whitened"`` ranks by cosine after whitening instead
    (`bridge_rna/scoring.py`). It is the same single pass: each block is
    centred on the ARCHS4 mean, multiplied by the per-query operator `v` in
    place of the query itself, and divided by the row norms stored at build
    time. Ties stay exact, because identical rows still produce identical
    scores, and the same tie rule then applies.
    """
    Q = np.asarray(q_mat, dtype=np.float32)
    if Q.ndim == 1:
        Q = Q.reshape(1, -1)
    if Q.size == 0 or Q.shape[0] == 0:
        raise RuntimeError("Query embedding is empty.")
    Q = Q / np.maximum(np.linalg.norm(Q, axis=1, keepdims=True), 1e-12)

    n = int(index_vecs.shape[0])
    d = int(index_vecs.shape[1])
    if Q.shape[1] != d:
        raise RuntimeError(
            f"Embedding dimension mismatch: query dim={Q.shape[1]} but ARCHS4 dim={d}")

    m = Q.shape[0]
    k = max(1, min(int(k), n))
    whitening = None
    if scoring == _scoring.WHITENED:
        whitening = _scoring.load_whitening()
        if whitening is None:
            raise RetrievalError("Whitened matching was requested but cache/whitening.npz "
                                 "is missing or does not match the index. "
                                 + _scoring.BUILD_HINT)
        Qt = _scoring.query_operator(Q, whitening)      # (d, m)
    else:
        Qt = np.ascontiguousarray(Q.T)                  # (d, m)
    if preserve_matrix_product and m == 1:
        Qt = np.column_stack((Qt, np.zeros((d, 1), dtype=Qt.dtype)))

    # The block *height* is derived from the query count, because the score
    # block is (rows x queries): a fixed 25,000 rows is 3.9 MB at m=39 and would
    # be far more if a caller ever batched hundreds. It resolves to exactly the
    # 25,000 the single-query scan always used for any m up to 2,000.
    chunk = int(max(2000, min(25000, int(block_bytes) // (4 * m))))

    best_idx = np.zeros((m, 0), dtype=np.int64)
    best_score = np.zeros((m, 0), dtype=np.float32)
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        x = np.asarray(index_vecs[start:end], dtype=np.float32)
        x /= (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)
        if whitening is None:
            block = (x @ Qt).T[:m]                      # (m, rows in block)
        else:
            # Centre first: x and mu are close, so subtracting mu . v after
            # the product would cancel away the digits that rank the rows.
            x -= whitening.mu
            block = (x @ Qt).T[:m] / whitening.row_norms[start:end]

        take = min(k, block.shape[1])
        part = _tie_exact_top(block, take)
        cand_idx = np.concatenate([best_idx, part + start], axis=1)
        cand_score = np.concatenate(
            [best_score, np.take_along_axis(block, part, axis=1)], axis=1)
        keep = min(k, cand_idx.shape[1])
        # Candidates are at most 2k per query, so an exact (-score, row) sort
        # is cheap here, and it is what makes the kept set independent of k.
        sel = np.lexsort((cand_idx, -cand_score), axis=-1)[:, :keep]
        best_idx = np.take_along_axis(cand_idx, sel, axis=1)
        best_score = np.take_along_axis(cand_score, sel, axis=1)
        if progress is not None:
            progress(end, n)

    # Best first, and an exact tie goes to the lower reference row. Ties are
    # not hypothetical here: one expression profile deposited under two GSM
    # accessions has a bit-identical vector, so both score exactly the same,
    # and `argsort`'s default is not stable. Without a tie key the order of
    # such a pair depended on how deep the scan ran, which would have let the
    # 1,000-deep score profile reorder a top 5 it was only meant to contain.
    order = np.lexsort((best_idx, -best_score), axis=-1)
    return (np.take_along_axis(best_idx, order, axis=1),
            np.take_along_axis(best_score, order, axis=1))


def _topk_cosine_from_memmap(index_vecs: np.memmap, q_vec: np.ndarray,
                             k: int, scoring: str = _scoring.COSINE,
                             ) -> tuple[np.ndarray, np.ndarray]:
    """One query vector's top-k. A one-row call into `_topk_cosine_matrix`.

    Verified against the previous standalone implementation on the real corpus
    before it was replaced: identical top-30 in order, maximum score difference
    exactly 0.0.
    """
    idx, score = _topk_cosine_matrix(index_vecs, np.asarray(q_vec).reshape(1, -1), k,
                                     scoring=scoring)
    return idx[0], score[0]


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


# --- The fourth query-vector source: a file the corpus has never seen --------

UPLOAD_MODE = "uploaded"


def embed_uploaded_counts(
    counts_path: str | Path,
    sample_column: str | None = None,
) -> np.ndarray:
    """Embed one uploaded OSDR counts file to its 512-d vector, live.

    The serving app never imports torch, so this shells out to
    `precompute/embed_upload.py`, exactly as the `demo` path shells out to
    `demo_osdr_top5.py`. The subprocess enforces invariant 1 (the gene-digest
    gate) before it produces anything, so a sample can never be embedded in a
    gene order the ARCHS4 index was not built in.

    Raises RetrievalError with a clean one-line message on any failure (missing
    model prerequisites, an unreadable file, no ortholog-mappable genes, a digest
    mismatch); the full detail is logged server-side.
    """
    missing, resolved = preflight_retrieval_requirements()
    needed = ("checkpoint", "orthologs", "canonical_genes", "mouse_exon_lengths")
    unresolved = [k for k in needed if resolved.get(k) is None]
    if unresolved:
        raise RetrievalError(
            "Cannot embed an uploaded sample: missing model prerequisites "
            f"({', '.join(unresolved)}). " + ("; ".join(missing) if missing else ""),
            detail="; ".join(missing),
        )

    counts_path = Path(counts_path)
    if not counts_path.exists():
        raise RetrievalError(f"Uploaded counts file not found: {counts_path}")

    with tempfile.TemporaryDirectory(prefix="osdr_upload_") as td:
        out_path = Path(td) / "query_vector.npy"
        cmd = [
            sys.executable,
            str(UPLOAD_EMBED_SCRIPT_PATH),
            "--counts", str(counts_path),
            "--out", str(out_path),
            "--sample", _safe_str(sample_column),
            "--checkpoint", str(resolved["checkpoint"]),
            "--orthologs", str(resolved["orthologs"]),
            "--canonical-genes", str(resolved["canonical_genes"]),
            "--mouse-exon-lengths", str(resolved["mouse_exon_lengths"]),
            "--device", "cpu",
        ]
        proc = subprocess.run(
            cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=600
        )
        if proc.returncode != 0 or not out_path.exists():
            msg = _safe_str(proc.stderr) or _safe_str(proc.stdout)
            print(
                "[embed_uploaded_counts] embed subprocess failed (returncode "
                f"{proc.returncode}). Full output below:\n{msg}",
                file=sys.stderr, flush=True,
            )
            # The script raises SystemExit("ABORT: ...") for the expected failures,
            # so the last non-empty line is the actionable reason, not a stack trace.
            reason = _last_nonempty_line(msg) or "Embedding the uploaded sample failed."
            reason = reason.replace("ABORT: ", "")
            raise RetrievalError(reason, detail=msg)

        vec = np.load(out_path).astype(np.float32).reshape(-1)
    if vec.size == 0:
        raise RetrievalError("The uploaded sample produced an empty embedding.")
    return vec


def run_uploaded_retrieval(
    counts_path: str | Path,
    sample_column: str | None,
    topk: int,
    entrez_email: str | None = None,
    enable_biopython_metadata: bool = True,
) -> pd.DataFrame:
    """Embed an uploaded sample live, then run the identical scan the cached path runs.

    The query vector is the only thing new here. The cosine scan, the offline
    annotation from `archs4_metadata.parquet`, and the `archs4_index` column are
    the same code the cached path uses, so an uploaded sample's hits carry the
    same schema - gse / title / tissue / species and a map position - as a
    precomputed OSDR sample's.
    """
    return run_uploaded_retrieval_with_neighborhood(
        counts_path,
        sample_column,
        topk,
        neighborhood_depth=topk,
        entrez_email=entrez_email,
        enable_biopython_metadata=enable_biopython_metadata,
    ).hits


def run_uploaded_retrieval_with_neighborhood(
    counts_path: str | Path,
    sample_column: str | None,
    topk: int,
    *,
    neighborhood_depth: int | None = None,
    entrez_email: str | None = None,
    enable_biopython_metadata: bool = True,
    scoring: str = _scoring.COSINE,
) -> UploadRetrieval:
    """Return requested upload hits, evidence neighborhood and score curve."""
    q_vec = embed_uploaded_counts(counts_path, sample_column)
    index_vecs, _, _ = _load_archs4_index()
    scan_k = _scan_depth(topk, neighborhood_depth)
    idx, score = _topk_cosine_from_memmap(index_vecs, q_vec, scan_k, scoring=scoring)
    ranked = _annotate_from_cache(idx, score)
    ranked["archs4_index"] = idx.astype(int)
    ranked = _with_plain_cosine(ranked, q_vec, scoring)
    hits, neighborhood = split_ranked_hits(ranked, topk, neighborhood_depth)
    if enable_biopython_metadata and _safe_str(entrez_email):
        hits = _enrich_hits_from_ncbi_eutils(hits, _safe_str(entrez_email))
    return UploadRetrieval(hits, neighborhood, score_profile(ranked))


# --- The fifth query-vector source: a whole experimental group ---------------

COHORT_MODE = "cohort"


def run_cohort_retrieval_with_neighborhood(
    sample_ids: Sequence[str],
    topk: int,
    *,
    neighborhood_depth: int | None = None,
    scoring: str = _scoring.COSINE,
) -> CohortRetrieval:
    """Pool a cohort and return requested hits plus one evidence prefix.

    The pooled vector is the only new thing. The cosine scan, the offline
    annotation from `archs4_metadata.parquet`, and the `archs4_index` column are
    the same code the cached path uses, so a cohort's hits carry the identical
    schema - gse / title / tissue / species and a map position - as one sample's.
    That is the same relationship file ingestion has to the cached path, and it
    is why nothing downstream of the hits frame needs to know a cohort produced
    it.

    Only the pooled vector is ranked. Returns `(hits, neighborhood, rows,
    profile)`; member vectors remain available for the existing cohort geometry
    without loading them twice.
    """
    from .cohorts import cohort_query_vector

    ids = [_safe_str(s) for s in sample_ids if _safe_str(s)]
    if not ids:
        raise RetrievalError("No samples were selected to pool.")

    rows, missing = cached_query_vectors(ids)
    if missing:
        raise RetrievalError(
            f"{len(missing)} of the {len(ids)} selected samples have no "
            "precomputed embedding, so the cohort cannot be pooled as described. "
            "Re-run precompute/embed_osdr.py.",
            detail="Missing: " + ", ".join(missing[:20]),
        )

    q_mat = cohort_query_vector(rows).reshape(1, -1)

    index_vecs, _, _ = _load_archs4_index()
    scan_k = _scan_depth(topk, neighborhood_depth)
    idx, score = _topk_cosine_matrix(
        index_vecs=index_vecs, q_mat=q_mat, k=scan_k,
        preserve_matrix_product=True, scoring=scoring)

    ranked = _annotate_from_cache(idx[0], score[0])
    ranked["archs4_index"] = idx[0].astype(int)
    ranked = _with_plain_cosine(ranked, q_mat[0], scoring)
    hits, neighborhood = split_ranked_hits(ranked, topk, neighborhood_depth)

    return CohortRetrieval(hits, neighborhood, rows, score_profile(ranked))


def run_cohort_retrieval(sample_ids: list[str] | tuple[str, ...], topk: int
                         ) -> tuple[pd.DataFrame, np.ndarray]:
    """Pool a cohort's cached vectors into one requested-depth result."""
    result = run_cohort_retrieval_with_neighborhood(
        sample_ids, topk, neighborhood_depth=topk
    )
    return result.hits, result.rows


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


# --- What one scan returns ----------------------------------------------------

# How deep the score-by-rank curve reaches. The memmap pass dominates a scan's
# cost at any k, so reaching 1,000 is measured at the same 0.4-0.5 s as the
# 250-deep scan this replaced. It is also the most a
# user can ask for, which means a request for 1,000 matches costs nothing extra.
SCORE_PROFILE_DEPTH = 1000
MAX_MATCHES = SCORE_PROFILE_DEPTH


class SampleRetrieval(NamedTuple):
    """One sample's requested hits, its evidence prefix, and its score curve."""

    hits: pd.DataFrame
    neighborhood: pd.DataFrame | None
    mode: str
    profile: list[float]


class CohortRetrieval(NamedTuple):
    hits: pd.DataFrame
    neighborhood: pd.DataFrame
    rows: np.ndarray
    profile: list[float]


class UploadRetrieval(NamedTuple):
    hits: pd.DataFrame
    neighborhood: pd.DataFrame
    profile: list[float]


def _scan_depth(topk: int, neighborhood_depth: int | None = None) -> int:
    return max(int(topk), int(neighborhood_depth or topk), SCORE_PROFILE_DEPTH)


def _ordered(ranked: pd.DataFrame) -> pd.DataFrame:
    """Best score first; an exact tie goes to the lower reference row.

    The same tie rule as `_topk_cosine_matrix`, restated because a ranked frame
    can arrive from a path that never went through it (the precomputed table).
    """
    if "archs4_index" in ranked.columns:
        ordered = ranked.sort_values(["score", "archs4_index"],
                                     ascending=[False, True], kind="mergesort")
    else:
        ordered = ranked.sort_values("score", ascending=False, kind="mergesort")
    return ordered.reset_index(drop=True)


def score_profile(ranked: pd.DataFrame,
                  depth: int = SCORE_PROFILE_DEPTH) -> list[float]:
    """The best `depth` scores of one scan, best first, as plain floats.

    Only scores, never identities: the curve says how quickly similarity falls
    away with rank, and the rows past the requested k are not results.
    """
    if ranked is None or ranked.empty:
        return []
    scores = _ordered(ranked)["score"].head(int(depth))
    return [float(s) for s in scores.to_numpy(dtype=np.float64)]


def reference_vectors(indices: Sequence[int]) -> np.ndarray:
    """Unit-normalized ARCHS4 vectors for memmap rows, in the order given.

    Float64, because the identical-profile rule compares cosine distances
    near 1e-6, which float32 accumulation would blur. Reading 1,000 rows was
    measured at about 1 ms once the scan has warmed the pages.
    """
    idx = np.asarray([int(i) for i in indices], dtype=np.int64)
    if idx.size == 0:
        return np.zeros((0, 0), dtype=np.float64)
    index_vecs, _, _ = _load_archs4_index()
    order = np.argsort(idx, kind="stable")
    block = np.empty((idx.size, int(index_vecs.shape[1])), dtype=np.float64)
    block[order] = np.asarray(index_vecs[idx[order]], dtype=np.float64)
    norms = np.linalg.norm(block, axis=1, keepdims=True)
    return block / np.maximum(norms, 1e-12)


def split_ranked_hits(
    ranked: pd.DataFrame,
    requested_k: int,
    neighborhood_depth: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split one ranked scan into the requested prefix and evidence prefix.

    The evidence prefix defaults to the requested prefix: since 2026-10-22 the
    map's neighborhood explorer is the retrieved matches, no more and no fewer.
    """
    ordered = _ordered(ranked)
    requested = ordered.head(max(1, int(requested_k))).copy().reset_index(drop=True)
    depth = requested_k if neighborhood_depth is None else neighborhood_depth
    neighborhood = (
        ordered.head(max(1, int(depth))).copy().reset_index(drop=True)
    )
    return requested, neighborhood


def search_hits_with_neighborhood(
    samples_df: pd.DataFrame,
    sample_id: str,
    topk: int,
    *,
    neighborhood_depth: int | None = None,
    entrez_email: str | None = None,
    enable_biopython_metadata: bool = True,
    scoring: str = _scoring.COSINE,
) -> SampleRetrieval:
    """Retrieve requested hits, a locatable evidence prefix, and a score curve.

    The demo and precomputed paths scan only the requested depth, so their
    curve is just the requested scores and they carry no neighborhood.
    """
    row = samples_df.loc[samples_df["sample_id"] == sample_id]
    if row.empty:
        raise ValueError(f"Unknown sample_id: {sample_id}")

    scan_k = _scan_depth(topk, neighborhood_depth)
    if cached_query_vector(sample_id) is not None:
        ranked = run_cached_query_retrieval(sample_id, scan_k, scoring=scoring)
        hits, neighborhood = split_ranked_hits(ranked, topk, neighborhood_depth)
        if enable_biopython_metadata and _safe_str(entrez_email):
            hits = _enrich_hits_from_ncbi_eutils(hits, _safe_str(entrez_email))
        return SampleRetrieval(hits, neighborhood, "cached", score_profile(ranked))

    hits, mode = search_hits(
        samples_df,
        sample_id,
        topk,
        entrez_email,
        enable_biopython_metadata,
    )
    return SampleRetrieval(hits, None, mode, score_profile(hits))



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
