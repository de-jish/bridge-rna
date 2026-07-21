#!/usr/bin/env python3
"""Phase 1: generate OSDR sample embeddings, gated on the canonical gene digest.

This is the highest-risk piece of Bridge Manifold: a subtle preprocessing
mismatch produces embeddings that look fine but are scientifically wrong. The
mitigation is to reproduce Bridge RNA's exact preprocessing (imported, not
reinvented) and to abort unless the canonical gene ordering hashes to the same
digest the ARCHS4 index was built with.

Recipe, in exact order (mirrors demo_osdr_top5.load_random_osdr_sample_vector):
  1. Filter OSDR metadata to Mus musculus rows with a counts path and a
     non-empty spaceflight factor - the same population Bridge RNA retrieves.
  2. Per sample: read the raw-count CSV, strip Ensembl version suffixes.
  3. Map mouse Ensembl IDs -> human symbols (one2one), sum duplicates.
  4. Reindex to the 15,165 canonical genes in canonical order, fill 0.
  5. TPM-normalize with mouse exon lengths, then log1p.
  6. Batch through ExpressionPerformer.encode(x, None, normalize=False).

Output:
  cache/osdr_expression.float32.npy          (N x 15165, the log1p-TPM stage)
  cache/osdr_expression_meta.parquet         (rows aligned to the matrix)
  cache/osdr_sample_embeddings.float32.npy   (N x 512)
  cache/osdr_metadata.parquet                (sample_key + color-by columns)

Measured cost: ~6.5 s/sample on CPU fp32 (10 threads), so the full run is a
multi-hour job. Two consequences shape this script. First, the expression stage
is cached separately, so a re-embed never re-reads the 71 counts CSVs. Second,
embedding writes into an on-disk memmap and records its progress, so an
interrupted run resumes instead of restarting.

Device: CPU fp32 is the fidelity baseline and the only viable option here. MPS
was measured and rejected: ``F.scaled_dot_product_attention`` has no fused
kernel for this shape on Metal, so it materializes the full 15,165 x 15,165
attention matrix and fails with a 6.85 GiB allocation. On CPU the flash SDPA
path keeps memory flat.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Make the manifold package importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from manifold import paths, preflight  # noqa: E402
from manifold.bridge_rna import load_bridge_rna_symbols  # noqa: E402

# Color-by columns to carry into the metadata parquet, mapped to clean keys.
OSDR_COLORBY_COLUMNS = {
    "spaceflight": "study.factor value.spaceflight",
    "tissue": "study.characteristics.material type",
    "strain": "study.characteristics.strain",
    "sex": "study.characteristics.sex",
    "genotype": "study.factor value.genotype",
    "study": "id.accession",
    "habitat": "study.parameter value.habitat",
    "duration": "study.parameter value.duration",
    "diet": "study.parameter value.diet",
}


def _resolve_counts_path(raw_path: str, osdr_data_dir: Path) -> Path:
    """Resolve a counts file path against common OSDR locations (from Bridge RNA)."""
    p = Path(str(raw_path))
    if p.is_absolute() and p.exists():
        return p
    candidates = [Path.cwd() / p, osdr_data_dir / p]
    if "raw" in p.parts:
        raw_tail = p.parts[p.parts.index("raw") + 1 :]
        if raw_tail:
            candidates.append(osdr_data_dir / "raw" / Path(*raw_tail))
    candidates.append(osdr_data_dir / "raw" / p.name)
    for c in candidates:
        if c.exists():
            return c
    return p if p.is_absolute() else (Path.cwd() / p)


def load_eligible_metadata() -> pd.DataFrame:
    """The OSDR population Bridge RNA retrieves over: mouse, counts present, spaceflight set."""
    meta = pd.read_csv(paths.OSDR_METADATA_TSV, sep="\t")
    meta = meta[
        meta["study.characteristics.organism"]
        .astype(str)
        .str.contains("Mus musculus", case=False, na=False)
    ].copy()
    meta = meta[meta["counts_path"].notna()].copy()

    sf = meta["study.factor value.spaceflight"]
    missing_sf = (
        sf.isna()
        | sf.astype(str).str.strip().eq("")
        | sf.astype(str).str.strip().str.lower().isin(["nan", "none", "na", "n/a"])
    )
    meta = meta[~missing_sf].copy()
    meta["sample_name"] = meta["id.sample name"].astype(str)
    meta["sample_key"] = meta["id.accession"].astype(str) + "|" + meta["sample_name"]
    if len(meta) == 0:
        raise RuntimeError("No eligible OSDR rows after filtering.")
    return meta.reset_index(drop=True)


def select_whole_studies(meta: pd.DataFrame, max_samples: int) -> pd.DataFrame:
    """Pick a development subset by including WHOLE studies until max_samples.

    Used only by ``--limit``. Whole studies are kept intact rather than taking a
    random sample of rows, because each OSD study is a coherent batch: a random
    row sample would shred exactly the cluster structure the map exists to show,
    and a dev cache built that way would validate nothing. Studies carrying a
    flight arm and more tissue variety are preferred so the small cache still
    exercises the spaceflight and tissue color-bys.
    """
    by_study = []
    for acc, rows in meta.groupby("id.accession", sort=False):
        tissues = rows["study.characteristics.material type"].astype(str).nunique()
        has_flight = bool(
            rows["study.factor value.spaceflight"]
            .astype(str)
            .str.contains("flight", case=False)
            .any()
        )
        by_study.append((acc, len(rows), tissues, has_flight))
    by_study.sort(key=lambda t: (-int(t[3]), -t[2], -t[1]))
    chosen, total = [], 0
    for acc, n, _, _ in by_study:
        if total >= max_samples:
            break
        chosen.append(acc)
        total += n
    sub = meta[meta["id.accession"].isin(chosen)].reset_index(drop=True)
    print(f"[subset] {len(chosen)} whole studies -> {len(sub)} samples "
          f"(target {max_samples})", flush=True)
    return sub


def _harmonize_categories(values: pd.Series) -> pd.Series:
    """Collapse casing/whitespace variants of the same category onto one label.

    OSDR metadata is curated per study, so the same value arrives as both
    ``Female`` and ``female``, which would otherwise split one biological group
    across two legend slots and two enrichment tests. Variants are folded onto
    the most frequent spelling, which keeps the label human-readable instead of
    forcing an arbitrary case convention. The unit suffix braces GeneLab writes
    (``38 {day}``) are unwrapped for display.
    """
    # fillna first: as of pandas 3.0 `astype(str)` leaves missing values as NA
    # rather than the literal "nan", so a later `.replace({"nan": ...})` never
    # sees them and a NA category survives all the way into the legend and the
    # enrichment tests.
    s = values.astype(str).fillna("Unknown")
    s = s.str.strip().str.replace(r"\s+", " ", regex=True)
    s = s.str.replace(r"\{(.+?)\}", r"\1", regex=True).str.strip()
    s = s.replace({"nan": "Unknown", "": "Unknown", "None": "Unknown",
                   "NA": "Unknown", "N/A": "Unknown"})
    s = s.fillna("Unknown")
    counts = s.value_counts()
    canonical: dict[str, str] = {}
    for label in counts.index:  # most frequent first
        canonical.setdefault(str(label).casefold(), str(label))
    return s.map(lambda v: canonical.get(str(v).casefold(), str(v)))


def _assert_gene_digest(canonical_genes: list[str], sym: dict) -> str:
    """The build gate: content-hashed gene order must match the ARCHS4 index's.

    Invariant 1. An OSDR embedding built against a different gene ordering is
    not degraded, it is meaningless - every gene's expression lands on the wrong
    embedding row - and nothing downstream would notice. Checked on every path
    that can produce or reuse an expression matrix, never just at first build.
    """
    digest = sym["canonical_gene_order_digest"](canonical_genes)
    expected = sym["CANONICAL_GENES_SHA256"]
    if digest != expected:
        raise SystemExit(
            "ABORT: canonical gene order digest mismatch.\n"
            f"  expected {expected}\n  found    {digest}\n"
            "An OSDR embedding built with the wrong gene order looks valid but is "
            "scientifically wrong. Refusing to produce it."
        )
    print(f"[gate] gene digest OK ({len(canonical_genes)} genes): {digest[:16]}...", flush=True)
    return digest


def build_expression_matrix(meta: pd.DataFrame, sym: dict,
                            canonical_genes: list[str] | None = None
                            ) -> tuple[np.ndarray, pd.DataFrame]:
    """Return an (N, 15165) log1p-TPM float32 matrix and the aligned metadata subset.

    Samples are grouped by counts file so each CSV is read exactly once, then
    every sample column in that file is processed. This is the same math as the
    Bridge RNA single-sample path, applied to the whole population.
    """
    ensembl_to_human, human_length_map = sym["build_mouse_to_human_maps"](
        paths.ORTHOLOGS_TXT, paths.MOUSE_EXON_LENGTHS_CSV
    )
    if canonical_genes is None:
        canonical_genes = pd.read_csv(
            paths.CANONICAL_GENES_CSV)["gene_symbol"].astype(str).tolist()
        _assert_gene_digest(canonical_genes, sym)

    normalize_tpm = sym["normalize_counts_to_tpm_single"]
    vectors: list[np.ndarray] = []
    kept_rows: list[int] = []
    missing_columns: list[tuple[str, str]] = []
    missing_files: list[str] = []

    grouped = meta.groupby(meta["counts_path"].astype(str), sort=False)
    n_groups = grouped.ngroups
    t0 = time.time()
    for gi, (raw_path, rows) in enumerate(grouped):
        counts_path = _resolve_counts_path(raw_path, paths.OSDR_DATA_DIR)
        if not counts_path.exists():
            print(f"[warn] counts file missing, skipping {len(rows)} rows: {counts_path}", flush=True)
            missing_files.extend([str(counts_path)] * len(rows))
            continue
        counts_df = pd.read_csv(counts_path, index_col=0)
        counts_df.index = counts_df.index.astype(str).str.replace(r"\..*$", "", regex=True)
        # Precompute the ensembl->human mapping for this file's index once.
        mapped_human = pd.Series(counts_df.index, index=counts_df.index).map(ensembl_to_human)
        keep = mapped_human.notna()
        if keep.sum() == 0:
            print(f"[warn] no ortholog-mappable genes in {counts_path.name}", flush=True)
            continue
        human_index = mapped_human[keep].values

        for row_idx, row in rows.iterrows():
            sample_name = str(row["sample_name"])
            if sample_name not in counts_df.columns:
                # The metadata TSV names a sample the counts matrix does not
                # carry a column for. Recorded rather than skipped in silence:
                # a corpus quietly 3% smaller than the metadata says is exactly
                # the kind of drift that makes later numbers unreproducible.
                missing_columns.append((counts_path.name, sample_name))
                continue
            counts = counts_df.loc[keep.values, sample_name].astype(np.float32)
            c = pd.Series(counts.values, index=human_index).groupby(level=0).sum()
            c = c.reindex(canonical_genes, fill_value=0.0).astype(np.float32)
            c_tpm = normalize_tpm(c, human_length_map)
            x = np.log1p(np.maximum(c_tpm.values.astype(np.float32), 0.0))
            vectors.append(x)
            kept_rows.append(row_idx)

        if (gi + 1) % 10 == 0 or gi + 1 == n_groups:
            print(
                f"[prep] {gi + 1}/{n_groups} files, {len(vectors)} samples "
                f"({time.time() - t0:.0f}s)",
                flush=True,
            )

    if not vectors:
        raise RuntimeError("No OSDR samples produced a valid expression vector.")

    dropped = len(meta) - len(vectors)
    if dropped:
        print(f"[drop] {dropped}/{len(meta)} eligible samples produced no vector: "
              f"{len(missing_files)} from absent counts files, "
              f"{len(missing_columns)} named a column the counts matrix lacks.", flush=True)
        for fname, sname in missing_columns[:5]:
            print(f"        e.g. {fname} has no column '{sname}'", flush=True)
        if len(missing_columns) > 5:
            print(f"        ... and {len(missing_columns) - 5} more", flush=True)

    X = np.stack(vectors).astype(np.float32)
    kept_meta = meta.loc[kept_rows].reset_index(drop=True)
    return X, kept_meta


def load_model(device: torch.device, sym: dict):
    ckpt = torch.load(paths.CHECKPOINT, map_location="cpu")
    cfg = dict(ckpt.get("config", {}))
    model = sym["ExpressionPerformer"](
        num_genes=15165,
        hidden_dim=int(cfg["hidden_dim"]),
        n_heads=int(cfg["num_heads"]),
        n_layers=int(cfg["num_layers"]),
        ffn_dim=int(cfg["ffn_dim"]),
        ree_base=float(cfg["ree_base"]),
        mask_token_id=float(cfg["mask_token"]),
        feature_type=str(cfg["feature_type"]),
        compute_type=str(cfg["compute_type"]),
        include_species_embedding=bool(cfg["include_species_embedding"]),
        num_species=2,
    )
    state = sym["_strip_module_prefix"](ckpt["model_state_dict"])
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model


def _eligible_fingerprint(meta: pd.DataFrame, gene_digest: str) -> str:
    """Identity of an expression build: which samples, in which gene order.

    Both halves matter. The sample set decides which rows exist; the gene digest
    decides what the columns mean. A cache built under a different canonical
    gene order is not stale-but-usable, it is wrong, so it must miss.
    """
    payload = gene_digest + "\n" + "\n".join(map(str, meta["sample_key"]))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def stage_expression(meta: pd.DataFrame, sym: dict, force: bool) -> tuple[np.ndarray, pd.DataFrame]:
    """Build (or reuse) the cached log1p-TPM matrix aligned to its metadata rows.

    Cached because the embedding stage is hours long and may need to be rerun;
    re-reading 71 counts CSVs each time is pure waste.

    The cache is keyed on a fingerprint of the *eligible* sample keys plus the
    canonical gene digest, recorded in a sidecar. Comparing against the eligible
    set rather than the cached (kept) rows matters: some eligible samples never
    produce a vector, so the two lists legitimately differ and a naive
    comparison would never hit. Carrying the gene digest in the key is what
    keeps invariant 1 intact on the reuse path - otherwise a rebuild under a
    changed gene order would silently inherit the old matrix.
    """
    x_path = paths.OSDR_EXPRESSION_NPY
    m_path = paths.OSDR_EXPRESSION_META_PARQUET
    k_path = paths.OSDR_EXPRESSION_KEY_JSON

    canonical_genes = pd.read_csv(paths.CANONICAL_GENES_CSV)["gene_symbol"].astype(str).tolist()
    gene_digest = _assert_gene_digest(canonical_genes, sym)
    fingerprint = _eligible_fingerprint(meta, gene_digest)

    if not force and x_path.exists() and m_path.exists() and k_path.exists():
        try:
            cached = json.loads(k_path.read_text())
        except (ValueError, OSError):
            cached = {}
        if cached.get("fingerprint") == fingerprint:
            cached_meta = pd.read_parquet(m_path)
            X = np.load(x_path, mmap_mode="r")
            if X.shape[0] == len(cached_meta):
                print(f"[prep] reusing cached expression matrix {X.shape} "
                      f"(fingerprint {fingerprint[:12]})", flush=True)
                return np.asarray(X), cached_meta
        print("[prep] cached expression matrix does not match this build; rebuilding",
              flush=True)

    X, kept_meta = build_expression_matrix(meta, sym, canonical_genes)
    np.save(x_path, X)
    kept_meta.to_parquet(m_path, index=False)
    k_path.write_text(json.dumps({
        "fingerprint": fingerprint,
        "gene_digest": gene_digest,
        "n_eligible": int(len(meta)),
        "n_kept": int(len(kept_meta)),
    }, indent=2))
    print(f"[prep] cached expression matrix -> {x_path.name} {X.shape}", flush=True)
    return X, kept_meta


def embed_matrix(model, X: np.ndarray, device: torch.device, batch_size: int,
                 resume: bool = True) -> np.ndarray:
    """Encode the (N, 15165) matrix to (N, 512), resumable across interruptions.

    Results stream into an on-disk float32 memmap and a sidecar records how many
    rows are final, so a killed run restarts where it stopped rather than
    repeating hours of inference. The SDPA backend is pinned to the fused
    kernels: the MATH fallback would materialize a 15,165 x 15,165 attention
    matrix per head and is not viable at this sequence length.
    """
    from torch.nn.attention import SDPBackend, sdpa_kernel

    n = X.shape[0]
    part_path = paths.OSDR_EMBEDDINGS_PARTIAL
    prog_path = paths.OSDR_EMBEDDINGS_PROGRESS

    # Identify the partial by the *content* of the matrix being embedded, not by
    # its row count. Two different OSDR populations can easily share a length,
    # and resuming across them would silently splice embeddings of one sample
    # set onto another.
    matrix_key = hashlib.sha256(
        np.ascontiguousarray(X[:: max(1, n // 64)], dtype=np.float32).tobytes()
    ).hexdigest()

    start_at = 0
    if resume and part_path.exists() and prog_path.exists():
        try:
            prog = json.loads(prog_path.read_text())
            if int(prog.get("n_total", -1)) == n and prog.get("matrix_key") == matrix_key:
                start_at = int(prog.get("n_done", 0))
            elif prog.get("matrix_key") not in (None, matrix_key):
                print("[embed] partial embedding belongs to a different expression "
                      "matrix; starting over", flush=True)
        except (ValueError, OSError):
            start_at = 0
    if start_at >= n and part_path.exists():
        print(f"[embed] resuming: all {n} rows already embedded", flush=True)
        return np.asarray(np.memmap(part_path, dtype=np.float32, mode="r", shape=(n, 512)))

    mode = "r+" if (part_path.exists() and start_at > 0) else "w+"
    out = np.memmap(part_path, dtype=np.float32, mode=mode, shape=(n, 512))
    if start_at:
        print(f"[embed] resuming at row {start_at}/{n}", flush=True)

    backends = [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]
    t0 = time.time()
    for start in range(start_at, n, batch_size):
        end = min(start + batch_size, n)
        xb = torch.from_numpy(np.ascontiguousarray(X[start:end])).to(device)
        with torch.no_grad(), sdpa_kernel(backends):
            emb = model.encode(xb, None, normalize=False)
        out[start:end] = emb.detach().float().cpu().numpy()
        out.flush()
        prog_path.write_text(json.dumps(
            {"n_done": end, "n_total": n, "matrix_key": matrix_key}))
        elapsed = time.time() - t0
        rate = (end - start_at) / max(elapsed, 1e-6)
        eta = (n - end) / max(rate, 1e-6)
        print(f"[embed] {end}/{n}  {rate:.2f} samp/s  elapsed {elapsed/60:.1f}m  "
              f"ETA {eta/60:.1f}m", flush=True)
    return np.asarray(out)


def write_metadata(kept_meta: pd.DataFrame) -> pd.DataFrame:
    """Project the raw OSDR columns onto the clean color-by keys the app uses."""
    out_meta = pd.DataFrame({"sample_key": kept_meta["sample_key"].to_numpy()})
    for clean, raw in OSDR_COLORBY_COLUMNS.items():
        if raw in kept_meta.columns:
            out_meta[clean] = _harmonize_categories(kept_meta[raw]).to_numpy()
        else:
            print(f"[warn] OSDR column absent, filling Unknown: {raw}", flush=True)
            out_meta[clean] = "Unknown"
    out_meta.to_parquet(paths.OSDR_METADATA_PARQUET, index=False)
    return out_meta


def main() -> None:
    ap = argparse.ArgumentParser(description="Embed OSDR samples with the ExpressionPerformer.")
    ap.add_argument("--device", default="cpu", choices=["cpu", "mps"],
                    help="cpu is the fidelity baseline; mps materializes attention and OOMs here.")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0,
                    help="Dev cache: embed only whole studies totalling ~N samples.")
    ap.add_argument("--no-resume", action="store_true",
                    help="Ignore any partial embedding and start over.")
    ap.add_argument("--rebuild-expression", action="store_true",
                    help="Re-read the counts CSVs even if the cached matrix matches.")
    ap.add_argument("--metadata-only", action="store_true",
                    help="Rewrite osdr_metadata.parquet from the cached expression "
                         "metadata and exit, without touching the embeddings.")
    args = ap.parse_args()

    paths.ensure_cache_dirs()

    if args.metadata_only:
        if not paths.OSDR_EXPRESSION_META_PARQUET.exists():
            raise SystemExit(
                f"--metadata-only needs {paths.OSDR_EXPRESSION_META_PARQUET.name}, "
                "which is written by a normal run."
            )
        kept_meta = pd.read_parquet(paths.OSDR_EXPRESSION_META_PARQUET)
        out_meta = write_metadata(kept_meta)
        print(f"[done] rewrote {paths.OSDR_METADATA_PARQUET.name} "
              f"({len(out_meta)} rows)", flush=True)
        return

    preflight.require(preflight.PRECOMPUTE_REQUIRED, "OSDR embedding")

    torch.set_num_threads(os.cpu_count() or 4)
    sym = load_bridge_rna_symbols()

    meta = load_eligible_metadata()
    print(f"[meta] {len(meta)} eligible OSDR samples across "
          f"{meta['id.accession'].nunique()} studies", flush=True)
    if args.limit:
        meta = select_whole_studies(meta, args.limit)

    X, kept_meta = stage_expression(meta, sym, force=args.rebuild_expression)
    print(f"[prep] expression matrix {X.shape}, nnz/row mean "
          f"{np.mean((X > 0).sum(1)):.0f}", flush=True)

    device = torch.device(args.device)
    model = load_model(device, sym)
    emb = embed_matrix(model, X, device, args.batch_size, resume=not args.no_resume)

    # Sanity: embeddings must be finite and must actually differ between samples.
    # An all-identical block would mean the expression signal never reached the
    # encoder, which is exactly the silent failure this script exists to avoid.
    assert np.isfinite(emb).all(), "non-finite OSDR embeddings"
    assert emb.shape[1] == 512, f"unexpected embedding dim {emb.shape[1]}"
    spread = float(np.std(emb, axis=0).mean())
    assert spread > 1e-4, f"OSDR embeddings are near-identical (mean per-dim std {spread:.2e})"
    norms = np.linalg.norm(emb, axis=1)
    print(f"[embed] done {emb.shape}; norms mean {norms.mean():.2f} "
          f"[{norms.min():.2f}, {norms.max():.2f}]; per-dim std {spread:.4f}", flush=True)

    np.save(paths.OSDR_EMBEDDINGS_NPY, np.asarray(emb, dtype=np.float32))
    out_meta = write_metadata(kept_meta)

    # The partial memmap is now redundant with the final npy.
    for p in (paths.OSDR_EMBEDDINGS_PARTIAL, paths.OSDR_EMBEDDINGS_PROGRESS):
        p.unlink(missing_ok=True)

    print(
        f"[done] wrote {paths.OSDR_EMBEDDINGS_NPY.name} and "
        f"{paths.OSDR_METADATA_PARQUET.name} ({len(out_meta)} rows)",
        flush=True,
    )


if __name__ == "__main__":
    main()
