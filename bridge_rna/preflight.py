"""Whether this machine can actually run a retrieval, and which files it would use.

The checks here are the difference between a fresh clone that fails at layout
time with an actionable list and one that looks healthy until someone clicks
Search. The gene-order digest check is the load-bearing one: a stand-in list of
the right length builds query vectors in a different gene space than the index,
which produces plausible-looking similarity scores that mean nothing.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

from .config import (
    DEMO_SCRIPT_PATH,
    EMBEDDING_DIR,
    OSDR_METADATA_PATH,
    ROOT,
)
from .util import _find_first_existing


def _infer_canonical_genes_from_checkpoint(checkpoint_path: Path) -> Path | None:
    """Create canonical_genes CSV sized to checkpoint embedding rows when absent.

    Uses data/ensembl/protein_coding_ortholog_genes.txt as the source list and
    truncates to checkpoint gene_embedding length.
    """
    source_txt = ROOT / "data" / "ensembl" / "protein_coding_ortholog_genes.txt"
    out_csv = ROOT / "data" / "ensembl" / "canonical_genes.inferred.csv"

    if not checkpoint_path.exists() or not source_txt.exists():
        return None

    if out_csv.exists():
        return out_csv

    try:
        import torch

        ckpt = torch.load(checkpoint_path, map_location="cpu")
        state = ckpt.get("model_state_dict", {})
        target_n = None
        for name, tensor in state.items():
            if "gene_embedding.weight" in name:
                target_n = int(tensor.shape[0])
                break
        if target_n is None or target_n <= 0:
            return None

        gene_df = pd.read_csv(source_txt, sep="\t", header=None, names=["gene_symbol"])
        genes = [str(g).strip() for g in gene_df["gene_symbol"].tolist() if str(g).strip()]

        deduped = []
        seen = set()
        for g in genes:
            if g not in seen:
                seen.add(g)
                deduped.append(g)

        if len(deduped) < target_n:
            return None

        out_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"gene_symbol": deduped[:target_n]}).to_csv(out_csv, index=False)
        return out_csv
    except Exception:
        return None


def _checkpoint_gene_count(checkpoint_path: Path) -> int | None:
    try:
        import torch

        ckpt = torch.load(checkpoint_path, map_location="cpu")
        state = ckpt.get("model_state_dict", {})
        for name, tensor in state.items():
            if "gene_embedding.weight" in name:
                return int(tensor.shape[0])
    except Exception:
        return None
    return None


def _checkpoint_attention_config(checkpoint_path: Path) -> tuple[str | None, str | None]:
    """Return (feature_type, compute_type) from checkpoint config when available."""
    try:
        import torch

        ckpt = torch.load(checkpoint_path, map_location="cpu")
        cfg = ckpt.get("config", {})
        feature = str(cfg.get("feature_type", "")).strip().lower() or None
        compute = str(cfg.get("compute_type", "")).strip().lower() or None
        return feature, compute
    except Exception:
        return None, None


def _canonical_matches_checkpoint(canonical_path: Path, checkpoint_path: Path) -> bool:
    expected = _checkpoint_gene_count(checkpoint_path)
    if expected is None:
        return True
    try:
        df = pd.read_csv(canonical_path)
        if "gene_symbol" not in df.columns:
            return False
        return int(df["gene_symbol"].astype(str).str.len().gt(0).sum()) == expected
    except Exception:
        return False


def _canonical_gene_order_is_authoritative(canonical_path: Path | None) -> bool:
    """True if this file carries the exact gene ordering the index was built with.

    The count check above cannot separate the real list from a stand-in of the
    same length, which is precisely how a scrambled gene space went unnoticed.
    Comparing the hashed ordering against the digest recorded alongside the
    model is the only test that distinguishes them, so it gates every
    user-facing claim that results are valid.
    """
    if canonical_path is None or not canonical_path.exists():
        return False
    try:
        from generate_archs4_embeddings import CANONICAL_GENES_SHA256, canonical_gene_order_digest

        genes = pd.read_csv(canonical_path)["gene_symbol"].astype(str).tolist()
        return canonical_gene_order_digest(genes) == CANONICAL_GENES_SHA256
    except Exception:
        return False


def _is_lfs_pointer(path: Path) -> bool:
    """True if ``path`` is an unfetched Git LFS pointer rather than real data.

    Pointer files are small text stubs beginning with a version line. Checking
    the size first keeps this cheap enough to run on every preflight, and means
    a real multi-hundred-megabyte artifact is never opened.
    """
    try:
        if not path.is_file() or path.stat().st_size > 1024:
            return False
        with path.open("rb") as fh:
            return fh.read(40).startswith(b"version https://git-lfs")
    except OSError:
        return False


def preflight_retrieval_requirements() -> tuple[list[str], dict[str, Path]]:
    """Return missing requirements and resolved runtime paths for demo retrieval."""
    missing: list[str] = []

    resolved_paths: dict[str, Path] = {}
    resolved_paths["checkpoint"] = _find_first_existing(
        [
            ROOT / "checkpoints_performer" / "r7hnr92k" / "best_model.pt",
            ROOT / "r7hnr92k" / "best_model.pt",
        ]
    )
    resolved_paths["orthologs"] = _find_first_existing([ROOT / "data" / "ensembl" / "orthologs_one2one.txt"])
    canonical_candidates = [
        ROOT / "data" / "archs4" / "train_orthologs" / "canonical_genes.csv",
        ROOT / "data" / "ensembl" / "canonical_genes.inferred.csv",
        ROOT / "data" / "ensembl" / "protein_coding_genes.csv",
    ]

    # Prefer a list whose ordering actually matches the index over one that
    # merely exists at the right path. Previously this took the first existing
    # candidate, which made the validating loop below unreachable and meant a
    # wrong-order file in the authoritative location would be trusted silently.
    canonical_candidate = next(
        (c for c in canonical_candidates if _canonical_gene_order_is_authoritative(c)),
        None,
    )

    if canonical_candidate is None and resolved_paths["checkpoint"] is not None:
        for candidate in canonical_candidates:
            if candidate.exists() and _canonical_matches_checkpoint(candidate, resolved_paths["checkpoint"]):
                canonical_candidate = candidate
                break

    if canonical_candidate is None:
        canonical_candidate = _find_first_existing(canonical_candidates)

    if canonical_candidate is None and resolved_paths["checkpoint"] is not None:
        canonical_candidate = _infer_canonical_genes_from_checkpoint(resolved_paths["checkpoint"])

    resolved_paths["canonical_genes"] = canonical_candidate
    resolved_paths["mouse_exon_lengths"] = _find_first_existing(
        [ROOT / "data" / "gencode" / "gencode_v49_mouse_gene_exon_lengths.csv"]
    )

    if importlib.util.find_spec("torch") is None:
        missing.append("python module: torch")
    if not DEMO_SCRIPT_PATH.exists():
        missing.append(f"file: {DEMO_SCRIPT_PATH}")
    if not EMBEDDING_DIR.exists():
        missing.append(f"directory: {EMBEDDING_DIR}")
    if not OSDR_METADATA_PATH.exists():
        missing.append(f"file: {OSDR_METADATA_PATH}")
    if not (ROOT / "generate_archs4_embeddings.py").exists():
        missing.append(f"file: {ROOT / 'generate_archs4_embeddings.py'}")
    if resolved_paths["checkpoint"] is not None:
        feature_type, compute_type = _checkpoint_attention_config(resolved_paths["checkpoint"])
        resolved_paths["feature_type"] = feature_type
        resolved_paths["compute_type"] = compute_type
        if feature_type is not None and feature_type != "flash":
            missing.append(
                f"unsupported checkpoint attention mode: feature_type={feature_type}. "
                "This app is configured for flash attention inference."
            )
        if feature_type not in (None, "flash") and not (ROOT / "slim_performer_model.py").exists():
            missing.append(f"file: {ROOT / 'slim_performer_model.py'}")

    osdr_data_dir = _find_first_existing([ROOT / "data" / "osdr", ROOT / "osdr"])
    if osdr_data_dir is None:
        missing.append("directory: missing required OSDR base directory (data/osdr or osdr)")
    else:
        resolved_paths["osdr_data_dir"] = osdr_data_dir

    for key in ["checkpoint", "orthologs", "canonical_genes", "mouse_exon_lengths", "osdr_data_dir"]:
        if resolved_paths[key] is None:
            missing.append(f"file: missing required {key}")

    # A clone without `git lfs pull` leaves ~130-byte pointer stubs in place of
    # the real binaries. Every path check above passes, then torch.load fails
    # deep inside retrieval with an error that says nothing about Git LFS.
    for label, candidate in (
        ("checkpoint", resolved_paths.get("checkpoint")),
        ("embedding index", EMBEDDING_DIR / "sample_embeddings.float16.mmap"),
        ("sample locations", EMBEDDING_DIR / "sample_locations.parquet"),
    ):
        if candidate is not None and _is_lfs_pointer(Path(candidate)):
            missing.append(
                f"Git LFS: {label} ({Path(candidate).name}) is an unfetched pointer stub. Run 'git lfs pull'."
            )

    return missing, resolved_paths
