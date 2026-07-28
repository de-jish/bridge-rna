#!/usr/bin/env python
"""Embed a single uploaded OSDR counts file to a 512-d ExpressionPerformer vector.

This is the live-embedding half of the Retrieve view's file-ingestion feature.
It is a subprocess by design. The serving app never imports torch - that is an
invariant pinned by `tests/test_app.py::test_the_serving_app_does_not_import_the_scientific_stack` -
so an uploaded sample is embedded the same way the `demo` path embeds an OSDR
catalog sample: by shelling out to a script that loads the checkpoint, embeds one
sample, writes the vector, and exits.

The preprocessing is not re-implemented here. It reuses the exact symbols
funnelled through `manifold.bridge_rna` (the same ones `embed_osdr.py` uses), so
an uploaded sample is embedded in byte-for-byte the same gene order, ortholog
mapping, TPM/log1p pipeline, and encode call as the 940,455-sample corpus it is
about to be compared against. **Invariant 1 (the gene-digest gate) is enforced
before any vector is produced**, so a sample can never be embedded in a gene
order the ARCHS4 index was not built in.

Input: a counts CSV/TSV with mouse Ensembl gene IDs in column 0 and one or more
sample columns (the OSDR counts-matrix format). Output: a float32 `.npy` of
shape (512,). A one-line JSON summary is printed to stdout on success so the
caller can report which column was embedded and how many genes mapped.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Make the manifold package importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from manifold import paths  # noqa: E402
from manifold.bridge_rna import load_bridge_rna_symbols  # noqa: E402


def _assert_gene_digest(canonical_genes: list[str], sym: dict) -> str:
    """Invariant 1: the content-hashed gene order must match the ARCHS4 index's.

    The same gate `embed_osdr.py` runs. An uploaded sample embedded in a
    different gene order is not degraded, it is meaningless - every gene's
    expression lands on the wrong input position - and nothing downstream would
    notice. Refuse rather than produce it.
    """
    digest = sym["canonical_gene_order_digest"](canonical_genes)
    expected = sym["CANONICAL_GENES_SHA256"]
    if digest != expected:
        raise SystemExit(
            "ABORT: canonical gene order digest mismatch.\n"
            f"  expected {expected}\n  found    {digest}\n"
            "An embedding built with the wrong gene order looks valid but is "
            "scientifically wrong. Refusing to produce it."
        )
    return digest


def _read_counts(counts_path: Path) -> pd.DataFrame:
    """Read a counts matrix, tolerating CSV, TSV, and gzip, gene IDs in column 0."""
    suffixes = {s.lower() for s in counts_path.suffixes}
    sep = "\t" if (".tsv" in suffixes or ".txt" in suffixes) else ","
    try:
        df = pd.read_csv(counts_path, sep=sep, index_col=0)
    except Exception as exc:  # noqa: BLE001 - surfaced as a clean one-liner upstream
        raise SystemExit(f"ABORT: could not read counts file: {exc}")
    if df.shape[1] == 0:
        raise SystemExit(
            "ABORT: the counts file has no sample columns. Expected mouse Ensembl "
            "gene IDs in the first column and one or more sample columns."
        )
    return df


def preprocess_counts(
    counts_path: Path,
    sample_column: str | None,
    orthologs_path: Path,
    exon_lengths_path: Path,
    canonical_genes: list[str],
    sym: dict,
) -> tuple[np.ndarray, str, int]:
    """Turn one sample column into the (15165,) log1p-TPM vector the model eats.

    This is the exact math of the Bridge RNA single-sample path
    (`demo_osdr_top5.py`) and the corpus path (`embed_osdr.py`): map mouse
    Ensembl -> human ortholog symbol, sum duplicates, reindex to the canonical
    genes, TPM-normalize, log1p.
    """
    ensembl_to_human, human_length_map = sym["build_mouse_to_human_maps"](
        orthologs_path, exon_lengths_path
    )
    normalize_tpm = sym["normalize_counts_to_tpm_single"]

    counts_df = _read_counts(counts_path)
    counts_df.index = counts_df.index.astype(str).str.replace(r"\..*$", "", regex=True)

    if not sample_column or str(sample_column).strip() == "":
        sample_column = str(counts_df.columns[0])
    if sample_column not in counts_df.columns:
        preview = list(map(str, counts_df.columns[:8]))
        raise SystemExit(
            f"ABORT: sample column '{sample_column}' is not in the counts file. "
            f"Available columns include: {preview}"
        )

    counts = counts_df[sample_column].astype(np.float32)
    mapped_human = pd.Series(counts.index, index=counts.index).map(ensembl_to_human)
    keep = mapped_human.notna()
    n_mapped = int(keep.sum())
    if n_mapped == 0:
        raise SystemExit(
            "ABORT: no genes mapped through the mouse->human ortholog table. This "
            "file does not look like mouse Ensembl gene IDs (e.g. ENSMUSG...). "
            "OSDR spaceflight data is Mus musculus; human-indexed matrices are not "
            "supported on this path."
        )

    c = counts.loc[keep.values].copy()
    c.index = mapped_human[keep].values
    c = c.groupby(level=0).sum()
    c = c.reindex(canonical_genes, fill_value=0.0).astype(np.float32)
    c_tpm = normalize_tpm(c, human_length_map)
    x = np.log1p(np.maximum(c_tpm.values.astype(np.float32), 0.0))
    return x, str(sample_column), n_mapped


def load_model(checkpoint_path: Path, device: torch.device, sym: dict):
    """Build the ExpressionPerformer from `ckpt['config']` (invariant 3)."""
    ckpt = torch.load(checkpoint_path, map_location="cpu")
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


def embed(x: np.ndarray, model, device: torch.device) -> np.ndarray:
    """Encode one (15165,) vector to (512,), exactly as the demo path does.

    Species is passed as None: the OSDR embedding path does not use the model's
    species embedding, and `normalize=False` matches the corpus build so cosine
    similarity is computed on the same raw direction the ARCHS4 index carries.
    """
    xt = torch.from_numpy(x[None, :]).to(device)
    with torch.no_grad(), torch.amp.autocast(
        device_type="cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")
    ):
        q = model.encode(xt, None, normalize=False)
    return q.detach().float().cpu().numpy().reshape(-1).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Embed one uploaded OSDR counts file to a 512-d vector."
    )
    ap.add_argument("--counts", required=True, help="Path to the uploaded counts file.")
    ap.add_argument("--out", required=True, help="Where to write the (512,) float32 .npy.")
    ap.add_argument("--sample", default="",
                    help="Which sample column to embed. Defaults to the first column.")
    ap.add_argument("--checkpoint", default=str(paths.CHECKPOINT))
    ap.add_argument("--orthologs", default=str(paths.ORTHOLOGS_TXT))
    ap.add_argument("--canonical-genes", default=str(paths.CANONICAL_GENES_CSV))
    ap.add_argument("--mouse-exon-lengths", default=str(paths.MOUSE_EXON_LENGTHS_CSV))
    ap.add_argument("--device", default="cpu",
                    help="cpu, mps, or cuda. Falls back to cpu when unavailable.")
    args = ap.parse_args()

    sym = load_bridge_rna_symbols()

    canonical_genes = (
        pd.read_csv(args.canonical_genes)["gene_symbol"].astype(str).tolist()
    )
    _assert_gene_digest(canonical_genes, sym)

    x, used_column, n_mapped = preprocess_counts(
        counts_path=Path(args.counts),
        sample_column=args.sample,
        orthologs_path=Path(args.orthologs),
        exon_lengths_path=Path(args.mouse_exon_lengths),
        canonical_genes=canonical_genes,
        sym=sym,
    )

    want = args.device
    if want.startswith("cuda") and not torch.cuda.is_available():
        want = "cpu"
    if want == "mps" and not (getattr(torch.backends, "mps", None)
                              and torch.backends.mps.is_available()):
        want = "cpu"
    device = torch.device(want)

    model = load_model(Path(args.checkpoint), device, sym)
    vec = embed(x, model, device)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, vec)

    print(json.dumps({
        "ok": True,
        "dim": int(vec.shape[0]),
        "sample_column": used_column,
        "genes_mapped": n_mapped,
        "out": str(out_path),
    }), flush=True)


if __name__ == "__main__":
    main()
