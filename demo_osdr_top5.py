#!/usr/bin/env python3
"""Demo: embed a random OSDR sample and retrieve top-k ARCHS4 nearest hits.

This script REUSES the prebuilt ARCHS4 embedding index in
``archs4_sample_embeddings_full/`` and only embeds one OSDR sample for retrieval.

All default paths are resolved relative to the repository root (the directory
containing this file), so the script behaves identically regardless of the
working directory it is invoked from.
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Dict, Tuple
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import torch

from generate_archs4_embeddings import ExpressionPerformer, _strip_module_prefix

try:
    import archs4py as a4
    _ARCHS4PY_AVAILABLE = True
except ImportError:
    _ARCHS4PY_AVAILABLE = False

try:
    from Bio import Entrez

    _BIOPYTHON_AVAILABLE = True
except ImportError:
    _BIOPYTHON_AVAILABLE = False


# Repository root. Every default path below is anchored here rather than to the
# process working directory, so the CLI works from anywhere on the filesystem.
ROOT = Path(__file__).resolve().parent

# Candidate locations for the canonical gene list, in priority order. The first
# entry is the authoritative list produced by the training pipeline; the second
# is the checkpoint-derived stand-in written by the Dash app. See
# resolve_canonical_genes() for why the distinction matters.
CANONICAL_GENE_CANDIDATES = (
    ROOT / "data" / "archs4" / "train_orthologs" / "canonical_genes.csv",
    ROOT / "data" / "ensembl" / "canonical_genes.inferred.csv",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Random OSDR sample -> top-k ARCHS4 retrieval demo")
    parser.add_argument(
        "--embedding-dir",
        type=Path,
        default=ROOT / "archs4_sample_embeddings_full",
        help="Directory containing embedding_manifest.json, sample_embeddings.*.mmap, sample_locations.parquet",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "checkpoints_performer" / "r7hnr92k" / "best_model.pt",
        help="Checkpoint used to generate embeddings.",
    )
    parser.add_argument("--topk", type=int, default=5, help="Number of nearest neighbors to return.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for OSDR sample pick.")
    parser.add_argument(
        "--osdr-sample-name",
        type=str,
        default=None,
        help="If set, use this exact OSDR id.sample name instead of random sampling.",
    )
    parser.add_argument(
        "--select-best",
        type=int,
        default=1,
        metavar="N",
        help="Try N random OSDR candidates and use the one with the highest top-1 similarity score (default: 1, i.e. first valid sample).",
    )
    parser.add_argument("--metric", choices=["cosine", "dot"], default="cosine")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--l2-normalize",
        action="store_true",
        default=True,
        help="L2-normalize query and index vectors before retrieval (recommended; improves hit quality).",
    )
    parser.add_argument(
        "--no-l2-normalize",
        dest="l2_normalize",
        action="store_false",
        help="Disable L2 normalization.",
    )
    parser.add_argument(
        "--archs4-h5",
        type=Path,
        default=ROOT / "data" / "archs4" / "human_gene_v2.5.h5",
        help=(
            "ARCHS4 human HDF5 file used to fetch hit metadata via archs4py. "
            "Optional and not bundled with the repository; download separately "
            "from https://archs4.org/download to enable metadata enrichment."
        ),
    )
    parser.add_argument(
        "--mouse-archs4-h5",
        type=Path,
        default=ROOT / "data" / "archs4" / "mouse_gene_v2.5.h5",
        help=(
            "ARCHS4 mouse HDF5 file for metadata lookup (searched after human h5). "
            "Optional and not bundled; see --archs4-h5."
        ),
    )
    parser.add_argument(
        "--biopython-metadata",
        action="store_true",
        help="Augment hit metadata using Biopython Entrez GEO lookups.",
    )
    parser.add_argument(
        "--entrez-email",
        type=str,
        default=None,
        help="Email required by NCBI Entrez when --biopython-metadata is enabled.",
    )
    parser.add_argument(
        "--entrez-api-key",
        type=str,
        default=None,
        help="Optional NCBI API key for faster Entrez requests.",
    )
    parser.add_argument(
        "--biopython-pubmed",
        action="store_true",
        help="Also fetch PubMed metadata for PubMed IDs linked to GEO hits (requires --biopython-metadata).",
    )

    parser.add_argument(
        "--osdr-data-dir",
        type=Path,
        default=ROOT / "data" / "osdr",
        help="Base OSDR data folder containing metadata/ and raw/",
    )

    parser.add_argument(
        "--osdr-metadata",
        type=Path,
        default=None,
        help="Optional metadata TSV path. Defaults to <osdr-data-dir>/metadata/selected_sample_metadata.tsv",
    )
    parser.add_argument(
        "--orthologs",
        type=Path,
        default=ROOT / "data" / "ensembl" / "orthologs_one2one.txt",
    )
    parser.add_argument(
        "--canonical-genes",
        type=Path,
        default=None,
        help=(
            "Canonical gene list matching the checkpoint's training gene order. "
            "Defaults to the first of CANONICAL_GENE_CANDIDATES that exists."
        ),
    )
    parser.add_argument(
        "--mouse-exon-lengths",
        type=Path,
        default=ROOT / "data" / "gencode" / "gencode_v49_mouse_gene_exon_lengths.csv",
    )
    parser.add_argument(
        "--save-report-prefix",
        type=Path,
        default=None,
        help=(
            "If set, save results to files using this prefix: "
            "<prefix>.report.md, <prefix>.top_hits.csv, <prefix>.archs4_metadata.csv"
        ),
    )
    return parser.parse_args()


def _resolve_counts_path(raw_path: str, osdr_data_dir: Path) -> Path:
    """Resolve a counts file path against common OSDR locations."""
    p = Path(str(raw_path))
    if p.is_absolute() and p.exists():
        return p

    repo_root = Path(__file__).resolve().parent
    candidates = [
        Path.cwd() / p,
        repo_root / p,
        osdr_data_dir / p,
    ]

    # Handle metadata rows that already include "data/osdr/raw/...".
    if "raw" in p.parts:
        raw_tail = p.parts[p.parts.index("raw") + 1 :]
        if raw_tail:
            candidates.append(osdr_data_dir / "raw" / Path(*raw_tail))

    # Handle rows that only provide filename.
    candidates.append(osdr_data_dir / "raw" / p.name)

    for c in candidates:
        if c.exists():
            return c

    return p if p.is_absolute() else (Path.cwd() / p)


def build_mouse_to_human_maps(orthologs_path: Path, mouse_exon_lengths_path: Path) -> Tuple[Dict[str, str], Dict[str, float]]:
    ortho = pd.read_csv(orthologs_path, sep="\t")
    ortho = ortho[ortho["Human homology type"] == "ortholog_one2one"].copy()
    ortho["Gene stable ID"] = ortho["Gene stable ID"].astype(str).str.split(".").str[0]

    ensembl_to_human = dict(zip(ortho["Gene stable ID"], ortho["Human gene name"]))

    mouse_lengths = pd.read_csv(mouse_exon_lengths_path).drop_duplicates("gene_symbol")
    mouse_lengths = mouse_lengths.set_index("gene_symbol")["exon_length"]

    human_length_map: Dict[str, float] = {}
    for _, row in ortho[["Human gene name", "Gene name"]].drop_duplicates().iterrows():
        human_gene = row["Human gene name"]
        mouse_gene = row["Gene name"]
        if mouse_gene in mouse_lengths.index and mouse_lengths[mouse_gene] > 0 and human_gene not in human_length_map:
            human_length_map[human_gene] = float(mouse_lengths[mouse_gene])

    return ensembl_to_human, human_length_map


def normalize_counts_to_tpm_single(counts: pd.Series, exon_len_by_human_gene: Dict[str, float]) -> pd.Series:
    lengths_bp = pd.Series(exon_len_by_human_gene, dtype=np.float64).reindex(counts.index)
    keep_mask = lengths_bp.notna() & (lengths_bp > 0)
    counts_use = counts.loc[keep_mask].astype(np.float32)
    lengths_kb = (lengths_bp.loc[keep_mask] / 1000.0).astype(np.float32)

    rate = counts_use / lengths_kb
    denom = float(rate.sum())
    if denom <= 0:
        return pd.Series(0.0, index=counts.index, dtype=np.float32)

    tpm = (rate / denom) * 1e6
    out = pd.Series(0.0, index=counts.index, dtype=np.float32)
    out.loc[tpm.index] = tpm.astype(np.float32)
    return out


def resolve_canonical_genes(explicit: Path | None) -> Path:
    """Locate the canonical gene list, warning when it is not the authoritative one.

    The gene list defines the row order of the query expression vector, and it
    must match the order used to build the ARCHS4 index. Only the training
    pipeline's ``canonical_genes.csv`` is authoritative. Any stand-in derived
    from the checkpoint reproduces the gene *count* but not the gene *order*,
    which silently misaligns every gene index and yields retrievals that look
    plausible while being meaningless. Warn loudly rather than fail, so the
    fallback stays usable for development but can never be mistaken for real.
    """
    if explicit is not None:
        if not explicit.exists():
            raise FileNotFoundError(f"Canonical gene list not found: {explicit}")
        return explicit

    for candidate in CANONICAL_GENE_CANDIDATES:
        if candidate.exists():
            if candidate.name != "canonical_genes.csv":
                print(
                    f"[WARN] Authoritative gene list {CANONICAL_GENE_CANDIDATES[0]} is missing.\n"
                    f"[WARN] Falling back to {candidate}, which is derived from the\n"
                    "[WARN] checkpoint's gene COUNT only and does not reproduce the training\n"
                    "[WARN] gene ORDER. Retrieval results from this fallback are NOT valid\n"
                    "[WARN] and must not be interpreted biologically."
                )
            return candidate

    raise FileNotFoundError(
        "No canonical gene list found. Looked for:\n  "
        + "\n  ".join(str(c) for c in CANONICAL_GENE_CANDIDATES)
    )


def load_random_osdr_sample_vector(args: argparse.Namespace) -> Tuple[np.ndarray, str, pd.Series]:
    osdr_data_dir = args.osdr_data_dir
    osdr_metadata = args.osdr_metadata or (osdr_data_dir / "metadata" / "selected_sample_metadata.tsv")
    canonical_genes_path = resolve_canonical_genes(args.canonical_genes)

    for p in [osdr_metadata, args.orthologs, canonical_genes_path, args.mouse_exon_lengths]:
        if not p.exists():
            raise FileNotFoundError(f"Required file not found: {p}")

    meta = pd.read_csv(osdr_metadata, sep="\t")
    meta = meta[meta["study.characteristics.organism"].astype(str).str.contains("Mus musculus", case=False, na=False)].copy()
    meta = meta[meta["counts_path"].notna()].copy()

    sf_raw = meta["study.factor value.spaceflight"]
    missing_sf_mask = (
        sf_raw.isna()
        | sf_raw.astype(str).str.strip().eq("")
        | sf_raw.astype(str).str.strip().str.lower().isin(["nan", "none", "na", "n/a"])
    )
    meta = meta[~missing_sf_mask].copy()

    meta["sample_name"] = meta["id.sample name"].astype(str)
    meta["sample_id"] = meta["id.accession"].astype(str) + "|" + meta["sample_name"]

    if len(meta) == 0:
        raise RuntimeError("No eligible OSDR rows after filtering.")

    ensembl_to_human, human_length_map = build_mouse_to_human_maps(args.orthologs, args.mouse_exon_lengths)
    canonical_genes = pd.read_csv(canonical_genes_path)["gene_symbol"].astype(str).tolist()

    requested_sample_name = (args.osdr_sample_name or "").strip()
    if requested_sample_name:
        selected = meta[meta["sample_name"] == requested_sample_name].copy()
        if len(selected) == 0:
            raise RuntimeError(
                f"Requested OSDR sample name not found after filtering: {requested_sample_name}"
            )
        candidate_rows = [selected.iloc[i] for i in range(len(selected))]
    else:
        rng = np.random.default_rng(args.seed)
        order = rng.permutation(len(meta))
        candidate_rows = [meta.iloc[int(idx)] for idx in order]

    for row in candidate_rows:
        counts_path = _resolve_counts_path(str(row["counts_path"]), osdr_data_dir)
        sample_name = str(row["sample_name"])
        sample_id = str(row["sample_id"])
        if not counts_path.exists():
            continue

        counts_df = pd.read_csv(counts_path, index_col=0)
        counts_df.index = counts_df.index.astype(str).str.replace(r"\..*$", "", regex=True)

        if sample_name not in counts_df.columns:
            continue

        counts = counts_df[sample_name].astype(np.float32)

        # Map mouse Ensembl genes -> human symbols, sum duplicates.
        mapped_human = pd.Series(counts.index, index=counts.index).map(ensembl_to_human)
        keep = mapped_human.notna()
        if keep.sum() == 0:
            continue

        c = counts.loc[keep.values].copy()
        c.index = mapped_human[keep].values
        c = c.groupby(level=0).sum()

        c = c.reindex(canonical_genes, fill_value=0.0).astype(np.float32)
        c_tpm = normalize_counts_to_tpm_single(c, human_length_map)

        # Match the embedding script's default (checkpoint normalization is log1p_tpm).
        x = np.log1p(np.maximum(c_tpm.values.astype(np.float32), 0.0))
        return x, sample_id, row

    if requested_sample_name:
        raise RuntimeError(
            f"Requested OSDR sample '{requested_sample_name}' was found but has no readable counts/columns after processing."
        )
    raise RuntimeError("Failed to locate a valid OSDR sample with readable counts.")


def build_model_and_query_embedding(args: argparse.Namespace, query_vec: np.ndarray) -> np.ndarray:
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = dict(ckpt.get("config", {}))

    model = ExpressionPerformer(
        num_genes=query_vec.shape[0],
        hidden_dim=int(cfg.get("hidden_dim", 512)),
        n_heads=int(cfg.get("num_heads", 8)),
        n_layers=int(cfg.get("num_layers", 4)),
        ffn_dim=int(cfg.get("ffn_dim", int(cfg.get("hidden_dim", 512)) * 4)),
        ree_base=float(cfg.get("ree_base", 100.0)),
        mask_token_id=float(cfg.get("mask_token", -10.0)),
        feature_type=str(cfg.get("feature_type", "sqr")),
        compute_type=str(cfg.get("compute_type", "iter")),
        include_species_embedding=bool(cfg.get("include_species_embedding", False)),
        num_species=2,
    )
    state = _strip_module_prefix(ckpt["model_state_dict"])
    model.load_state_dict(state, strict=False)

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    model.to(device)
    model.eval()

    x = torch.from_numpy(query_vec[None, :]).to(device)
    with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
        q = model.encode(x, None, normalize=False)
    return q.detach().float().cpu().numpy().reshape(-1)


def topk_search(index_vecs: np.ndarray, q: np.ndarray, k: int, metric: str, l2_normalize: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    k = int(min(max(k, 1), index_vecs.shape[0]))

    if l2_normalize:
        q = q / (float(np.linalg.norm(q)) + 1e-12)
        norms = np.linalg.norm(index_vecs, axis=1, keepdims=True) + 1e-12
        index_vecs = index_vecs / norms

    if metric == "cosine":
        # After L2 norm, cosine == dot product; keep the manual path for when l2=False.
        if not l2_normalize:
            qn = float(np.linalg.norm(q)) + 1e-12
            xnorm = np.linalg.norm(index_vecs, axis=1) + 1e-12
            scores = (index_vecs @ q) / (xnorm * qn)
        else:
            scores = index_vecs @ q
    else:
        scores = index_vecs @ q

    top_idx = np.argpartition(scores, -k)[-k:]
    top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
    return top_idx, scores[top_idx]


def fetch_archs4_metadata(geo_accessions: list, human_h5: Path, mouse_h5: Path | None = None) -> pd.DataFrame:
    """Pull sample metadata from ARCHS4 HDF5 files (human and/or mouse)."""
    if not _ARCHS4PY_AVAILABLE:
        print(
            "[WARN] archs4py is not installed, so hits are reported as bare GSM\n"
            "[WARN] accessions without tissue/title metadata. Retrieval itself is\n"
            "[WARN] unaffected. To enable enrichment:\n"
            "[WARN]   python -m pip install -r requirements-optional.txt\n"
            "[WARN] and download the ARCHS4 HDF5 files from https://archs4.org/download"
        )
        return pd.DataFrame({"geo_accession": geo_accessions})

    missing_h5 = [p for p in (human_h5, mouse_h5) if p is not None and not p.exists()]
    if missing_h5 and not any(p is not None and p.exists() for p in (human_h5, mouse_h5)):
        print(
            "[WARN] archs4py is installed but no ARCHS4 HDF5 file was found at:\n"
            + "".join(f"[WARN]   {p}\n" for p in missing_h5)
            + "[WARN] These files are not bundled with the repository. Download them\n"
            "[WARN] from https://archs4.org/download and pass --archs4-h5 /\n"
            "[WARN] --mouse-archs4-h5, or omit metadata enrichment."
        )

    frames = []
    remaining = list(geo_accessions)

    for h5_path, species_label in [(human_h5, "human"), (mouse_h5, "mouse")]:
        if not remaining:
            break
        if h5_path is None or not h5_path.exists():
            continue
        try:
            df = a4.meta.samples(str(h5_path), remaining)
            if isinstance(df, pd.DataFrame) and not df.empty:
                df["species"] = species_label
                frames.append(df)
                found = set(df["geo_accession"].tolist()) if "geo_accession" in df.columns else set(df.index.tolist())
                remaining = [g for g in remaining if g not in found]
        except Exception as e:
            print(f"[WARN] archs4py metadata fetch from {h5_path.name} failed: {e}")

    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame({"geo_accession": geo_accessions})


def fetch_geo_metadata_biopython(geo_accessions: list[str], email: str, api_key: str | None = None) -> pd.DataFrame:
    """Fetch additional GEO metadata for GSM accessions via Biopython Entrez."""
    if not _BIOPYTHON_AVAILABLE:
        print("[WARN] biopython is not installed - skipping Entrez metadata enrichment.")
        return pd.DataFrame({"geo_accession": geo_accessions})

    def _entrez_get(obj, key: str, default=None):
        if obj is None:
            return default
        try:
            getter = getattr(obj, "get", None)
            if callable(getter):
                return getter(key, default)
        except Exception:
            pass
        try:
            return obj[key] if key in obj else default
        except Exception:
            return default

    def _as_list(value):
        if value is None:
            return []
        if isinstance(value, (list, tuple)):
            return list(value)
        return [value]

    def _fetch_geo_soft_fields(gse_accession: str, ftp_link: str | None) -> dict[str, str | None]:
        out = {
            "geo_summary_biopython": None,
            "geo_overall_design_biopython": None,
            "geo_abstract_biopython": None,
        }
        if not gse_accession:
            return out

        gse = str(gse_accession).strip()
        if not gse:
            return out
        if not gse.upper().startswith("GSE"):
            gse = f"GSE{gse}"

        if ftp_link and str(ftp_link).strip().startswith("ftp://"):
            base = str(ftp_link).strip().rstrip("/")
            soft_url = f"{base}/soft/{gse}_family.soft.gz"
        else:
            prefix = f"{gse[:-3]}nnn" if len(gse) > 3 else gse
            soft_url = f"ftp://ftp.ncbi.nlm.nih.gov/geo/series/{prefix}/{gse}/soft/{gse}_family.soft.gz"

        try:
            req = Request(soft_url, headers={"User-Agent": "bridge-rna-biopython-metadata/1.0"})
            with urlopen(req, timeout=20) as resp:
                raw = resp.read()
            text = gzip.decompress(raw).decode("utf-8", errors="replace")
        except Exception:
            return out

        for line in text.splitlines():
            if line.startswith("!Series_summary"):
                val = line.split("=", 1)[1].strip() if "=" in line else ""
                if val:
                    out["geo_summary_biopython"] = (
                        (out["geo_summary_biopython"] + " " + val).strip()
                        if out["geo_summary_biopython"]
                        else val
                    )
            elif line.startswith("!Series_overall_design"):
                val = line.split("=", 1)[1].strip() if "=" in line else ""
                if val:
                    out["geo_overall_design_biopython"] = (
                        (out["geo_overall_design_biopython"] + " " + val).strip()
                        if out["geo_overall_design_biopython"]
                        else val
                    )

        # GEO does not expose a strict "abstract" field; use summary as the closest equivalent.
        out["geo_abstract_biopython"] = out["geo_summary_biopython"]
        return out

    Entrez.email = str(email).strip()
    if api_key:
        Entrez.api_key = str(api_key).strip()

    rows: list[dict] = []
    for gsm in geo_accessions:
        record = {
            "geo_accession": gsm,
            "geo_title_biopython": None,
            "geo_summary_biopython": None,
            "geo_overall_design_biopython": None,
            "geo_abstract_biopython": None,
            "geo_taxon_biopython": None,
            "geo_gse_biopython": None,
            "geo_platform_biopython": None,
            "geo_entry_type_biopython": None,
            "geo_gds_type_biopython": None,
            "geo_pdat_biopython": None,
            "geo_n_samples_biopython": None,
            "geo_ftp_link_biopython": None,
            "geo_pubmed_ids_biopython": None,
        }

        try:
            with Entrez.esearch(db="gds", term=f"{gsm}[ACCN]", retmax=1) as handle:
                search = Entrez.read(handle)
            id_list = _as_list(_entrez_get(search, "IdList", []))
            id_list = [str(x) for x in id_list if str(x).strip()]
            if not id_list:
                rows.append(record)
                continue

            with Entrez.esummary(db="gds", id=id_list[0], retmode="xml") as handle:
                summary = Entrez.read(handle)

            doc_set = _entrez_get(summary, "DocumentSummarySet", None)
            docs = _as_list(_entrez_get(doc_set, "DocumentSummary", [])) if doc_set is not None else []
            if not docs:
                docs = _as_list(_entrez_get(summary, "DocumentSummary", []))
            if not docs and isinstance(summary, (list, tuple)):
                docs = list(summary)
            if not docs:
                rows.append(record)
                continue

            doc = docs[0]
            record["geo_title_biopython"] = str(_entrez_get(doc, "title", "") or "").strip() or None
            record["geo_summary_biopython"] = str(_entrez_get(doc, "summary", "") or "").strip() or None
            record["geo_taxon_biopython"] = str(_entrez_get(doc, "taxon", "") or "").strip() or None
            record["geo_gse_biopython"] = str(_entrez_get(doc, "GSE", "") or "").strip() or None
            record["geo_platform_biopython"] = str(_entrez_get(doc, "GPL", "") or "").strip() or None
            record["geo_entry_type_biopython"] = str(_entrez_get(doc, "entryType", "") or "").strip() or None
            record["geo_gds_type_biopython"] = str(_entrez_get(doc, "gdsType", "") or "").strip() or None
            record["geo_pdat_biopython"] = str(_entrez_get(doc, "PDAT", "") or "").strip() or None
            record["geo_ftp_link_biopython"] = str(_entrez_get(doc, "FTPLink", "") or "").strip() or None

            soft_fields = _fetch_geo_soft_fields(record["geo_gse_biopython"], record["geo_ftp_link_biopython"])
            if soft_fields.get("geo_summary_biopython"):
                record["geo_summary_biopython"] = soft_fields["geo_summary_biopython"]
            record["geo_overall_design_biopython"] = soft_fields.get("geo_overall_design_biopython")
            record["geo_abstract_biopython"] = soft_fields.get("geo_abstract_biopython") or record["geo_summary_biopython"]

            n_samples = _entrez_get(doc, "n_samples", None)
            if n_samples is not None:
                try:
                    record["geo_n_samples_biopython"] = int(n_samples)
                except Exception:
                    record["geo_n_samples_biopython"] = str(n_samples).strip() or None

            pubmed_ids = _as_list(_entrez_get(doc, "PubMedIds", []))
            if pubmed_ids:
                cleaned_pubmed_ids = []
                for x in pubmed_ids:
                    try:
                        token = str(int(x))
                    except Exception:
                        token = str(x).strip()
                        if token.startswith("IntegerElement("):
                            token = token.split("(", 1)[1].split(",", 1)[0].strip()
                    if token:
                        cleaned_pubmed_ids.append(token)
                record["geo_pubmed_ids_biopython"] = ";".join(cleaned_pubmed_ids) or None

        except Exception as e:
            print(f"[WARN] Entrez lookup failed for {gsm}: {e}")

        rows.append(record)

    return pd.DataFrame(rows)


def fetch_pubmed_metadata_biopython(pubmed_ids: list[str], email: str, api_key: str | None = None) -> pd.DataFrame:
    """Fetch PubMed summary metadata for a list of PubMed IDs via Biopython Entrez."""
    if not _BIOPYTHON_AVAILABLE:
        print("[WARN] biopython is not installed - skipping PubMed enrichment.")
        return pd.DataFrame(columns=["pubmed_id"])

    def _entrez_get(obj, key: str, default=None):
        if obj is None:
            return default
        try:
            getter = getattr(obj, "get", None)
            if callable(getter):
                return getter(key, default)
        except Exception:
            pass
        try:
            return obj[key] if key in obj else default
        except Exception:
            return default

    Entrez.email = str(email).strip()
    if api_key:
        Entrez.api_key = str(api_key).strip()

    unique_pmids = []
    seen = set()
    for pmid in pubmed_ids:
        s = str(pmid).strip()
        if s and s not in seen:
            seen.add(s)
            unique_pmids.append(s)

    rows = []
    for pmid in unique_pmids:
        rec = {
            "pubmed_id": pmid,
            "pubmed_title_biopython": None,
            "pubmed_journal_biopython": None,
            "pubmed_pub_date_biopython": None,
            "pubmed_doi_biopython": None,
            "pubmed_authors_biopython": None,
        }
        try:
            with Entrez.esummary(db="pubmed", id=pmid, retmode="xml") as handle:
                summary = Entrez.read(handle)

            if isinstance(summary, (list, tuple)):
                docs = list(summary)
            else:
                doc_set = _entrez_get(summary, "DocumentSummarySet", None)
                docs = _entrez_get(doc_set, "DocumentSummary", []) if doc_set is not None else []
                if not isinstance(docs, list):
                    docs = [docs] if docs else []

            if docs:
                doc = docs[0]
                rec["pubmed_title_biopython"] = str(_entrez_get(doc, "Title", "") or "").strip() or None
                rec["pubmed_journal_biopython"] = str(_entrez_get(doc, "FullJournalName", "") or "").strip() or None
                rec["pubmed_pub_date_biopython"] = str(_entrez_get(doc, "PubDate", "") or "").strip() or None
                rec["pubmed_doi_biopython"] = str(_entrez_get(doc, "DOI", "") or "").strip() or None

                authors = _entrez_get(doc, "AuthorList", [])
                if not isinstance(authors, list):
                    authors = [authors] if authors else []
                author_tokens = [str(a).strip() for a in authors if str(a).strip()]
                rec["pubmed_authors_biopython"] = "; ".join(author_tokens) or None
        except Exception as e:
            print(f"[WARN] PubMed lookup failed for {pmid}: {e}")

        rows.append(rec)

    return pd.DataFrame(rows)


def save_retrieval_report(
    prefix: Path,
    query_sample_id: str,
    args: argparse.Namespace,
    osdr_row: pd.Series,
    osdr_meta_cols: list[str],
    hits_display: pd.DataFrame,
    archs4_meta_display: pd.DataFrame,
) -> None:
    """Save retrieval outputs to readable markdown and CSV files."""
    base = prefix if prefix.suffix == "" else prefix.with_suffix("")
    base.parent.mkdir(parents=True, exist_ok=True)

    report_path = Path(f"{base}.report.md")
    hits_csv_path = Path(f"{base}.top_hits.csv")
    meta_csv_path = Path(f"{base}.archs4_metadata.csv")

    hits_display.to_csv(hits_csv_path, index=False)
    archs4_meta_display.to_csv(meta_csv_path, index=False)

    lines = []
    lines.append("# OSDR to ARCHS4 Retrieval Report")
    lines.append("")
    lines.append("## Query")
    lines.append(f"- sample_id: {query_sample_id}")
    lines.append(f"- metric: {args.metric}")
    lines.append(f"- l2_normalize: {args.l2_normalize}")
    lines.append(f"- topk: {args.topk}")
    lines.append("")

    if osdr_meta_cols:
        lines.append("## OSDR Sample Metadata")
        for col in osdr_meta_cols:
            val = osdr_row[col]
            if pd.notna(val) and str(val).strip():
                lines.append(f"- {col}: {val}")
        lines.append("")

    lines.append("## Top Hits (ARCHS4)")
    lines.append("```text")
    lines.append(hits_display.to_string(index=False))
    lines.append("```")
    lines.append("")

    lines.append("## ARCHS4 Hit Metadata")
    lines.append("```text")
    lines.append(archs4_meta_display.to_string(index=False))
    lines.append("```")
    lines.append("")
    lines.append("## Saved Files")
    lines.append(f"- report: {report_path}")
    lines.append(f"- top_hits_csv: {hits_csv_path}")
    lines.append(f"- metadata_csv: {meta_csv_path}")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[INFO] Saved report: {report_path}")
    print(f"[INFO] Saved top hits CSV: {hits_csv_path}")
    print(f"[INFO] Saved metadata CSV: {meta_csv_path}")


def main() -> None:
    args = parse_args()

    manifest_path = args.embedding_dir / "embedding_manifest.json"
    metadata_path = args.embedding_dir / "sample_locations.parquet"
    if not manifest_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            f"Missing embedding outputs in {args.embedding_dir}. Need embedding_manifest.json and sample_locations.parquet"
        )

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    n = int(manifest["total_samples"])
    d = int(manifest["embedding_dim"])
    dtype = np.float16 if manifest.get("embedding_dtype", "float16") == "float16" else np.float32

    emb_path = args.embedding_dir / f"sample_embeddings.{manifest.get('embedding_dtype', 'float16')}.mmap"
    if not emb_path.exists():
        raise FileNotFoundError(f"Embedding memmap not found: {emb_path}")

    print(f"[INFO] Loading index vectors: {emb_path} shape=({n}, {d})", flush=True)
    vecs = np.memmap(emb_path, dtype=dtype, mode="r", shape=(n, d))
    vecs_np = np.asarray(vecs, dtype=np.float32)

    if args.osdr_sample_name:
        print(f"[INFO] Using requested OSDR sample: {args.osdr_sample_name}", flush=True)
        q_vec, best_sample_id, best_osdr_row = load_random_osdr_sample_vector(args)
        best_q_emb = build_model_and_query_embedding(args, q_vec)
    else:
        n_candidates = max(1, args.select_best)
        print(f"[INFO] Evaluating {n_candidates} OSDR candidate(s) to find best match...", flush=True)

        best_q_emb, best_sample_id, best_osdr_row, best_score = None, None, None, -np.inf
        seen_seeds = set()
        attempts = 0
        while len(seen_seeds) < n_candidates:
            seed = args.seed + attempts
            attempts += 1
            if seed in seen_seeds:
                continue
            seen_seeds.add(seed)
            # Temporarily override seed on args to get a different sample each time.
            args.seed = seed
            try:
                q_vec, sample_id, osdr_row = load_random_osdr_sample_vector(args)
            except RuntimeError:
                continue
            q_emb = build_model_and_query_embedding(args, q_vec)
            _, scores = topk_search(vecs_np, q_emb, 1, args.metric, l2_normalize=args.l2_normalize)
            top_score = float(scores[0])
            print(f"  candidate {len(seen_seeds)}/{n_candidates}: {sample_id}  top-1 score={top_score:.4f}", flush=True)
            if top_score > best_score:
                best_score = top_score
                best_q_emb = q_emb
                best_sample_id = sample_id
                best_osdr_row = osdr_row

    idx, scores = topk_search(vecs_np, best_q_emb, args.topk, args.metric, l2_normalize=args.l2_normalize)
    query_sample_id, osdr_row, q_emb = best_sample_id, best_osdr_row, best_q_emb

    meta = pd.read_parquet(metadata_path)
    hits = meta.iloc[idx].copy()
    hits["score"] = scores

    print("\n=== Query (OSDR) ===")
    print(f"sample_id: {query_sample_id}")
    print(f"metric: {args.metric}  l2_normalize: {args.l2_normalize}")
    print(f"k: {args.topk}")
    osdr_meta_cols = [c for c in [
        "id.accession", "id.sample name",
        "study.characteristics.organism", "study.characteristics.strain",
        "study.characteristics.sex", "study.characteristics.age at launch",
        "study.characteristics.material type", "study.characteristics.tissue",
        "study.factor value.spaceflight", "study.factor value.treatment",
        "study.parameter value.duration", "study.parameter value.habitat",
        "study.parameter value.diet",
        "investigation.study assays.study assay technology type",
    ] if c in osdr_row.index]
    if osdr_meta_cols:
        print("\nOSDR sample metadata:")
        for col in osdr_meta_cols:
            val = osdr_row[col]
            if pd.notna(val) and str(val).strip():
                print(f"  {col}: {val}")

    print("\n=== Top Hits (ARCHS4) ===")
    cols = ["score", "geo_accession", "global_index", "shard_file", "row_in_shard"]
    hits_display = hits[cols].copy()
    print(hits_display.to_string(index=False))

    # Pull rich metadata for each hit GSM accession via archs4py.
    gsm_ids = hits["geo_accession"].dropna().astype(str).tolist()
    print("\n=== ARCHS4 Hit Metadata ===")
    archs4_meta = fetch_archs4_metadata(gsm_ids, args.archs4_h5, getattr(args, 'mouse_archs4_h5', None))

    if args.biopython_metadata:
        if not args.entrez_email:
            raise RuntimeError("--entrez-email is required when --biopython-metadata is enabled")
        print("[INFO] Fetching Entrez GEO metadata via Biopython...", flush=True)
        bio_meta = fetch_geo_metadata_biopython(gsm_ids, args.entrez_email, args.entrez_api_key)
        archs4_meta = archs4_meta.merge(bio_meta, on="geo_accession", how="left")

        if args.biopython_pubmed:
            # Use the first linked PMID per GEO accession for compact tabular reporting.
            if "geo_pubmed_ids_biopython" not in bio_meta.columns:
                print("[WARN] BioPython GEO metadata did not include geo_pubmed_ids_biopython; skipping PubMed enrichment.", flush=True)
            else:
                first_pmid = bio_meta[["geo_accession", "geo_pubmed_ids_biopython"]].copy()
                first_pmid["pubmed_id"] = (
                    first_pmid["geo_pubmed_ids_biopython"]
                    .fillna("")
                    .astype(str)
                    .str.split(";")
                    .str[0]
                    .str.strip()
                )
                pmids = [p for p in first_pmid["pubmed_id"].tolist() if p]
                if pmids:
                    print("[INFO] Fetching PubMed summaries via Biopython...", flush=True)
                    pubmed_meta = fetch_pubmed_metadata_biopython(pmids, args.entrez_email, args.entrez_api_key)
                    first_pmid = first_pmid.merge(pubmed_meta, on="pubmed_id", how="left")
                    archs4_meta = archs4_meta.merge(
                        first_pmid[
                            [
                                "geo_accession",
                                "pubmed_id",
                                "pubmed_title_biopython",
                                "pubmed_journal_biopython",
                                "pubmed_pub_date_biopython",
                                "pubmed_doi_biopython",
                                "pubmed_authors_biopython",
                            ]
                        ],
                        on="geo_accession",
                        how="left",
                    )

    if args.biopython_pubmed and not args.biopython_metadata:
        raise RuntimeError("--biopython-pubmed requires --biopython-metadata")

    # Show the most informative columns if present; fall back to whatever is available.
    priority_cols = [
        "geo_accession",
        "species",
        "title",
        "source_name_ch1",
        "characteristics_ch1",
        "tissue",
        "cell_type",
        "cell_line",
        "disease",
        "treatment",
        "organism_ch1",
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
        "pubmed_pub_date_biopython",
        "pubmed_journal_biopython",
        "pubmed_doi_biopython",
        "pubmed_authors_biopython",
        "pubmed_title_biopython",
    ]
    show_cols = [c for c in priority_cols if c in archs4_meta.columns]
    if not show_cols:
        show_cols = archs4_meta.columns.tolist()
    archs4_meta_display = archs4_meta[show_cols].copy()
    with pd.option_context("display.max_colwidth", 80, "display.width", 220):
        print(archs4_meta_display.to_string(index=False))

    if args.save_report_prefix is not None:
        save_retrieval_report(
            prefix=args.save_report_prefix,
            query_sample_id=query_sample_id,
            args=args,
            osdr_row=osdr_row,
            osdr_meta_cols=osdr_meta_cols,
            hits_display=hits_display,
            archs4_meta_display=archs4_meta_display,
        )


if __name__ == "__main__":
    main()
