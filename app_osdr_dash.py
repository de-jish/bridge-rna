#!/usr/bin/env python3
"""Dash MVP for OSDR -> ARCHS4 analog retrieval visualization.

All searches run through the existing demo retrieval script for real hits.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import time
import gzip
from pathlib import Path
from typing import Any
import importlib.util
import numpy as np
import requests

try:
    from osdr_metadata import get_study_summary
except Exception:
    get_study_summary = None

import dash
from dash import Dash, Input, Output, State, dcc, html
import pandas as pd
import plotly.graph_objects as go


ROOT = Path(__file__).resolve().parent
OSDR_METADATA_PATH = (
    ROOT / "data" / "osdr" / "metadata" / "selected_sample_metadata.tsv"
    if (ROOT / "data" / "osdr" / "metadata" / "selected_sample_metadata.tsv").exists()
    else (ROOT / "osdr" / "metadata" / "selected_sample_metadata.tsv")
)
DEMO_SCRIPT_PATH = ROOT / "demo_osdr_top5.py"
EMBEDDING_DIR = ROOT / "archs4_sample_embeddings_full"
GENERIC_ENTREZ_EMAIL = os.environ.get("GENERIC_ENTREZ_EMAIL", "noreply@example.com")
DEFAULT_ENTREZ_EMAIL = os.environ.get("ENTREZ_EMAIL", GENERIC_ENTREZ_EMAIL)
NCBI_API_KEY = os.environ.get("NCBI_API_KEY", "")
AI_PROMPT_PATH = ROOT / "prompts" / "ai_summary_prompt.txt"

# AI provider selection: default to local Ollama for now.
AI_SUMMARY_PROVIDER = os.environ.get("AI_SUMMARY_PROVIDER", "ollama").strip().lower()

# Bedrock settings (kept for optional later use).
BEDROCK_API_URL = os.environ.get("BEDROCK_API_URL", "")
BEDROCK_API_KEY = os.environ.get("BEDROCK_API_KEY", "")
BEDROCK_API_KEY_HEADER = os.environ.get("BEDROCK_API_KEY_HEADER", "x-api-key")
BEDROCK_PAYLOAD_KEY = os.environ.get("BEDROCK_PAYLOAD_KEY", "query").strip() or "query"

# Ollama settings (new default path).
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gemma3:4b")
OLLAMA_TIMEOUT_SECONDS = int(os.environ.get("OLLAMA_TIMEOUT_SECONDS", "180"))
PRECOMPUTED_QUERY_EMBEDDING_CANDIDATES = [
    ROOT / "osdr_query_embeddings.parquet",
    ROOT / "data" / "osdr" / "metadata" / "osdr_query_embeddings.parquet",
    ROOT / "data" / "osdr" / "metadata" / "selected_sample_embeddings.parquet",
    ROOT / "osdr" / "metadata" / "osdr_query_embeddings.parquet",
    ROOT / "osdr" / "metadata" / "selected_sample_embeddings.parquet",
]

_ARCHS4_CACHE: dict[str, Any] = {}
_OSDR_QUERY_CACHE: dict[str, Any] = {}
_OSDR_STUDY_SUMMARY_CACHE: dict[str, dict[str, str]] = {}


def _load_ai_prompt_template() -> str:
    if AI_PROMPT_PATH.exists():
        return AI_PROMPT_PATH.read_text(encoding="utf-8")
    return (
        "You are a computational biologist helping interpret transcriptomic retrieval results.\n\n"
        "### OSDR Query Sample\n{osdr_metadata}\n\n"
        "### Retrieved GEO Hits\n{retrieved_hits_table}\n\n"
        "### GEO Study Context\n{geo_summaries}\n"
    )


def _format_osdr_query_text(query_row: pd.Series) -> str:
    study_summary = _fetch_osdr_study_summary(_safe_str(query_row.get("study_id", "")))
    lines = [
        f"sample_id: {_safe_str(query_row.get('sample_id', ''))}",
        f"study_id: {_safe_str(query_row.get('study_id', ''))}",
        f"sample_name: {_safe_str(query_row.get('sample_name', ''))}",
        f"tissue: {_safe_str(query_row.get('tissue', ''))}",
        f"condition: {_safe_str(query_row.get('condition', ''))}",
        f"strain: {_safe_str(query_row.get('strain', ''))}",
        f"sex: {_safe_str(query_row.get('sex', ''))}",
        f"duration: {_safe_str(query_row.get('duration', ''))}",
        f"study_title: {_safe_str(study_summary.get('study_title', ''))}",
        f"study_publication_title: {_safe_str(study_summary.get('study_publication_title', ''))}",
        f"study_description: {_safe_str(study_summary.get('study_description', ''))}",
        f"study_protocol_description: {_safe_str(study_summary.get('study_protocol_description', ''))}",
    ]
    return "\n".join(lines)


def _format_hits_table_text(hits_df: pd.DataFrame) -> str:
    if hits_df.empty:
        return "No hits available."
    cols = [
        "gsm",
        "gse",
        "score",
        "species",
        "title",
        "source_name",
        "characteristics",
        "pubmed_ids",
    ]
    avail = [c for c in cols if c in hits_df.columns]
    return hits_df[avail].head(20).to_string(index=False)


def _format_geo_context_text(hits_df: pd.DataFrame) -> str:
    if hits_df.empty:
        return "No GEO summaries available."
    lines = []
    for _, r in hits_df.head(20).iterrows():
        lines.append(f"GSM: {_safe_str(r.get('gsm', ''))}")
        lines.append(f"GSE: {_safe_str(r.get('gse', ''))}")
        lines.append(f"Title: {_safe_str(r.get('title', ''))}")
        lines.append(f"Summary: {_safe_str(r.get('geo_summary', ''))}")
        lines.append(f"Overall design: {_safe_str(r.get('geo_design', ''))}")
        lines.append("-")
    return "\n".join(lines)


def _call_bedrock_summary(prompt: str) -> str:
    """Call Bedrock-compatible endpoint. URL/API key intentionally user-configured."""
    if not BEDROCK_API_URL:
        return (
            "AI Summary endpoint not configured. Set BEDROCK_API_URL (and optionally BEDROCK_API_KEY) "
            "in your environment, then rerun."
        )

    headers = {"Content-Type": "application/json"}
    if BEDROCK_API_KEY:
        # Support both API Gateway key auth and bearer-token adapters.
        headers[BEDROCK_API_KEY_HEADER] = BEDROCK_API_KEY
        headers["Authorization"] = f"Bearer {BEDROCK_API_KEY}"

    def _payload_for_key(key: str, text: str) -> dict[str, Any]:
        k = (key or "query").strip()
        if k == "messages":
            return {"messages": [{"role": "user", "content": text}]}
        if k == "messages_content_array":
            return {"messages": [{"role": "user", "content": [{"text": text}]}]}
        return {k: text}

    def _sanitize_query_text(text: str) -> str:
        # Keep endpoint input strict: printable ASCII-ish text with normalized whitespace.
        cleaned = "".join(ch if (ch == "\n" or ch == "\t" or 32 <= ord(ch) <= 126) else " " for ch in text)
        cleaned = re.sub(r"[ \t]+", " ", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip()

    def _extract_text_response(resp: requests.Response) -> str:
        try:
            data = resp.json() if resp.text else {}
        except Exception:
            data = {}

        if isinstance(data, dict):
            # Legacy widget format: {"answer": [{"text": "..."}], "source_urls": [...]}
            ans = data.get("answer")
            if isinstance(ans, list) and ans:
                first = ans[0]
                if isinstance(first, dict):
                    t = _safe_str(first.get("text"))
                    if t:
                        return t

            for key in ["summary", "output", "text", "completion", "response", "answer"]:
                val = data.get(key)
                if isinstance(val, str) and val.strip():
                    return val.strip()

            body = data.get("body")
            if isinstance(body, str):
                try:
                    body_json = json.loads(body)
                    if isinstance(body_json, dict):
                        for key in ["summary", "output", "text", "completion", "response", "answer"]:
                            val = body_json.get(key)
                            if isinstance(val, str) and val.strip():
                                return val.strip()
                except Exception:
                    if body.strip():
                        return body.strip()

        txt = _safe_str(resp.text)
        return txt or "AI Summary returned no text."

    payload_key = BEDROCK_PAYLOAD_KEY
    primary_text = prompt if payload_key != "query" else _sanitize_query_text(prompt)
    payload = _payload_for_key(payload_key, primary_text)

    try:
        resp = requests.post(BEDROCK_API_URL, headers=headers, json=payload, timeout=120)
    except Exception as exc:
        return f"AI Summary call failed: {_safe_str(exc)}"

    if resp.status_code >= 400:
        err_text = _safe_str(resp.text)
        if len(err_text) > 800:
            err_text = err_text[:800] + "..."
        # Endpoint-specific recovery: retry once with strict query contract and condensed text.
        if resp.status_code == 400 and "input_03" in err_text:
            retry_key = "query"
            retry_text = _sanitize_query_text(prompt)
            retry_payload = _payload_for_key(retry_key, retry_text)
            try:
                retry_resp = requests.post(BEDROCK_API_URL, headers=headers, json=retry_payload, timeout=120)
                if retry_resp.status_code < 400:
                    return _extract_text_response(retry_resp)
                retry_err = _safe_str(retry_resp.text)
                if len(retry_err) > 800:
                    retry_err = retry_err[:800] + "..."
                return (
                    f"AI Summary call failed: HTTP {retry_resp.status_code} after sanitize+shorten retry with payload key '{retry_key}': "
                    f"{retry_err}"
                )
            except Exception as retry_exc:
                return f"AI Summary call failed during sanitize+shorten retry: {_safe_str(retry_exc)}"

        return f"AI Summary call failed: HTTP {resp.status_code} using payload key '{payload_key}': {err_text}"
    return _extract_text_response(resp)


def _call_ollama_summary(prompt: str) -> str:
    if not OLLAMA_BASE_URL:
        return "AI Summary call failed: OLLAMA_BASE_URL is not configured."

    def _get_available_models() -> list[str]:
        try:
            resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=20)
            if resp.status_code >= 400:
                return []
            data = resp.json() if resp.text else {}
            models = data.get("models", []) if isinstance(data, dict) else []
            names = []
            for model in models:
                if isinstance(model, dict):
                    name = _safe_str(model.get("name"))
                    if name:
                        names.append(name)
            return names
        except Exception:
            return []

    def _pick_model() -> str:
        available = _get_available_models()
        if OLLAMA_MODEL in available:
            return OLLAMA_MODEL
        if available:
            preferred = ["llama3", "qwen", "gpt-oss", "codestral", "mistral", "gemma"]
            for pref in preferred:
                for name in available:
                    if pref in name.lower():
                        return name
            return available[0]
        return OLLAMA_MODEL

    def _generate(model_name: str) -> requests.Response:
        url = f"{OLLAMA_BASE_URL}/api/generate"
        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
        }
        return requests.post(url, json=payload, timeout=OLLAMA_TIMEOUT_SECONDS)

    try:
        resp = _generate(_pick_model())
    except Exception as exc:
        return f"AI Summary call failed (Ollama): {_safe_str(exc)}"

    if resp.status_code >= 400:
        err = _safe_str(resp.text)
        if len(err) > 800:
            err = err[:800] + "..."
        if resp.status_code == 404 and "not found" in err.lower():
            available = _get_available_models()
            if available:
                fallback = _pick_model()
                if fallback != OLLAMA_MODEL:
                    try:
                        retry = _generate(fallback)
                        if retry.status_code < 400:
                            data = retry.json() if retry.text else {}
                            if isinstance(data, dict):
                                response_text = _safe_str(data.get("response"))
                                if response_text:
                                    return f"[Using fallback model: {fallback}]\n\n{response_text}"
                            return f"[Using fallback model: {fallback}]\n\n" + (_safe_str(retry.text) or "AI Summary returned no text.")
                    except Exception as retry_exc:
                        return f"AI Summary call failed (Ollama fallback): {_safe_str(retry_exc)}"
        return f"AI Summary call failed (Ollama): HTTP {resp.status_code}: {err}"

    try:
        data = resp.json() if resp.text else {}
    except Exception:
        data = {}

    if isinstance(data, dict):
        response_text = _safe_str(data.get("response"))
        if response_text:
            return response_text

    return _safe_str(resp.text) or "AI Summary returned no text."


def _call_ai_summary(prompt: str) -> str:
    provider = AI_SUMMARY_PROVIDER
    if provider == "bedrock":
        return _call_bedrock_summary(prompt)
    if provider == "ollama":
        return _call_ollama_summary(prompt)
    return (
        f"AI Summary provider '{provider}' is not supported. "
        "Use AI_SUMMARY_PROVIDER=ollama or AI_SUMMARY_PROVIDER=bedrock."
    )


def _find_first_existing(paths: list[Path]) -> Path | None:
    for p in paths:
        if p.exists():
            return p
    return None


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

    canonical_candidate = _find_first_existing(
        [
            ROOT / "data" / "archs4" / "train_orthologs" / "canonical_genes.csv",
            ROOT / "data" / "ensembl" / "canonical_genes.inferred.csv",
        ]
    )

    if canonical_candidate is None and resolved_paths["checkpoint"] is not None:
        for candidate in canonical_candidates:
            if candidate.exists() and _canonical_matches_checkpoint(candidate, resolved_paths["checkpoint"]):
                canonical_candidate = candidate
                break

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

    return missing, resolved_paths


def _safe_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value).strip()


def _last_nonempty_line(text: str) -> str:
    """Return the last non-empty line of a subprocess error blob.

    For a Python traceback this is the actual exception line
    (e.g. "RuntimeError: Requested OSDR sample not found") rather than the
    surrounding stack frames, which keeps the UI status message clean.
    """
    for line in reversed(_safe_str(text).splitlines()):
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


class RetrievalError(RuntimeError):
    """Retrieval failure carrying a clean message plus the full raw detail.

    ``str(err)`` is the short one-line message shown in the status banner;
    ``err.detail`` holds the full traceback/output for the collapsible panel.
    """

    def __init__(self, message: str, detail: str = "") -> None:
        super().__init__(message)
        self.detail = detail or message


def _fetch_osdr_study_summary(study_id: str) -> dict[str, str]:
    """Fetch OSDR study summary via osdr_metadata.py with in-process caching."""
    sid = _safe_str(study_id)
    if not sid:
        return {}

    cached = _OSDR_STUDY_SUMMARY_CACHE.get(sid)
    if cached is not None:
        return cached

    if get_study_summary is None:
        _OSDR_STUDY_SUMMARY_CACHE[sid] = {}
        return {}

    try:
        summary = get_study_summary(sid)
        out = {
            "dataset_id": _safe_str(summary.get("dataset_id", sid)),
            "study_title": _safe_str(summary.get("study_title", "")),
            "study_description": _safe_str(summary.get("study_description", "")),
            "study_publication_title": _safe_str(summary.get("study_publication_title", "")),
            "study_protocol_description": _safe_str(summary.get("study_protocol_description", "")),
        }
        _OSDR_STUDY_SUMMARY_CACHE[sid] = out
        return out
    except Exception:
        _OSDR_STUDY_SUMMARY_CACHE[sid] = {}
        return {}


def _build_osdr_query_metadata_block(query: pd.Series) -> list[Any]:
    """Appendable OSDR metadata section for the right panel."""
    study_id = _safe_str(query.get("study_id", ""))
    summary = _fetch_osdr_study_summary(study_id)
    study_title = _safe_str(summary.get("study_title", ""))
    study_description = _safe_str(summary.get("study_description", ""))
    study_publication_title = _safe_str(summary.get("study_publication_title", ""))
    protocol = _safe_str(summary.get("study_protocol_description", ""))
    section = _detail_section(
        "OSDR study",
        [
            _detail_row("Study ID", study_id, mono=True),
            _detail_row("Study title", study_title),
        ],
    )
    blocks: list[Any] = [section] if section else []
    if study_description:
        blocks.append(_detail_text_block("Study description", study_description, collapsible=True))
    if study_publication_title:
        blocks.append(_detail_text_block("Publication title", study_publication_title, collapsible=True))
    if protocol:
        blocks.append(_detail_text_block("Protocol description", protocol, collapsible=True))
    return blocks


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


def load_osdr_samples(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")

    out = pd.DataFrame()
    out["sample_name"] = df.get("id.sample name", "")
    out["study_id"] = df.get("id.accession", "")
    out["tissue"] = df.get("study.characteristics.material type", "")
    out["condition"] = df.get("study.factor value.spaceflight", "")
    out["strain"] = df.get("study.characteristics.strain", "")
    out["sex"] = df.get("study.characteristics.sex", "")
    out["duration"] = df.get("study.parameter value.duration", "")
    out["counts_path"] = df.get("counts_path", "")
    out["sample_id"] = out["study_id"].astype(str) + "|" + out["sample_name"].astype(str)

    keep = out["sample_name"].astype(str).str.len() > 0
    out = out[keep].drop_duplicates(subset=["sample_id"]).reset_index(drop=True)
    return out


def _first_non_empty(row: pd.Series, candidates: list[str]) -> str:
    for col in candidates:
        if col in row.index:
            value = _safe_str(row[col])
            if value:
                return value
    return ""


def _extract_gse(value: str) -> str:
    m = re.search(r"(GSE\d+)", value or "", flags=re.IGNORECASE)
    return m.group(1).upper() if m else ""


def _enrich_hits_from_ncbi_eutils(hits_df: pd.DataFrame, entrez_email: str) -> pd.DataFrame:
    """Enrich hit rows with GEO metadata via NCBI E-utilities.

    This fallback works even without local ARCHS4 H5 metadata or Biopython.
    """
    email = _safe_str(entrez_email)
    if hits_df.empty or not email:
        return hits_df

    esearch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    esummary_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
    session = requests.Session()

    base_params = {
        "tool": "osdr_dash_app",
        "email": email,
    }
    if NCBI_API_KEY:
        base_params["api_key"] = NCBI_API_KEY

    def _first_pmid_string(pmids_val: Any) -> str:
        if pmids_val is None:
            return ""
        if isinstance(pmids_val, list):
            for p in pmids_val:
                s = _safe_str(p)
                if s:
                    return s
            return ""
        return _safe_str(pmids_val)

    def _pubmed_details(pmid: str) -> tuple[str, str, str, str]:
        if not pmid:
            return "", "", "", ""
        p = dict(base_params)
        p.update({"db": "pubmed", "id": pmid, "retmode": "json"})
        try:
            r = None
            for attempt in range(3):
                r = session.get(esummary_url, params=p, timeout=20)
                if r.status_code == 200:
                    break
                if r.status_code == 429:
                    time.sleep(0.7 * (attempt + 1))
            if r is None or r.status_code != 200:
                return "", "", "", ""
            j = r.json()
            uid = (j.get("result", {}).get("uids") or [None])[0]
            d = j.get("result", {}).get(uid, {}) if uid else {}
            title = _safe_str(d.get("title", ""))
            journal = _safe_str(d.get("fulljournalname", ""))
            pub_date = _safe_str(d.get("pubdate", ""))

            doi = ""
            for aid in d.get("articleids", []) if isinstance(d.get("articleids", []), list) else []:
                if isinstance(aid, dict) and _safe_str(aid.get("idtype", "")).lower() == "doi":
                    doi = _safe_str(aid.get("value", ""))
                    break
            return title, journal, pub_date, doi
        except Exception:
            return "", "", "", ""

    def _soft_summary_and_design(gse: str, ftp_link: str) -> tuple[str, str]:
        gse_u = _safe_str(gse).upper()
        if not gse_u:
            return "", ""
        if not gse_u.startswith("GSE"):
            gse_u = f"GSE{gse_u}"

        summary_parts: list[str] = []
        design_parts: list[str] = []

        candidates: list[str] = []
        ftp = _safe_str(ftp_link)
        if ftp.startswith("ftp://"):
            http_base = "https://" + ftp[len("ftp://") :].rstrip("/")
            candidates.append(f"{http_base}/soft/{gse_u}_family.soft.gz")

        prefix = f"{gse_u[:-3]}nnn" if len(gse_u) > 3 else gse_u
        candidates.append(
            f"https://ftp.ncbi.nlm.nih.gov/geo/series/{prefix}/{gse_u}/soft/{gse_u}_family.soft.gz"
        )

        for url in candidates:
            try:
                rr = session.get(url, timeout=25)
                if rr.status_code != 200:
                    continue
                text = gzip.decompress(rr.content).decode("utf-8", errors="replace")
                for line in text.splitlines():
                    if line.startswith("!Series_summary"):
                        val = line.split("=", 1)[1].strip() if "=" in line else ""
                        if val:
                            summary_parts.append(val)
                    elif line.startswith("!Series_overall_design"):
                        val = line.split("=", 1)[1].strip() if "=" in line else ""
                        if val:
                            design_parts.append(val)
                break
            except Exception:
                continue

        return " ".join(summary_parts).strip(), " ".join(design_parts).strip()

    rows = []
    for gsm in hits_df["gsm"].astype(str).dropna().unique().tolist():
        rec = {
            "gsm": gsm,
            "gse_ncbi": "",
            "title_ncbi": "",
            "summary_ncbi": "",
            "design_ncbi": "",
            "pubmed_ids_ncbi": "",
            "platform_ncbi": "",
            "taxon_ncbi": "",
            "entry_type_ncbi": "",
            "gds_type_ncbi": "",
            "pdat_ncbi": "",
            "n_samples_ncbi": "",
            "ftp_link_ncbi": "",
            "pubmed_title_ncbi": "",
            "pubmed_journal_ncbi": "",
            "pubmed_pub_date_ncbi": "",
            "pubmed_doi_ncbi": "",
        }
        try:
            p1 = dict(base_params)
            p1.update({"db": "gds", "term": f"{gsm}[ACCN]", "retmode": "json", "retmax": 1})
            r1 = None
            for attempt in range(3):
                r1 = session.get(esearch_url, params=p1, timeout=20)
                if r1.status_code == 200:
                    break
                if r1.status_code == 429:
                    time.sleep(0.7 * (attempt + 1))
            if r1 is None or r1.status_code != 200:
                rows.append(rec)
                continue
            j1 = r1.json()
            ids = j1.get("esearchresult", {}).get("idlist", [])
            if not ids:
                rows.append(rec)
                continue

            p2 = dict(base_params)
            p2.update({"db": "gds", "id": ids[0], "retmode": "json"})
            r2 = None
            for attempt in range(3):
                r2 = session.get(esummary_url, params=p2, timeout=20)
                if r2.status_code == 200:
                    break
                if r2.status_code == 429:
                    time.sleep(0.7 * (attempt + 1))
            if r2 is None or r2.status_code != 200:
                rows.append(rec)
                continue
            j2 = r2.json()
            uid = (j2.get("result", {}).get("uids") or [None])[0]
            doc = j2.get("result", {}).get(uid, {}) if uid else {}

            gse_raw = _safe_str(doc.get("gse", ""))
            rec["gse_ncbi"] = f"GSE{gse_raw}" if gse_raw and not gse_raw.upper().startswith("GSE") else gse_raw.upper()
            rec["title_ncbi"] = _safe_str(doc.get("title", ""))
            rec["summary_ncbi"] = _safe_str(doc.get("summary", ""))
            rec["platform_ncbi"] = _safe_str(doc.get("gpl", ""))
            rec["taxon_ncbi"] = _safe_str(doc.get("taxon", ""))
            rec["entry_type_ncbi"] = _safe_str(doc.get("entrytype", ""))
            rec["gds_type_ncbi"] = _safe_str(doc.get("gdstype", ""))
            rec["pdat_ncbi"] = _safe_str(doc.get("pdat", ""))
            rec["n_samples_ncbi"] = _safe_str(doc.get("n_samples", ""))
            rec["ftp_link_ncbi"] = _safe_str(doc.get("ftplink", ""))

            pmids = doc.get("pubmedids", [])
            if isinstance(pmids, list):
                rec["pubmed_ids_ncbi"] = ";".join([_safe_str(p) for p in pmids if _safe_str(p)])
            else:
                rec["pubmed_ids_ncbi"] = _safe_str(pmids)

            pmid_one = _first_pmid_string(pmids)
            t, jn, pub_date_val, doi = _pubmed_details(pmid_one)
            rec["pubmed_title_ncbi"] = t
            rec["pubmed_journal_ncbi"] = jn
            rec["pubmed_pub_date_ncbi"] = pub_date_val
            rec["pubmed_doi_ncbi"] = doi

            # Pull richer overall design/summary from GEO family SOFT when possible.
            soft_summary, soft_design = _soft_summary_and_design(rec["gse_ncbi"], rec["ftp_link_ncbi"])
            if soft_summary:
                rec["summary_ncbi"] = soft_summary
            rec["design_ncbi"] = soft_design
        except Exception:
            pass

        rows.append(rec)
        # Be polite to NCBI public API.
        time.sleep(0.12)

    if not rows:
        return hits_df

    enrich = pd.DataFrame(rows)
    out = hits_df.merge(enrich, on="gsm", how="left")

    # Fill only missing/blank values from NCBI fallback.
    out["gse"] = out.apply(lambda r: _safe_str(r.get("gse")) or _safe_str(r.get("gse_ncbi")), axis=1)
    out["title"] = out.apply(lambda r: _safe_str(r.get("title")) or _safe_str(r.get("title_ncbi")), axis=1)
    out["geo_summary"] = out.apply(lambda r: _safe_str(r.get("geo_summary")) or _safe_str(r.get("summary_ncbi")), axis=1)
    out["geo_design"] = out.apply(lambda r: _safe_str(r.get("geo_design")) or _safe_str(r.get("design_ncbi")), axis=1)
    out["pubmed_ids"] = out.apply(lambda r: _safe_str(r.get("pubmed_ids")) or _safe_str(r.get("pubmed_ids_ncbi")), axis=1)

    out["geo_platform_biopython"] = out.apply(lambda r: _safe_str(r.get("geo_platform_biopython")) or _safe_str(r.get("platform_ncbi")), axis=1)
    out["geo_taxon_biopython"] = out.apply(lambda r: _safe_str(r.get("geo_taxon_biopython")) or _safe_str(r.get("taxon_ncbi")), axis=1)
    out["geo_entry_type_biopython"] = out.apply(lambda r: _safe_str(r.get("geo_entry_type_biopython")) or _safe_str(r.get("entry_type_ncbi")), axis=1)
    out["geo_gds_type_biopython"] = out.apply(lambda r: _safe_str(r.get("geo_gds_type_biopython")) or _safe_str(r.get("gds_type_ncbi")), axis=1)
    out["geo_pdat_biopython"] = out.apply(lambda r: _safe_str(r.get("geo_pdat_biopython")) or _safe_str(r.get("pdat_ncbi")), axis=1)
    out["geo_n_samples_biopython"] = out.apply(lambda r: _safe_str(r.get("geo_n_samples_biopython")) or _safe_str(r.get("n_samples_ncbi")), axis=1)
    out["geo_ftp_link_biopython"] = out.apply(lambda r: _safe_str(r.get("geo_ftp_link_biopython")) or _safe_str(r.get("ftp_link_ncbi")), axis=1)

    out["pubmed_title_biopython"] = out.apply(lambda r: _safe_str(r.get("pubmed_title_biopython")) or _safe_str(r.get("pubmed_title_ncbi")), axis=1)
    out["pubmed_journal_biopython"] = out.apply(lambda r: _safe_str(r.get("pubmed_journal_biopython")) or _safe_str(r.get("pubmed_journal_ncbi")), axis=1)
    out["pubmed_pub_date_biopython"] = out.apply(lambda r: _safe_str(r.get("pubmed_pub_date_biopython")) or _safe_str(r.get("pubmed_pub_date_ncbi")), axis=1)
    out["pubmed_doi_biopython"] = out.apply(lambda r: _safe_str(r.get("pubmed_doi_biopython")) or _safe_str(r.get("pubmed_doi_ncbi")), axis=1)

    for c in [
        "gse_ncbi",
        "title_ncbi",
        "summary_ncbi",
        "design_ncbi",
        "pubmed_ids_ncbi",
        "platform_ncbi",
        "taxon_ncbi",
        "entry_type_ncbi",
        "gds_type_ncbi",
        "pdat_ncbi",
        "n_samples_ncbi",
        "ftp_link_ncbi",
        "pubmed_title_ncbi",
        "pubmed_journal_ncbi",
        "pubmed_pub_date_ncbi",
        "pubmed_doi_ncbi",
    ]:
        if c in out.columns:
            out = out.drop(columns=[c])
    return out


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
    row = samples_df.loc[samples_df["sample_id"] == sample_id]
    if row.empty:
        raise ValueError(f"Unknown sample_id: {sample_id}")
    sample_row = row.iloc[0]
    sample_name = _safe_str(sample_row["sample_name"])

    q_file = _find_precomputed_query_embedding_file()
    if q_file is not None:
        return run_precomputed_query_retrieval(sample_id=sample_id, sample_name=sample_name, topk=topk), "precomputed"

    return (
        run_real_retrieval(
            sample_name=sample_name,
            topk=topk,
            entrez_email=entrez_email,
            enable_biopython_metadata=enable_biopython_metadata,
        ),
        "demo",
    )


# Plotly can't read CSS variables, so the graph palette is mirrored here from the
# light-theme tokens in assets/style.css. Keep these in sync with :root.
GRAPH_THEME = {
    "paper_bg": "#ffffff",
    "plot_bg": "#ffffff",
    "grid": "#e6ecf5",
    "text_primary": "#1a2432",
    "text_secondary": "#5a6b7e",
    "query": "#0bab9f",       # --accent-teal (query stands apart from its hits)
    "gsm": "#2b7fff",         # --accent (GSM hit nodes)
    "gse": "#d9791b",         # --accent-warm (GSE study nodes)
    "edge": "rgba(43, 127, 255, 0.42)",
    "edge_gse": "rgba(217, 121, 27, 0.35)",
    "marker_line": "#ffffff",
    "font_sans": "Inter, 'Segoe UI', -apple-system, sans-serif",
}


def _empty_network_figure(message: str = "Run a search to build the retrieval network.") -> go.Figure:
    """A clean, axis-free placeholder that matches the workspace card."""
    fig = go.Figure()
    fig.update_layout(
        paper_bgcolor=GRAPH_THEME["paper_bg"],
        plot_bgcolor=GRAPH_THEME["plot_bg"],
        margin={"l": 20, "r": 20, "t": 20, "b": 20},
        xaxis={"visible": False, "range": [0, 1]},
        yaxis={"visible": False, "range": [0, 1]},
        height=560,
        annotations=[
            {
                "text": message,
                "x": 0.5,
                "y": 0.5,
                "xref": "paper",
                "yref": "paper",
                "showarrow": False,
                "font": {"family": GRAPH_THEME["font_sans"], "size": 15, "color": GRAPH_THEME["text_secondary"]},
            }
        ],
    )
    return fig


def _edge_width(scores: pd.Series) -> list[float]:
    if len(scores) == 0:
        return []
    smin = float(scores.min())
    smax = float(scores.max())
    if abs(smax - smin) < 1e-12:
        return [3.0] * len(scores)
    return [1.5 + 6.5 * ((float(s) - smin) / (smax - smin)) for s in scores]


def build_network_figure(query: pd.Series, hits_df: pd.DataFrame) -> go.Figure:
    gse_values = [g for g in hits_df["gse"].astype(str).tolist() if g]
    gse_unique = sorted(dict.fromkeys(gse_values))

    node_rows = []
    edge_rows = []

    q_id = _safe_str(query["sample_id"])
    q_label = _safe_str(query["sample_name"])
    node_rows.append(
        {
            "node_id": q_id,
            "label": q_label,
            "kind": "query",
            "x": 0.0,
            "y": 0.0,
            "size": 28,
            "color": GRAPH_THEME["query"],
            "symbol": "star",
            "hover": f"OSDR query<br>{q_label}<br>{q_id}",
        }
    )

    y_space = 1.4
    gsm_count = len(hits_df)
    gsm_y_start = (gsm_count - 1) * 0.5 * y_space
    widths = _edge_width(hits_df["score"]) if "score" in hits_df else [3.0] * len(hits_df)

    for i, (_, row) in enumerate(hits_df.iterrows()):
        y = gsm_y_start - i * y_space
        score = float(row["score"])
        gsm = _safe_str(row["gsm"])
        gse = _safe_str(row.get("gse", ""))
        hover = (
            f"{gsm}<br>"
            f"{_safe_str(row.get('source_name', ''))}<br>"
            f"{_safe_str(row.get('characteristics', ''))}<br>"
            f"Score: {score:.3f}<br>"
            f"{gse}"
        )

        node_rows.append(
            {
                "node_id": gsm,
                "label": gsm,
                "kind": "gsm",
                "x": 1.0,
                "y": y,
                "size": 16 + max(0.0, (score - float(hits_df["score"].min())) * 20.0),
                "color": GRAPH_THEME["gsm"],
                "symbol": "circle",
                "hover": hover,
            }
        )

        edge_rows.append(
            {
                "x0": 0.0,
                "y0": 0.0,
                "x1": 1.0,
                "y1": y,
                "width": widths[i],
                "color": GRAPH_THEME["edge"],
            }
        )

        if gse:
            g_idx = gse_unique.index(gse)
            gse_y_start = (len(gse_unique) - 1) * 0.5 * 2.3
            g_y = gse_y_start - g_idx * 2.3
            if not any(n["node_id"] == gse for n in node_rows):
                node_rows.append(
                    {
                        "node_id": gse,
                        "label": gse,
                        "kind": "gse",
                        "x": 2.1,
                        "y": g_y,
                        "size": 19,
                        "color": GRAPH_THEME["gse"],
                        "symbol": "diamond",
                        "hover": f"GEO series {gse}",
                    }
                )

            edge_rows.append(
                {
                    "x0": 1.0,
                    "y0": y,
                    "x1": 2.1,
                    "y1": g_y,
                    "width": max(1.0, widths[i] * 0.7),
                    "color": GRAPH_THEME["edge_gse"],
                }
            )

    fig = go.Figure()
    for e in edge_rows:
        fig.add_trace(
            go.Scatter(
                x=[e["x0"], e["x1"]],
                y=[e["y0"], e["y1"]],
                mode="lines",
                line={"width": e["width"], "color": e["color"]},
                hoverinfo="skip",
                showlegend=False,
            )
        )

    node_df = pd.DataFrame(node_rows)

    # Declutter: with many GSM hits, 30 always-on labels collide, so at high
    # node counts we keep labels only for the query + GSE studies and rely on
    # hover for individual GSM ids.
    gsm_count = int((node_df["kind"] == "gsm").sum())
    if gsm_count > 12:
        node_df["display_label"] = node_df.apply(
            lambda r: "" if r["kind"] == "gsm" else r["label"], axis=1
        )
    else:
        node_df["display_label"] = node_df["label"]

    fig.add_trace(
        go.Scatter(
            x=node_df["x"],
            y=node_df["y"],
            mode="markers+text",
            text=node_df["display_label"],
            textposition="top center",
            textfont={"family": GRAPH_THEME["font_sans"], "size": 11, "color": GRAPH_THEME["text_secondary"]},
            hovertemplate="%{customdata[2]}<extra></extra>",
            customdata=node_df[["kind", "node_id", "hover"]].values,
            marker={
                "size": node_df["size"],
                "color": node_df["color"],
                "symbol": node_df["symbol"],
                "line": {"width": 1.5, "color": GRAPH_THEME["marker_line"]},
            },
            showlegend=False,
        )
    )

    fig.update_layout(
        margin={"l": 16, "r": 16, "t": 16, "b": 16},
        paper_bgcolor=GRAPH_THEME["paper_bg"],
        plot_bgcolor=GRAPH_THEME["plot_bg"],
        font={"family": GRAPH_THEME["font_sans"], "color": GRAPH_THEME["text_primary"]},
        xaxis={"visible": False},
        yaxis={"visible": False},
        clickmode="event+select",
        hoverlabel={"font": {"family": GRAPH_THEME["font_sans"], "size": 12}, "bgcolor": "#ffffff", "bordercolor": GRAPH_THEME["grid"]},
        autosize=True,
        height=None,
    )
    return fig


def build_bar_figure(hits_df: pd.DataFrame) -> go.Figure:
    display = hits_df.sort_values("score", ascending=True)
    labels = [f"{g} ({s})" for g, s in zip(display["gsm"], display["gse"].replace("", "no GSE"))]
    fig = go.Figure(
        go.Bar(
            x=display["score"],
            y=labels,
            orientation="h",
            marker={
                "color": display["score"],
                "colorscale": "Blues",
                "line": {"color": "#1f3d7a", "width": 0.8},
            },
            hovertemplate="%{y}<br>Score: %{x:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        title={"text": "Top retrieved analogs by cosine similarity", "font": {"family": GRAPH_THEME["font_sans"], "size": 15, "color": GRAPH_THEME["text_primary"]}},
        margin={"l": 20, "r": 20, "t": 56, "b": 30},
        paper_bgcolor=GRAPH_THEME["paper_bg"],
        plot_bgcolor=GRAPH_THEME["plot_bg"],
        font={"family": GRAPH_THEME["font_sans"], "color": GRAPH_THEME["text_secondary"]},
        xaxis={"title": "Similarity", "gridcolor": GRAPH_THEME["grid"], "zerolinecolor": GRAPH_THEME["grid"]},
        yaxis_title="",
        height=420,
    )
    return fig


def _detail_row(label: str, value: Any, mono: bool = False) -> Any:
    """A single label / value row for the details panel."""
    text = _safe_str(value)
    cls = "value" + (" mono" if mono else "")
    val = html.Span(text, className=cls) if text else html.Span("—", className=cls + " empty")
    return html.Div(className="detail-row", children=[html.Span(label, className="label"), val])


def _detail_link_row(label: str, url: str) -> Any:
    url = _safe_str(url)
    if not url:
        return _detail_row(label, "")
    href = url if re.match(r"^(https?|ftp)://", url) else f"ftp://{url}"
    return html.Div(
        className="detail-row",
        children=[
            html.Span(label, className="label"),
            html.Span(className="value", children=html.A(url, href=href, target="_blank")),
        ],
    )


def _detail_section(title: str, rows: list[Any]) -> Any | None:
    rows = [r for r in rows if r is not None]
    if not rows:
        return None
    return html.Div(className="details-section", children=[html.Div(title, className="details-section-title"), *rows])


def _detail_text_block(title: str, text: str, collapsible: bool = False, placeholder: str = "Not available.") -> Any:
    """Full-width long-form text block; collapsible for multi-paragraph fields."""
    text = _safe_str(text)
    if collapsible and text:
        return html.Details(
            className="detail-collapse",
            children=[html.Summary(title), html.Div(text, className="detail-block-body")],
        )
    return html.Div(
        className="detail-block",
        children=[
            html.Div(title, className="detail-block-title"),
            html.Div(text or placeholder, className="detail-block-body"),
        ],
    )


def _details_head(kicker: str, heading: str, score: float | None = None) -> Any:
    children: list[Any] = [
        html.Div(
            children=[
                html.Div(kicker, className="details-kicker"),
                html.H3(heading, className="details-heading"),
            ]
        )
    ]
    if score is not None:
        children.append(html.Span(f"{score:.4f}", className="score-badge"))
    return html.Div(className="details-head", children=children)


AUTHORITATIVE_GENE_LIST = ROOT / "data" / "archs4" / "train_orthologs" / "canonical_genes.csv"


def build_gene_list_banner() -> Any:
    """Persistent banner shown when retrieval is running on a stand-in gene list.

    demo_osdr_top5.py prints this warning, but the app captures the subprocess
    output and only reads it when the process fails, so on a successful run the
    warning is discarded and never reaches the person looking at the results.
    The check here is a cheap existence test rather than a preflight call,
    because it runs at import time on every page load.
    """
    if AUTHORITATIVE_GENE_LIST.exists():
        return None

    return html.Div(
        className="invalid-banner",
        children=[
            html.Span("Results are not scientifically valid", className="invalid-banner-title"),
            html.Span(
                "The authoritative gene list is missing, so retrieval is running on a "
                "stand-in that reproduces the model's gene count but not its training "
                "gene order. Query vectors are built in a different gene space than the "
                "ARCHS4 index, so similarity scores look plausible but are not "
                "meaningful and must not be interpreted biologically.",
                className="invalid-banner-body",
            ),
        ],
    )


def build_status_banner(message: str, kind: str = "info", detail: str | None = None) -> Any:
    """One-line status banner. ``kind`` is info | good | error.

    When ``detail`` is provided (e.g. a full error blob), a collapsed
    "Show details" disclosure is appended so debugging text stays out of the
    primary viewport but remains reachable.
    """
    children: list[Any] = [html.Span(message, className="status-banner-text")]
    if detail and _safe_str(detail) and _safe_str(detail) != _safe_str(message):
        children.append(
            html.Details(
                className="status-details",
                children=[
                    html.Summary("Show details"),
                    html.Pre(_safe_str(detail), className="status-details-pre"),
                ],
            )
        )
    return html.Div(children, className=f"status-banner status-{kind}")


def _build_query_details(query: pd.Series, compact: bool) -> list[Any]:
    """Details for the OSDR query node. ``compact`` omits the finer biology rows."""
    heading = _safe_str(query.get("sample_name")) or _safe_str(query.get("sample_id")) or "OSDR query"
    biology_rows = [
        _detail_row("Species", "Mus musculus"),
        _detail_row("Tissue", _safe_str(query.get("tissue"))),
        _detail_row("Condition", _safe_str(query.get("condition"))),
    ]
    if not compact:
        biology_rows += [
            _detail_row("Strain", _safe_str(query.get("strain"))),
            _detail_row("Sex", _safe_str(query.get("sex"))),
            _detail_row("Duration", _safe_str(query.get("duration"))),
        ]
    parts: list[Any] = [
        _details_head("OSDR query", heading),
        _detail_section(
            "Identity",
            [
                _detail_row("Sample ID", _safe_str(query.get("sample_id")), mono=True),
                _detail_row("Study ID", _safe_str(query.get("study_id")), mono=True),
            ],
        ),
        _detail_section("Biology", biology_rows),
    ]
    parts += _build_osdr_query_metadata_block(query)
    return [p for p in parts if p is not None]


def build_details_panel(query: pd.Series, selected_payload: dict[str, Any] | None, hits_df: pd.DataFrame) -> list[Any]:
    node_kind = _safe_str(selected_payload.get("kind")) if selected_payload else ""
    node_id = _safe_str(selected_payload.get("node_id")) if selected_payload else ""

    if not selected_payload or node_kind == "query":
        return _build_query_details(query, compact=not selected_payload)

    if node_kind == "gse":
        df = hits_df[hits_df["gse"] == node_id]
        examples = ", ".join(df["gsm"].head(8).astype(str).tolist())
        return [
            _details_head("GSE study", node_id),
            _detail_section(
                "Overview",
                [
                    _detail_row("Connected GSM hits", str(len(df))),
                    _detail_row("Example GSMs", examples),
                ],
            ),
            html.P("Click an individual GSM node for full GEO fields.", className="details-empty-hint"),
        ]

    df = hits_df[hits_df["gsm"] == node_id]
    if df.empty:
        return [
            _details_head("Details", "No metadata"),
            html.P("No metadata found for the selected node.", className="details-empty"),
        ]

    r = df.iloc[0]
    species = _first_non_empty(r, ["species", "geo_taxon_biopython"])
    source_name = _first_non_empty(r, ["source_name", "source_name_ch1"])
    characteristics = _first_non_empty(r, ["characteristics", "characteristics_ch1"])
    gse = _first_non_empty(r, ["gse", "series_id", "geo_gse_biopython"])
    platform = _first_non_empty(r, ["geo_platform_biopython", "platform_ncbi"])
    entry_type = _first_non_empty(r, ["geo_entry_type_biopython", "entry_type_ncbi"])
    gds_type = _first_non_empty(r, ["geo_gds_type_biopython", "gds_type_ncbi"])
    pdat = _first_non_empty(r, ["geo_pdat_biopython", "pdat_ncbi"])
    n_samples = _first_non_empty(r, ["geo_n_samples_biopython", "n_samples_ncbi"])
    ftp_link = _first_non_empty(r, ["geo_ftp_link_biopython", "ftp_link_ncbi"])

    title = _first_non_empty(r, ["title", "geo_title_biopython"])
    geo_summary = _first_non_empty(r, ["geo_summary", "geo_summary_biopython", "geo_abstract_biopython"])
    geo_design = _first_non_empty(r, ["geo_design", "geo_overall_design_biopython", "design_ncbi"])
    pubmed_ids = _first_non_empty(r, ["pubmed_ids", "geo_pubmed_ids_biopython", "pubmed_id"])
    pubmed_title = _first_non_empty(r, ["pubmed_title_biopython", "pubmed_title_ncbi"])
    pubmed_journal = _first_non_empty(r, ["pubmed_journal_biopython", "pubmed_journal_ncbi"])
    pubmed_date = _first_non_empty(r, ["pubmed_pub_date_biopython", "pubmed_pub_date_ncbi"])
    pubmed_doi = _first_non_empty(r, ["pubmed_doi_biopython", "pubmed_doi_ncbi"])

    parts: list[Any] = [
        _details_head("ARCHS4 hit · GSM", _safe_str(r.get("gsm")), score=float(r.get("score", 0.0))),
        _detail_section(
            "Identity",
            [
                _detail_row("GSM", _safe_str(r.get("gsm")), mono=True),
                _detail_row("GSE", gse, mono=True),
                _detail_row("Title", title),
            ],
        ),
        _detail_section(
            "Biology",
            [
                _detail_row("Species", species),
                _detail_row("Source name", source_name),
                _detail_row("Characteristics", characteristics),
            ],
        ),
        _detail_section(
            "Platform & series",
            [
                _detail_row("Platform", platform),
                _detail_row("Entry type", entry_type),
                _detail_row("GDS type", gds_type),
                _detail_row("Release date", pdat),
                _detail_row("Series sample count", n_samples),
                _detail_link_row("FTP link", ftp_link),
            ],
        ),
    ]

    if _safe_str(geo_summary) or _safe_str(geo_design):
        context = html.Div(className="details-section", children=[html.Div("Study context", className="details-section-title")])
        blocks = [c for c in [
            _detail_text_block("GEO summary", geo_summary, collapsible=True) if _safe_str(geo_summary) else None,
            _detail_text_block("Overall design", geo_design, collapsible=True) if _safe_str(geo_design) else None,
        ] if c is not None]
        context.children = context.children + blocks
        parts.append(context)

    pub_rows = [
        _detail_row("PubMed IDs", pubmed_ids, mono=True),
        _detail_row("Title", pubmed_title),
        _detail_row("Journal / date", " ".join(x for x in [pubmed_journal, pubmed_date] if x)),
        _detail_row("DOI", pubmed_doi),
    ]
    if any(_safe_str(v) for v in [pubmed_ids, pubmed_title, pubmed_journal, pubmed_date, pubmed_doi]):
        parts.append(_detail_section("Publication", pub_rows))

    return [p for p in parts if p is not None]


samples_df = load_osdr_samples(OSDR_METADATA_PATH)
study_options = sorted(samples_df["study_id"].dropna().astype(str).unique().tolist())
default_study = study_options[0] if study_options else ""
default_samples = samples_df[samples_df["study_id"] == default_study]
default_sample_id = default_samples.iloc[0]["sample_id"] if not default_samples.empty else ""


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


def _eligible_osdr_count(df: pd.DataFrame) -> int | None:
    """OSDR samples eligible for retrieval: mouse counts present + a spaceflight
    condition present (mirrors the demo script's eligibility filter)."""
    try:
        counts_ok = df["counts_path"].astype(str).str.len() > 0
        condition_ok = df["condition"].astype(str).str.len() > 0
        return int((counts_ok & condition_ok).sum())
    except Exception:
        return None


def _format_count(value: int | None) -> str:
    return f"{value:,}" if isinstance(value, int) else "—"


ARCHS4_SAMPLE_COUNT = _archs4_sample_count()
ELIGIBLE_OSDR_COUNT = _eligible_osdr_count(samples_df)


app: Dash = Dash(__name__)
app.title = "Bridge RNA · OSDR → ARCHS4 Explorer"

# Pull in Inter + JetBrains Mono (local dev app, so a Google Fonts <link> is fine).
app.index_string = """<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
        {%favicon%}
        {%css%}
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>"""

def build_graph_legend() -> Any:
    """Horizontal legend strip explaining node shapes/colors + edge encoding."""
    return html.Div(
        className="graph-legend",
        children=[
            html.Div(className="legend-item", children=[
                html.Span(className="legend-swatch legend-swatch--star"),
                html.Span("OSDR query"),
            ]),
            html.Div(className="legend-item", children=[
                html.Span(className="legend-swatch legend-swatch--circle"),
                html.Span("GSM sample (ARCHS4 hit)"),
            ]),
            html.Div(className="legend-item", children=[
                html.Span(className="legend-swatch legend-swatch--diamond"),
                html.Span("GSE study"),
            ]),
            html.Span(className="legend-divider"),
            html.Div(className="legend-note", children=[
                html.Span(className="legend-edge"),
                html.Span("edge width = similarity score"),
            ]),
        ],
    )


app.layout = html.Div(
    className="app-root",
    children=[
        html.Header(
            className="app-header",
            children=[
                html.Div(
                    className="app-brand",
                    children=[
                        html.Div("BR", className="app-brand-mark"),
                        html.Div(
                            className="app-brand-text",
                            children=[
                                html.H1("Bridge RNA", className="app-title"),
                                html.P(
                                    "OSDR → ARCHS4 transcriptomic analog retrieval",
                                    className="app-subtitle",
                                ),
                            ],
                        ),
                    ],
                ),
                html.Div(
                    className="app-header-meta",
                    children=[
                        html.Div(
                            className="header-stat",
                            title="Earth-based samples in the ARCHS4 embedding index",
                            children=[
                                html.Span(_format_count(ARCHS4_SAMPLE_COUNT), className="header-stat-value"),
                                html.Span("ARCHS4 samples", className="header-stat-label"),
                            ],
                        ),
                        html.Div(className="header-stat-divider"),
                        html.Div(
                            className="header-stat",
                            title="OSDR samples eligible for retrieval (mouse counts + spaceflight condition)",
                            children=[
                                html.Span(_format_count(ELIGIBLE_OSDR_COUNT), className="header-stat-value header-stat-value--accent"),
                                html.Span("Eligible OSDR samples", className="header-stat-label"),
                            ],
                        ),
                    ],
                ),
            ],
        ),
        build_gene_list_banner(),
        html.Div(
            className="app-grid",
            children=[
                # ---- Left: tool panel ----
                html.Aside(
                    className="sidebar",
                    children=[
                        html.H2("Search controls", className="sidebar-title"),
                        html.Div(
                            className="control-group",
                            children=[
                                html.Div("Query sample", className="control-group-title"),
                                html.Div(
                                    className="control",
                                    children=[
                                        html.Label("OSDR study", className="control-label"),
                                        dcc.Dropdown(
                                            id="study-dropdown",
                                            options=[{"label": s, "value": s} for s in study_options],
                                            value=default_study,
                                            clearable=False,
                                        ),
                                    ],
                                ),
                                html.Div(
                                    className="control",
                                    children=[
                                        html.Label("OSDR sample", className="control-label"),
                                        dcc.Dropdown(id="sample-dropdown", clearable=False),
                                    ],
                                ),
                                html.Div(id="sample-preview", className="sample-preview"),
                            ],
                        ),
                        html.Div(
                            className="control-group",
                            children=[
                                html.Div("Retrieval", className="control-group-title"),
                                html.Div(
                                    className="control",
                                    children=[
                                        html.Label("Top-k neighbors", className="control-label"),
                                        html.Div(
                                            className="control-slider",
                                            children=[
                                                dcc.Slider(
                                                    id="topk-slider",
                                                    min=3, max=30, step=1, value=5,
                                                    marks={3: "3", 5: "5", 10: "10", 20: "20", 30: "30"},
                                                ),
                                            ],
                                        ),
                                    ],
                                ),
                            ],
                        ),
                        html.Details(
                            className="control-group advanced-group",
                            children=[
                                html.Summary(
                                    className="advanced-summary",
                                    children=[
                                        html.Span("Metadata enrichment", className="control-group-title"),
                                        html.Span("Optional", className="advanced-badge"),
                                    ],
                                ),
                                html.Div(
                                    className="advanced-body",
                                    children=[
                                        html.Div(
                                            className="control",
                                            children=[
                                                html.Label(
                                                    [
                                                        "Entrez email ",
                                                        html.Span("(GEO / PubMed lookups)", className="control-hint"),
                                                    ],
                                                    className="control-label",
                                                ),
                                                dcc.Input(
                                                    id="entrez-email-input",
                                                    type="email",
                                                    value=DEFAULT_ENTREZ_EMAIL,
                                                    placeholder="name@domain.com",
                                                    className="dash-input",
                                                ),
                                            ],
                                        ),
                                        dcc.Checklist(
                                            id="biopython-toggle",
                                            options=[{"label": " Enrich with Biopython (GEO + PubMed)", "value": "on"}],
                                            value=["on"],
                                            className="dash-checklist",
                                        ),
                                    ],
                                ),
                            ],
                        ),
                        html.Div(
                            className="control-group",
                            children=[
                                html.Button("Search", id="search-button", n_clicks=0, className="btn-primary"),
                                html.Div(id="query-running-indicator", className="running-indicator"),
                                html.Div(
                                    id="search-status",
                                    children=build_status_banner("Select a sample and run a search.", kind="info"),
                                ),
                            ],
                        ),
                        dcc.Store(id="hits-store"),
                        dcc.Store(id="selected-node-store"),
                    ],
                ),
                # ---- Center: workspace (the main event) ----
                html.Main(
                    className="workspace",
                    children=[
                        html.Div(
                            className="panel panel--canvas",
                            children=[
                                html.Div(
                                    className="panel-header",
                                    children=[
                                        html.Span(className="panel-dot"),
                                        html.Div(
                                            children=[
                                                html.H2("Retrieval network", className="panel-title"),
                                                html.P(
                                                    "OSDR query → nearest ARCHS4 GSM samples → GSE studies",
                                                    className="panel-subtitle",
                                                ),
                                            ],
                                        ),
                                    ],
                                ),
                                build_graph_legend(),
                                html.Div(
                                    className="graph-wrap",
                                    children=[
                                        dcc.Graph(
                                            id="network-graph",
                                            className="dash-graph",
                                            figure=_empty_network_figure(),
                                            config={"displaylogo": False, "responsive": True},
                                            style={"height": "100%"},
                                        ),
                                    ],
                                ),
                            ],
                        ),
                    ],
                ),
                # ---- Right: inspector ----
                html.Aside(
                    className="inspector",
                    children=[
                        html.Div(id="details-panel", className="panel details-panel"),
                        html.Div(
                            className="panel ai-panel",
                            children=[
                                html.Div(
                                    className="panel-header",
                                    children=[
                                        html.Span(className="panel-dot panel-dot--warm"),
                                        html.H2("AI hypothesis", className="panel-title"),
                                        html.Span("Beta", className="app-header-chip"),
                                    ],
                                ),
                                html.Button(
                                    "Generate AI summary",
                                    id="ai-summary-button",
                                    n_clicks=0,
                                    className="btn-secondary",
                                ),
                                html.Div(id="ai-summary-status", className="ai-status"),
                                dcc.Markdown(id="ai-summary-output", className="ai-output"),
                            ],
                        ),
                    ],
                ),
            ],
        ),
    ],
)


@app.callback(
    Output("sample-dropdown", "options"),
    Output("sample-dropdown", "value"),
    Input("study-dropdown", "value"),
)
def update_sample_options(study_id: str):
    filtered = samples_df[samples_df["study_id"] == study_id].copy()
    opts = []
    for _, r in filtered.iterrows():
        label = f"{_safe_str(r['sample_name'])} | {_safe_str(r['condition'])} | {_safe_str(r['tissue'])}"
        opts.append({"label": label, "value": _safe_str(r["sample_id"])})
    value = opts[0]["value"] if opts else None
    return opts, value


@app.callback(
    Output("sample-preview", "children"),
    Input("sample-dropdown", "value"),
)
def update_sample_preview(sample_id: str):
    """Instant local summary of the selected OSDR sample, shown before any search."""
    empty = html.P("Select a sample to preview its metadata.", className="sample-preview-empty")
    if not sample_id:
        return empty
    match = samples_df.loc[samples_df["sample_id"].astype(str) == str(sample_id)]
    if match.empty:
        return empty
    row = match.iloc[0]

    def _tidy(value: Any) -> str:
        # Unwrap ISA-Tab unit annotations, e.g. "37 {day}" -> "37 day".
        return re.sub(r"\s*\{([^}]*)\}", r" \1", _safe_str(value)).strip()

    fields = [
        ("Study", _tidy(row.get("study_id"))),
        ("Tissue", _tidy(row.get("tissue"))),
        ("Spaceflight", _tidy(row.get("condition"))),
        ("Strain", _tidy(row.get("strain"))),
        ("Sex", _tidy(row.get("sex"))),
        ("Duration", _tidy(row.get("duration"))),
    ]
    detail_rows = [
        html.Div(
            className="sample-preview-row",
            children=[
                html.Span(label, className="sample-preview-key"),
                html.Span(value, className="sample-preview-val"),
            ],
        )
        for label, value in fields
        if value
    ]
    return html.Div(
        className="sample-preview-card",
        children=[
            html.Div(_safe_str(row.get("sample_name")), className="sample-preview-name"),
            html.Div(className="sample-preview-grid", children=detail_rows),
        ],
    )


@app.callback(
    Output("network-graph", "figure"),
    Output("hits-store", "data"),
    Output("search-status", "children"),
    Input("search-button", "n_clicks"),
    State("sample-dropdown", "value"),
    State("topk-slider", "value"),
    State("entrez-email-input", "value"),
    State("biopython-toggle", "value"),
    running=[
        (Output("search-button", "disabled"), True, False),
        (Output("query-running-indicator", "children"), "Query running... retrieving nearest neighbors and metadata.", ""),
    ],
)
def run_search(
    _: int,
    sample_id: str,
    topk: int,
    entrez_email: str | None,
    biopython_toggle: list[str] | None,
):
    if not sample_id:
        return (
            _empty_network_figure("Select an OSDR sample, then run a search."),
            None,
            build_status_banner("Select a sample to start.", kind="info"),
        )

    q_row = samples_df.loc[samples_df["sample_id"] == sample_id].iloc[0]
    enable_biopython = bool(biopython_toggle and "on" in biopython_toggle)
    email_value = _safe_str(entrez_email) or GENERIC_ENTREZ_EMAIL
    try:
        hits_df, mode = search_hits(
            samples_df=samples_df,
            sample_id=sample_id,
            topk=int(topk),
            entrez_email=email_value,
            enable_biopython_metadata=enable_biopython,
        )
    except Exception as exc:
        detail = getattr(exc, "detail", "") or _safe_str(exc)
        return (
            _empty_network_figure("Retrieval failed - see status for details."),
            None,
            build_status_banner(
                _last_nonempty_line(_safe_str(exc)) or "Retrieval failed.",
                kind="error",
                detail=detail,
            ),
        )

    network = build_network_figure(query=q_row, hits_df=hits_df)

    status_message = (
        f"Retrieved {len(hits_df)} hits using precomputed OSDR query embeddings."
        if mode == "precomputed"
        else (
            f"Retrieved {len(hits_df)} hits using real demo script output"
            + (" + Biopython metadata enrichment." if (enable_biopython and _safe_str(entrez_email)) else ".")
        )
    )
    status = build_status_banner(status_message, kind="good")
    payload = {
        "sample_id": sample_id,
        "entrez_email": email_value,
        "biopython_enabled": bool(enable_biopython),
        "hits": hits_df.to_dict(orient="records"),
    }
    return network, payload, status


@app.callback(
    Output("ai-summary-output", "children"),
    Output("ai-summary-status", "children"),
    Input("ai-summary-button", "n_clicks"),
    State("hits-store", "data"),
    running=[
        (Output("ai-summary-button", "disabled"), True, False),
        (Output("ai-summary-status", "children"), "Generating hypothesis...", ""),
        (Output("ai-summary-status", "className"), "ai-status ai-status--loading", "ai-status"),
    ],
    prevent_initial_call=True,
)
def generate_ai_summary(_: int, hits_payload: dict[str, Any] | None):
    if not hits_payload:
        return "", "Run a retrieval first so metadata is available."

    sample_id = _safe_str(hits_payload.get("sample_id"))
    q_match = samples_df.loc[samples_df["sample_id"] == sample_id]
    if q_match.empty:
        return "", "Selected query sample is missing from local metadata."

    query_row = q_match.iloc[0]
    hits_df = pd.DataFrame(hits_payload.get("hits", []))
    if not _safe_str(hits_payload.get("entrez_email")):
        hits_payload["entrez_email"] = GENERIC_ENTREZ_EMAIL

    prompt_template = _load_ai_prompt_template()
    prompt = prompt_template.format(
        osdr_metadata=_format_osdr_query_text(query_row),
        retrieved_hits_table=_format_hits_table_text(hits_df),
        geo_summaries=_format_geo_context_text(hits_df),
    )

    summary = _call_ai_summary(prompt)
    return summary, ""


@app.callback(
    Output("selected-node-store", "data"),
    Input("network-graph", "clickData"),
)
def select_node(click_data: dict[str, Any] | None):
    if not click_data:
        return None
    points = click_data.get("points", [])
    if not points:
        return None
    custom = points[0].get("customdata")
    if not custom or len(custom) < 2:
        return None
    return {"kind": custom[0], "node_id": custom[1]}


@app.callback(
    Output("details-panel", "children"),
    Input("hits-store", "data"),
    Input("selected-node-store", "data"),
)
def render_details(hits_payload: dict[str, Any] | None, selected_node: dict[str, Any] | None):
    if not hits_payload:
        return [
            _details_head("Inspector", "Details"),
            html.P("Run a search to load the retrieval network.", className="details-empty"),
            html.P("Then click any node - the query, a GSM hit, or a GSE study - to inspect its metadata here.", className="details-empty-hint"),
        ]

    sample_id = _safe_str(hits_payload.get("sample_id"))
    entrez_email = _safe_str(hits_payload.get("entrez_email")) or GENERIC_ENTREZ_EMAIL
    biopython_enabled = bool(hits_payload.get("biopython_enabled", False))
    q_row = samples_df.loc[samples_df["sample_id"] == sample_id].iloc[0]
    hits_df = pd.DataFrame(hits_payload.get("hits", []))

    # If a GSM is clicked and fields are blank, enrich that one on demand.
    if selected_node and _safe_str(selected_node.get("kind")) == "gsm" and not hits_df.empty and biopython_enabled and entrez_email:
        gsm = _safe_str(selected_node.get("node_id"))
        one = hits_df[hits_df["gsm"] == gsm]
        if not one.empty:
            r = one.iloc[0]
            has_core = any(
                _safe_str(r.get(c))
                for c in ["gse", "title", "geo_summary", "pubmed_ids"]
            )
            if not has_core:
                enriched_one = _enrich_hits_from_ncbi_eutils(one.copy(), entrez_email)
                for col in enriched_one.columns:
                    if col in hits_df.columns:
                        hits_df.loc[hits_df["gsm"] == gsm, col] = enriched_one.iloc[0][col]

    return build_details_panel(query=q_row, selected_payload=selected_node, hits_df=hits_df)


if __name__ == "__main__":
    # Keep hot-reload for development, but hide the floating dev-tools toolbar so
    # it doesn't overlap the UI during a demo. Set DASH_DEBUG=0 to disable reload.
    debug = os.environ.get("DASH_DEBUG", "1") not in ("0", "false", "False")
    app.run(debug=debug, dev_tools_ui=False, host="0.0.0.0", port=8050)
