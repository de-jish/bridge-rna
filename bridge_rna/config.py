"""Paths, environment knobs, and the constants the rest of the package reads.

Everything configurable lives here so a deployment is a matter of environment
variables rather than of edits scattered through the app.
"""

from __future__ import annotations

import os
from pathlib import Path

# bridge_rna/config.py -> the repository root is its parent's parent.
ROOT = Path(__file__).resolve().parent.parent


OSDR_METADATA_PATH = (
    ROOT / "data" / "osdr" / "metadata" / "selected_sample_metadata.tsv"
    if (ROOT / "data" / "osdr" / "metadata" / "selected_sample_metadata.tsv").exists()
    else (ROOT / "osdr" / "metadata" / "selected_sample_metadata.tsv")
)
DEMO_SCRIPT_PATH = ROOT / "demo_osdr_top5.py"
# The live-embedding subprocess for uploaded OSDR counts files. It lives under
# precompute/ so that all torch-touching code stays behind that boundary and the
# serving app never imports the scientific stack (pinned by a test).
UPLOAD_EMBED_SCRIPT_PATH = ROOT / "precompute" / "embed_upload.py"
# Cap for an uploaded counts matrix. A whole-transcriptome counts file for a
# handful of samples is a few MB; 200 MB is generous headroom and also the
# Flask MAX_CONTENT_LENGTH the app sets so an oversized upload fails cleanly.
MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", str(200 * 1024 * 1024)))
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
