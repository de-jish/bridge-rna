"""The optional AI hypothesis: prompt assembly and the two providers.

This is the one feature that depends on software the repository cannot install
for you, so every failure path answers with what to do rather than with a
connection traceback.
"""

from __future__ import annotations

import json
import re
from typing import Any

import pandas as pd
import requests

from .config import (
    AI_PROMPT_PATH,
    AI_SUMMARY_PROVIDER,
    BEDROCK_API_KEY,
    BEDROCK_API_KEY_HEADER,
    BEDROCK_API_URL,
    BEDROCK_PAYLOAD_KEY,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    OLLAMA_TIMEOUT_SECONDS,
)
from .osdr import _fetch_osdr_study_summary
from .util import _safe_str


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

    # AI summaries are the one feature that depends on software this repository
    # cannot install for you, so failures here are the most likely thing a new
    # user hits. Answer with what to do rather than with a connection traceback.
    try:
        resp = _generate(_pick_model())
    except requests.exceptions.ConnectionError:
        return (
            f"Could not reach Ollama at {OLLAMA_BASE_URL}.\n\n"
            "AI summaries are optional. Retrieval, the network graph, and all metadata "
            "on this page work without them.\n\n"
            "To enable them, install Ollama from https://ollama.com and then run:\n"
            "    ollama serve\n"
            f"    ollama pull {OLLAMA_MODEL}\n\n"
            "To use AWS Bedrock instead, set AI_SUMMARY_PROVIDER=bedrock and BEDROCK_API_URL."
        )
    except requests.exceptions.Timeout:
        return (
            f"Ollama did not respond within {OLLAMA_TIMEOUT_SECONDS} seconds.\n\n"
            "A local model is usually slowest on its first call, while the weights load "
            "into memory. Try again, or set OLLAMA_MODEL to a smaller model."
        )
    except Exception as exc:
        return f"AI Summary call failed (Ollama): {_safe_str(exc)}"

    if resp.status_code >= 400:
        err = _safe_str(resp.text)
        if len(err) > 800:
            err = err[:800] + "..."
        if resp.status_code == 404 and "not found" in err.lower():
            available = _get_available_models()
            if not available:
                return (
                    f"Ollama is running at {OLLAMA_BASE_URL}, but no models are installed.\n\n"
                    "Pull one first:\n"
                    f"    ollama pull {OLLAMA_MODEL}"
                )
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
