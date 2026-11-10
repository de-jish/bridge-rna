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


class SummaryFailure(str):
    """A provider's answer that is not a summary: an error or a setup instruction.

    It is still the text to show, so it subclasses `str` and every message
    reads exactly as before. What it adds is a type the caller can test, so the
    status line beside it never says "Summary of the N matches shown" when no
    summary was produced, which it used to do.
    """


def _failure(text: str) -> SummaryFailure:
    return SummaryFailure(text)


def _no_summary_text(raw: str = "", *, prefix: str = "") -> SummaryFailure:
    """A reply that arrived but held no summary, with what it did hold."""
    raw = _safe_str(raw)
    if len(raw) > 400:
        raw = raw[:400] + "..."
    detail = f"\n\nThe service replied: {raw}" if raw else ""
    return _failure(f"{prefix}AI Summary returned no text.{detail}")


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
    # A whitened result arrives with its score renamed `whitened_score` and
    # the plain `cosine` beside it, so the model never reads one as the other.
    cols = [
        "gsm",
        "gse",
        "score",
        "whitened_score",
        "cosine",
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
        return _failure(
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
            structured = isinstance(data, (dict, list))
        except Exception:
            data, structured = {}, False

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
        # A JSON reply with none of the fields above is an envelope with no
        # summary in it (an error, an empty completion), not a summary to show
        # under "Summary of the N matches". A plain-text reply is the summary.
        if structured or not txt:
            return _no_summary_text(txt)
        return txt

    payload_key = BEDROCK_PAYLOAD_KEY
    primary_text = prompt if payload_key != "query" else _sanitize_query_text(prompt)
    payload = _payload_for_key(payload_key, primary_text)

    try:
        resp = requests.post(BEDROCK_API_URL, headers=headers, json=payload, timeout=120)
    except Exception as exc:
        return _failure(f"AI Summary call failed: {_safe_str(exc)}")

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
                return _failure(
                    f"AI Summary call failed: HTTP {retry_resp.status_code} after sanitize+shorten retry with payload key '{retry_key}': "
                    f"{retry_err}"
                )
            except Exception as retry_exc:
                return _failure(
                    f"AI Summary call failed during sanitize+shorten retry: {_safe_str(retry_exc)}")

        return _failure(
            f"AI Summary call failed: HTTP {resp.status_code} using payload key '{payload_key}': {err_text}")
    return _extract_text_response(resp)


def _ollama_unreachable_message() -> SummaryFailure:
    return _failure(
        f"Could not reach Ollama at {OLLAMA_BASE_URL}.\n\n"
        "AI summaries are optional. Retrieval, the network graph, and all metadata "
        "on this page work without them.\n\n"
        "To enable them, install Ollama from https://ollama.com and then run:\n"
        "    ollama serve\n"
        f"    ollama pull {OLLAMA_MODEL}\n\n"
        "To use AWS Bedrock instead, set AI_SUMMARY_PROVIDER=bedrock and BEDROCK_API_URL."
    )


def _ollama_no_model_message() -> SummaryFailure:
    return _failure(
        f"Ollama is running at {OLLAMA_BASE_URL}, but no models are installed.\n\n"
        "AI summaries are optional. Retrieval and all metadata on this page work "
        "without them.\n\n"
        "Pull one first:\n"
        f"    ollama pull {OLLAMA_MODEL}"
    )


def _ollama_models(timeout: float = 5) -> list[str] | None:
    """The names of the models Ollama has installed, or None if it did not say.

    Raises `requests.exceptions.RequestException` when Ollama cannot be reached,
    so the caller can tell "not running" from "running with nothing pulled".
    An answer that cannot be read is None, never an empty list: only an
    explicit empty list is evidence that no model is installed.
    """
    resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=timeout)
    try:
        if resp is None or resp.status_code >= 400:
            return None
        data = resp.json() if resp.text else None
    except Exception:
        return None
    models = data.get("models") if isinstance(data, dict) else None
    if not isinstance(models, list):
        return None
    names = [_safe_str(m.get("name")) for m in models
             if isinstance(m, dict) and _safe_str(m.get("name"))]
    # Entries we cannot name are still models: only an empty list says "none".
    if models and not names:
        return None
    return names


def unavailable_reason() -> str | None:
    """Why a summary cannot be produced at all, before any work is done for it.

    The caller assembles the prompt from study abstracts fetched over the
    network - one round trip per accession, and a gzip download per series -
    which is the slow part of the whole feature. Doing that first and *then*
    discovering there is no model to send it to made "install Ollama" a message
    that arrived up to a minute after the click, in the state a fresh clone
    starts in. The precondition is knowable in a few milliseconds, so it is
    checked first.

    "Knowable" has two halves for Ollama, and until 2026-11-10 only the first
    was checked: whether it answers, and whether it has any model to run. A
    running Ollama with nothing pulled - a fresh install, and this project's
    demo machine - passed the check, so the click spent 30 to 90 s fetching
    abstracts before the call itself said "no models are installed". Both
    halves come from the same local `/api/tags` request.

    Returns None when there is nothing to report, which includes the case where
    the answer is not cheaply knowable: Bedrock has no probe short of a real
    request, so a configured endpoint is taken at its word and any failure
    surfaces through the normal path. An Ollama answer that cannot be read is
    treated the same way rather than as "no models".
    """
    if AI_SUMMARY_PROVIDER == "ollama":
        if not OLLAMA_BASE_URL:
            return _failure("AI Summary call failed: OLLAMA_BASE_URL is not configured.")
        try:
            models = _ollama_models(timeout=5)
        except requests.exceptions.ReadTimeout:
            # Connected but slow: a busy Ollama is not an absent one. The call
            # itself waits OLLAMA_TIMEOUT_SECONDS and says so if it gives up.
            # (ConnectTimeout is not caught here: it is a ConnectionError too,
            # and a host that cannot be connected to is unreachable.)
            return None
        except requests.exceptions.RequestException:
            return _ollama_unreachable_message()
        if models == []:
            return _ollama_no_model_message()
        return None
    if AI_SUMMARY_PROVIDER == "bedrock" and not BEDROCK_API_URL:
        return _failure("AI Summary call failed: AI_SUMMARY_PROVIDER is 'bedrock' but "
                        "BEDROCK_API_URL is not set.")
    return None


def _call_ollama_summary(prompt: str) -> str:
    if not OLLAMA_BASE_URL:
        return _failure("AI Summary call failed: OLLAMA_BASE_URL is not configured.")

    def _get_available_models() -> list[str]:
        try:
            return _ollama_models(timeout=20) or []
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
        return _ollama_unreachable_message()
    except requests.exceptions.Timeout:
        return _failure(
            f"Ollama did not respond within {OLLAMA_TIMEOUT_SECONDS} seconds.\n\n"
            "A local model is usually slowest on its first call, while the weights load "
            "into memory. Try again, or set OLLAMA_MODEL to a smaller model."
        )
    except Exception as exc:
        return _failure(f"AI Summary call failed (Ollama): {_safe_str(exc)}")

    if resp.status_code >= 400:
        err = _safe_str(resp.text)
        if len(err) > 800:
            err = err[:800] + "..."
        if resp.status_code == 404 and "not found" in err.lower():
            available = _get_available_models()
            if not available:
                return _ollama_no_model_message()
            if available:
                fallback = _pick_model()
                if fallback != OLLAMA_MODEL:
                    try:
                        retry = _generate(fallback)
                        if retry.status_code < 400:
                            label = f"[Using fallback model: {fallback}]\n\n"
                            text = _ollama_response_text(retry, prefix=label)
                            return text if isinstance(text, SummaryFailure) else label + text
                    except Exception as retry_exc:
                        return _failure(
                            f"AI Summary call failed (Ollama fallback): {_safe_str(retry_exc)}")
        return _failure(f"AI Summary call failed (Ollama): HTTP {resp.status_code}: {err}")

    return _ollama_response_text(resp)


def _ollama_response_text(resp: requests.Response, *, prefix: str = "") -> str:
    """The `response` field of an `/api/generate` reply, or why there is none.

    Ollama always answers that endpoint with JSON. A reply whose `response` is
    empty or missing - an empty completion, or `{"error": ...}` with HTTP 200 -
    is not a summary, and used to be shown as one, raw JSON and all, under
    "Summary of the N matches shown".
    """
    try:
        data = resp.json() if resp.text else None
    except Exception:
        data = None
    if isinstance(data, dict):
        text = _safe_str(data.get("response"))
        if text:
            return text
        error = _safe_str(data.get("error"))
        if error:
            return _failure(f"{prefix}AI Summary call failed (Ollama): {error}")
    return _no_summary_text(resp.text, prefix=prefix)


def _call_ai_summary(prompt: str) -> str:
    """The summary text, or a `SummaryFailure` saying why there is none."""
    provider = AI_SUMMARY_PROVIDER
    if provider == "bedrock":
        return _call_bedrock_summary(prompt)
    if provider == "ollama":
        return _call_ollama_summary(prompt)
    return _failure(
        f"AI Summary provider '{provider}' is not supported. "
        "Use AI_SUMMARY_PROVIDER=ollama or AI_SUMMARY_PROVIDER=bedrock."
    )
