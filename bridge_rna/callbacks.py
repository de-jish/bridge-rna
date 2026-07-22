"""Retrieval callbacks, registered onto the shell's Dash app.

These were module-level `@app.callback` decorators against a module-level app
object, which is exactly what a router cannot mount. They are the same bodies,
one indent level deeper, inside a `register(app)` the shell calls once.
"""

from __future__ import annotations

import re
from typing import Any

import pandas as pd
from dash import Input, Output, State, html

from .ai import (
    _call_ai_summary,
    _format_geo_context_text,
    _format_hits_table_text,
    _format_osdr_query_text,
    _load_ai_prompt_template,
)
from .config import GENERIC_ENTREZ_EMAIL
from .figures import _empty_network_figure, build_network_figure
from .geo import _enrich_hits_from_ncbi_eutils
from .layout import samples_df
from .panels import _details_head, build_details_panel, build_status_banner
from .retrieval import search_hits
from .util import _last_nonempty_line, _safe_str


def register(app) -> None:
    """Attach every retrieval callback to `app`. Called once by the shell."""

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
