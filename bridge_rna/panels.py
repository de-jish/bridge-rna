"""The inspector: detail rows, sections, banners, and the details panel.

Everything here returns Dash components and reads no data of its own; the
callbacks hand it the query row and the hits frame.
"""

from __future__ import annotations

import re
from typing import Any

import pandas as pd
from dash import html

from .config import ROOT
from .osdr import _fetch_osdr_study_summary
from .preflight import (
    _canonical_gene_order_is_authoritative,
    preflight_retrieval_requirements,
)
from .util import _first_non_empty, _safe_str



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

    The test is on the gene *ordering*, not on whether a file exists. An
    existence check would clear the banner for any file sitting at the
    authoritative path, including a wrong-order one -- the same failure the
    banner exists to announce.
    """
    if _canonical_gene_order_is_authoritative(AUTHORITATIVE_GENE_LIST):
        return None

    detail = (
        "The authoritative gene list is missing, so retrieval is running on a stand-in."
        if not AUTHORITATIVE_GENE_LIST.exists()
        else "The gene list in place does not match the ordering the ARCHS4 index was built with."
    )

    return html.Div(
        className="invalid-banner",
        children=[
            html.Span("Results are not scientifically valid", className="invalid-banner-title"),
            html.Span(
                f"{detail} A list that reproduces the model's gene count but not its "
                "training gene order builds query vectors in a different gene space "
                "than the ARCHS4 index, so similarity scores look plausible but are "
                "not meaningful and must not be interpreted biologically.",
                className="invalid-banner-body",
            ),
        ],
    )


def build_setup_banner() -> Any:
    """Persistent banner listing unmet prerequisites, shown before any search.

    preflight_retrieval_requirements() is otherwise consulted only inside
    run_real_retrieval, which raises on failure. A fresh clone whose Git LFS
    payload never arrived therefore looks completely healthy: the app serves,
    the sample dropdowns populate from ordinary Git files, and nothing hints
    at a problem until someone picks a sample, clicks Search, and waits for
    the error. Surfacing it at layout time costs one preflight call at import.
    """
    try:
        missing, _ = preflight_retrieval_requirements()
    except Exception as exc:  # never let a diagnostic stop the app from serving
        missing = [f"preflight check failed: {exc}"]

    if not missing:
        return None

    return html.Div(
        className="setup-banner",
        children=[
            html.Span("Setup incomplete, so retrieval cannot run", className="setup-banner-title"),
            html.Span(
                "The interface loaded, but the files retrieval depends on are not ready. "
                "Everything else on this page works; searching will fail until these are resolved.",
                className="setup-banner-body",
            ),
            html.Ul(className="setup-banner-list", children=[html.Li(m) for m in missing]),
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
