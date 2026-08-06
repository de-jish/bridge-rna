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


# --- The cohort card: how much to trust this pooled query --------------------


#: What each role is called on the rail. A comparison pools two cohorts and
#: draws both, in the network figure and on the map, and the only thing binding
#: a card to those glyphs is the color of the dot beside this name - so
#: `.cohort-role-dot` in assets/retrieve.css mirrors GRAPH_THEME's `cohort_a`
#: and `cohort_b` exactly, and a test pins the pair.
COHORT_ROLES = {"a": "Cohort A", "b": "Cohort B"}


def build_cohort_card(cohort, geometry, role: str = "",
                      contrast: str = "") -> Any:
    """Size and result stability for one pooled cohort.

    One number, and that is the point. **Stability** is a property of *k*: the
    measured agreement between this cohort's top-5 and the top-5 it would have
    produced with any one animal left out. It says how far to trust the list,
    and it is quoted rather than reduced to a word, because "low confidence"
    tells a researcher nothing they can act on while "3 samples, 0.51" tells
    them exactly how far down the list to stop reading.

    A second stat sat beside it and was removed: `R̄`, the vMF resultant
    length, labelled "Group tightness". Measured over all 212 real cohorts it
    is near-constant at a median 0.9991, and no lower for a cohort of two than
    for one of thirty, so it never changed a decision while looking like a
    grade. The per-member leave-one-out cosine stays, in the member list on the
    rail, because that one varies and points at a specific animal.

    ``role`` names which arm of a comparison this is, and is empty when only
    one cohort is being pooled - a lone card needs no letter, and adding one
    would be chrome for a distinction that does not exist yet. When it is set
    the card gains a colored role line and the cohort's own name, because a
    comparison runs **two** pooled queries and describing only the selected one
    was the gap this closed: the second query got a color on two canvases and
    a size nowhere. ``contrast`` is the one facet the two differ in, stated on
    the second card because it is a property of the pair.
    """
    from .cohorts import (
        LOW_N_THRESHOLD,
        SINGLE_SAMPLE_STABILITY,
        STABILITY_BY_K,
        TIER_LOW_N,
        TIER_SINGLETON,
    )

    role_class = f" is-{role}" if role in COHORT_ROLES else ""
    k = geometry.size
    tier = geometry.tier
    if tier == TIER_SINGLETON:
        return html.Div(
            className="cohort-card cohort-card--empty" + role_class,
            children=html.Div(
                "Pooling needs at least two samples. Select a larger cohort, or "
                "use Sample mode for a single one.",
                className="cohort-card-note"),
        )

    stability = geometry.stability
    low = tier == TIER_LOW_N
    best = STABILITY_BY_K[max(STABILITY_BY_K)]

    rows: list[Any] = [
        html.Div(
            className="cohort-stat",
            children=[
                html.Div(
                    className="cohort-stat-head",
                    children=[
                        html.Span("Result stability", className="cohort-stat-label"),
                        html.Span(f"{stability:.2f}", className="cohort-stat-value"),
                    ],
                ),
                html.Div(
                    className="cohort-meter",
                    children=html.Div(
                        className="cohort-meter-fill" + (" is-low" if low else ""),
                        style={"width": f"{max(0.0, min(1.0, stability)) * 100:.1f}%"},
                    ),
                ),
                html.Div(
                    f"Measured share of the top 5 that survives dropping any one "
                    f"of these {k} samples. One sample alone scores "
                    f"{SINGLE_SAMPLE_STABILITY:.2f}.",
                    className="cohort-stat-note"),
            ],
        ),
    ]

    if low:
        # Amber, not red, and the same reason the map's coverage bar is amber:
        # a small cohort is working correctly, not failing.
        rows.insert(0, html.Div(
            className="cohort-flag",
            children=[
                html.Span(f"Only {k} sample{'s' if k != 1 else ''} in this cohort",
                          className="cohort-flag-title"),
                html.Span(
                    f"Pooled results reach {stability:.2f} stability at this size, "
                    f"against {best:.2f} at {LOW_N_THRESHOLD * 3} or more. Read "
                    "the hits as a neighbourhood, not as a ranking.",
                    className="cohort-flag-body"),
            ],
        ))

    head: list[Any] = []
    if role in COHORT_ROLES:
        head = [
            html.Div(
                className="cohort-role",
                children=[
                    html.Span(className="cohort-role-dot"),
                    html.Span(COHORT_ROLES[role], className="cohort-role-name"),
                    (html.Span(f"differs by {contrast}",
                               className="cohort-role-contrast")
                     if contrast else None),
                ],
            ),
            html.Div(cohort.label, className="cohort-card-title"),
        ]

    return html.Div(
        className="cohort-card" + role_class,
        children=[
            *head,
            html.Div(
                className="cohort-card-head",
                children=[
                    html.Span(f"{k}", className="cohort-card-k"),
                    html.Span("samples pooled into one query",
                              className="cohort-card-k-label"),
                ],
            ),
            *rows,
        ],
    )


def build_cohort_details(query: pd.Series, role: str = "") -> list[Any]:
    """Inspector view of a pooled cohort query node.

    The single-sample panel would render this as one sample with a blank name,
    so a cohort gets its own: what defined it, how many went in, what came back
    out about its spread, and which members were excluded.

    ``role`` names which arm of a comparison this is, and only a comparison
    passes one. Without it the panel opened on cohort A and read as *the* pooled
    query rather than as one of two, with nothing saying the other star on the
    canvas leads to its twin. It is the same letter and the same dot the rail's
    cards carry, so the card, the star and this heading name one thing.
    """
    label = _safe_str(query.get("cohort_label")) or "Cohort"
    study = _safe_str(query.get("study_id"))
    members = [m for m in _safe_str(query.get("members")).split("\n") if m]
    excluded = [m for m in _safe_str(query.get("excluded")).split("\n") if m]
    outliers = {m for m in _safe_str(query.get("outliers")).split("\n") if m}
    kicker = ("Pooled OSDR cohort" if role not in COHORT_ROLES
              else f"Pooled OSDR cohort · {COHORT_ROLES[role]}")

    parts: list[Any] = [
        _details_head(kicker, label),
        _detail_section(
            "Definition",
            [
                _detail_row("Study", study, mono=True),
                _detail_row("Grouped by", _safe_str(query.get("grouped_by"))),
                _detail_row("Samples pooled", str(len(members))),
                _detail_row("Result stability", _safe_str(query.get("stability"))),
            ],
        ),
    ]

    if members:
        parts.append(html.Div(
            className="details-section",
            children=[
                html.Div(f"Pooled members ({len(members)})",
                         className="details-section-title"),
                html.Ul(
                    className="cohort-member-list",
                    children=[
                        html.Li(
                            className="cohort-member" + (" is-outlier" if m in outliers else ""),
                            children=[
                                html.Span(m, className="cohort-member-name"),
                                (html.Span("furthest from the rest",
                                           className="cohort-member-tag")
                                 if m in outliers else None),
                            ],
                        )
                        for m in members
                    ],
                ),
            ],
        ))

    if excluded:
        parts.append(html.Div(
            className="details-section",
            children=[
                html.Div(f"Excluded by you ({len(excluded)})",
                         className="details-section-title"),
                html.Ul(className="cohort-member-list cohort-member-list--excluded",
                        children=[html.Li(m, className="cohort-member") for m in excluded]),
            ],
        ))

    parts.append(_detail_text_block(
        "How this query was built",
        "Each member's cached 512-d embedding was L2-normalized and averaged, "
        "then the mean was normalized again - the maximum-likelihood mean "
        "direction for vectors compared by cosine, so every animal gets one "
        "vote regardless of how concentrated its transcriptome is. The pooled "
        "vector was then scored against every ARCHS4 sample by exactly the scan "
        "a single-sample search uses.",
        collapsible=True))

    return [p for p in parts if p is not None]


OSDR_ACCESSION_RE = re.compile(r"^(OSD|GLDS)-\d+$")


def _build_osdr_query_metadata_block(query: pd.Series) -> list[Any]:
    """Appendable OSDR metadata section for the right panel.

    Only an OSDR accession has an OSDR study behind it. An uploaded sample
    carries the synthesized study_id "Uploaded file", which the Identity
    section already shows, so rendering an "OSDR study" section for it repeats
    that row, adds a "Study title —" that can never fill, and sends a lookup
    for a study that does not exist. Say nothing instead.
    """
    study_id = _safe_str(query.get("study_id", ""))
    if not OSDR_ACCESSION_RE.match(study_id):
        return []
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


def _build_query_details(query: pd.Series, compact: bool,
                         role: str = "") -> list[Any]:
    """Details for the OSDR query node. ``compact`` omits the finer biology rows."""
    # A pooled cohort is a query too, and rendering it through the single-sample
    # path would show one blank sample name where a group belongs.
    if _safe_str(query.get("is_cohort")) == "1":
        return build_cohort_details(query, role=role)
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


def build_details_panel(query: pd.Series, selected_payload: dict[str, Any] | None,
                        hits_df: pd.DataFrame,
                        query_b: pd.Series | None = None) -> list[Any]:
    node_kind = _safe_str(selected_payload.get("kind")) if selected_payload else ""
    node_id = _safe_str(selected_payload.get("node_id")) if selected_payload else ""

    if not selected_payload or node_kind == "query":
        # A comparison is on screen exactly when there is a second query, so
        # the letters appear only when there are two arms to tell apart.
        return _build_query_details(query, compact=not selected_payload,
                                    role="a" if query_b is not None else "")

    # The comparison figure draws a second query star, tagged `query2`. Without
    # this it fell through to the GSM lookup, found nothing, and reported "No
    # metadata found" for a node the figure had just drawn.
    if node_kind == "query2":
        if query_b is None:
            return [
                _details_head("Details", "No metadata"),
                html.P("This retrieval carries no second cohort.",
                       className="details-empty"),
            ]
        # `compact` has no default, and False is what the `query` branch above
        # resolves to whenever a node was actually clicked.
        return _build_query_details(query_b, compact=False, role="b")

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
