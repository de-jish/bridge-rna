"""The retrieval view: search controls, the network canvas, and the inspector.

This used to be a module-level `app.layout = ...`. It is a function now because
the view is mounted by the router in `app.py` rather than being the whole page,
and because the shared header moved up into the shell - the retrieval view no
longer draws its own.

Counts in the header are computed once at import, not per request; they read a
manifest and a TSV that do not change while the app is running.
"""

from __future__ import annotations

from typing import Any

from dash import dcc, html

from .config import DEFAULT_ENTREZ_EMAIL, OSDR_METADATA_PATH
from .figures import _empty_network_figure
from .osdr import _eligible_osdr_count, load_osdr_samples
from .panels import build_gene_list_banner, build_setup_banner, build_status_banner
from .retrieval import _archs4_sample_count
from .util import _format_count, _safe_str

samples_df = load_osdr_samples(OSDR_METADATA_PATH)
study_options = sorted(samples_df["study_id"].dropna().astype(str).unique().tolist())
default_study = study_options[0] if study_options else ""
default_samples = samples_df[samples_df["study_id"] == default_study]
default_sample_id = default_samples.iloc[0]["sample_id"] if not default_samples.empty else ""

ARCHS4_SAMPLE_COUNT = _archs4_sample_count()
ELIGIBLE_OSDR_COUNT = _eligible_osdr_count(samples_df)


def _initial_study() -> str:
    """The study to open on, honouring a `/?q=<sample_id>` link on cold load.

    Reads the request directly so a pasted or bookmarked deep link lands on the
    right study without waiting for a callback. `flask.request` is absent when
    Dash builds the layout outside a request (once, at startup, to validate the
    callback graph), so the default study is the fallback.
    """
    try:
        from urllib.parse import parse_qs

        from flask import request

        sample_id = _safe_str(parse_qs(request.query_string.decode()).get("q", [""])[0])
        if sample_id:
            match = samples_df.loc[samples_df["sample_id"].astype(str) == sample_id]
            if not match.empty:
                return _safe_str(match.iloc[0]["study_id"])
    except Exception:
        pass
    return default_study


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
                html.Span("edge width = cosine similarity"),
            ]),
        ],
    )


def build_view() -> html.Div:
    """The retrieval view, everything below the shared header.

    The view is built per request (the shell's `serve_layout` is a function),
    so a cold-loaded `/?q=<sample_id>` link can pick the right study *here*,
    before any callback runs. That is what makes a pasted or bookmarked link
    work on the initial load: the study dropdown's live callback keeps
    `prevent_initial_call`, because firing it on load fought the dropdown's own
    initialization and left both dropdowns empty, so the cold-load case is
    handled by choosing the initial value rather than by a callback.
    """
    initial_study = _initial_study()
    return html.Div(
        className="app-root",
        children=[
            build_setup_banner(),
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
                                                value=initial_study,
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
                                    html.Div("Or upload a sample", className="control-group-title"),
                                    dcc.Upload(
                                        id="upload-counts",
                                        className="upload-dropzone",
                                        multiple=False,
                                        children=html.Div([
                                            html.Div("Drop a counts file or click to browse",
                                                     className="upload-dropzone-title"),
                                            html.Div(
                                                "CSV/TSV, mouse Ensembl gene IDs in column 1, "
                                                "samples in columns.",
                                                className="upload-dropzone-hint"),
                                        ]),
                                    ),
                                    html.Div(id="upload-preview", className="sample-preview"),
                                    html.Div(
                                        id="upload-column-control",
                                        className="control",
                                        style={"display": "none"},
                                        children=[
                                            html.Label("Sample column", className="control-label"),
                                            dcc.Dropdown(id="upload-sample-column", clearable=False),
                                        ],
                                    ),
                                    html.Button(
                                        "Embed & search uploaded sample",
                                        id="upload-search-button",
                                        className="btn-secondary",
                                        n_clicks=0,
                                        disabled=True,
                                    ),
                                    html.Div(id="upload-running-indicator", className="running-indicator"),
                                    dcc.Store(id="upload-store"),
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
                                            # Off by default now. Every hit
                                            # already arrives with its GEO
                                            # series, title, source name and
                                            # tissue from the local cache; this
                                            # adds study abstracts, overall
                                            # design, and PubMed records, at a
                                            # network round trip per accession.
                                            # Leaving it on made a 0.8 s search
                                            # take 11 s, for text most searches
                                            # never open. The inspector fetches
                                            # it for the one hit you click, and
                                            # the AI panel fetches it for all of
                                            # them before it writes.
                                            dcc.Checklist(
                                                id="biopython-toggle",
                                                options=[{
                                                    "label": " Fetch study abstracts and publications during search",
                                                    "value": "on",
                                                }],
                                                value=[],
                                                className="dash-checklist",
                                            ),
                                            html.Div(
                                                "Adds roughly two seconds per hit. "
                                                "Off, a search is local and instant, and "
                                                "abstracts are fetched for a hit when you "
                                                "open it or when the AI hypothesis needs them.",
                                                className="control-hint",
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
                                    # The second act, offered only once there is
                                    # something to see there. Hidden until a
                                    # search succeeds; a link to "see your
                                    # results on the map" before any results
                                    # exist would be an empty promise.
                                    dcc.Link(
                                        id="see-on-map",
                                        href="/map",
                                        className="btn-ghost",
                                        style={"display": "none"},
                                        children="See these hits on the map →",
                                    ),
                                ],
                            ),
                            # hits-store lives on the shell (app.py): it
                            # has to outlive this view so the map can draw
                            # the retrieval. selected-node-store is genuinely
                            # local - which node of *this* network is open.
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
