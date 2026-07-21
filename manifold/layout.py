"""The Dash layout: header, left control rail, center plot, right readout panel.

The chrome matches Bridge RNA (light scientific instrument); the plot canvas is
dark navy. Controls are declarative here; behavior lives in callbacks.py.
"""

from __future__ import annotations

from dash import dcc, html

from . import data

COLOR_BY_OPTIONS = [
    {"label": "Species (both corpora)", "value": "species"},
    {"label": "Flight vs Ground (OSDR)", "value": "flight_status"},
    {"label": "Spaceflight arm (OSDR)", "value": "spaceflight"},
    {"label": "Tissue (OSDR)", "value": "tissue"},
    {"label": "Strain (OSDR)", "value": "strain"},
    {"label": "Sex (OSDR)", "value": "sex"},
    {"label": "Genotype (OSDR)", "value": "genotype"},
    {"label": "Study (OSDR)", "value": "study"},
    {"label": "Habitat (OSDR)", "value": "habitat"},
    {"label": "Duration (OSDR)", "value": "duration"},
    {"label": "Diet (OSDR)", "value": "diet"},
]


def color_by_label(value: str) -> str:
    """The human-facing name for a color-by, for the legend heading.

    The legend showed the raw field key otherwise, so a user picking
    "Flight vs Ground (OSDR)" got a legend titled "flight_status".
    """
    for opt in color_by_options():
        if opt["value"] == value:
            return opt["label"]
    return value.replace("_", " ").title()


def color_by_options() -> list[dict]:
    """Color-by menu, including the ARCHS4 tissue join only once it is built.

    The join is an optional tens-of-GB download, so the option is added rather
    than shown-and-disabled: an entry that can never be selected on this machine
    is menu clutter, and the badge on the plot already names the script to run.
    """
    options = list(COLOR_BY_OPTIONS)
    if data.archs4_tissue_available():
        options.insert(1, {"label": "Tissue (ARCHS4)", "value": "archs4_tissue"})
    return options


def _segmented(id_: str, options: list[dict], value: str):
    """A radio group styled as a segmented pill control.

    Only the container carries a class. Dash renders each option as a label with
    its own structural classes and marks the chosen one `.selected`, and
    `labelClassName` lands on an inner text span rather than the label itself -
    so the stylesheet targets Dash's classes scoped under `.bm-seg` instead.
    """
    return dcc.RadioItems(id=id_, options=options, value=value, className="bm-seg")


def control_rail() -> html.Div:
    n_archs4, n_osdr, total = data.counts()
    umap_ok = data.method_available("umap")
    method_options = [
        {"label": "UMAP", "value": "umap", "disabled": not umap_ok},
        {"label": "PCA", "value": "pca"},
    ]
    default_method = "umap" if umap_ok else "pca"

    return html.Div(
        className="bm-rail",
        children=[
            html.Div(className="bm-group", children=[
                html.Div("Projection", className="bm-group-label"),
                _segmented("method", method_options, default_method),
                html.Div(
                    "UMAP preserves local neighborhoods, not global distances. "
                    "Cluster sizes and gaps are not quantitative.",
                    className="bm-hint",
                ),
            ]),
            html.Div(className="bm-group", children=[
                html.Div("Dimensions", className="bm-group-label"),
                _segmented("dims", [
                    {"label": "2D", "value": "2d"},
                    {"label": "3D", "value": "3d"},
                ], "2d"),
            ]),
            html.Div(className="bm-group", children=[
                html.Div("Color by", className="bm-group-label"),
                dcc.Dropdown(
                    id="color-by",
                    options=color_by_options(),
                    value="flight_status",
                    clearable=False,
                    className="bm-dropdown",
                ),
                html.Div(
                    "OSDR fields color the spaceflight overlay; the ARCHS4 cloud "
                    "stays neutral unless you color by species.",
                    className="bm-hint",
                ),
            ]),
            html.Div(className="bm-group", children=[
                html.Div("Layers", className="bm-group-label"),
                dcc.Checklist(
                    id="layers",
                    options=[
                        {"label": f"ARCHS4 cloud ({n_archs4:,})", "value": "archs4"},
                        {"label": f"OSDR overlay ({n_osdr:,})", "value": "osdr"},
                        {"label": f"Density underlay (all {total:,})", "value": "density"},
                    ],
                    value=["archs4", "osdr", "density"],
                    className="bm-checklist",
                ),
            ]),
            html.Div(className="bm-group", children=[
                html.Div("ARCHS4 point budget", className="bm-group-label"),
                _segmented("budget", [
                    {"label": "60k", "value": "60000"},
                    {"label": "100k", "value": "100000"},
                    {"label": "150k", "value": "150000"},
                ], "100000"),
                html.Div(
                    "Live glyphs are a stratified sample over the density raster. "
                    "Zoom re-samples the visible window.",
                    className="bm-hint",
                ),
            ]),
        ],
    )


def legend_panel() -> html.Div:
    """The floating color key.

    Every part is declared statically and filled by callbacks, rather than the
    whole panel being rebuilt as callback output. Dash validates its callback
    graph against the initial layout, so components that only ever exist as
    callback output cannot be checked - a typo in one of their ids fails
    silently at runtime instead of loudly at startup.
    """
    return html.Div(
        id="legend", className="bm-legend", style={"display": "none"},
        children=[
            html.Div(id="legend-title", className="bm-legend-title"),
            dcc.Input(id="legend-search", className="bm-legend-search",
                      placeholder="filter categories…", type="text",
                      debounce=False, style={"display": "none"}),
            html.Div(id="legend-list", className="bm-legend-list"),
        ],
    )


def readout_panel() -> html.Div:
    return html.Div(
        className="bm-readout",
        id="readout",
        children=[
            html.Div("Selection readout", className="bm-readout-title"),
            html.Div(
                "Lasso a region to test whether the selected samples are "
                "meaningfully related. Every statistic is computed in the "
                "original 512-d cosine space, not from the 2D projection.",
                className="bm-readout-sub",
            ),
            html.Div(id="readout-body", children=_empty_readout()),
        ],
    )


def _empty_readout():
    return html.Div(className="bm-empty", children=[
        html.Div("◎", className="bm-empty-icon"),
        html.Div("No selection yet."),
        html.Div("Draw a lasso on the map to begin.", style={"marginTop": "4px"}),
    ])


def header() -> html.Div:
    n_archs4, n_osdr, total = data.counts()
    return html.Div(className="bm-header", children=[
        html.Div(className="bm-logo", children=[
            html.Span(className="bm-dot"),
            html.Span("Bridge Manifold"),
        ]),
        html.Div("the exploratory map for Bridge RNA", className="bm-sub"),
        html.Div(className="bm-spacer"),
        html.Div(className="bm-count", children=[
            html.Span("corpus "),
            html.B(f"{total:,}"),
            html.Span(f"  ·  ARCHS4 {n_archs4:,}  ·  OSDR {n_osdr:,}"),
        ]),
    ])


def build_layout() -> html.Div:
    return html.Div(className="bm-app", children=[
        header(),
        html.Div(className="bm-body", children=[
            control_rail(),
            html.Div(className="bm-plot-wrap", children=[
                html.Div(id="plot-badges", className="bm-plot-badges"),
                legend_panel(),
                # dcc.Loading wraps its children in two nested divs of its own.
                # Both need an explicit full height or the chain from
                # .bm-plot-wrap collapses and the graph falls back to Plotly's
                # default 450 px, leaving half the canvas empty.
                #
                # delay_show keeps the map on screen during the short rebuilds
                # that follow a color-by change or a zoom step. Blanking a
                # 100k-point scatter behind a spinner for a few hundred
                # milliseconds on every interaction makes the whole map feel
                # like it is reloading; the spinner should only appear when the
                # wait is long enough to need explaining.
                dcc.Loading(
                    type="circle", color="#22c7bd",
                    delay_show=600,
                    overlay_style={"visibility": "visible", "opacity": 0.45},
                    parent_className="bm-plot-loading",
                    children=dcc.Graph(
                        id="manifold-graph",
                        className="dash-graph",
                        clear_on_unhover=True,
                        config={
                            "displaylogo": False,
                            "scrollZoom": True,
                            "displayModeBar": True,
                            "modeBarButtonsToRemove": ["select2d", "autoScale2d"],
                        },
                        style={"height": "100%"},
                    ),
                ),
            ]),
            readout_panel(),
        ]),
        dcc.Store(id="viewport-store"),
        dcc.Store(id="legend-store"),
    ])
