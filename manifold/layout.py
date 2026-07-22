"""The Dash layout: header, left control rail, and the plot.

The chrome matches Bridge RNA (light scientific instrument); the plot canvas is
dark navy. Controls are declarative here; behaviour lives in callbacks.py.

The shell is two columns. It used to be three, with a right-hand panel reporting
statistics for a lasso selection; that feature is gone and the plot took the
space back.
"""

from __future__ import annotations

from dash import dcc, html

from . import colorby, data


def color_by_label(value: str) -> str:
    """The human-facing name for a color-by, for the legend heading."""
    return colorby.get(value).label


def _segmented(id_: str, options: list[dict], value: str):
    """A radio group styled as a segmented pill control.

    Only the container carries a class. Dash renders each option as a label with
    its own structural classes and marks the chosen one `.selected`, and
    `labelClassName` lands on an inner text span rather than the label itself -
    so the stylesheet targets Dash's classes scoped under `.bm-seg` instead.
    """
    return dcc.RadioItems(id=id_, options=options, value=value, className="bm-seg")


def control_rail() -> html.Div:
    n_archs4, n_osdr, _ = data.counts()
    umap_ok = data.method_available("umap")
    method_options = [
        {"label": "UMAP", "value": "umap", "disabled": not umap_ok},
        {"label": "PCA", "value": "pca"},
    ]

    return html.Div(
        className="bm-rail",
        children=[
            html.Div(className="bm-group", children=[
                html.Div("Projection", className="bm-group-label"),
                _segmented("method", method_options, "umap" if umap_ok else "pca"),
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
            # The color-by group carries its own coverage readout, so the answer
            # to "how much of this map is this field actually colouring?" sits
            # next to the control that decides it rather than being inferred
            # from how grey the plot looks.
            html.Div(className="bm-group", children=[
                html.Div("Color by", className="bm-group-label"),
                dcc.Dropdown(
                    id="color-by",
                    options=colorby.menu_options(),
                    value=colorby.default_key(),
                    clearable=False,
                    className="bm-dropdown",
                ),
                html.Div(id="coverage", className="bm-coverage"),
                html.Div(id="color-by-hint", className="bm-hint"),
            ]),
            html.Div(className="bm-group", children=[
                html.Div("Layers", className="bm-group-label"),
                dcc.Checklist(
                    id="layers",
                    options=[
                        {"label": f"ARCHS4 cloud ({n_archs4:,})", "value": "archs4"},
                        {"label": f"OSDR overlay ({n_osdr:,})", "value": "osdr"},
                    ],
                    value=["archs4", "osdr"],
                    className="bm-checklist",
                ),
            ]),
            # The budget used to top out at 150,000 because a density raster was
            # carrying the other 790,455 points underneath. With the raster gone
            # the glyph sample is the only thing representing them, so the
            # ceiling is now the whole corpus. It is affordable: measured on the
            # real corpus, every tier costs the same ~0.6 s to build, and even
            # all 942,563 points serialize in 0.15 s to an 11.3 MB payload.
            html.Div(className="bm-group", children=[
                html.Div("ARCHS4 point budget", className="bm-group-label"),
                _segmented("budget", [
                    {"label": "100k", "value": "100000"},
                    {"label": "250k", "value": "250000"},
                    {"label": "500k", "value": "500000"},
                    {"label": "All", "value": str(n_archs4)},
                ], str(n_archs4)),
                html.Div(
                    "Every glyph is one sample. Below the full corpus the cloud "
                    "is a species-stratified sample, and zoom re-samples the "
                    "visible window. 3-D caps the cloud at 40,000 whatever is "
                    "chosen here, because rotation cost grows with glyph count; "
                    "the badge on the plot always states what was drawn.",
                    className="bm-hint",
                ),
            ]),
            # The measured cross-corpus batch effect used to be disclosed only
            # inside the lasso readout. It is a property of the map itself, not
            # of any selection, so it belongs on the map's controls.
            html.Div(className="bm-caution", children=[
                html.B("Reading across corpora."),
                " OSDR and ARCHS4 were embedded on different hardware and in "
                "different precisions. OSDR samples sharing neither study nor "
                "tissue still neighbour each other 54x above chance, so distance "
                "between the two corpora is partly technical. Compare within a "
                "corpus, not across.",
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


def build_view() -> html.Div:
    """The map view, everything below the shared header.

    The map used to be its own app and drew its own header, with a corpus count
    strip reading "corpus 942,563 · ARCHS4 940,455 · OSDR 2,108". The shell
    draws the header now and that function is deleted rather than kept unused:
    both of its counts already appear on the control rail, in the Layers group,
    beside the toggle that decides whether each corpus is drawn at all.
    """
    return html.Div(className="bm-app", children=[
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
                            # No selection feature exists, so neither selection
                            # tool is offered. Leaving lasso2d on the modebar
                            # would let a user draw a marquee that does nothing.
                            "modeBarButtonsToRemove": [
                                "select2d", "lasso2d", "autoScale2d"],
                        },
                        style={"height": "100%"},
                    ),
                ),
            ]),
        ]),
        dcc.Store(id="viewport-store"),
        dcc.Store(id="legend-store"),
    ])
