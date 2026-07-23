"""The Dash layout: header, left control rail, and the plot.

The chrome matches Bridge RNA (light scientific instrument); the plot canvas is
dark navy. Controls are declarative here; behaviour lives in callbacks.py.

The shell is two columns. It used to be three, with a right-hand panel reporting
statistics for a lasso selection; that feature is gone and the plot took the
space back.
"""

from __future__ import annotations

from dash import dcc, html

from . import colorby, data, render


def color_by_label(value: str) -> str:
    """The human-facing name for a color-by, for the legend heading."""
    return colorby.get(value).label


def budget_options(dims: str) -> list[dict]:
    """The ARCHS4 point-budget tiers, which depend on the dimensionality.

    In 2-D the glyph cloud can carry the whole corpus, so the tiers climb to
    "All". In 3-D the cloud is capped at ``render.SCATTER3D_ARCHS4_CAP`` so
    rotation stays smooth, so the tiers stop there rather than offering a "500k"
    or "All" pill the 3-D view would silently redraw as 40,000 - a control that
    lies about what it does is worse than one that offers less.
    """
    n_archs4, _, _ = data.counts()
    if dims == "3d":
        cap = render.SCATTER3D_ARCHS4_CAP
        step = cap // 4
        vals = [step, step * 2, step * 3, cap]
        return [{"label": f"{v // 1000}k", "value": str(v)} for v in vals]
    return [
        {"label": "100k", "value": "100000"},
        {"label": "250k", "value": "250000"},
        {"label": "500k", "value": "500000"},
        {"label": "All", "value": str(n_archs4)},
    ]


def default_budget(dims: str) -> str:
    """The tier a fresh view of ``dims`` starts on: everything in 2-D, the cap
    in 3-D."""
    n_archs4, _, _ = data.counts()
    return str(render.SCATTER3D_ARCHS4_CAP) if dims == "3d" else str(n_archs4)


def resolve_budget(dims: str, current: str | None) -> str:
    """The budget value to use for ``dims``, preserving ``current`` if it is
    still one of that dimensionality's tiers and falling back to the default
    otherwise. This is what a dimensionality switch runs so a 3-D "All" cannot
    survive into a view that cannot honour it."""
    valid = {o["value"] for o in budget_options(dims)}
    return current if current in valid else default_budget(dims)


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
            # In 2-D the glyph sample can carry the whole corpus (the density
            # raster it used to sit above is gone, so the sample is the only
            # thing drawing those points). In 3-D the tiers stop at the rotation
            # cap; budget_options() decides, and callbacks.sync_budget_to_dims
            # swaps the tiers when the dimensionality changes.
            html.Div(className="bm-group", children=[
                html.Div("ARCHS4 point budget", className="bm-group-label"),
                _segmented("budget", budget_options("2d"), default_budget("2d")),
                html.Div(
                    "One glyph per sample; zoom re-samples the visible window.",
                    className="bm-hint",
                ),
            ]),
            # Clicking an OSDR point offers a retrieval for it. Hidden until
            # something is clicked; the map is read rather than driven, so this
            # is an offer that appears in response to interest, not a control
            # sitting on the rail waiting to be understood.
            html.Div(id="picked-group", className="bm-group",
                     style={"display": "none"}, children=[
                html.Div("Selected sample", className="bm-group-label"),
                html.Div(id="picked-label", className="bm-picked"),
                dcc.Link(id="picked-link", href="/", className="bm-button",
                         children="Retrieve its Earth analogs →"),
            ]),
            # Shown only when there is a retrieval to show. Declared here
            # rather than created by a callback so Dash can validate the
            # callback graph against it at startup: a component that exists
            # only as callback output fails silently at runtime if its id is
            # mistyped, which is the same reason the legend is built this way.
            html.Div(id="retrieval-group", className="bm-group",
                     style={"display": "none"}, children=[
                html.Div("Your retrieval", className="bm-group-label"),
                dcc.Checklist(
                    id="show-retrieval",
                    options=[{"label": " Show it on the map", "value": "on"}],
                    value=["on"],
                    className="bm-checklist",
                ),
                html.Div(id="retrieval-summary", className="bm-hint"),
                html.Button("Frame the retrieval", id="frame-retrieval",
                            n_clicks=0, className="bm-button"),
                # This is the caveat that keeps the feature honest, and it is
                # placed with the control rather than in a tooltip because the
                # temptation to read rank off the picture is immediate.
                html.Div(
                    "Hits are ranked by cosine distance in 512 dimensions. "
                    "This map is a projection into two or three of them and "
                    "does not preserve those distances, so how far a hit sits "
                    "from the query here is not its rank, and no line is drawn "
                    "between them. Hover a hit for both orderings.",
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
