#!/usr/bin/env python3
"""Bridge RNA - one app, two views.

    python app.py              # http://127.0.0.1:8050

`/` retrieves the closest Earth analogs for one NASA OSDR spaceflight sample.
`/map` draws all 942,563 points of the joint embedding space the retrieval
searched. They are two views of one instrument rather than two apps: a
retrieval's query and its hits are drawn in place on the map, and a point on
the map launches a retrieval, because the two halves address the same sample by
the same key and the same ARCHS4 hit by the same integer.

The retrieval half needs the model checkpoint and the 963 MB embedding memmap;
the map half needs only its precomputed cache. Neither is required for the
other to work, and the shell reports what is missing rather than failing.
"""

from __future__ import annotations

import argparse
import os
import sys

from dash import Dash, Input, Output, dcc, html

from bridge_rna import callbacks as rna_callbacks
from bridge_rna import layout as rna_layout
from bridge_rna.config import ROOT
from bridge_rna.util import _format_count

# --- Routes -----------------------------------------------------------------
# One place defining every view, so the nav, the router, and the title all read
# from the same list and cannot drift apart.
ROUTES = [
    {"path": "/", "key": "retrieve", "label": "Retrieve",
     "title": "Bridge RNA - retrieve Earth analogs for a NASA sample"},
    {"path": "/map", "key": "map", "label": "Map",
     "title": "Bridge RNA - the embedding map"},
]
DEFAULT_ROUTE = ROUTES[0]


def _route_for(pathname: str | None) -> dict:
    path = (pathname or "/").rstrip("/") or "/"
    return next((r for r in ROUTES if r["path"] == path), DEFAULT_ROUTE)


def _manifold_available() -> bool:
    """Whether the map's precomputed cache exists on this machine.

    The map is not degradable: without coordinates there is nothing to draw. The
    nav shows the tab regardless and the view explains what to run, because
    hiding it makes the app look like it never had a map.
    """
    try:
        from manifold import paths as mpaths

        return mpaths.POINTS_META_PARQUET.exists() and mpaths.COORDS_PCA2.exists()
    except Exception:
        return False


def header(active: str) -> html.Header:
    """The shared chrome. Both views hang below it and neither draws its own."""
    return html.Header(
        className="app-header",
        children=[
            dcc.Link(
                href="/", className="app-brand",
                children=[
                    html.Div("BR", className="app-brand-mark"),
                    html.Div(
                        className="app-brand-text",
                        children=[
                            html.H1("Bridge RNA", className="app-title"),
                            html.P("NASA spaceflight transcriptomes, against all of Earth's",
                                   className="app-subtitle"),
                        ],
                    ),
                ],
            ),
            html.Nav(
                className="app-nav",
                children=[
                    dcc.Link(
                        r["label"], href=r["path"],
                        className="app-nav-item" + (" is-active" if r["key"] == active else ""),
                    )
                    for r in ROUTES
                ],
            ),
            html.Div(
                className="app-header-meta",
                children=[
                    html.Div(
                        className="header-stat",
                        title="Earth-based samples in the ARCHS4 embedding index",
                        children=[
                            html.Span(_format_count(rna_layout.ARCHS4_SAMPLE_COUNT),
                                      className="header-stat-value"),
                            html.Span("ARCHS4 samples", className="header-stat-label"),
                        ],
                    ),
                    html.Div(className="header-stat-divider"),
                    html.Div(
                        className="header-stat",
                        title="OSDR samples eligible for retrieval (mouse counts + spaceflight condition)",
                        children=[
                            html.Span(_format_count(rna_layout.ELIGIBLE_OSDR_COUNT),
                                      className="header-stat-value header-stat-value--accent"),
                            html.Span("Eligible OSDR samples", className="header-stat-label"),
                        ],
                    ),
                ],
            ),
        ],
    )


def map_unavailable_view() -> html.Div:
    """What `/map` shows on a clone whose precompute has not been run."""
    return html.Div(
        className="app-root",
        children=[
            html.Div(
                className="setup-banner",
                children=[
                    html.Span("The map has not been built on this machine",
                              className="setup-banner-title"),
                    html.Span(
                        "The map is drawn from precomputed coordinates rather than "
                        "computed on demand, so it needs a one-time build. Retrieval "
                        "works without it.",
                        className="setup-banner-body"),
                    html.Ul(className="setup-banner-list", children=[
                        html.Li("python precompute/embed_osdr.py"),
                        html.Li("python precompute/build_projections.py"),
                        html.Li("python precompute/fetch_archs4_meta.py"),
                    ]),
                ],
            ),
        ],
    )


def build_app() -> Dash:
    app = Dash(
        __name__,
        assets_folder=str(ROOT / "assets"),
        title="Bridge RNA",
        update_title=None,
        # The router mounts one view at a time, so callbacks for the other one
        # necessarily reference ids that are not in the tree yet.
        suppress_callback_exceptions=True,
    )
    app.index_string = INDEX_STRING

    def view_for(route: dict):
        if route["key"] == "map":
            if not _manifold_available():
                return map_unavailable_view()
            from manifold import layout as manifold_layout

            return manifold_layout.build_view()
        return rna_layout.build_view()

    def serve_layout():
        """Build the page for the path this request actually asked for.

        Dash calls a layout *function* once per request, so the requested view
        is in the first HTML response instead of arriving a callback round trip
        later. Beyond the round trip it removes, this is what stops a direct
        link to /map from painting the retrieval view first and then replacing
        it.

        `flask.request` is absent when the layout is built outside a request -
        Dash does exactly that once at startup to validate the callback graph -
        so the default route is the fallback rather than an error.
        """
        try:
            from flask import request

            route = _route_for(request.path)
        except Exception:
            route = DEFAULT_ROUTE
        return html.Div(
            className="app-shell",
            children=[
                dcc.Location(id="url", refresh=False),
                html.Div(header(route["key"]), id="app-header-slot"),
                html.Div(view_for(route), id="page-content"),
            ],
        )

    app.layout = serve_layout

    @app.callback(
        Output("page-content", "children"),
        Output("app-header-slot", "children"),
        Input("url", "pathname"),
        # The initial view is already in the served layout; without this the
        # router would rebuild it on load and pay the cost this design avoids.
        prevent_initial_call=True,
    )
    def render_page(pathname: str | None):
        route = _route_for(pathname)
        return view_for(route), header(route["key"])

    rna_callbacks.register(app)
    if _manifold_available():
        from manifold import callbacks as manifold_callbacks

        manifold_callbacks.register(app)
    return app


INDEX_STRING = """<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
        {%favicon%}
        {%css%}
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>"""


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Bridge RNA (retrieval + the embedding map).")
    parser.add_argument(
        "--host", default=os.environ.get("DASH_HOST", "127.0.0.1"),
        help="Interface to bind. Defaults to 127.0.0.1 (this machine only). "
             "Use 0.0.0.0 to expose the app on your network, but only on a "
             "network you trust and never together with --debug.")
    parser.add_argument(
        "--port", type=int, default=int(os.environ.get("DASH_PORT", "8050")),
        help="Port to serve on (default: 8050).")
    parser.add_argument(
        "--debug", action="store_true", default=_env_flag("DASH_DEBUG", False),
        help="Enable hot reload and the Werkzeug debugger. Development only: "
             "the debugger exposes an interactive Python console to anyone who "
             "can reach the port.")
    args = parser.parse_args()

    # The debugger's console executes arbitrary Python for any client that can
    # reach it, so binding it off-loopback hands out remote code execution.
    # Refuse rather than warn: this is not a combination anyone wants by accident.
    if args.debug and args.host not in ("127.0.0.1", "localhost", "::1"):
        parser.error(
            f"refusing to run the debugger on {args.host}: the Werkzeug console "
            "executes arbitrary code for anyone who can reach the port. Use "
            "--host 127.0.0.1, or drop --debug to serve on this interface.")

    if args.host == "0.0.0.0":
        print(f"[WARN] Serving on all interfaces (port {args.port}). Anyone who can "
              "reach this machine can use the app and the data it exposes.",
              file=sys.stderr, flush=True)

    app = build_app()
    print(f"[bridge-rna] serving on http://{args.host}:{args.port}", flush=True)
    app.run(debug=args.debug, dev_tools_ui=False, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
