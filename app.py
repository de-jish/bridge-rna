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

from dash import Dash, Input, Output, State, dcc, html, no_update

from bridge_rna import callbacks as rna_callbacks
from bridge_rna import layout as rna_layout
from bridge_rna.config import ROOT
from bridge_rna.util import _format_count

# --- Routes -----------------------------------------------------------------
# One place defining every view, so the nav, the router, and the title all read
# from the same list and cannot drift apart.
ROUTES = [
    {"path": "/", "key": "retrieve", "label": "Retrieve",
     "title": "Bridge RNA - find related ARCHS4 samples"},
    {"path": "/map", "key": "map", "label": "Map",
     "title": "Bridge RNA - the embedding map"},
]
DEFAULT_ROUTE = ROUTES[0]


def _route_for(pathname: str | None) -> dict:
    path = (pathname or "/").rstrip("/") or "/"
    return next((r for r in ROUTES if r["path"] == path), DEFAULT_ROUTE)


def navigation_for(pathname: str | None, painted: str | None) -> dict | None:
    """The route to swap to, or None when the view on screen already serves it.

    Split out of the router callback so the decision can be tested without Dash
    plumbing, and because it is the whole of what that callback decides.
    """
    route = _route_for(pathname)
    return None if route["key"] == painted else route


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
                            html.P("Compare mouse OSDR RNA-seq with ARCHS4",
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
                        title="NASA OSDR samples the app can retrieve; they are also the OSDR points drawn on the map",
                        children=[
                            html.Span(_format_count(rna_layout.ELIGIBLE_OSDR_COUNT),
                                      className="header-stat-value header-stat-value--accent"),
                            html.Span("Retrievable OSDR samples", className="header-stat-label"),
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
                # Cross-view state lives on the shell, not in a view.
                #
                # `hits-store` began inside the retrieval sidebar, which was
                # fine while that was the whole app. Under the router a view is
                # destroyed when you leave it, so a retrieval held there would
                # not survive the walk over to the map - and the map showing
                # you where your hits landed is the entire reason the two
                # halves are one app. session storage additionally keeps a
                # result across a reload, which matters more now that a
                # retrieval is a half-second rather than a coffee break.
                dcc.Store(id="hits-store", storage_type="session"),
                # Which view the server actually painted into `page-content`
                # above. The router compares against this so it can decline to
                # repaint a view that is already on screen - see `render_page`.
                dcc.Store(id="route-store", data=route["key"]),
            ],
        )

    app.layout = serve_layout

    @app.callback(
        Output("page-content", "children"),
        Output("app-header-slot", "children"),
        Output("route-store", "data"),
        Input("url", "pathname"),
        State("route-store", "data"),
        # The initial view is already in the served layout; without this the
        # router would rebuild it on load and pay the cost this design avoids.
        prevent_initial_call=True,
    )
    def render_page(pathname: str | None, painted: str | None):
        """Swap the view, and *only* when the view actually has to change.

        `prevent_initial_call` is not enough on its own, and the gap it leaves
        was a live bug rather than a wasted round trip. `dcc.Location` publishes
        `pathname` once it mounts, which Dash sees as a change, so this callback
        ran on every page load and rebuilt the view the server had already
        painted. Rebuilding the retrieval view is not free - it reads the OSDR
        catalog and renders the sample preview - so its response landed a few
        hundred milliseconds after load, on top of whatever the user had done in
        the meantime.

        What that looked like: arrive on the retrieval view, click Cohort, watch
        the cohort panel open and then close itself, leaving Sample mode and no
        Search button. Reproduced 6 times out of 6, and the click was innocent -
        its own callback ran and returned the right thing, then this one
        overwrote the whole subtree it had just updated.

        So the route the server painted is recorded in `route-store`, and a
        pathname that resolves to that same route is not a navigation.
        """
        route = navigation_for(pathname, painted)
        if route is None:
            return no_update, no_update, no_update
        return view_for(route), header(route["key"]), route["key"]

    # Cap the request body so an oversized counts upload fails at the server with
    # a 413 rather than being silently truncated into a malformed matrix.
    from bridge_rna.config import MAX_UPLOAD_BYTES

    app.server.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES

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
