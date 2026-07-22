#!/usr/bin/env python3
"""Bridge Manifold - the serving Dash app.

Loads only small precomputed artifacts (coordinate parquets, the point identity
table, the OSDR labels, and the ARCHS4 GEO metadata join) and draws them.
It never runs the model, never runs UMAP, and never opens the
963 MB ARCHS4 embedding memmap, so BRIDGE_RNA_ROOT is needed to build the cache
but not to serve it. Run the `precompute/` scripts first to build the cache.

    python app_manifold.py            # http://127.0.0.1:8051
"""

from __future__ import annotations

import argparse

from dash import Dash

from manifold import callbacks, layout, paths, preflight

def build_app() -> Dash:
    app = Dash(
        __name__,
        assets_folder=str(paths.ASSETS_DIR),
        title="Bridge Manifold",
        update_title=None,
        suppress_callback_exceptions=True,
    )
    app.layout = layout.build_layout()
    callbacks.register(app)
    return app

def main() -> None:
    ap = argparse.ArgumentParser(description="Bridge Manifold exploratory map.")
    ap.add_argument("--host", default="127.0.0.1", help="Bind host (loopback by default).")
    ap.add_argument("--port", type=int, default=8051)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    if args.host not in ("127.0.0.1", "localhost") and not args.debug:
        print(f"[warn] binding to {args.host} exposes the app beyond loopback.", flush=True)

    preflight.require(preflight.APP_REQUIRED, "serving app")

    app = build_app()
    print(f"[bridge-manifold] serving on http://{args.host}:{args.port}", flush=True)
    app.run(host=args.host, port=args.port, debug=args.debug)

if __name__ == "__main__":
    main()
