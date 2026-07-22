"""Wiring: controls -> figure, zoom -> level-of-detail, legend filter."""

from __future__ import annotations

from urllib.parse import quote

from dash import Input, Output, State, html, no_update

from . import colorby, data, layout, render

# The legend search only earns its space once the list is long enough to be
# hard to scan; below that it is chrome.
LEGEND_SEARCH_MIN_ITEMS = 8


def _retrieval_overlay(hits_payload: dict | None) -> dict | None:
    """Turn a stored retrieval into point indices on this map.

    This is the whole cross-view translation, and it is this short because
    there is nothing to translate. An ARCHS4 hit carries `archs4_index`, its
    row in the embedding memmap, and ARCHS4 occupies rows `0..n_archs4-1` of
    the map's global point order - so the row *is* the point. The OSDR query is
    found by its `sample_key`, the same string the retrieval calls `sample_id`.

    Returns None when there is nothing to draw, or when the hits predate the
    `archs4_index` column: a retrieval whose hits cannot be located is not
    drawn at all rather than drawn in the wrong place.
    """
    if not hits_payload:
        return None
    hits = hits_payload.get("hits") or []
    n_archs4, n_osdr, _ = data.counts()

    points, labels, scores = [], [], []
    for hit in hits:
        idx = hit.get("archs4_index")
        if idx is None:
            continue
        idx = int(idx)
        if 0 <= idx < n_archs4:
            points.append(idx)
            labels.append(str(hit.get("gsm") or ""))
            scores.append(float(hit.get("score") or 0.0))

    query_point, query_label = None, ""
    sample_id = str(hits_payload.get("sample_id") or "")
    if sample_id:
        meta = data.osdr_metadata()
        if "sample_key" in meta.columns:
            match = meta.index[meta["sample_key"].astype(str) == sample_id]
            if len(match):
                query_point = n_archs4 + int(match[0])
                query_label = sample_id

    if not points and query_point is None:
        return None
    return {"hit_points": points, "hit_labels": labels, "hit_scores": scores,
            "query_point": query_point, "query_label": query_label}


def _viewport_from_relayout(relayout: dict | None):
    """Interpret a Plotly relayout event as a viewport change.

    Three outcomes, and the distinction matters: a concrete window (re-stratify
    the sample inside it), ``None`` (reset to the full corpus), or the sentinel
    ``"unchanged"`` for events that are not zooms at all - a hover, a legend
    click, a drag-mode switch - which must leave the current sample alone rather
    than triggering a full resample.
    """
    if not relayout:
        return None
    keys = ["xaxis.range[0]", "xaxis.range[1]", "yaxis.range[0]", "yaxis.range[1]"]
    if all(k in relayout for k in keys):
        x0, x1, y0, y1 = (relayout[k] for k in keys)
        return (min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1))
    if relayout.get("xaxis.autorange") or relayout.get("autosize"):
        return None
    return "unchanged"


def _frame_for(hits_payload, method: str, dims: str):
    """A viewport containing the query and every hit, with room to breathe.

    At full-corpus zoom a retrieval's points are a few pixels apart - the hits
    really are that close, which is the finding - so the rings overlap into one
    illegible mark. Framing is what makes them separately readable, and it is
    offered as an action rather than done automatically: arriving already
    zoomed in would hide the thing worth seeing first, which is *where in the
    whole corpus* the query landed.

    Returns None (the whole map) if there is nothing to frame, or in 3-D, where
    the viewport store does not drive the camera.
    """
    if dims != "2d":
        return None
    overlay = _retrieval_overlay(hits_payload)
    if overlay is None:
        return None
    points = list(overlay["hit_points"])
    if overlay["query_point"] is not None:
        points.append(overlay["query_point"])
    coords = data.coords(method, "2d")
    if not points or coords.shape[0] == 0:
        return None

    import numpy as np

    xy = coords[np.asarray(points), :2]
    x0, y0 = xy.min(axis=0)
    x1, y1 = xy.max(axis=0)
    # A tight cluster would otherwise frame to a zero-width window. The pad is
    # a share of the span with an absolute floor, so a single point still gets
    # a sensible window.
    pad_x = max((x1 - x0) * 0.6, 0.35)
    pad_y = max((y1 - y0) * 0.6, 0.35)
    return [float(x0 - pad_x), float(x1 + pad_x),
            float(y0 - pad_y), float(y1 + pad_y)]


def coverage_children(color_by: str):
    """The coverage readout under the color-by control.

    States the exact number of points the selected field colours. This is the
    control that answers "why is so much of my map not coloured?" before the
    user has to ask it, and it is why an OSDR-only field is no longer
    indistinguishable from a broken one.
    """
    spec = colorby.get(color_by)
    covered, total = colorby.coverage(color_by)
    pct = covered / total * 100 if total else 0.0
    whole_map = covered >= total

    bar = html.Div(className="bm-coverage-bar", children=html.Div(
        className="bm-coverage-fill" + ("" if whole_map else " partial"),
        style={"width": f"{max(pct, 1.5):.1f}%"}))

    if whole_map:
        text = f"Colours all {covered:,} points."
    elif covered:
        text = (f"Colours {covered:,} of {total:,} points ({pct:.1f}%). "
                "ARCHS4 is drawn as faint context.")
    else:
        text = "No data for this field on this machine."

    children = [bar, html.Div(text, className="bm-coverage-text")]
    missing = spec.missing_hint()
    if missing:
        children.append(html.Div(missing, className="bm-coverage-fix"))
    return children


def register(app):
    @app.callback(
        Output("manifold-graph", "figure"),
        Output("plot-badges", "children"),
        Output("legend", "style"),
        Output("legend-title", "children"),
        Output("legend-search", "style"),
        Output("legend-store", "data"),
        Output("coverage", "children"),
        Output("color-by-hint", "children"),
        Input("method", "value"),
        Input("dims", "value"),
        Input("color-by", "value"),
        Input("layers", "value"),
        Input("budget", "value"),
        Input("viewport-store", "data"),
        # The retrieval lives on the shell, so the map can draw a search that
        # was run before the user walked over here. An Input rather than a
        # State: running a new retrieval should redraw the map that is showing
        # the old one.
        Input("hits-store", "data"),
        Input("show-retrieval", "value"),
    )
    def update_figure(method, dims, color_by, layers, budget, viewport,
                      hits_payload, show_retrieval):
        vp = tuple(viewport) if viewport else None
        retrieval = (_retrieval_overlay(hits_payload)
                     if (show_retrieval and "on" in show_retrieval) else None)
        fig, legend_data, badges = render.build_figure(
            method, dims, color_by, layers or [], budget,
            vp if dims == "2d" else None, retrieval=retrieval)
        legend_data["title"] = layout.color_by_label(color_by)
        # Persist zoom in 2D by pinning axis ranges to the viewport.
        if vp and dims == "2d":
            fig.update_layout(xaxis=dict(range=[vp[0], vp[1]]),
                              yaxis=dict(range=[vp[2], vp[3]]))
        badge_children = [_badge(b) for b in badges]
        items = legend_data.get("items", [])
        legend_style = {} if items else {"display": "none"}
        search_style = ({} if len(items) >= LEGEND_SEARCH_MIN_ITEMS
                        else {"display": "none"})
        title = f"Color · {legend_data['title']}"
        return (fig, badge_children, legend_style, title, search_style,
                legend_data, coverage_children(color_by),
                colorby.get(color_by).hint)

    @app.callback(
        Output("picked-group", "style"),
        Output("picked-label", "children"),
        Output("picked-link", "href"),
        Input("manifold-graph", "clickData"),
    )
    def pick_osdr_point(click_data):
        """Offer a retrieval for a clicked OSDR point.

        Only the OSDR overlay carries customdata - the ARCHS4 cloud has none,
        deliberately, because 940,455 rows of it cost about 600 KB of dead
        payload per figure. So a click that returns no customdata is a click on
        the cloud, and nothing is offered rather than something being guessed.

        The link goes to `/?q=<sample_id>` rather than mutating a store,
        because it is a navigation: it should be a real link that middle-clicks
        into a new tab and shows its destination on hover.
        """
        points = (click_data or {}).get("points") or []
        custom = points[0].get("customdata") if points else None
        if not custom:
            return {"display": "none"}, "", "/"
        sample_key = str(custom[0])
        # An OSDR key is "<accession>|<sample name>". Anything else is not one.
        if "|" not in sample_key:
            return {"display": "none"}, "", "/"
        study, name = sample_key.split("|", 1)
        return (
            {},
            [html.B(name), html.Span(f"  ·  {study}", className="bm-picked-study")],
            f"/?q={quote(sample_key)}",
        )

    @app.callback(
        Output("retrieval-group", "style"),
        Output("retrieval-summary", "children"),
        Output("frame-retrieval", "style"),
        Input("hits-store", "data"),
        Input("dims", "value"),
    )
    def show_retrieval_group(hits_payload, dims):
        """Reveal the retrieval control only when there is a retrieval.

        An always-visible control that does nothing until you have searched
        somewhere else is worse than no control: it reads as broken.

        The same argument hides the frame button in 3-D. Framing works by
        pinning the 2-D axis ranges, which the 3-D camera ignores, so the
        button would have been a click with no visible effect - the thing this
        map removed the lasso for.
        """
        overlay = _retrieval_overlay(hits_payload)
        if overlay is None:
            return {"display": "none"}, "", {"display": "none"}
        n = len(overlay["hit_points"])
        query = overlay["query_label"] or "an OSDR sample"
        frame_style = {} if dims == "2d" else {"display": "none"}
        return {}, [
            html.B(query.split("|")[-1]),
            f" and its {n} nearest ARCHS4 neighbour{'s' if n != 1 else ''}, "
            "drawn where they sit in the space.",
        ], frame_style

    @app.callback(
        Output("viewport-store", "data"),
        Input("manifold-graph", "relayoutData"),
        Input("method", "value"),
        Input("dims", "value"),
        # Framing the retrieval is a viewport change, so it is an Input to the
        # callback that already owns the viewport rather than a second writer.
        # Two callbacks writing one Output is a race Dash only sometimes
        # rejects, and `test_every_output_has_exactly_one_writer` pins it.
        Input("frame-retrieval", "n_clicks"),
        State("hits-store", "data"),
        State("method", "value"),
        State("dims", "value"),
        prevent_initial_call=True,
    )
    def update_viewport(relayout, method, dims, _frame_clicks, hits_payload,
                        method_state, dims_state):
        from dash import ctx
        if ctx.triggered_id in ("method", "dims"):
            return None  # reset zoom when the projection changes
        if ctx.triggered_id == "frame-retrieval":
            return _frame_for(hits_payload, method_state, dims_state)
        vp = _viewport_from_relayout(relayout)
        if vp == "unchanged":
            return no_update
        if vp is None:
            return None
        return list(vp)

    @app.callback(
        Output("legend-list", "children"),
        Input("legend-store", "data"),
        Input("legend-search", "value"),
    )
    def render_legend(legend_data, query):
        """Render the legend rows, filtered by the search box.

        One callback owns this output: the figure callback publishes categories
        to legend-store and this renders them, so a search term survives a
        re-render and there is no second writer to race with.
        """
        return filtered_legend_rows(legend_data, query)


def _badge(html_text: str):
    # badges may carry <b> tags; render via a small parser.
    return html.Div(className="bm-badge", children=_html_with_bold(html_text))


def _html_with_bold(text: str):
    parts = []
    rest = text
    while "<b>" in rest and "</b>" in rest:
        pre, rest = rest.split("<b>", 1)
        bold, rest = rest.split("</b>", 1)
        if pre:
            parts.append(pre)
        parts.append(html.B(bold))
    if rest:
        parts.append(rest)
    return parts


def _legend_rows(items):
    return [
        html.Div(className="bm-legend-item", children=[
            html.Span(className="bm-legend-swatch", style={"background": it["color"]}),
            html.Span(it["label"], title=it["label"],
                      style={"overflow": "hidden", "textOverflow": "ellipsis"}),
            html.Span(f"{it['count']:,}", className="bm-legend-count"),
        ])
        for it in items
    ]


def filtered_legend_rows(legend_data, query):
    """Legend rows matching ``query`` (case-insensitive substring on the label)."""
    items = (legend_data or {}).get("items", [])
    if query:
        needle = query.strip().lower()
        items = [i for i in items if needle in str(i["label"]).lower()]
        if not items:
            return html.Div("no matching categories", className="bm-legend-empty")
    return _legend_rows(items)
