"""Wiring: controls -> figure, zoom -> level-of-detail, legend filter."""

from __future__ import annotations

from dash import Input, Output, State, html, no_update

from . import colorby, layout, render

# The legend search only earns its space once the list is long enough to be
# hard to scan; below that it is chrome.
LEGEND_SEARCH_MIN_ITEMS = 8


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
    )
    def update_figure(method, dims, color_by, layers, budget, viewport):
        vp = tuple(viewport) if viewport else None
        fig, legend_data, badges = render.build_figure(
            method, dims, color_by, layers or [], budget, vp if dims == "2d" else None)
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
        Output("viewport-store", "data"),
        Input("manifold-graph", "relayoutData"),
        Input("method", "value"),
        Input("dims", "value"),
        State("viewport-store", "data"),
        prevent_initial_call=True,
    )
    def update_viewport(relayout, method, dims, current):
        from dash import ctx
        if ctx.triggered_id in ("method", "dims"):
            return None  # reset zoom when the projection changes
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
