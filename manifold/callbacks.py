"""Wiring: controls -> figure, zoom -> level-of-detail, lasso -> 512-d readout."""

from __future__ import annotations

import numpy as np
from dash import Input, Output, State, html, no_update

from . import coherence, layout, render


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


def selection_indices(selected: dict | None) -> np.ndarray:
    """Global corpus indices for a Plotly lasso selection.

    Points arrive grouped by trace and every drawn glyph carries its global
    index in ``customdata``. Anything without customdata is skipped rather than
    guessed at, and duplicates are left for the readout to collapse.
    """
    if not selected or not selected.get("points"):
        return np.empty(0, dtype=np.int64)
    out = []
    for p in selected["points"]:
        cd = p.get("customdata")
        if cd is None:
            continue
        out.append(int(cd[0]) if isinstance(cd, (list, tuple)) else int(cd))
    return np.array(out, dtype=np.int64)


def register(app):
    @app.callback(
        Output("manifold-graph", "figure"),
        Output("plot-badges", "children"),
        Output("legend", "style"),
        Output("legend-title", "children"),
        Output("legend-search", "style"),
        Output("legend-store", "data"),
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
        return fig, badge_children, legend_style, title, search_style, legend_data

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
        Output("readout-body", "children"),
        Input("manifold-graph", "selectedData"),
        Input("method", "value"),
        Input("dims", "value"),
        Input("color-by", "value"),
        Input("budget", "value"),
        Input("layers", "value"),
        prevent_initial_call=True,
    )
    def update_readout(selected, method, dims, color_by, budget, layers):
        from dash import ctx

        return readout_for(selected, triggered_id=ctx.triggered_id)

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


# The legend search only earns its space once the list is long enough to be
# hard to scan; below that it is chrome.
LEGEND_SEARCH_MIN_ITEMS = 8


def readout_for(selected, triggered_id="manifold-graph"):
    """The readout panel contents for a selection.

    Any control change rebuilds the figure and clears Plotly's selection, but
    ``selectedData`` does not fire on that path. So a change triggered by
    anything other than the graph itself clears the panel: leaving the previous
    verdict up would attribute statistics to a lasso that is no longer on
    screen, possibly over points the new view does not even draw.
    """
    if triggered_id != "manifold-graph":
        return _empty()
    idx = selection_indices(selected)
    if len(idx) == 0:
        return _empty()
    return _render_readout(coherence.analyze_selection(idx))


def filtered_legend_rows(legend_data, query):
    """Legend rows matching ``query`` (case-insensitive substring on the label)."""
    items = (legend_data or {}).get("items", [])
    if query:
        needle = query.strip().lower()
        items = [i for i in items if needle in str(i["label"]).lower()]
        if not items:
            return html.Div("no matching categories", className="bm-legend-empty")
    return _legend_rows(items)


def _empty():
    return html.Div(className="bm-empty", children=[
        html.Div("◎", className="bm-empty-icon"),
        html.Div("No points selected."),
        html.Div("Draw a lasso on the map to test coherence.", style={"marginTop": "4px"}),
    ])


def _render_readout(r: dict):
    if r["status"] == "too_small":
        return html.Div(className="bm-empty", children=[
            html.Div("◔", className="bm-empty-icon"),
            html.Div(f"Only {r['n']} point(s) selected."),
            html.Div(f"Select at least {r['min']} to compute statistics.",
                     style={"marginTop": "4px"}),
        ])

    coh = r["cohesion"]
    knn = r["knn"]
    children = [
        html.Div(r["verdict"], className=f"bm-verdict {r['verdict_class']}"),
        html.Div(className="bm-stat-grid", children=[
            _stat("Selected", f"{r['n']:,}"),
            _stat("Cohesion z", f"{coh['z']:.1f}"),
            _stat("Empirical p", _p_text(coh["emp_p"])),
            _stat("kNN-purity", f"{knn['fold']:.1f}×" if knn else "—"),
        ]),
    ]

    if r["enrichment"]:
        children.append(html.Div("Enriched features", className="bm-section-h"))
        for e in r["enrichment"]:
            children.append(html.Div(className="bm-enrich-row", children=[
                html.Div(className="bm-enrich-cat", children=[
                    html.Span(f"{e['category']}  "),
                    html.Span(e["field"], className="field"),
                ]),
                html.Span(f"{e['fold']:.1f}×", className="bm-enrich-fold"),
                html.Span(_q_text(e["q"]), className="bm-enrich-q"),
            ]))
    else:
        children.append(html.Div(
            "No categorical feature enriched at q < 0.05.", className="bm-note"))

    if r.get("batch"):
        b = r["batch"]
        cls = "bm-note warn" if b["flag"] else "bm-note"
        msg = (f"Top study {b['top_study']} is {b['fraction']*100:.0f}% of the OSDR "
               f"selection — coherence may be batch-driven." if b["flag"]
               else f"Low batch confound: top study is {b['fraction']*100:.0f}% of the "
                    f"OSDR selection.")
        children.append(html.Div(msg, className=cls))

    if r.get("cross_dataset"):
        children.append(html.Div(
            f"Selection mixes {r['n_archs4']:,} ARCHS4 and {r['n_osdr']:,} OSDR points. "
            "Cross-corpus proximity is confounded by the fp32-vs-bf16 precision batch "
            "effect; treat inter-dataset closeness cautiously.",
            className="bm-note warn"))

    return html.Div(children)


def _stat(k, v):
    return html.Div(className="bm-stat", children=[
        html.Div(k, className="k"), html.Div(v, className="v")])


def _p_text(p):
    if p <= 1e-3:
        return f"{p:.0e}".replace("e-0", "e-")
    return f"{p:.3f}"


def _q_text(q):
    if q < 1e-10:
        return "q<1e-10"
    if q < 1e-3:
        return f"q={q:.0e}"
    return f"q={q:.2f}"
