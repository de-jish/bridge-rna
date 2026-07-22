"""Plotly figures for the retrieval view.

Plotly cannot read CSS variables, so the palette is mirrored from the light
theme tokens in assets/style.css. Keep GRAPH_THEME in sync with :root there.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from .util import _safe_str


GRAPH_THEME = {
    "paper_bg": "#ffffff",
    "plot_bg": "#ffffff",
    "grid": "#e6ecf5",
    "text_primary": "#1a2432",
    "text_secondary": "#5a6b7e",
    "query": "#0bab9f",       # --accent-teal (query stands apart from its hits)
    "gsm": "#2b7fff",         # --accent (GSM hit nodes)
    "gse": "#d9791b",         # --accent-warm (GSE study nodes)
    "edge": "rgba(43, 127, 255, 0.42)",
    "edge_gse": "rgba(217, 121, 27, 0.35)",
    "marker_line": "#ffffff",
    "font_sans": "Inter, 'Segoe UI', -apple-system, sans-serif",
}


def _empty_network_figure(message: str = "Run a search to build the retrieval network.") -> go.Figure:
    """A clean, axis-free placeholder that matches the workspace card."""
    fig = go.Figure()
    fig.update_layout(
        paper_bgcolor=GRAPH_THEME["paper_bg"],
        plot_bgcolor=GRAPH_THEME["plot_bg"],
        margin={"l": 20, "r": 20, "t": 20, "b": 20},
        xaxis={"visible": False, "range": [0, 1]},
        yaxis={"visible": False, "range": [0, 1]},
        height=560,
        annotations=[
            {
                "text": message,
                "x": 0.5,
                "y": 0.5,
                "xref": "paper",
                "yref": "paper",
                "showarrow": False,
                "font": {"family": GRAPH_THEME["font_sans"], "size": 15, "color": GRAPH_THEME["text_secondary"]},
            }
        ],
    )
    return fig


# Cosine similarities among retrieved hits live in a narrow high band - the
# top five for the exemplar query span 0.9970 to 0.9954, a spread of 0.0016 -
# and every edge maps onto this fixed domain rather than onto the min and max
# of the current result set. A min-max rescale made the thinnest hit 1.5 px and
# the thickest 8 px *regardless of the actual scores*, so a 0.0016 spread and a
# 0.4 spread drew identically and the width encoded rank, not similarity. On a
# fixed domain, near-equal scores draw near-equal widths - which is the honest
# picture, and the same reason the map draws every hit ring identically.
EDGE_WIDTH_DOMAIN = (0.90, 1.0)
EDGE_WIDTH_RANGE = (1.5, 8.0)


def _edge_width(scores: pd.Series) -> list[float]:
    lo, hi = EDGE_WIDTH_DOMAIN
    wlo, whi = EDGE_WIDTH_RANGE
    span = hi - lo
    out = []
    for s in scores:
        frac = min(1.0, max(0.0, (float(s) - lo) / span))
        out.append(wlo + (whi - wlo) * frac)
    return out


def build_network_figure(query: pd.Series, hits_df: pd.DataFrame) -> go.Figure:
    gse_values = [g for g in hits_df["gse"].astype(str).tolist() if g]
    gse_unique = sorted(dict.fromkeys(gse_values))

    node_rows = []
    edge_rows = []

    q_id = _safe_str(query["sample_id"])
    q_label = _safe_str(query["sample_name"])
    node_rows.append(
        {
            "node_id": q_id,
            "label": q_label,
            "kind": "query",
            "x": 0.0,
            "y": 0.0,
            "size": 28,
            "color": GRAPH_THEME["query"],
            "symbol": "star",
            "hover": f"OSDR query<br>{q_label}<br>{q_id}",
        }
    )

    y_space = 1.4
    gsm_count = len(hits_df)
    gsm_y_start = (gsm_count - 1) * 0.5 * y_space
    widths = _edge_width(hits_df["score"]) if "score" in hits_df else [3.0] * len(hits_df)

    for i, (_, row) in enumerate(hits_df.iterrows()):
        y = gsm_y_start - i * y_space
        score = float(row["score"])
        gsm = _safe_str(row["gsm"])
        gse = _safe_str(row.get("gse", ""))
        # Join only the fields that have content. source_name and characteristics
        # come from the optional archs4py HDF5 enrichment, so without it they are
        # empty strings, and joining unconditionally left blank lines stranded in
        # the middle of every tooltip.
        hover = "<br>".join(
            part
            for part in (
                gsm,
                _safe_str(row.get("source_name", "")),
                _safe_str(row.get("characteristics", "")),
                f"Score: {score:.3f}",
                gse,
            )
            if part
        )

        node_rows.append(
            {
                "node_id": gsm,
                "label": gsm,
                "kind": "gsm",
                "x": 1.0,
                "y": y,
                "size": 16 + max(0.0, (score - float(hits_df["score"].min())) * 20.0),
                "color": GRAPH_THEME["gsm"],
                "symbol": "circle",
                "hover": hover,
            }
        )

        edge_rows.append(
            {
                "x0": 0.0,
                "y0": 0.0,
                "x1": 1.0,
                "y1": y,
                "width": widths[i],
                "color": GRAPH_THEME["edge"],
            }
        )

        if gse:
            g_idx = gse_unique.index(gse)
            gse_y_start = (len(gse_unique) - 1) * 0.5 * 2.3
            g_y = gse_y_start - g_idx * 2.3
            if not any(n["node_id"] == gse for n in node_rows):
                node_rows.append(
                    {
                        "node_id": gse,
                        "label": gse,
                        "kind": "gse",
                        "x": 2.1,
                        "y": g_y,
                        "size": 19,
                        "color": GRAPH_THEME["gse"],
                        "symbol": "diamond",
                        "hover": f"GEO series {gse}",
                    }
                )

            edge_rows.append(
                {
                    "x0": 1.0,
                    "y0": y,
                    "x1": 2.1,
                    "y1": g_y,
                    "width": max(1.0, widths[i] * 0.7),
                    "color": GRAPH_THEME["edge_gse"],
                }
            )

    fig = go.Figure()
    for e in edge_rows:
        fig.add_trace(
            go.Scatter(
                x=[e["x0"], e["x1"]],
                y=[e["y0"], e["y1"]],
                mode="lines",
                line={"width": e["width"], "color": e["color"]},
                hoverinfo="skip",
                showlegend=False,
            )
        )

    node_df = pd.DataFrame(node_rows)

    # Declutter: with many GSM hits, 30 always-on labels collide, so at high
    # node counts we keep labels only for the query + GSE studies and rely on
    # hover for individual GSM ids.
    gsm_count = int((node_df["kind"] == "gsm").sum())
    if gsm_count > 12:
        node_df["display_label"] = node_df.apply(
            lambda r: "" if r["kind"] == "gsm" else r["label"], axis=1
        )
    else:
        node_df["display_label"] = node_df["label"]

    fig.add_trace(
        go.Scatter(
            x=node_df["x"],
            y=node_df["y"],
            mode="markers+text",
            text=node_df["display_label"],
            textposition="top center",
            textfont={"family": GRAPH_THEME["font_sans"], "size": 11, "color": GRAPH_THEME["text_secondary"]},
            hovertemplate="%{customdata[2]}<extra></extra>",
            customdata=node_df[["kind", "node_id", "hover"]].values,
            marker={
                "size": node_df["size"],
                "color": node_df["color"],
                "symbol": node_df["symbol"],
                "line": {"width": 1.5, "color": GRAPH_THEME["marker_line"]},
            },
            # Let labels on the outermost nodes spill into the margin instead of
            # being cut off at the plot edge. The axis padding below sizes the
            # plot so this is a backstop for narrow viewports, not the main fix.
            cliponaxis=False,
            showlegend=False,
        )
    )

    # Labels are centered on their node, so half of each one overhangs the node
    # it belongs to. The query sits at the far left (x=0.0) and carries the
    # longest text -- OSDR sample names run past 25 characters -- so Plotly's
    # autorange, which pads by only a few percent of the data extent, renders it
    # clipped. Pad each side by the overhang of the widest label anchored there.
    x_chars_per_unit = 88.0  # ~11px glyphs across the 0.0-2.1 node span
    def _label_overhang(kind: str) -> float:
        widest = max((len(str(v)) for v in node_df.loc[node_df["kind"] == kind, "label"]), default=0)
        return widest / (2.0 * x_chars_per_unit)

    fig.update_layout(
        margin={"l": 48, "r": 48, "t": 16, "b": 16},
        paper_bgcolor=GRAPH_THEME["paper_bg"],
        plot_bgcolor=GRAPH_THEME["plot_bg"],
        font={"family": GRAPH_THEME["font_sans"], "color": GRAPH_THEME["text_primary"]},
        xaxis={
            "visible": False,
            "range": [0.0 - _label_overhang("query") - 0.04, 2.1 + _label_overhang("gse") + 0.04],
        },
        yaxis={"visible": False},
        # "event", not "event+select". Clicking a node opens it in the
        # inspector; it does not select anything. With "+select" Plotly applied
        # its selection styling on every click, fading all the *other* nodes to
        # near-invisible - so inspecting one hit made the rest of the retrieval
        # look like it had been dismissed. clickData fires either way.
        clickmode="event",
        # The font colour must be set explicitly. Plotly only auto-contrasts the
        # hover text when it also picks the background; forcing bgcolor to white
        # while leaving the colour unset makes it inherit the trace colour, so
        # tooltips rendered pale blue on white and were effectively unreadable.
        hoverlabel={
            "font": {"family": GRAPH_THEME["font_sans"], "size": 12, "color": GRAPH_THEME["text_primary"]},
            "bgcolor": "#ffffff",
            "bordercolor": GRAPH_THEME["grid"],
        },
        autosize=True,
        height=None,
    )
    return fig


def build_bar_figure(hits_df: pd.DataFrame) -> go.Figure:
    display = hits_df.sort_values("score", ascending=True)
    labels = [f"{g} ({s})" for g, s in zip(display["gsm"], display["gse"].replace("", "no GSE"))]
    fig = go.Figure(
        go.Bar(
            x=display["score"],
            y=labels,
            orientation="h",
            marker={
                "color": display["score"],
                "colorscale": "Blues",
                "line": {"color": "#1f3d7a", "width": 0.8},
            },
            hovertemplate="%{y}<br>Score: %{x:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        title={"text": "Top retrieved analogs by cosine similarity", "font": {"family": GRAPH_THEME["font_sans"], "size": 15, "color": GRAPH_THEME["text_primary"]}},
        margin={"l": 20, "r": 20, "t": 56, "b": 30},
        paper_bgcolor=GRAPH_THEME["paper_bg"],
        plot_bgcolor=GRAPH_THEME["plot_bg"],
        font={"family": GRAPH_THEME["font_sans"], "color": GRAPH_THEME["text_secondary"]},
        xaxis={"title": "Similarity", "gridcolor": GRAPH_THEME["grid"], "zerolinecolor": GRAPH_THEME["grid"]},
        yaxis_title="",
        height=420,
        # Pinned for the same reason as the network graph: never leave hover
        # text colour to Plotly's fallback once a background is specified.
        hoverlabel={
            "font": {"family": GRAPH_THEME["font_sans"], "size": 12, "color": GRAPH_THEME["text_primary"]},
            "bgcolor": "#ffffff",
            "bordercolor": GRAPH_THEME["grid"],
        },
    )
    return fig
