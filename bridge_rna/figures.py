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
    # Accessions are set in mono for the same reason the rail's measured values
    # are: GSM6431262 and GSM6431263 differ in one glyph, and a proportional
    # font is where that difference goes to hide.
    "font_mono": "JetBrains Mono, 'SF Mono', Consolas, monospace",
    # The comparison view's own three roles, named. They used to borrow `gsm`,
    # `gse` and `query`, which mean a hit node, a study node and the query in
    # the single-query network - so the same key meant two things depending on
    # which figure was being built.
    #
    # Cohort A is teal and "retrieved by both" is blue, which is a swap from
    # what shipped first, and it fixes a real inconsistency rather than a
    # preference: teal is the query star in the single-query network and the
    # query mark on the map, so giving it to "shared" meant running a
    # comparison silently recolored the star the previous search drew teal.
    # Both views now agree that teal is cohort A and warm is cohort B, and each
    # renders "both" the way its canvas supports - a third color on white, a
    # doubled mark on the map's navy.
    "cohort_a": "#0bab9f",
    "cohort_b": "#d9791b",
    "cohort_shared": "#2b7fff",
    "edge_cohort_a": "rgba(11, 171, 159, 0.42)",
    "edge_cohort_b": "rgba(217, 121, 27, 0.35)",
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

#: Above this many hit nodes the comparison network stops writing accessions on
#: the figure. Its two arms share one vertical rhythm, so the node count is
#: `2 * k` in the worst case and the labels would collide long before k = 30.
COMPARISON_MAX_LABELS = 20


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
                # Constant, and that is the fix. It used to be
                # `16 + (score - min(score)) * 20`, which is a second encoding
                # of the quantity the edge width already carries, on a
                # different scale, keyed nowhere - the legend names the edge
                # and says nothing about node size. It was also the min-max
                # rescale `_edge_width` exists to avoid, and it inherited that
                # rescale's dishonesty in reverse: over the 0.0016 spread these
                # scores actually have, it varied the diameter by three
                # hundredths of a pixel, so it looked like a constant while
                # claiming to be a measurement. One quantity, one channel, and
                # that channel is in the key.
                "size": 16,
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
        # The font color must be set explicitly. Plotly only auto-contrasts the
        # hover text when it also picks the background; forcing bgcolor to white
        # while leaving the color unset makes it inherit the trace color, so
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


def build_comparison_figure(query_a: pd.Series, hits_a: pd.DataFrame,
                            query_b: pd.Series, hits_b: pd.DataFrame) -> go.Figure:
    """Two pooled cohorts, two independent queries, one picture of their overlap.

    The single-query network puts one query on the left and fans its hits to the
    right. This puts one cohort at each corner of the left edge and splits the
    hit column into three bands: what only the first retrieved, what both did,
    and what only the second did. The height of the middle band *is* the answer
    to the question the comparison asks - do these two arms land in the same
    part of Earth's transcriptome space.

    The GSE column is dropped here. With two queries it would triple the edges
    for a grouping that is not what this view is asking about, and the study
    behind any hit is one click away in the inspector.

    Nothing about this figure is a difference vector. Each cohort was scored
    against ARCHS4 on its own, and what is drawn is set overlap between two real
    result lists.
    """
    a_ids = list(dict.fromkeys(hits_a["gsm"].astype(str).tolist()))
    b_ids = list(dict.fromkeys(hits_b["gsm"].astype(str).tolist()))
    shared = [g for g in a_ids if g in set(b_ids)]
    a_only = [g for g in a_ids if g not in set(shared)]
    b_only = [g for g in b_ids if g not in set(shared)]

    score_a = {str(g): float(s) for g, s in zip(hits_a["gsm"], hits_a["score"])}
    score_b = {str(g): float(s) for g, s in zip(hits_b["gsm"], hits_b["score"])}
    meta = {}
    for df in (hits_a, hits_b):
        for _, r in df.iterrows():
            meta.setdefault(_safe_str(r["gsm"]), r)

    # One vertical rhythm across all three bands, so band height reads as count.
    step = 1.0
    total = len(a_only) + len(shared) + len(b_only)
    gap = step * 1.4  # a visible break between bands
    height = (total - 1) * step + 2 * gap if total else 0.0
    top = height / 2.0

    positions: dict[str, float] = {}
    y = top
    for g in a_only:
        positions[g] = y
        y -= step
    y -= gap
    for g in shared:
        positions[g] = y
        y -= step
    y -= gap
    for g in b_only:
        positions[g] = y
        y -= step

    # Each query sits level with the centre of the hits it alone retrieved, so
    # the edges fan rather than cross.
    def _centre(group: list[str], fallback: float) -> float:
        ys = [positions[g] for g in group]
        return sum(ys) / len(ys) if ys else fallback

    qa_y = _centre(a_only + shared, top)
    qb_y = _centre(b_only + shared, -top)

    fig = go.Figure()

    def _edges(ids: list[str], qy: float, color: str, scores: dict[str, float]):
        for g in ids:
            fig.add_trace(go.Scatter(
                x=[0.0, 1.0], y=[qy, positions[g]], mode="lines",
                line={"width": _edge_width([scores.get(g, 0.95)])[0] * 0.6,
                      "color": color},
                hoverinfo="skip", showlegend=False))

    _edges(a_only + shared, qa_y, GRAPH_THEME["edge_cohort_a"], score_a)
    _edges(b_only + shared, qb_y, GRAPH_THEME["edge_cohort_b"], score_b)

    def _hover(g: str) -> str:
        r = meta.get(g)
        bits = [g]
        if r is not None:
            bits += [p for p in (_safe_str(r.get("source_name", "")),
                                 _safe_str(r.get("tissue", ""))) if p]
        if g in score_a:
            bits.append(f"{_safe_str(query_a.get('cohort_label'))}: "
                        f"{score_a[g]:.4f}")
        if g in score_b:
            bits.append(f"{_safe_str(query_b.get('cohort_label'))}: "
                        f"{score_b[g]:.4f}")
        return "<br>".join(bits)

    bands = [
        (a_only, _safe_str(query_a.get("cohort_label")) or "First cohort only",
         GRAPH_THEME["cohort_a"], 15),
        (shared, "Retrieved by both", GRAPH_THEME["cohort_shared"], 19),
        (b_only, _safe_str(query_b.get("cohort_label")) or "Second cohort only",
         GRAPH_THEME["cohort_b"], 15),
    ]
    # The single-query network names every hit on the face of the figure. This
    # one named none of them, so the accessions existed only in a tooltip and a
    # comparison could not be read on paper or in a screenshot at all - and it
    # is the figure most likely to end up in both. They come back, up to the
    # point where they would stop being readable: the bands share one vertical
    # rhythm at `step`, so 60 nodes at k=30 would overlap where 10 at the
    # default k=5 sit clear. Same rule, and the same reason, as the map's
    # `RETRIEVAL_MAX_NUMERALS`.
    label_hits = total <= COMPARISON_MAX_LABELS
    for ids, name, color, size in bands:
        if not ids:
            continue
        fig.add_trace(go.Scatter(
            x=[1.0] * len(ids), y=[positions[g] for g in ids],
            mode="markers+text" if label_hits else "markers", name=name,
            text=list(ids) if label_hits else None,
            textposition="middle right",
            textfont={"family": GRAPH_THEME["font_mono"], "size": 10,
                      "color": GRAPH_THEME["text_secondary"]},
            marker={"size": size, "color": color, "symbol": "circle",
                    "line": {"width": 1.5, "color": GRAPH_THEME["marker_line"]}},
            customdata=[["gsm", g, _hover(g)] for g in ids],
            hovertemplate="%{customdata[2]}<extra></extra>",
            cliponaxis=False, showlegend=True))

    for query, qy, color, kind in (
        (query_a, qa_y, GRAPH_THEME["cohort_a"], "query"),
        (query_b, qb_y, GRAPH_THEME["cohort_b"], "query2"),
    ):
        label = _safe_str(query.get("cohort_label")) or _safe_str(query.get("sample_name"))
        fig.add_trace(go.Scatter(
            x=[0.0], y=[qy], mode="markers+text", text=[label],
            textposition="middle left",
            textfont={"family": GRAPH_THEME["font_sans"], "size": 11,
                      "color": GRAPH_THEME["text_secondary"]},
            marker={"size": 28, "color": color, "symbol": "star",
                    "line": {"width": 1.5, "color": GRAPH_THEME["marker_line"]}},
            customdata=[[kind, _safe_str(query.get("sample_id")),
                         f"{_safe_str(query.get('sample_name'))}<br>"
                         f"{len(_safe_str(query.get('members')).splitlines())} "
                         "samples pooled"]],
            hovertemplate="%{customdata[2]}<extra></extra>",
            cliponaxis=False, showlegend=False))

    widest = max((len(_safe_str(q.get("cohort_label")))
                  for q in (query_a, query_b)), default=0)
    pad = widest / 60.0 + 0.08

    fig.update_layout(
        margin={"l": 48, "r": 48, "t": 16, "b": 16},
        paper_bgcolor=GRAPH_THEME["paper_bg"],
        plot_bgcolor=GRAPH_THEME["plot_bg"],
        font={"family": GRAPH_THEME["font_sans"], "color": GRAPH_THEME["text_primary"]},
        xaxis={"visible": False, "range": [-pad, 1.35]},
        yaxis={"visible": False},
        clickmode="event",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.0,
                "xanchor": "left", "x": 0.0,
                "font": {"size": 11, "color": GRAPH_THEME["text_secondary"]},
                "bgcolor": "rgba(0,0,0,0)"},
        hoverlabel={
            "font": {"family": GRAPH_THEME["font_sans"], "size": 12,
                     "color": GRAPH_THEME["text_primary"]},
            "bgcolor": "#ffffff",
            "bordercolor": GRAPH_THEME["grid"],
        },
        autosize=True,
        height=None,
    )
    return fig
