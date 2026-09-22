"""Plotly figures for the retrieval view.

Plotly cannot read CSS variables, so the palette is mirrored from the light
theme tokens in assets/00-tokens.css. Keep GRAPH_THEME in sync with :root there.
"""

from __future__ import annotations

import textwrap

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .util import _safe_str


GRAPH_THEME = {
    "paper_bg": "#ffffff",
    "plot_bg": "#ffffff",
    "grid": "#d1d1d1",
    "text_primary": "#17171b",
    "text_secondary": "#444447",
    "query": "#d83933",       # --network-query: requested research-app adaptation
    "gsm": "#0b3d91",         # --network-sample
    "gse": "#58585b",         # --network-study
    "edge": "#858589",        # --network-edge, >3:1 against the white canvas
    "edge_gse": "#858589",
    "marker_line": "#ffffff",
    "font_sans": "Public Sans Web, 'Segoe UI', sans-serif",
    # Accessions are set in mono for the same reason the rail's measured values
    # are: GSM6431262 and GSM6431263 differ in one glyph, and a proportional
    # font is where that difference goes to hide.
    "font_mono": "DM Mono, Consolas, monospace",
    # The comparison view's own three roles, named. They used to borrow `gsm`,
    # `gse` and `query`, which mean a hit node, a study node and the query in
    # the single-query network - so the same key meant two things depending on
    # which figure was being built.
    #
    # These hues identify cohort membership, rather than node type. Preserve
    # their separate A/B/shared meaning across comparison, cards and Map.
    "cohort_a": "#0bab9f",
    "cohort_b": "#d9791b",
    "cohort_shared": "#2b7fff",
    "edge_cohort_a": "rgba(11, 171, 159, 0.62)",
    "edge_cohort_b": "rgba(217, 121, 27, 0.60)",
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


# Edges express connections; cosine scores remain in hover and the inspector.
EDGE_WIDTH = 3.0

#: Above this many hit nodes the comparison network stops writing accessions on
#: the figure. Its two arms share one vertical rhythm, so the node count is
#: `2 * k` in the worst case and the labels would collide long before k = 30.
COMPARISON_MAX_LABELS = 20


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
            "size": 30,
            # White-filled circle with a red outline hides edges below its
            # center, unlike Plotly's transparent circle-open marker.
            "color": GRAPH_THEME["paper_bg"],
            "symbol": "circle",
            "hover": f"Query<br>{q_label}<br>{q_id}",
        }
    )

    y_space = 1.4
    gsm_count = len(hits_df)
    gsm_y_start = (gsm_count - 1) * 0.5 * y_space

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
                # Node size is constant; scores are numerical metadata.
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
                "width": EDGE_WIDTH,
                "color": GRAPH_THEME["edge"],
                "nodes": [q_id, gsm],
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
                    "width": EDGE_WIDTH,
                    "color": GRAPH_THEME["edge_gse"],
                    "nodes": [gsm, gse],
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
                meta={"network_edge": e["nodes"]},
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
            # Line breaks keep long sample identifiers inside laptop-sized plots.
            # Customdata, hover identifiers, node positions and axes are unchanged.
            text=["<br>".join(textwrap.wrap(str(label), width=20,
                                          break_on_hyphens=False))
                  if kind == "query" else label
                  for label, kind in zip(node_df["display_label"], node_df["kind"])],
            textposition="top center",
            textfont={"family": GRAPH_THEME["font_sans"], "size": 11, "color": GRAPH_THEME["text_secondary"]},
            hovertemplate="%{customdata[2]}<extra></extra>",
            customdata=node_df[["kind", "node_id", "hover"]].values,
            marker={
                "size": node_df["size"],
                "opacity": 1.0,  # Match the solid legend swatches, not Plotly's array default.
                "color": node_df["color"],
                "symbol": node_df["symbol"],
                "line": {"width": [2.5 if k == "query" else 1.5 for k in node_df["kind"]],
                         "color": [GRAPH_THEME["query"] if k == "query" else GRAPH_THEME["marker_line"]
                                   for k in node_df["kind"]]},
            },
            # Presentation-only metadata for local hover and
            # inspector emphasis. Node customdata and scientific data stay intact.
            meta={"network_nodes": True},
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

    def _edges(ids: list[str], qy: float, color: str):
        for g in ids:
            fig.add_trace(go.Scatter(
                x=[0.0, 1.0], y=[qy, positions[g]], mode="lines",
                line={"width": EDGE_WIDTH,
                      "color": color},
                hoverinfo="skip", showlegend=False))

    _edges(a_only + shared, qa_y, GRAPH_THEME["edge_cohort_a"])
    _edges(b_only + shared, qb_y, GRAPH_THEME["edge_cohort_b"])

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


# ---------------------------------------------------------------------------
# Ranked similarity profile
#
# Two rows sharing one x axis: the decay curve, and the rank-to-rank change
# beneath it. They are one figure rather than two so the strip reads as a
# gutter of the curve, and so a hover resolves against the same rank in both.
#
# The y axis is **fitted to the displayed slice, never fixed**. Span over the
# top 250 runs 0.00027 at the 5th percentile to 0.01503 at the 95th across 120
# real queries, a 55x spread, so one window would draw most samples as a flat
# line against the top of the panel. A fitted axis magnifies, so the span is
# published in the panel beside it - that number is what stops a magnified
# nothing from reading as a separation.
# ---------------------------------------------------------------------------

#: Markers stop being individual points and fuse into a bar below this pitch.
PROFILE_MARKER_MIN_PITCH_PX = 7.0
#: A plausible plot width, used only to decide whether markers can be drawn at
#: all. The figure is responsive, so this is a floor rather than a measurement.
PROFILE_ASSUMED_PLOT_PX = 620.0

PROFILE_HEIGHT = 300


def _profile_hover(frame: pd.DataFrame) -> list[str]:
    """One tooltip per rank. Values lead, labels follow."""
    out = []
    for rank, row in enumerate(frame.itertuples(index=False), start=1):
        tissue = _safe_str(getattr(row, "tissue", "")) or "Unknown"
        species = _safe_str(getattr(row, "species", "")) or "Unknown"
        gse = _safe_str(getattr(row, "gse", ""))
        out.append(
            f"<b>{getattr(row, 'score'):.4f}</b>  cosine<br>"
            f"Rank {rank:,}<br>"
            f"{_safe_str(getattr(row, 'gsm', ''))}"
            + (f" · {gse}" if gse else "")
            + f"<br>{tissue} · {species}"
        )
    return out



def _profile_tick_step(span: float) -> float:
    """A 1 / 2 / 5 step giving roughly four or five gridlines over `span`."""
    if span <= 0:
        return 0.001
    import math
    raw = span / 5.0
    mag = 10 ** math.floor(math.log10(raw))
    for m in (1.0, 2.0, 5.0, 10.0):
        if m * mag >= raw:
            return m * mag
    return 10.0 * mag


def _profile_tick_format(step: float) -> str:
    """Enough decimals to distinguish two adjacent ticks, and no more."""
    import math
    if step <= 0:
        return ".4f"
    return f".{max(2, min(6, int(math.ceil(-math.log10(step)))))}f"


def build_similarity_profile_figure(
    frame: pd.DataFrame,
    *,
    shape=None,
    height: int = PROFILE_HEIGHT,
) -> go.Figure:
    """The ranked similarity profile for one query, over `frame`'s ranks.

    `frame` is the already-sliced, already-sorted profile prefix. `shape` is
    the matching `bridge_rna.profile.Shape`, used only for the step annotation
    on the rare corpus where a step clears the floors.
    """
    from plotly.subplots import make_subplots

    scores = frame["score"].to_numpy(dtype=float)
    n = int(scores.size)
    ranks = np.arange(1, n + 1)

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.76, 0.24], vertical_spacing=0.10,
    )

    # --- the curve ---------------------------------------------------------
    # Markers only where the pitch can carry them. At Top 1,000 in this panel
    # they would fuse into a solid bar and stop being points at all.
    pitch = PROFILE_ASSUMED_PLOT_PX / max(n - 1, 1)
    fig.add_trace(
        go.Scatter(
            x=ranks, y=scores,
            mode="lines+markers" if pitch >= PROFILE_MARKER_MIN_PITCH_PX else "lines",
            line={"color": GRAPH_THEME["gsm"], "width": 2, "shape": "linear"},
            marker={"size": 5, "color": GRAPH_THEME["gsm"],
                    "line": {"width": 1.2, "color": GRAPH_THEME["marker_line"]}},
            hovertext=_profile_hover(frame),
            hovertemplate="%{hovertext}<extra></extra>",
            customdata=frame.get("archs4_index", pd.Series([None] * n)).tolist(),
            name="cosine",
            showlegend=False,
        ),
        row=1, col=1,
    )

    # --- the change strip --------------------------------------------------
    # Scaled to the largest *ordinary* change, meaning the largest with any
    # detected step excluded. Scaling to the true maximum would put one large
    # step against a typical tiny one and flatten every other bar onto the
    # baseline, losing exactly the fine structure the fitted axis compresses.
    deltas = np.diff(scores) * -1.0
    deltas = np.clip(deltas, 0.0, None)
    mid = ranks[:-1] + 0.5
    step_rank = shape.step.rank if (shape is not None and shape.step and shape.step.found) else None
    ordinary = deltas.copy()
    if step_rank is not None:
        near = np.abs(np.arange(1, n) - step_rank) <= 1
        ordinary = deltas[~near] if (~near).any() else deltas
    cap = float(ordinary.max()) if ordinary.size and ordinary.max() > 0 else (
        float(deltas.max()) if deltas.size and deltas.max() > 0 else 1e-9)
    fig.add_trace(
        go.Bar(
            x=mid, y=np.minimum(deltas, cap),
            marker={"color": GRAPH_THEME["gsm"], "line": {"width": 0}},
            opacity=0.38,
            hovertemplate="Change after rank %{x:.0f}<extra></extra>",
            showlegend=False,
        ),
        row=2, col=1,
    )

    # --- the step annotation, drawn only when there is one -----------------
    if step_rank is not None:
        fig.add_vline(
            x=step_rank + 0.5, line={"color": GRAPH_THEME["text_secondary"],
                                     "width": 1.5, "dash": "dash"},
        )
        fig.add_annotation(
            x=step_rank + 0.5, y=float(scores.min()),
            xref="x", yref="y",
            text=f"Largest local drop<br>Rank {step_rank:,}",
            showarrow=False, xanchor="left", xshift=8, yanchor="bottom",
            align="left", bgcolor=GRAPH_THEME["paper_bg"],
            bordercolor=GRAPH_THEME["grid"], borderwidth=1, borderpad=5,
            font={"family": GRAPH_THEME["font_sans"], "size": 11,
                  "color": GRAPH_THEME["text_primary"]},
        )

    lo, hi = float(scores.min()), float(scores.max())
    pad = max((hi - lo) * 0.08, 1e-6)
    # Plotly's automatic tick step lands on 0.01 for a typical 0.024 span, which
    # is two gridlines on the whole plot. Span runs 0.00027 to 0.01503 across the
    # corpus, so the step has to be derived from the range rather than left to a
    # default tuned for numbers three orders of magnitude larger.
    dtick = _profile_tick_step(hi - lo + 2 * pad)

    fig.update_layout(
        height=height,
        paper_bgcolor=GRAPH_THEME["paper_bg"],
        plot_bgcolor=GRAPH_THEME["plot_bg"],
        margin={"l": 62, "r": 14, "t": 8, "b": 36},
        bargap=0.15,
        hovermode="closest",
        hoverlabel={
            "bgcolor": GRAPH_THEME["paper_bg"],
            "bordercolor": GRAPH_THEME["grid"],
            "font": {"family": GRAPH_THEME["font_sans"], "size": 12,
                     "color": GRAPH_THEME["text_primary"]},
            "align": "left",
        },
        font={"family": GRAPH_THEME["font_sans"], "color": GRAPH_THEME["text_secondary"]},
        showlegend=False,
    )
    axis = {
        "gridcolor": GRAPH_THEME["grid"], "griddash": "solid", "zeroline": False,
        "showline": False, "ticks": "",
        "tickfont": {"family": GRAPH_THEME["font_mono"], "size": 10,
                     "color": GRAPH_THEME["text_secondary"]},
    }
    fig.update_yaxes(**axis, range=[lo - pad, hi + pad], dtick=dtick,
                     tickformat=_profile_tick_format(dtick),
                     title={"text": "Cosine similarity",
                            "font": {"family": GRAPH_THEME["font_sans"], "size": 11,
                                     "color": GRAPH_THEME["text_secondary"]}},
                     row=1, col=1)
    fig.update_yaxes(**{**axis, "showgrid": False}, rangemode="tozero",
                     showticklabels=False,
                     title={"text": "Change",
                            "font": {"family": GRAPH_THEME["font_sans"], "size": 11,
                                     "color": GRAPH_THEME["text_secondary"]}},
                     row=2, col=1)
    fig.update_xaxes(**axis, showgrid=False, range=[0.5, n + 0.5], row=1, col=1)
    fig.update_xaxes(**axis, showgrid=False, range=[0.5, n + 0.5],
                     title={"text": "Match rank",
                            "font": {"family": GRAPH_THEME["font_sans"], "size": 11,
                                     "color": GRAPH_THEME["text_secondary"]}},
                     row=2, col=1)
    return fig


def empty_profile_figure(message: str) -> go.Figure:
    """The profile panel's own placeholder, at the panel's own height."""
    fig = _empty_network_figure(message)
    fig.update_layout(height=PROFILE_HEIGHT)
    return fig
