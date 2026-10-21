"""Plotly figures for the retrieval results panel.

Three views draw the rows the results table currently shows (see
docs/retrieval-results.md, "Plots" and "Contracts"):

- `score_rank_figure`: the top-1,000 score profile on a log rank axis, with the
  retrieved k marked on it. It answers whether the top k stands apart from the
  ranks below it or sits on a flat curve.
- `similarity_figure`: pairwise cosine among up to 50 filtered matches, ordered
  by average linkage, with the dendrogram above.
- `series_figure`: matches per GEO series among the filtered rows.

The fourth view, the network, is `figures.build_network_figure`, unchanged.

Every clickable mark carries `customdata` of `[kind, node_id]` with `kind`
"gsm" or "gse", the network's convention, so one selection callback serves all
four views. Context marks (the profile line, the k rule, selection rings, the
dendrogram, total labels) set `hoverinfo="skip"` and carry no customdata:
Plotly emits no click for a skipped trace, so they can never clear a selection.

The panel is a light HDS surface, so these figures share the network's white
canvas, type and hover label rather than the map's navy plot canvas. Pure
plotly and numpy: no Dash, no memmap, no scipy.
"""

from __future__ import annotations

import html
import math
from typing import Any, Iterable

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .figures import GRAPH_THEME, _empty_network_figure
from .similarity import (IDENTICAL_PROFILE_DISTANCE, average_linkage, cosine_matrix,
                         dendrogram_segments, leaf_order)

#: The similarity view clusters at most this many rows; the caller slices the
#: filtered rows in table order. Fifty GSM labels is the most an axis can carry
#: at a legible size, and O(n^3) linkage is still instant there.
SIMILARITY_MAX_ROWS = 50

#: The series view draws this many GEO series; `series_counts` returns them all
#: so the panel can say how many were left out.
SERIES_MAX_BARS = 20

#: The network view is drawn from this many filtered rows. Above it the hit
#: column is unreadable (see `figures.build_network_figure`).
NETWORK_MAX_ROWS = 30

# Local theme additions. They belong beside GRAPH_THEME in figures.py and are
# kept here only because this change may not edit that file; move them there
# when it is next touched.
#
# HDS reserves NASA blue (#1c67e3) for controls and says "never for dataviz",
# so a match in the table wears the blue *shade*, the same hue the network
# gives a sample node. The neutrals are HDS carbon steps: the profile line is
# carbon-40 and a filtered-out match is a carbon-20 fill with a carbon-40 rim,
# deliberately recessive because it is context, not the result. The selection
# ring is carbon-90, the `--network-emphasis` stroke the network uses for the
# selected node.
_MATCH = GRAPH_THEME["gsm"]          # #0b3d91, nasa-blue-shade
_PROFILE = "#959599"                 # carbon-40
_MUTED_FILL = "#d1d1d1"              # carbon-20
_MUTED_LINE = "#959599"              # carbon-40
_SELECTED = GRAPH_THEME["text_primary"]  # carbon-90
_RULE = "#58585b"                    # carbon-60
_SURFACE = GRAPH_THEME["paper_bg"]

# Sequential ramp for cosine: one hue, light to dark, ending on nasa-blue-shade.
# Nine stops interpolated in OKLab from #e6eefa (L 0.947) to #0b3d91 (L 0.386),
# so lightness falls in equal steps and Plotly's RGB interpolation between
# neighbouring stops cannot bend the hue. The light end stays a visible tint so
# the least similar pair never vanishes into the white gaps between cells.
COSINE_SCALE = [
    [0.000, "#e6eefa"],
    [0.125, "#c9d7ee"],
    [0.250, "#adc1e1"],
    [0.375, "#91abd5"],
    [0.500, "#7795c8"],
    [0.625, "#5c7fba"],
    [0.750, "#4269ad"],
    [0.875, "#28549f"],
    [1.000, "#0b3d91"],
]

_NO_SERIES = "No series recorded"
_TITLE_MAX = 80
_LOG_TICKS = (1, 2, 5)

# Figure heights in px. Each view needs a height its marks can be read at, so
# the figure states it and the panel's container should follow it rather than
# clip it (a fixed container shorter than the figure hides the axis band).
SCORE_HEIGHT = 360
_SERIES_ROW_PX = 26
_DENDRO_PX = 84


# ---------------------------------------------------------------- shared style

def _revision(results: dict[str, Any] | None) -> str:
    """A `uirevision` that changes with the search and not with the filters.

    Plotly keeps a reader's zoom while `uirevision` is unchanged, so keying it
    on what was searched (rather than on what the filters left) means a filter
    change redraws in place and a new search starts from the full view.
    """
    if not results:
        return "results:none"
    parts = (results.get("mode"), results.get("query_id"), results.get("k"),
             bool(results.get("comparison")), results.get("label_a"),
             results.get("label_b"), results.get("reference_count"))
    return "results:" + "|".join("" if p is None else str(p) for p in parts)


def _font(size: int = 11, color: str | None = None, mono: bool = False) -> dict:
    return {"family": GRAPH_THEME["font_mono" if mono else "font_sans"],
            "size": size,
            "color": color or GRAPH_THEME["text_secondary"]}


def _axis(**overrides: Any) -> dict:
    """A recessive axis: hairline grid, no zero line, numerals in DM Mono."""
    axis = {
        "showgrid": True,
        "gridcolor": GRAPH_THEME["grid"],
        "gridwidth": 1,
        "zeroline": False,
        "showline": False,
        "ticks": "",
        "tickfont": _font(11, mono=True),
        "title": {"font": _font(12), "standoff": 8},
        "automargin": True,
    }
    axis.update(overrides)
    return axis


def _base_layout(revision: str, height: int, *, legend: bool,
                 margin: dict | None = None) -> dict:
    """The layout every result figure shares with the network figures."""
    return {
        "paper_bgcolor": GRAPH_THEME["paper_bg"],
        "plot_bgcolor": GRAPH_THEME["plot_bg"],
        "font": {"family": GRAPH_THEME["font_sans"],
                 "color": GRAPH_THEME["text_primary"]},
        "height": height,
        "autosize": True,
        "margin": margin or {"l": 12, "r": 16, "t": 40 if legend else 14, "b": 12},
        "showlegend": legend,
        "legend": {"orientation": "h", "yanchor": "bottom", "y": 1.0,
                   "xanchor": "left", "x": 0.0, "font": _font(11),
                   "bgcolor": "rgba(0,0,0,0)", "itemclick": False,
                   "itemdoubleclick": False},
        # "event", not "event+select", for the reason given in
        # figures.build_network_figure: select styling fades every other mark.
        "clickmode": "event",
        "hovermode": "closest",
        # The font color must be explicit; see figures.build_network_figure.
        "hoverlabel": {"font": {"family": GRAPH_THEME["font_sans"], "size": 12,
                                "color": GRAPH_THEME["text_primary"]},
                       "bgcolor": "#ffffff",
                       "bordercolor": GRAPH_THEME["grid"],
                       "align": "left"},
        # Nothing consumes a box or lasso selection, so neither is offered.
        "modebar": {"remove": ["select2d", "lasso2d", "autoScale2d"]},
        "uirevision": revision,
    }


def _esc(value: Any) -> str:
    """Source metadata as literal text inside Plotly's pseudo-HTML."""
    return html.escape("" if value is None else str(value), quote=False)


def _trim(text: str, limit: int = _TITLE_MAX) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."


def _labels(results: dict[str, Any]) -> tuple[str, str]:
    a = str(results.get("label_a") or "").strip() or "Cohort A"
    b = str(results.get("label_b") or "").strip() or "Cohort B"
    return a, b


def _decimals(span: float, ticks: int = 5) -> int:
    """Decimals that keep `ticks` evenly spaced tick labels distinct."""
    if not span or span <= 0 or not math.isfinite(span):
        return 4
    return int(min(6, max(2, -math.floor(math.log10(span / ticks)))))


def _best_rank(row: dict[str, Any]) -> int | None:
    ranks = [r for r in (row.get("rank_a"), row.get("rank_b")) if r is not None]
    return min(ranks) if ranks else None


def _visible_gsms(visible: Iterable[dict[str, Any]] | None) -> set[str]:
    return {str(r.get("gsm")) for r in (visible or [])}


# ------------------------------------------------------------- score by rank

def _log_ticks(n: int) -> tuple[list[int], list[str]]:
    vals: list[int] = []
    decade = 1
    while decade <= n:
        vals += [m * decade for m in _LOG_TICKS if m * decade <= n]
        decade *= 10
    if vals and n / vals[-1] >= 1.2:
        vals.append(n)
    return vals, [f"{v:,}" for v in vals]


def _score_hover(row: dict[str, Any], arm: str, results: dict[str, Any],
                 in_table: bool) -> str:
    lines = [f"<b>{_esc(row.get('gsm'))}</b>"]
    if results.get("comparison"):
        label_a, label_b = _labels(results)
        arms = (("a", label_a), ("b", label_b))
        # The marked arm first, then the other, so the hover reads across both.
        for key, label in sorted(arms, key=lambda kv: kv[0] != arm):
            rank, score = row.get(f"rank_{key}"), row.get(f"score_{key}")
            if rank is None or score is None:
                lines.append(f"{_esc(label)}: not in its top {int(results.get('k') or 0):,}")
            else:
                lines.append(f"{_esc(label)}: rank {int(rank):,}, score {float(score):.6f}")
    else:
        lines.append(f"Rank {int(row['rank_a']):,}, score {float(row['score_a']):.6f}")
    if row.get("gse"):
        lines.append(_esc(row["gse"]))
    if row.get("title"):
        lines.append(_esc(_trim(row["title"])))
    if not in_table:
        lines.append("<i>Filtered out of the table</i>")
    return "<br>".join(lines)


def score_rank_figure(results: dict[str, Any] | None,
                      visible: list[dict[str, Any]] | None,
                      selected_gsm: str | None = None,
                      revision: str | None = None) -> go.Figure:
    """Score by rank: the top-1,000 profile with the retrieved k marked on it.

    Scores at the top of a 940,455-sample index are flat (rank 1 near 0.997,
    rank 1,000 near 0.963), so the y axis is fitted to the data rather than
    starting at 0, and x is a log rank axis so the first ten ranks get as much
    width as the next ninety. Rows the filters keep are drawn in the match hue;
    retrieved rows they hide stay on the curve, muted, because hiding a row
    from the table does not change where it ranked.
    """
    rows = (results or {}).get("rows") or []
    if not results or not rows:
        return _empty_network_figure("Run a search to plot match scores by rank.")

    comparison = bool(results.get("comparison"))
    in_table = _visible_gsms(visible)
    k = int(results.get("k") or 0)
    arms = ["a", "b"] if comparison else ["a"]
    label_a, label_b = _labels(results)
    style = {
        "a": {"color": GRAPH_THEME["cohort_a"] if comparison else _MATCH,
              "symbol": "circle", "label": label_a},
        "b": {"color": GRAPH_THEME["cohort_b"], "symbol": "square", "label": label_b},
    }

    profiles = {arm: [float(s) for s in (results.get(f"profile_{arm}") or [])]
                for arm in arms}
    ranked = {arm: [r for r in rows
                    if r.get(f"rank_{arm}") is not None and r.get(f"score_{arm}") is not None]
              for arm in arms}
    depth = max([len(p) for p in profiles.values()]
                + [int(r[f"rank_{arm}"]) for arm in arms for r in ranked[arm]]
                + [k, 2])
    max_k = max([len(ranked[arm]) for arm in arms] + [k])
    size = 9 if max_k <= 60 else 7 if max_k <= 250 else 5
    # A surface ring keeps neighbouring markers apart until they are dense
    # enough that the rings, not the fills, would set the color of the curve.
    ring = 1.0 if size > 5 else 0.0

    fig = go.Figure()
    for arm in arms:
        prof = profiles[arm]
        if prof:
            fig.add_trace(go.Scatter(
                x=list(range(1, len(prof) + 1)), y=prof, mode="lines",
                line={"color": style[arm]["color"] if comparison else _PROFILE,
                      "width": 1.5},
                opacity=0.55 if comparison else 1.0,
                hoverinfo="skip", showlegend=False,
                meta={"role": "profile", "arm": arm}))

    any_muted = any(r["gsm"] not in in_table for arm in arms for r in ranked[arm])
    muted_keyed = False  # one legend row for the muted state, whichever arm
    # Muted markers first so the rows in the table draw on top of them.
    for shown in (False, True):
        for arm in arms:
            pts = [r for r in ranked[arm] if (r["gsm"] in in_table) == shown]
            if not pts and not (shown and comparison):
                continue
            sym = style[arm]["symbol"]
            sz = size - 1 if sym == "square" else size
            if shown:
                marker = {"color": style[arm]["color"], "size": sz, "symbol": sym,
                          "line": {"color": _SURFACE, "width": ring}}
                name = style[arm]["label"] if comparison else "In the table"
                legend = comparison or any_muted
            else:
                marker = {"color": _MUTED_FILL, "size": sz, "symbol": sym,
                          "line": {"color": _MUTED_LINE, "width": 1}}
                name = "Filtered out"
                legend, muted_keyed = not muted_keyed, True
            fig.add_trace(go.Scatter(
                x=[int(r[f"rank_{arm}"]) for r in pts],
                y=[float(r[f"score_{arm}"]) for r in pts],
                mode="markers", name=name, marker=marker,
                customdata=[["gsm", str(r["gsm"])] for r in pts],
                hovertext=[_score_hover(r, arm, results, shown) for r in pts],
                hovertemplate="%{hovertext}<extra></extra>",
                cliponaxis=False, showlegend=bool(legend),
                legendrank=(1 if arm == "a" else 2) if shown else 3,
                meta={"role": "in_table" if shown else "filtered_out", "arm": arm}))

    if selected_gsm:
        for arm in arms:
            hit = [r for r in ranked[arm] if str(r["gsm"]) == str(selected_gsm)]
            if not hit:
                continue
            sym = style[arm]["symbol"]
            fig.add_trace(go.Scatter(
                x=[int(hit[0][f"rank_{arm}"])], y=[float(hit[0][f"score_{arm}"])],
                mode="markers",
                marker={"symbol": f"{sym}-open", "size": size + 9,
                        "color": _SELECTED, "line": {"color": _SELECTED, "width": 2}},
                hoverinfo="skip", showlegend=False, cliponaxis=False,
                meta={"role": "selection", "arm": arm}))

    ys = [y for p in profiles.values() for y in p] + [
        float(r[f"score_{arm}"]) for arm in arms for r in ranked[arm]]
    lo, hi = min(ys), max(ys)
    span = hi - lo
    pad = span * 0.06 if span > 0 else max(abs(hi) * 1e-4, 1e-6)
    tickvals, ticktext = _log_ticks(depth)

    reference = results.get("reference_count")
    x_title = (f"Rank among {int(reference):,} reference samples"
               if reference else "Rank among reference samples")
    legend = comparison or any_muted
    layout = _base_layout(revision or _revision(results), SCORE_HEIGHT, legend=legend)
    layout.update(
        xaxis=_axis(type="log", range=[math.log10(0.82), math.log10(depth * 1.18)],
                    tickmode="array", tickvals=tickvals, ticktext=ticktext,
                    title={"text": x_title, "font": _font(12), "standoff": 8}),
        yaxis=_axis(range=[lo - pad, hi + pad],
                    tickformat=f".{_decimals(span)}f", nticks=6,
                    title={"text": "Cosine similarity", "font": _font(12), "standoff": 8}),
        # Box zoom on the curve; panning a log axis by drag is easy to lose.
        dragmode="zoom",
    )
    fig.update_layout(**layout)

    if k >= 1:
        # A rule at k, labelled with k, so the reader can see where the result
        # list ends and how the curve continues past it. On a log axis Plotly 6
        # places a shape by data value but an annotation by log10 of it; both
        # were checked in a rendered figure.
        at_edge = k >= depth / 1.25
        fig.add_shape(type="line", xref="x", yref="paper",
                      x0=k, x1=k, y0=0, y1=1,
                      line={"color": _RULE, "width": 1}, layer="below")
        fig.add_annotation(x=math.log10(k), y=1, xref="x", yref="paper",
                           text=f"Top {k:,}", showarrow=False,
                           xanchor="right" if at_edge else "left",
                           xshift=-5 if at_edge else 5, yanchor="top",
                           font=_font(11, GRAPH_THEME["text_secondary"]))
    return fig


# ------------------------------------------------------------------ similarity

def _rank_text(row: dict[str, Any]) -> str:
    a, b = row.get("rank_a"), row.get("rank_b")
    if b is None and a is not None:
        return f"rank {int(a):,}"
    parts = [f"rank {arm.upper()} {int(r):,}" for arm, r in (("a", a), ("b", b))
             if r is not None]
    return ", ".join(parts) or "no rank"


def similarity_figure(rows: list[dict[str, Any]], unit_vectors: np.ndarray,
                      selected_gsm: str | None = None,
                      revision: str | None = None) -> go.Figure:
    """Pairwise cosine among matches, clustered, with the dendrogram above.

    Rows are reordered by average linkage on 1 - cosine
    (`similarity.average_linkage`), so near profiles sit together and an
    identical pair (one profile under two GSM accessions) is adjacent and
    merges at height 0. The color scale runs from the least similar pair shown
    to 1, because every match is already close to the query and a scale from 0
    would paint the whole matrix one color.
    """
    rows = list(rows or [])[:SIMILARITY_MAX_ROWS]
    unit = np.asarray(unit_vectors, dtype=np.float64)
    if unit.ndim != 2:
        unit = unit.reshape(len(rows), -1) if len(rows) else np.zeros((0, 0))
    unit = unit[: len(rows)]
    if len(rows) != unit.shape[0]:
        raise ValueError(f"{len(rows)} rows but {unit.shape[0]} vectors")
    n = len(rows)
    if n < 2:
        return _empty_network_figure(
            "Similarity needs at least two filtered matches with a stored profile.")

    cos = cosine_matrix(unit)
    merges = average_linkage(1.0 - cos)
    order = leaf_order(merges, n)
    segments = dendrogram_segments(merges, n)
    ordered = [rows[i] for i in order]
    z = cos[np.ix_(order, order)]
    gsms = [str(r.get("gsm")) for r in ordered]

    off = z[~np.eye(n, dtype=bool)]
    zmin = float(off.min())
    if zmin >= 1.0:
        zmin = 1.0 - 1e-6
    span = 1.0 - zmin

    cell = 14 if n > 30 else 18 if n > 15 else 26
    heat_px = n * cell
    gap_px = 6
    top, bottom = 14, 84
    height = top + _DENDRO_PX + gap_px + heat_px + bottom
    plot_px = _DENDRO_PX + gap_px + heat_px
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[_DENDRO_PX / plot_px, heat_px / plot_px],
                        vertical_spacing=gap_px / plot_px)

    xs: list[float | None] = []
    ys: list[float | None] = []
    for x0, y0, x1, y1 in segments:
        xs += [x0, x1, None]
        ys += [y0, y1, None]
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="lines", line={"color": _RULE, "width": 1.25},
        hoverinfo="skip", showlegend=False, cliponaxis=False,
        meta={"role": "dendrogram"}), row=1, col=1)

    # A merge below the identical-profile distance is one profile under two
    # accessions. At the dendrogram's scale its bracket is a flat line on the
    # baseline, easy to miss, so its crossbar is redrawn heavier in ink.
    ix: list[float | None] = []
    iy: list[float | None] = []
    for i, m in enumerate(merges):
        if m.height < IDENTICAL_PROFILE_DISTANCE:
            x0, y0, x1, y1 = segments[3 * i + 2]
            ix += [x0, x1, None]
            iy += [y0, y1, None]
    if ix:
        fig.add_trace(go.Scatter(
            x=ix, y=iy, mode="lines", line={"color": _SELECTED, "width": 3},
            hoverinfo="skip", showlegend=False, cliponaxis=False,
            meta={"role": "identical"}), row=1, col=1)

    pos = list(range(n))
    hover = [[(f"<b>{z[i, j]:.6f}</b> cosine<br>"
               f"{_esc(gsms[i])} ({_rank_text(ordered[i])})<br>"
               f"{_esc(gsms[j])} ({_rank_text(ordered[j])})")
              for j in range(n)] for i in range(n)]
    decimals = _decimals(span, 4)
    fig.add_trace(go.Heatmap(
        x=pos, y=pos, z=z.tolist(), zmin=zmin, zmax=1.0,
        colorscale=COSINE_SCALE, xgap=1, ygap=1,
        customdata=[[["gsm", gsms[i]] for _ in range(n)] for i in range(n)],
        hovertext=hover, hovertemplate="%{hovertext}<extra></extra>",
        colorbar={"title": {"text": "Cosine", "side": "top", "font": _font(11)},
                  "thickness": 10, "outlinewidth": 0, "ticks": "",
                  "tickfont": _font(10, mono=True), "tickformat": f".{decimals}f",
                  # Below the tick, so the 1.0 label clears the title above.
                  "ticklabelposition": "outside bottom",
                  "nticks": 5, "x": 1.0, "xanchor": "left", "xpad": 10,
                  "len": heat_px / plot_px, "lenmode": "fraction",
                  "y": 0.0, "yanchor": "bottom"},
        meta={"role": "cosine"}), row=2, col=1)

    selected = [i for i, g in enumerate(gsms) if selected_gsm and g == str(selected_gsm)]
    ticktext = [f"<b>{_esc(g)}</b>" if i in selected else _esc(g)
                for i, g in enumerate(gsms)]
    label_size = 9 if n > 30 else 10 if n > 15 else 11
    tickfont = _font(label_size, GRAPH_THEME["text_secondary"], mono=True)

    layout = _base_layout(revision or "similarity:" + "|".join(sorted(gsms)), height,
                          legend=False,
                          margin={"l": 12, "r": 12, "t": top, "b": 12})
    layout.update(dragmode="zoom")
    fig.update_layout(**layout)
    height_max = max((s[3] for s in segments), default=0.0) or 1e-6
    fig.update_yaxes(row=1, col=1, **_axis(
        range=[-0.06 * height_max, height_max * 1.08], nticks=3,
        tickformat=f".{_decimals(height_max, 2)}f",
        tickfont=_font(9, mono=True), fixedrange=True,
        title={"text": "Distance", "font": _font(10), "standoff": 4}))
    fig.update_xaxes(row=1, col=1, showgrid=False, zeroline=False, showline=False,
                     showticklabels=False, ticks="")
    fig.update_xaxes(row=2, col=1, range=[-0.5, n - 0.5], tickmode="array",
                     tickvals=pos, ticktext=ticktext, tickangle=-90,
                     tickfont=tickfont, showgrid=False, zeroline=False,
                     showline=False, ticks="", automargin=True)
    fig.update_yaxes(row=2, col=1, range=[n - 0.5, -0.5], tickmode="array",
                     tickvals=pos, ticktext=ticktext, tickfont=tickfont,
                     showgrid=False, zeroline=False, showline=False, ticks="",
                     automargin=True)

    for i in selected:
        # Outline the selected sample's row and column, in the selection ink.
        for x0, x1, y0, y1 in ((-0.5, n - 0.5, i - 0.5, i + 0.5),
                               (i - 0.5, i + 0.5, -0.5, n - 0.5)):
            fig.add_shape(type="rect", xref="x2", yref="y2", x0=x0, x1=x1,
                          y0=y0, y1=y1, line={"color": _SELECTED, "width": 1.5},
                          fillcolor="rgba(0,0,0,0)", layer="above")
    return fig


# ---------------------------------------------------------------------- series

def series_counts(results: dict[str, Any] | None,
                  visible: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Matches per GEO series among `visible`, every series, most matches first.

    Each entry has `gse` ("" for rows with no series recorded), `label`,
    `count` and `best_rank` (the best rank in either cohort, or None). A
    comparison adds `a_only`, `both` and `b_only`. Ties in count go to the
    better best rank, then to the accession, so the order is deterministic.
    The "" entry is included so the caller can count it or not.
    """
    comparison = bool((results or {}).get("comparison"))
    groups: dict[str, dict[str, Any]] = {}
    for row in visible or []:
        gse = str(row.get("gse") or "").strip()
        entry = groups.setdefault(gse, {
            "gse": gse, "label": gse or _NO_SERIES, "count": 0, "best_rank": None,
            **({"a_only": 0, "both": 0, "b_only": 0} if comparison else {})})
        entry["count"] += 1
        best = _best_rank(row)
        if best is not None and (entry["best_rank"] is None or best < entry["best_rank"]):
            entry["best_rank"] = best
        if comparison:
            in_a, in_b = row.get("rank_a") is not None, row.get("rank_b") is not None
            key = "both" if in_a and in_b else "a_only" if in_a else "b_only"
            entry[key] += 1
    return sorted(groups.values(), key=lambda e: (
        -e["count"],
        e["best_rank"] if e["best_rank"] is not None else math.inf,
        e["gse"] == "", e["gse"]))


def _series_hover(entry: dict[str, Any], results: dict[str, Any]) -> str:
    n = entry["count"]
    lines = [f"<b>{_esc(entry['label'])}</b>",
             f"{n:,} {'match' if n == 1 else 'matches'} in the table"]
    if results.get("comparison"):
        label_a, label_b = _labels(results)
        lines += [f"Only {_esc(label_a)}: {entry['a_only']:,}",
                  f"Both: {entry['both']:,}",
                  f"Only {_esc(label_b)}: {entry['b_only']:,}"]
    if entry.get("best_rank") is not None:
        lines.append(f"Best rank {entry['best_rank']:,}")
    return "<br>".join(lines)


def series_figure(results: dict[str, Any] | None,
                  visible: list[dict[str, Any]] | None,
                  selected_gsm: str | None = None,
                  selected_gse: str | None = None,
                  revision: str | None = None) -> go.Figure:
    """Matches per GEO series among the filtered rows, top 20, largest on top.

    A count of matches, not a claim that a series is over-represented: nothing
    here is compared against the series' size in the reference collection.
    """
    if not results or not (results.get("rows") or []):
        return _empty_network_figure("Run a search to count matches per GEO series.")
    entries = series_counts(results, visible)
    if not entries:
        return _empty_network_figure("No matches pass the current filters.")
    shown = entries[:SERIES_MAX_BARS]
    comparison = bool(results.get("comparison"))

    target = str(selected_gse or "").strip() or None
    if target is None and selected_gsm:
        pool = list(visible or []) + list(results.get("rows") or [])
        hit = next((r for r in pool if str(r.get("gsm")) == str(selected_gsm)), None)
        if hit is not None and str(hit.get("gse") or "").strip():
            target = str(hit.get("gse")).strip()

    cats = [e["label"] for e in shown]
    fig = go.Figure()

    if comparison:
        label_a, label_b = _labels(results)
        parts = (("a_only", f"Only {label_a}", GRAPH_THEME["cohort_a"]),
                 ("both", "Both", GRAPH_THEME["cohort_shared"]),
                 ("b_only", f"Only {label_b}", GRAPH_THEME["cohort_b"]))
    else:
        parts = (("count", "Matches", _MATCH),)

    for key, name, color in parts:
        # The no-series bar is its own trace with no customdata: there is no
        # series to open, and Plotly cannot make one bar of a trace inert.
        for clickable in (True, False):
            group = [e for e in shown if bool(e["gse"]) == clickable]
            if not group:
                continue
            sel = [bool(target) and e["gse"] == target for e in group]
            fig.add_trace(go.Bar(
                orientation="h", y=[e["label"] for e in group],
                x=[e[key] for e in group], name=name,
                marker={"color": color,
                        # A 1px surface line on each side leaves a 2px gap
                        # between stacked segments; the selected series is
                        # outlined in the selection ink instead.
                        "line": {"color": [_SELECTED if s else _SURFACE for s in sel],
                                 "width": [2 if s else 1 for s in sel]}},
                customdata=[["gse", e["gse"]] for e in group] if clickable else None,
                hovertext=[_series_hover(e, results) for e in group],
                hovertemplate="%{hovertext}<extra></extra>",
                text=None if comparison else [f"{e['count']:,}" for e in group],
                textposition="outside", cliponaxis=False,
                textfont=_font(11, GRAPH_THEME["text_secondary"], mono=True),
                showlegend=comparison and clickable,
                legendgroup=key,
                meta={"role": "series", "part": key, "clickable": clickable}))

    longest = max(e["count"] for e in shown)
    if comparison:
        # Totals at the bar ends; the segments' own counts are in the hover and
        # the table, since a label inside a narrow segment would not fit. The
        # small offset is the gap a single bar's "outside" label gets for free.
        fig.add_trace(go.Scatter(
            x=[e["count"] + longest * 0.012 for e in shown], y=cats, mode="text",
            text=[f"{e['count']:,}" for e in shown], textposition="middle right",
            textfont=_font(11, GRAPH_THEME["text_secondary"], mono=True),
            hoverinfo="skip", showlegend=False, cliponaxis=False,
            meta={"role": "totals"}))

    def _tick(e: dict[str, Any]) -> str:
        if not e["gse"]:
            # A sentence, not an accession, so it is set in the sans.
            return (f'<span style="font-family:{GRAPH_THEME["font_sans"]}">'
                    f'{_esc(e["label"])}</span>')
        return f"<b>{_esc(e['label'])}</b>" if e["gse"] == target else _esc(e["label"])

    ticktext = [_tick(e) for e in shown]
    top = 40 if comparison else 10
    height = top + len(shown) * _SERIES_ROW_PX + 16
    layout = _base_layout(revision or _revision(results), height, legend=comparison,
                          margin={"l": 12, "r": 36, "t": top, "b": 6})
    layout.update(
        barmode="stack", bargap=0.34,
        # A 3px rounded data-end, square at the baseline (Plotly rounds only
        # the far end of a stack).
        barcornerradius=3,
        # Every bar carries its count, so the value axis would only repeat it.
        xaxis=_axis(range=[0, longest * 1.12], showgrid=False,
                    showticklabels=False, fixedrange=True),
        yaxis=_axis(type="category", categoryorder="array", categoryarray=cats,
                    autorange="reversed", tickmode="array", tickvals=cats,
                    ticktext=ticktext, showgrid=False, fixedrange=True,
                    tickfont=_font(11, GRAPH_THEME["text_primary"], mono=True),
                    ticklabelstandoff=6),
        dragmode=False,
    )
    fig.update_layout(**layout)
    return fig
