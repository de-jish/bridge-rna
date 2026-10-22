"""Where each label of the retrieval network goes.

The network is three columns - the query at x = 0, its hits at x = 1, their
GEO series at x = 2.1 - and every edge runs left to right. Two of the three
label kinds therefore have an answer that holds for any result: nothing ever
leaves the query to the left, and nothing ever leaves a study to the right, so
the query's name sits on its left and each study's accession on its right.

A hit has an edge on both sides, and which of its six candidate spots is clear
depends on where its neighbors' edges run. That is decided here by scoring
each spot against every edge, marker and already-placed label, in pixels,
because a label is a pixel-sized object and its clearance cannot be judged in
data units. The figure does not know the canvas it will be drawn on, so a spot
is scored on each canvas the retrieval view actually produces (`FRAMES`) and
the lowest total wins. Labels sat centered above every node until 2026-10-21,
which put the query's name and most accessions on top of their own edges.

Nothing here touches a score, a rank, a node position or an edge. It decides
where text goes and how much margin the figure reserves for it.
"""

from __future__ import annotations

import re
from typing import Iterable

#: Labels are accessions and sample names, set in mono like the rail's
#: identifiers, which is also what makes their width predictable.
FONT_SIZE = 11
#: DM Mono's advance is 0.6 em: a 10-character accession measured 66 px.
CHAR_PX = 0.6 * FONT_SIZE
#: One label line's box height at 11 px, measured.
LINE_PX = 15.0
#: Where a hit's caps sit inside that box. An edge passing through the box's
#: empty ascent or descent band is not touching any text.
_INK_INSET_PX = 3.0
#: How close an edge may come to a glyph. Half the 4.5 px emphasized stroke,
#: plus a pixel of air.
_CLEARANCE_PX = 3.25

#: Widest line of a wrapped query name, in characters.
QUERY_WRAP = 16
#: Air between a side label and the canvas edge.
_EDGE_AIR_PX = 4
#: Room above the top row and below the bottom one. Autorange already keeps
#: an outermost node a marker radius plus 5% of the axis inside the plot, so
#: a label set above or below it needs only this much more.
ROW_PX = 16
#: Plotly's autorange adds this share of the axis length past the outermost
#: point of any trace drawn with markers or text (`padded` in its autorange).
_AUTORANGE_PAD = 0.05

#: (width, height) of `#network-graph`'s canvas, measured in the running app
#: at 1680x1335 (the README capture), 1440x1000, 1280x800, and - below the
#: 1180 px breakpoint, where the network becomes a 320 px strip - at 1180,
#: 768 and 390 px wide. A placement is chosen to suit all of them at once.
FRAMES: tuple[tuple[int, int], ...] = (
    (926, 1081), (686, 746), (526, 546), (1098, 320), (686, 320), (324, 320))

#: In preference order, which breaks ties: centered first because it keeps the
#: label over its own node, then inside the query fan, then toward the studies.
HIT_POSITIONS = ("top center", "bottom center", "top left", "bottom left",
                 "top right", "bottom right")


def wrap_identifier(label: str, width: int = QUERY_WRAP) -> list[str]:
    """Wrap an identifier onto lines of at most `width` characters.

    Breaks go after `_` or a space so that `Mmus_C57-6J_EYE_FLT_Rep1_M23`
    reads as `Mmus_C57-6J_EYE_` / `FLT_Rep1_M23` rather than splitting a
    field; a single field longer than `width` is cut hard.
    """
    label = str(label)
    lines: list[str] = []
    line = ""
    for token in re.findall(r"[^_ ]*[_ ]?", label):
        if not token:
            continue
        while len(token) > width:
            if line:
                lines.append(line)
                line = ""
            lines.append(token[:width])
            token = token[width:]
        if len(line) + len(token) > width:
            lines.append(line)
            line = token
        else:
            line += token
    if line:
        lines.append(line)
    return [ln.rstrip() for ln in lines]


def text_px(lines: Iterable[str]) -> float:
    return max((len(ln) for ln in lines), default=0) * CHAR_PX


def _ppad(size: float) -> float:
    """The pixel room Plotly's autorange keeps for a marker: `size / 1.6`."""
    return size / 1.6


def _offset(size: float) -> float:
    """Plotly's distance from a marker's center to text set beside it."""
    return _ppad(size) + 1


def label_box(position: str, size: float, lines: list[str]) -> tuple[float, float, float, float]:
    """The label's box relative to its node's center, as (x0, y0, x1, y1).

    Screen orientation: y grows downward. Reproduces Plotly's own
    `textPointPosition` offsets, checked against the rendered boxes of all
    eight positions.
    """
    d, w, h = _offset(size), text_px(lines), LINE_PX * max(len(lines), 1)
    vertical, horizontal = position.split()
    x0 = {"left": -d - w, "center": -w / 2, "right": d}[horizontal]
    y0 = {"top": -(d - 1) - h, "bottom": d - 3, "middle": -h / 2}[vertical]
    return x0, y0, x0 + w, y0 + h


def margins(query_lines: list[str], query_size: float,
            gse_lines: list[list[str]], gse_size: float) -> dict[str, int]:
    """Pixel margins that hold the query's name and the widest study accession.

    Autorange keeps each outermost marker `_ppad` inside the plot, so a side
    label needs the rest of its offset plus its own width.
    """
    left = _offset(query_size) - _ppad(query_size) + text_px(query_lines)
    right = max((_offset(gse_size) - _ppad(gse_size) + text_px(lines) for lines in gse_lines),
                default=0.0)
    return {"l": round(left + _EDGE_AIR_PX), "r": round(right + _EDGE_AIR_PX),
            "t": ROW_PX, "b": ROW_PX}


def _autorange(values: list[float], sizes: list[float], length: float):
    """Map data to pixels along one axis the way Plotly's autorange does.

    The outermost point at each end is kept `_ppad(size)` plus 5% of the axis
    inside the plot, so its marker - and its click and hover target, which
    Plotly only tests inside the plot area - is never cut by the plot edge.
    Fitting the axes exactly to the nodes was tried and left the top hit,
    the query and every study half-clickable.
    """
    lo, hi = min(values), max(values)
    pad_lo = max(_ppad(s) for v, s in zip(values, sizes) if v == lo) + _AUTORANGE_PAD * length
    pad_hi = max(_ppad(s) for v, s in zip(values, sizes) if v == hi) + _AUTORANGE_PAD * length
    if hi == lo:
        return lambda v: length / 2
    return lambda v: pad_lo + (v - lo) / (hi - lo) * (length - pad_lo - pad_hi)


def _segment_meets_box(a, b, box) -> bool:
    """Liang-Barsky: does segment a-b enter the axis-aligned box?"""
    (x0, y0), (dx, dy) = a, (b[0] - a[0], b[1] - a[1])
    t0, t1 = 0.0, 1.0
    for p, q in ((-dx, x0 - box[0]), (dx, box[2] - x0), (-dy, y0 - box[1]), (dy, box[3] - y0)):
        if p == 0:
            if q < 0:
                return False
            continue
        t = q / p
        if p < 0:
            if t > t1:
                return False
            t0 = max(t0, t)
        else:
            if t < t0:
                return False
            t1 = min(t1, t)
    return True


def _boxes_meet(a, b) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def _circle_meets_box(c, r, box) -> bool:
    nx, ny = min(max(c[0], box[0]), box[2]), min(max(c[1], box[1]), box[3])
    return (nx - c[0]) ** 2 + (ny - c[1]) ** 2 < r * r


def place(nodes: list[dict], edges: list[dict], margin: dict[str, int],
          frames: Iterable[tuple[int, int]] = FRAMES) -> dict[str, str]:
    """Choose a text position for every labeled node.

    `nodes` carry `node_id`, `kind`, `x`, `y`, `size` and `lines` (empty for
    a node drawn without a label); `edges` carry `x0, y0, x1, y1` and `nodes`.
    Returns `{node_id: textposition}`.

    A hit is never put where one of its own edges runs. Among the rest, each
    candidate costs one point per frame for every edge that comes within
    `_CLEARANCE_PX` of its text, every label box it overlaps and every marker
    it covers; hits are placed top to bottom, each against those already set.
    """
    frames = tuple(frames)
    out: dict[str, str] = {}
    for n in nodes:
        if n["kind"] == "query":
            out[n["node_id"]] = "middle left"
        elif n["kind"] == "gse":
            out[n["node_id"]] = "middle right"
    hits = sorted((n for n in nodes if n["kind"] == "gsm" and n["lines"]), key=lambda n: -n["y"])
    if not hits:
        return out

    xs, ys, sizes = ([n[k] for n in nodes] for k in ("x", "y", "size"))

    def projector(width, height):
        to_x = _autorange(xs, sizes, width - margin["l"] - margin["r"])
        to_y = _autorange([-y for y in ys], sizes, height - margin["t"] - margin["b"])
        return lambda x, y: (margin["l"] + to_x(x), margin["t"] + to_y(-y))

    projections = [(projector(w, h), w, h) for w, h in frames]

    def boxed(node, position, to_px):
        cx, cy = to_px(node["x"], node["y"])
        x0, y0, x1, y1 = label_box(position, node["size"], node["lines"])
        return cx + x0, cy + y0, cx + x1, cy + y1

    def ink(box):
        c = _CLEARANCE_PX
        return box[0] - c, box[1] + _INK_INSET_PX - c, box[2] + c, box[3] - _INK_INSET_PX + c

    by_id = {n["node_id"]: n for n in nodes}
    placed = {n["node_id"]: out[n["node_id"]] for n in nodes if n["node_id"] in out and n["lines"]}
    for hit in hits:
        own = [e for e in edges if hit["node_id"] in e["nodes"]]
        best, best_cost = None, None
        for rank, position in enumerate(HIT_POSITIONS):
            cost, blocked = rank * 1e-3, False
            for to_px, w, h in projections:
                box = boxed(hit, position, to_px)
                text = ink(box)
                for e in edges:
                    if _segment_meets_box(to_px(e["x0"], e["y0"]), to_px(e["x1"], e["y1"]), text):
                        if e in own:
                            blocked = True
                            break
                        cost += 1
                if blocked:
                    break
                for n in nodes:
                    if n is not hit and _circle_meets_box(to_px(n["x"], n["y"]), n["size"] / 2 + 1.5, box):
                        cost += 1
                for other_id, other_pos in placed.items():
                    if _boxes_meet(box, boxed(by_id[other_id], other_pos, to_px)):
                        cost += 1
                if box[0] < _EDGE_AIR_PX or box[2] > w - _EDGE_AIR_PX or box[1] < 0 or box[3] > h:
                    cost += 1
            if blocked:
                continue
            if best_cost is None or cost < best_cost:
                best, best_cost = position, cost
        # Every candidate blocked by its own edges cannot happen in this layout
        # (a hit's two edges leave on opposite sides), but fall back sanely.
        out[hit["node_id"]] = placed[hit["node_id"]] = best or "top center"
    return out
