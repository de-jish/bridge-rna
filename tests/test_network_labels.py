"""Label placement for the retrieval network.

The geometry constants below were measured in the running app (Chromium,
DM Mono at 11 px) on 2026-10-21; `tests/e2e_network_style.py` checks the same
property - no edge touches a label's glyphs - on the rendered figure.
"""
import random

import pandas as pd
import pytest

from bridge_rna import network_labels as nl
from bridge_rna.figures import LABEL_HALO, build_network_figure


QUERY = "OSD-100|Mmus_C57-6J_EYE_FLT_Rep1_M23"


def _readme_network():
    """The README's OSD-100 retrieval: five hits across three GEO series."""
    hits = pd.DataFrame({
        "gsm": ["GSM6431263", "GSM6204794", "GSM6431262", "GSM6431234", "GSM4256053"],
        "gse": ["GSE210492", "GSE205070", "GSE210492", "GSE210492", "GSE143281"],
        "score": [.997, .996, .995, .994, .993],
    })
    return pd.Series({"sample_id": QUERY, "sample_name": QUERY.split("|")[1]}), hits


def _capture(query, hits):
    """Run the figure build and keep what it handed the placer."""
    seen = {}
    real = nl.place

    def spy(nodes, edges, margin, frames=nl.FRAMES):
        seen.update(nodes=nodes, edges=edges, margin=margin)
        return real(nodes, edges, margin, frames)

    nl.place = spy
    try:
        fig = build_network_figure(query, hits)
    finally:
        nl.place = real
    return fig, seen


def _projector(nodes, margin, width, height):
    to_x = nl._autorange([n["x"] for n in nodes], [n["size"] for n in nodes],
                         width - margin["l"] - margin["r"])
    to_y = nl._autorange([-n["y"] for n in nodes], [n["size"] for n in nodes],
                         height - margin["t"] - margin["b"])
    return lambda x, y: (margin["l"] + to_x(x), margin["t"] + to_y(-y))


def _ink(node, position, to_px):
    cx, cy = to_px(node["x"], node["y"])
    x0, y0, x1, y1 = nl.label_box(position, node["size"], node["lines"])
    return cx + x0, cy + y0 + nl._INK_INSET_PX, cx + x1, cy + y1 - nl._INK_INSET_PX


def _touching(nodes, edges, positions, margin, frame, own_only=False):
    to_px = _projector(nodes, margin, *frame)
    found = []
    for n in nodes:
        if not n["lines"]:
            continue
        box = _ink(n, positions[n["node_id"]], to_px)
        grown = (box[0] - 1.5, box[1] - 1.5, box[2] + 1.5, box[3] + 1.5)
        for e in edges:
            if own_only and n["node_id"] not in e["nodes"]:
                continue
            if nl._segment_meets_box(to_px(e["x0"], e["y0"]), to_px(e["x1"], e["y1"]), grown):
                found.append((n["node_id"], tuple(e["nodes"])))
    return found


def test_a_sample_name_wraps_at_its_field_boundaries():
    assert nl.wrap_identifier("Mmus_C57-6J_EYE_FLT_Rep1_M23") == ["Mmus_C57-6J_EYE_", "FLT_Rep1_M23"]
    assert nl.wrap_identifier("RR10_KDN_WT_BSL_B11") == ["RR10_KDN_WT_BSL_", "B11"]
    assert nl.wrap_identifier("short") == ["short"]
    assert nl.wrap_identifier("") == []
    # A field longer than a line is cut, and nothing is ever lost.
    long = "A" * 40 + "_tail"
    lines = nl.wrap_identifier(long)
    assert all(len(ln) <= nl.QUERY_WRAP for ln in lines)
    assert "".join(lines) == long


@pytest.mark.parametrize("position, box", [
    ("top left", (-77, -25, -11, -10)), ("top center", (-33, -25, 33, -10)),
    ("top right", (11, -25, 77, -10)), ("bottom left", (-77, 8, -11, 23)),
    ("bottom center", (-33, 8, 33, 23)), ("bottom right", (11, 8, 77, 23)),
    ("middle left", (-77, -7.5, -11, 7.5)), ("middle right", (11, -7.5, 77, 7.5)),
])
def test_label_boxes_match_what_plotly_draws(position, box):
    """Measured: a 10-character accession beside a 16 px hit marker."""
    assert nl.label_box(position, 16, ["GSM6431263"]) == pytest.approx(box, abs=1)


def test_the_pixel_model_matches_the_rendered_figure():
    """Node centers measured in the browser at 1280x800 (a 526x546 canvas)."""
    query, hits = _readme_network()
    _, seen = _capture(query, hits)
    to_px = _projector(seen["nodes"], seen["margin"], 526, 546.406)
    measured = {QUERY: (147, 273), "GSM6431263": (283, 52), "GSM6204794": (283, 162),
                "GSM6431262": (283, 273), "GSM6431234": (283, 384), "GSM4256053": (283, 494),
                "GSE143281": (433, 91), "GSE205070": (433, 273), "GSE210492": (433, 455)}
    for n in seen["nodes"]:
        assert to_px(n["x"], n["y"]) == pytest.approx(measured[n["node_id"]], abs=1.5), n["node_id"]


def test_the_query_and_every_study_are_labeled_on_their_open_side():
    query, hits = _readme_network()
    fig, _ = _capture(query, hits)
    nodes = fig.data[-1]
    for (kind, node_id, _), position in zip(nodes.customdata, nodes.textposition):
        if kind == "query":
            assert position == "middle left"
        elif kind == "gse":
            assert position == "middle right"


@pytest.mark.parametrize("frame", nl.FRAMES[:2], ids=["1680", "1440"])
def test_no_edge_touches_a_label_in_the_readme_network(frame):
    """The reported defect: edges struck through the query's name and accessions."""
    query, hits = _readme_network()
    fig, seen = _capture(query, hits)
    positions = dict(zip((c[1] for c in fig.data[-1].customdata), fig.data[-1].textposition))
    assert _touching(seen["nodes"], seen["edges"], positions, seen["margin"], frame) == []


def test_no_label_ever_sits_on_its_own_edges():
    """Across random results of every labeled size, on every canvas."""
    rng = random.Random(20261021)
    for trial in range(60):
        k = rng.randint(1, 12)
        studies = [f"GSE{rng.randint(1, 999999)}" for _ in range(rng.randint(1, k))]
        hits = pd.DataFrame({
            "gsm": [f"GSM{rng.randint(1000000, 9999999)}" for _ in range(k)],
            "gse": [rng.choice(studies + [""]) for _ in range(k)],
            "score": sorted((rng.random() for _ in range(k)), reverse=True),
        })
        fig, seen = _capture(pd.Series({"sample_id": QUERY, "sample_name": "x" * rng.randint(4, 40)}), hits)
        positions = dict(zip((c[1] for c in fig.data[-1].customdata), fig.data[-1].textposition))
        for frame in nl.FRAMES:
            assert _touching(seen["nodes"], seen["edges"], positions, seen["margin"], frame,
                             own_only=True) == [], (trial, frame)


def test_the_margins_hold_the_side_labels_on_every_canvas():
    query, hits = _readme_network()
    fig, seen = _capture(query, hits)
    positions = dict(zip((c[1] for c in fig.data[-1].customdata), fig.data[-1].textposition))
    for width, height in nl.FRAMES:
        to_px = _projector(seen["nodes"], seen["margin"], width, height)
        for n in seen["nodes"]:
            cx, cy = to_px(n["x"], n["y"])
            x0, y0, x1, y1 = nl.label_box(positions[n["node_id"]], n["size"], n["lines"])
            assert cx + x0 >= 3 and cx + x1 <= width - 3, (n["node_id"], width)
            assert cy + y0 >= 0 and cy + y1 <= height, (n["node_id"], height)


def test_labels_are_mono_with_a_solid_knockout_and_markers_stay_clickable():
    query, hits = _readme_network()
    fig, _ = _capture(query, hits)
    nodes = fig.data[-1]
    assert nodes.textfont.family.startswith("DM Mono")
    assert nodes.textfont.shadow == LABEL_HALO
    # Every shadow is unblurred, so together they paint a solid outline: a
    # blurred halo let a 3 px edge show through the glyphs at full strength.
    assert all(part.split()[2] == "0" for part in LABEL_HALO.split(", "))
    # Autorange, not an exact fit: Plotly hit-tests only inside the plot area,
    # so an axis fitted to the nodes leaves the outermost markers half-clickable.
    assert fig.layout.xaxis.range is None and fig.layout.yaxis.range is None
