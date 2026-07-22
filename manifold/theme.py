"""Visual language for Bridge Manifold.

Bridge RNA is a light scientific-instrument theme; Bridge Manifold matches its
chrome exactly and departs in one deliberate place: a dark navy *plot canvas*
so the WebGL glyphs have contrast.

The categorical palette was validated with the dataviz skill's checker against
the navy plot surface (`#0e1d34`): all eleven hues sit in the OKLCH L 0.48-0.67
band, clear the chroma floor, pass the adjacent-pair CVD floor (worst ΔE 8.4),
the normal-vision floor (worst ΔE 15.4), and >= 3:1 contrast on the surface.
Perfect all-pairs CVD separation is impossible past a few categories on a
scatter, so high-cardinality color-bys lean on secondary encoding - a searchable
legend, hover that names the exact category, and a distinct OSDR symbol.
"""

from __future__ import annotations

# --- Bridge RNA chrome tokens (reused verbatim; REFERENCE.md section 9) -----
BG_CANVAS = "#eef2f7"
BG_PANEL = "#ffffff"
BG_PANEL_RAISED = "#f4f7fb"
BG_INSET = "#f5f8fc"
TEXT_PRIMARY = "#1a2432"
TEXT_SECONDARY = "#5a6b7e"
TEXT_MUTED = "#8a99ac"
ACCENT = "#2b7fff"
ACCENT_HOVER = "#1f6ff0"
ACCENT_TEAL = "#0bab9f"
ACCENT_WARM = "#d9791b"
HEADER_BG = "#14294a"
HEADER_FG = "#f3f7fc"
HEADER_LINE = "#22c7bd"
STATUS_GOOD = "#1f9d57"
STATUS_ERROR = "#d64545"
STATUS_WARN = "#b7791f"

# --- The one deliberate departure: a dark navy plot canvas ------------------
PLOT_BG = "#0e1d34"
PLOT_GRID = "#1c3252"
PLOT_AXIS = "#2a456b"
PLOT_TEXT = "#c7d6ea"

# --- Categorical palette (validated against PLOT_BG) ------------------------
# Slot order is the CVD-safety mechanism; do not shuffle without re-validating.
CATEGORICAL = [
    "#3987e5",  # 1 blue
    "#d95926",  # 2 orange
    "#199e70",  # 3 aqua
    "#c98500",  # 4 yellow
    "#d55181",  # 5 magenta
    "#008300",  # 6 green
    "#9085e9",  # 7 violet
    "#e66767",  # 8 red
    "#1b95a3",  # 9 cyan
    "#7d9a3c",  # 10 olive
    "#d84f96",  # 11 pink
]
# The neutral end of the palette. Two greys, because "Other" and "Unknown" are
# different answers: something was recorded and could not be placed, versus
# nothing was recorded at all. Unknown is the dimmer of the two so absence
# recedes furthest.
OTHER_COLOR = "#7f8ea3"
UNKNOWN_COLOR = "#56657a"
UNKNOWN_LABEL = "Unknown"


def residual_color(label: str) -> str:
    """Grey for a category that carries no information."""
    return UNKNOWN_COLOR if label == UNKNOWN_LABEL else OTHER_COLOR

# ARCHS4 drawn purely as spatial context, when the selected field describes only
# OSDR. Deliberately close to the plot background: it must read as scenery
# rather than as a category, because the whole point of the context state is
# that these points have no value under this field.
ARCHS4_CONTEXT = "#43597c"

# OSDR overlay marker: distinct symbol with a white ring so it pops above cloud.
OSDR_SYMBOL = "diamond"
OSDR_OUTLINE = "#ffffff"
# Single-color OSDR overlay, used when the color-by describes ARCHS4 only. Warm
# against the cool ARCHS4 palette, so the spaceflight corpus stays findable
# without competing for a categorical slot.
OSDR_HIGHLIGHT = "#f2a03d"

# --- A retrieval, drawn on the map -----------------------------------------
# The corpus recedes rather than these glyphs growing without limit: a mark big
# enough to find unaided in 942,563 points would be big enough to misrepresent
# where the sample actually sits.
#
# The query keeps the teal it wears as the star in the retrieval network, so
# the one glyph a returning user already recognises survives the trip. It is
# not in CATEGORICAL, so it cannot be mistaken for a legend row.
RETRIEVAL_QUERY = "#0bab9f"      # == ACCENT_TEAL, the network graph's query star
RETRIEVAL_QUERY_HALO = "rgba(11, 171, 159, 0.50)"
RETRIEVAL_QUERY_SIZE = 20.0
RETRIEVAL_QUERY_HALO_SIZE = 46.0

# Hits are an **open white ring**, and both halves of that matter.
#
# White because the obvious choice, the retrieval network's own blue #2b7fff,
# measures **1.03:1** against CATEGORICAL[0] (#3987e5, Blood / immune) - the
# largest bucket on the map at 155,761 points. A hit landing anywhere in 16.6%
# of the corpus would have been invisible. White is 3.64:1 against that bucket
# and 16.9:1 against the plot background.
#
# Open because it leaves the ARCHS4 point underneath showing its own tissue
# colour. One glyph then carries two independent measurements at once: that
# the sample was retrieved (512-d cosine, from the model) and what GEO's free
# text calls it (40 keyword rules, which know nothing about any embedding).
# Whether those two agree is visible in a single mark, for no extra ink.
RETRIEVAL_HIT_RING = "#ffffff"
RETRIEVAL_HIT_SIZE = 20.0
RETRIEVAL_HIT_LINE = 2.2

# Sizes above are pixels, not data units, so a halo never reads as a radius in
# map space and does not swell on zoom.

# How far the corpus recedes while a retrieval is drawn. The default ARCHS4
# opacity is 0.55, so this is a clear step back rather than an erasure. 0.22 was
# tried first and went too far: framed on a retrieval the surrounding region is
# sparse, and at 0.22 its tissue colours vanished - which defeats the open ring,
# whose whole purpose is to let the hit's tissue colour show through it.
RETRIEVAL_DIM_ARCHS4 = 0.35
# OSDR recedes as far as ARCHS4 while a retrieval is shown. Keeping it brighter
# was tried and looked wrong: 2,108 white-ringed diamonds at 0.40 are far more
# prominent than the handful of white rings marking the actual hits, so the
# thing the user came to see loses to the thing they did not ask about. The
# query is drawn separately as a star, so the OSDR corpus receding does not
# take the query with it.
RETRIEVAL_DIM_OSDR = 0.30

# Rank numerals stop here. Past 25 the rings stay and the numerals would be
# unreadable overlapping text rather than information.
RETRIEVAL_MAX_NUMERALS = 25


def color_for_index(i: int) -> str:
    """Categorical color for the i-th distinct category (wraps into Other-grey)."""
    if i < len(CATEGORICAL):
        return CATEGORICAL[i]
    return OTHER_COLOR


def base_figure_layout(is_3d: bool = False) -> dict:
    """A Plotly layout dict carrying the dark-navy plot theme."""
    axis = dict(
        showgrid=True,
        gridcolor=PLOT_GRID,
        zeroline=False,
        showline=False,
        color=PLOT_TEXT,
        tickfont=dict(color=PLOT_TEXT, size=10),
        showspikes=False,
    )
    layout = dict(
        paper_bgcolor=PLOT_BG,
        plot_bgcolor=PLOT_BG,
        font=dict(color=PLOT_TEXT, family="system-ui, -apple-system, 'Segoe UI', sans-serif"),
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False,
        # Pan, not select. There is no selection feature: the map is read, not
        # queried, and a drag that draws a marquee doing nothing would be a
        # promise the app does not keep. scrollZoom supplies the zoom.
        dragmode="pan",
        hovermode="closest",
        uirevision="keep",
    )
    if is_3d:
        scene_axis = dict(
            showgrid=True,
            gridcolor=PLOT_GRID,
            zeroline=False,
            showbackground=True,
            backgroundcolor=PLOT_BG,
            color=PLOT_TEXT,
        )
        layout["scene"] = dict(
            xaxis=scene_axis, yaxis=scene_axis, zaxis=scene_axis,
            bgcolor=PLOT_BG,
        )
    else:
        layout["xaxis"] = dict(axis, visible=False)
        layout["yaxis"] = dict(axis, visible=False, scaleanchor="x", scaleratio=1)
    return layout
