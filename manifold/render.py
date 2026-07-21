"""Figure construction: layered WebGL scatter over a density underlay.

Layers, back to front (IMPLEMENTATION.md section 6):
  1. Density underlay - a precomputed raster of all 940k points as a layout
     image, so the global manifold shape is always visible.
  2. ARCHS4 background - a stratified WebGL sample, split into categorical
     traces when colored by species, otherwise a neutral cloud.
  3. OSDR overlay - all OSDR points, larger diamonds with a white ring, colored
     by the selected field, always on top.

Every drawn point carries its global corpus index in `customdata`, so a lasso
selection recovers exact indices for the 512-d readout.
"""

from __future__ import annotations

import base64
from functools import lru_cache

import numpy as np
import plotly.graph_objects as go

from . import data, sampling, theme

ARCHS4_SIZE = 3.4
OSDR_SIZE = 8.5
TOP_N = 11
SCATTER3D_ARCHS4_CAP = 40000


@lru_cache(maxsize=4)
def _density_data_uri(name: str) -> str:
    path = data.paths.DENSITY_DIR / f"{name}.png"
    if not path.exists():
        return ""
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _archs4_species_indices(coords_xy, budget, viewport):
    """Sampled ARCHS4 indices (global == local for the ARCHS4 block)."""
    n_archs4, _, _ = data.counts()
    species = data.points_meta()["species_id"].to_numpy()[:n_archs4]
    mask = None
    if viewport is not None:
        mask = sampling.viewport_mask(coords_xy[:n_archs4], viewport)
    idx = sampling.stratified_archs4_sample(species, budget, seed=7, mask=mask)
    return idx


# Hover for the OSDR overlay: the sample it is, then what it is under the
# current color-by. Read from the rich customdata rows built below.
OSDR_HOVER = ("<b>%{customdata[1]}</b>", "%{customdata[2]}")


def _osdr_customdata(osdr_global: np.ndarray, color_by: str) -> list[list]:
    """Rows of [global_index, sample_key, category] for the OSDR overlay.

    Only OSDR carries the richer payload. The 100k-point ARCHS4 background
    keeps a bare list of ints, because it has no hover and the extra strings
    would be pure figure weight.
    """
    meta = data.osdr_metadata()
    keys = meta["sample_key"].astype(str).to_numpy() if "sample_key" in meta.columns \
        else np.array([f"OSDR {i}" for i in range(len(meta))])
    values = data.osdr_field_values(color_by).to_numpy() \
        if color_by in meta.columns or color_by in data.OSDR_FIELDS \
        else np.full(len(meta), "-")
    n_archs4, _, _ = data.counts()
    local = osdr_global - n_archs4
    return [[int(g), str(keys[i]), str(values[i])] for g, i in zip(osdr_global, local)]


def _categorical_traces(coords, idx, values, is_3d, size, symbol, outline,
                        hover_lines=(), customdata=None):
    """Build one trace per Top-N category (+ Other) for a set of points."""
    values = np.asarray(values).astype(str)
    order = [c for c, _ in _rank_categories(values)]
    top = order[:TOP_N]
    traces = []
    legend = []

    def rows_for(mask):
        if customdata is None:
            return None
        return [customdata[i] for i in np.where(mask)[0]]

    for i, cat in enumerate(top):
        sel = values == cat
        if not sel.any():
            continue
        color = theme.color_for_index(i)
        traces.append(_scatter(coords, idx[sel], color, is_3d, size, symbol, outline,
                               name=cat, hover_lines=hover_lines, customdata=rows_for(sel)))
        legend.append({"label": cat, "color": color, "count": int(sel.sum())})
    other_mask = ~np.isin(values, top)
    if other_mask.any():
        traces.append(_scatter(coords, idx[other_mask], theme.OTHER_COLOR, is_3d, size,
                               symbol, outline, name="Other", hover_lines=hover_lines,
                               customdata=rows_for(other_mask)))
        legend.append({"label": "Other", "color": theme.OTHER_COLOR,
                       "count": int(other_mask.sum())})
    return traces, legend


def _rank_categories(values):
    vals, counts = np.unique(values, return_counts=True)
    ranked = sorted(zip(vals, counts), key=lambda t: -t[1])
    # Push Unknown to the end so it never claims a bright slot.
    ranked = [t for t in ranked if t[0] not in ("Unknown", "nan", "None")] + \
             [t for t in ranked if t[0] in ("Unknown", "nan", "None")]
    return ranked


def _scatter(coords, idx, color, is_3d, size, symbol, outline, name,
             hover_lines=(), customdata=None):
    idx = np.asarray(idx)
    x = coords[idx, 0]
    y = coords[idx, 1]
    # customdata must be a plain Python list, not a numpy array. plotly.py
    # serializes numpy arrays as base64 typed-array specs, and Dash builds its
    # `selectedData` payload by indexing the *user* data - which, for an encoded
    # array, yields undefined. The result is a lasso whose points arrive at the
    # server with no customdata at all, so every selection reads as empty and
    # the entire readout silently never fires. Element 0 is always the global
    # corpus index, whether the row is a bare int or a richer hover payload.
    if customdata is None:
        customdata = idx.tolist()

    # Hover is the dominant per-frame cost at 100k glyphs, so the ARCHS4
    # background disables it outright. `hoverinfo="skip"` alone is not enough:
    # a hovertemplate overrides it, which is how the background cloud ended up
    # showing a label. The two must be turned off together.
    hover_on = bool(hover_lines)
    hovertemplate = ("<br>".join(hover_lines) + "<extra></extra>") if hover_on else None

    if is_3d:
        z = coords[idx, 2]
        return go.Scatter3d(
            x=x, y=y, z=z, mode="markers", name=name,
            marker=dict(size=size * 0.5, color=color, opacity=0.85,
                        line=dict(width=0)),
            customdata=customdata,
            hovertemplate=hovertemplate,
            hoverinfo=None if hover_on else "skip",
            showlegend=False,
        )
    line = dict(width=1.1, color=outline) if outline else dict(width=0)
    return go.Scattergl(
        x=x, y=y, mode="markers", name=name,
        marker=dict(size=size, color=color, opacity=(0.95 if outline else 0.55),
                    symbol=symbol, line=line),
        customdata=customdata,
        hovertemplate=hovertemplate,
        hoverinfo=None if hover_on else "skip",
        showlegend=False,
    )


def build_figure(method, dims, color_by, layers, budget, viewport):
    is_3d = dims == "3d"
    coords = data.coords(method, dims)
    n_archs4, n_osdr, total = data.counts()
    fig = go.Figure()
    legend_data = {"title": color_by, "items": []}
    badges = []

    if coords.shape[0] == 0:
        fig.update_layout(**theme.base_figure_layout(is_3d))
        fig.add_annotation(text=f"{method.upper()} coordinates not built yet",
                           showarrow=False, font=dict(color=theme.PLOT_TEXT, size=15))
        return fig, legend_data, [f"{method.upper()} not available"]

    coords_xy = coords[:, :2]

    # --- Layer 1: density underlay (2D only) -------------------------------
    images = []
    if "density" in layers and not is_3d:
        uri = _density_data_uri(data.METHODS[method]["density"])
        ext = data.stats().get(f"density_{data.METHODS[method]['density']}")
        if uri and ext:
            images.append(dict(
                source=uri, xref="x", yref="y",
                x=ext["x0"], y=ext["y1"],
                sizex=ext["x1"] - ext["x0"], sizey=ext["y1"] - ext["y0"],
                sizing="stretch", layer="below", opacity=0.85,
            ))

    # --- Layer 2: ARCHS4 background ----------------------------------------
    if "archs4" in layers:
        idx = _archs4_species_indices(coords_xy, int(budget), viewport)
        if is_3d and len(idx) > SCATTER3D_ARCHS4_CAP:
            idx = np.random.default_rng(1).choice(idx, SCATTER3D_ARCHS4_CAP, replace=False)

        if color_by == "species":
            sp = data.species_labels()[idx]
            for label, color in theme.SPECIES_COLORS.items():
                sel = sp == label
                if sel.any():
                    fig.add_trace(_scatter(coords, idx[sel], color, is_3d, ARCHS4_SIZE,
                                           "circle", None, name=label.title(), hover_lines=()))
                    legend_data["items"].append(
                        {"label": f"{label.title()} (ARCHS4)", "color": color,
                         "count": int(sel.sum())})
        elif color_by == "archs4_tissue":
            tissue = data.archs4_tissue()
            if tissue is None:
                # The optional HDF5 join was never built. Say so on the plot
                # rather than silently drawing a flat cloud the user will read
                # as "ARCHS4 has no tissue structure".
                fig.add_trace(_scatter(coords, idx, theme.ARCHS4_NEUTRAL, is_3d, ARCHS4_SIZE,
                                       "circle", None, name="ARCHS4", hover_lines=()))
                badges.append("ARCHS4 tissue not built - run "
                              "<b>precompute/fetch_archs4_meta.py</b>")
            else:
                traces, leg = _categorical_traces(
                    coords, idx, tissue[idx], is_3d, ARCHS4_SIZE, "circle", None,
                    hover_lines=())
                for t in traces:
                    fig.add_trace(t)
                legend_data["items"].extend(leg)
        else:
            fig.add_trace(_scatter(coords, idx, theme.ARCHS4_NEUTRAL, is_3d, ARCHS4_SIZE,
                                   "circle", None, name="ARCHS4", hover_lines=()))
        badges.append(f"ARCHS4 live: <b>{len(idx):,}</b>")

    # --- Layer 3: OSDR overlay ---------------------------------------------
    if "osdr" in layers and n_osdr > 0:
        osdr_global = np.arange(n_archs4, n_archs4 + n_osdr)
        osdr_rows = _osdr_customdata(osdr_global, color_by)
        if color_by == "species":
            sp = data.species_labels()[osdr_global]
            for label, color in theme.SPECIES_COLORS.items():
                sel = sp == label
                if sel.any():
                    fig.add_trace(_scatter(
                        coords, osdr_global[sel], color, is_3d, OSDR_SIZE,
                        theme.OSDR_SYMBOL, theme.OSDR_OUTLINE, name=f"OSDR {label}",
                        hover_lines=OSDR_HOVER,
                        customdata=[osdr_rows[i] for i in np.where(sel)[0]]))
                    # Contribute to the legend so it survives the ARCHS4 layer
                    # being switched off; otherwise a colored plot has no key.
                    if "archs4" not in layers:
                        legend_data["items"].append(
                            {"label": f"{label.title()} (OSDR)", "color": color,
                             "count": int(sel.sum())})
        elif color_by == "archs4_tissue":
            # An ARCHS4-only color-by. OSDR keeps its distinct glyph in a single
            # highlight color so the spaceflight corpus stays locatable without
            # borrowing ARCHS4's tissue vocabulary, which is a different
            # controlled list and would map to the wrong hues.
            fig.add_trace(_scatter(coords, osdr_global, theme.OSDR_HIGHLIGHT, is_3d,
                                   OSDR_SIZE, theme.OSDR_SYMBOL, theme.OSDR_OUTLINE,
                                   name="OSDR", hover_lines=OSDR_HOVER,
                                   customdata=osdr_rows))
            legend_data["items"].append(
                {"label": "OSDR (all)", "color": theme.OSDR_HIGHLIGHT, "count": n_osdr})
        else:
            values = data.osdr_field_values(color_by).to_numpy()
            traces, leg = _categorical_traces(
                coords, osdr_global, values, is_3d, OSDR_SIZE,
                theme.OSDR_SYMBOL, theme.OSDR_OUTLINE, hover_lines=OSDR_HOVER,
                customdata=osdr_rows)
            for t in traces:
                fig.add_trace(t)
            legend_data["items"].extend(leg)
        badges.append(f"OSDR: <b>{n_osdr:,}</b>")

    layout = theme.base_figure_layout(is_3d)
    if images:
        layout["images"] = images
    fig.update_layout(**layout)
    return fig, legend_data, badges
