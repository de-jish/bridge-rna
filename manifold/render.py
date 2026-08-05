"""Figure construction: layered WebGL scatter.

Layers, back to front:
  1. ARCHS4 background - a WebGL sample of the 940,455-point corpus, split into
     categorical traces by the selected field.
  2. OSDR overlay - all OSDR points, larger diamonds with a white ring, always
     on top so the 2,108 spaceflight samples stay findable in 940k.

There used to be a third layer underneath both: a precomputed density raster of
all 942,563 points, placed as a layout image. It is gone. Everything drawn here
is now a real glyph at a real sample's coordinates, which is why the point
budget goes all the way to the whole corpus.

Two decisions here are what keep the map honest.

*One palette for both corpora.* Categories are ranked once over the whole
covered population and every layer draws from that single mapping, so a liver in
GEO and a liver in OSDR are the same colour. Ranking per layer - the previous
behaviour - silently gave the same category two different colours whenever the
two corpora had different category orderings, which is a legend that lies.

*A corpus a field does not describe is drawn as context, not as data.* Picking
an OSDR-only field used to paint 940,455 uniform grey glyphs, which reads as
"ARCHS4 was measured and has no structure here". Instead those points are drawn
in one deliberately faint context colour at 0.35 opacity, outside the legend, so
they read as scenery rather than as a category. See manifold/colorby.py.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from . import colorby, data, sampling, theme

ARCHS4_SIZE = 3.4
ARCHS4_CONTEXT_SIZE = 2.6
OSDR_SIZE = 8.5
TOP_N = 11

# 3-D keeps a cap the 2-D view no longer needs, and it was re-measured rather
# than inherited when the 2-D budget was raised to the whole corpus. `Scatter3d`
# has no equivalent of `Scattergl`'s fast path, and the cost that matters is
# rotation, not first paint. Measured in a headless browser over the real 3-D
# coordinates, first paint barely moves (1.1 s at 40k, 1.9 s at 400k) but a
# twelve-step camera drag scales linearly with glyph count: 5.6 s at 42k, 10.4 s
# at 102k, 18.5 s at 202k, 31.4 s at 402k. Spinning it is the whole point of a
# 3-D view, so the cap stays where rotation stays usable.
SCATTER3D_ARCHS4_CAP = 40000

# Label for everything past the palette's capacity, merged with any residual
# category ("Other", "Unknown") so the legend has one grey row rather than two.
OVERFLOW = "Other"


def _archs4_sample_indices(coords_xy, budget, viewport):
    """Sampled ARCHS4 indices (global == local for the ARCHS4 block)."""
    n_archs4, _, _ = data.counts()
    species = data.points_meta()["species_id"].to_numpy()[:n_archs4]
    mask = None
    if viewport is not None:
        mask = sampling.viewport_mask(coords_xy[:n_archs4], viewport)
    return sampling.stratified_archs4_sample(species, budget, seed=7, mask=mask)


# Hover for the OSDR overlay: the sample it is, then what it is under the
# current color-by.
OSDR_HOVER = ("<b>%{customdata[0]}</b>", "%{customdata[1]}")


def _category_plan(values: np.ndarray) -> tuple[dict, list[dict]]:
    """Rank categories once over the whole covered population.

    Returns a lookup mapping every raw category to its *display* category, and
    the legend rows. The counts here are whole-corpus counts, and they do two
    jobs: they rank the categories - which fixes each one's colour and legend
    order stably, independent of budget or zoom - and they back the
    point-to-slot consistency check. They are not what the user sees: build_figure
    overrides each row's displayed count with the number of points that field
    actually plots in the current view, because a legend count is read as "how
    many of these are on screen", not "how many exist".

    Residual categories keep their own legend rows rather than being folded into
    the overflow bucket. "Unknown" and "Other" are different facts - we were
    never told, versus we were told something that could not be placed - and
    manifold/tissue.py goes to some trouble to keep them apart, so throwing the
    distinction away at the last step would waste it. They share the neutral
    end of the palette and always sort last, so they still never outrank a
    category that carries information.
    """
    covered = values[values != colorby.NOT_COVERED]
    if covered.size == 0:
        return {}, []

    uniq, counts = np.unique(covered.astype(str), return_counts=True)
    ranked = sorted(zip(uniq.tolist(), counts.tolist()), key=lambda t: -t[1])
    primary = [t for t in ranked if not colorby.is_residual(t[0])]
    residual = [t for t in ranked if colorby.is_residual(t[0])]

    top = primary[:TOP_N]
    lookup = {cat: cat for cat, _ in top}
    legend = [{"label": cat, "color": theme.color_for_index(i), "count": n}
              for i, (cat, n) in enumerate(top)]

    # Grey rows, keyed by display label so a genuine "Other" category and the
    # overflow bucket - which share a name by construction - become one row
    # instead of two identical-looking ones.
    grey: dict[str, int] = {}
    for cat, n in residual:
        label = cat if cat and cat not in ("nan", "None") else theme.UNKNOWN_LABEL
        lookup[cat] = label
        grey[label] = grey.get(label, 0) + n
    for cat, n in primary[TOP_N:]:
        lookup[cat] = OVERFLOW
        grey[OVERFLOW] = grey.get(OVERFLOW, 0) + n

    for label, n in sorted(grey.items(), key=lambda t: (t[0] == theme.UNKNOWN_LABEL, -t[1])):
        legend.append({"label": label, "color": theme.residual_color(label), "count": n})
    return lookup, legend


NOT_COVERED_CODE = -1


def _display_codes(values: np.ndarray, lookup: dict, legend: list[dict]) -> np.ndarray:
    """Legend slot for every point, as one compact integer array.

    Deliberately integer codes rather than the obvious array of display-label
    strings, for two reasons that both bite at 942,563 points.

    Memory: under pandas 3.0 a string Series materializes a *fresh* Python str
    per element on ``.to_numpy()``, so the string version of this array held
    942,563 distinct objects to represent 13 distinct values - 127 MB per
    colour-by, measured, which across the registry would have made the memoized
    plan below cost more than a gigabyte. The codes cost 1.9 MB.

    Speed: ``codes == slot`` is a vectorized integer compare, where
    ``labels == "Liver"`` over an object array is 942,563 Python string
    comparisons, once per category.

    Points the field says nothing about get ``NOT_COVERED_CODE``, which matches
    no legend slot, so they are drawn by the context path rather than silently
    folded into the overflow bucket.
    """
    slot = {row["label"]: i for i, row in enumerate(legend)}
    overflow = slot.get(OVERFLOW, NOT_COVERED_CODE)
    raw_codes, uniques = pd.factorize(values, sort=False)
    lut = np.array(
        [NOT_COVERED_CODE if u == colorby.NOT_COVERED
         else slot.get(lookup.get(str(u), OVERFLOW), overflow)
         for u in uniques],
        dtype=np.int16)
    return lut[raw_codes]


@lru_cache(maxsize=len(colorby.REGISTRY))
def _colour_plan(key: str) -> tuple[np.ndarray, list[dict]]:
    """The (legend slot per point, legend rows) for a colour-by, cached.

    This is the dominant per-figure cost - resolving one label array over all
    942,563 points, ranking the categories, and assigning each point a slot runs
    about 0.8 s for Tissue - and none of it depends on the projection, the
    dimensionality, the point budget, or the viewport. Only the colour-by key
    changes the answer.

    Caching it here is what keeps a zoom, a budget change, or a switch between
    PCA and UMAP cheap now that those redraw the whole corpus rather than a
    100,000-point sample. The registry is small and fixed and each entry is
    1.9 MB, so every key can be held at once. This inherits the same assumption
    the loaders in data.py already make: cache artifacts do not change while the
    app is running.
    """
    values = colorby.labels(key)
    lookup, legend = _category_plan(values)
    return _display_codes(values, lookup, legend), legend


def _legend_with_drawn_counts(legend: list[dict],
                              drawn_codes: list[np.ndarray]) -> list[dict]:
    """Legend rows re-counted to the points actually plotted.

    The colour plan ranks categories over the whole corpus, which fixes each
    row's colour and order. The count, though, is read as "how many of these
    are on screen", so it is recomputed here from the slot codes of the drawn
    points - the ARCHS4 sample the budget and zoom selected, plus the OSDR
    overlay - and a category with nothing currently drawn drops out of the key.
    Colours and order are untouched, so a category keeps its colour and its
    place whether or not it happens to be on screen right now.
    """
    if drawn_codes:
        stacked = np.concatenate(drawn_codes)
        stacked = stacked[stacked >= 0]
        counts = np.bincount(stacked, minlength=len(legend))
    else:
        counts = np.zeros(len(legend), dtype=int)
    return [{**legend[slot], "count": int(counts[slot])}
            for slot in range(len(legend)) if counts[slot] > 0]


def _scatter(coords, idx, color, is_3d, size, symbol, outline, name,
             hover_lines=(), customdata=None, opacity=None):
    idx = np.asarray(idx)
    x = coords[idx, 0]
    y = coords[idx, 1]

    # Hover is the dominant per-frame cost at 100k glyphs, so the ARCHS4
    # background disables it outright. `hoverinfo="skip"` alone is not enough:
    # a hovertemplate overrides it, which is how the background cloud ended up
    # showing a label. The two must be turned off together.
    hover_on = bool(hover_lines)
    hovertemplate = ("<br>".join(hover_lines) + "<extra></extra>") if hover_on else None

    if is_3d:
        return go.Scatter3d(
            x=x, y=y, z=coords[idx, 2], mode="markers", name=name,
            marker=dict(size=size * 0.5, color=color,
                        opacity=0.85 if opacity is None else opacity,
                        line=dict(width=0)),
            customdata=customdata,
            hovertemplate=hovertemplate,
            hoverinfo=None if hover_on else "skip",
            showlegend=False,
        )
    line = dict(width=1.1, color=outline) if outline else dict(width=0)
    if opacity is None:
        opacity = 0.95 if outline else 0.55
    return go.Scattergl(
        x=x, y=y, mode="markers", name=name,
        marker=dict(size=size, color=color, opacity=opacity,
                    symbol=symbol, line=line),
        customdata=customdata,
        hovertemplate=hovertemplate,
        hoverinfo=None if hover_on else "skip",
        showlegend=False,
    )


# How far the residual buckets recede in the ARCHS4 cloud. The tissue vocabulary
# has a long tail, so "Other" legitimately holds about a third of the corpus, and
# at full weight a third of the map paints grey *over* the categories that do
# carry information. Receding it is the honest way to fix that: the legend still
# reports the true count, nothing is hidden, but points with no usable label stop
# competing with points that have one. Adding more palette hues would be the
# wrong fix - the eleven are already at the limit of what stays separable on a
# scatter, and the dataviz rule is to fold the tail into Other, not to invent
# colours for it.
RESIDUAL_OPACITY = 0.26
RESIDUAL_SIZE_SCALE = 0.82


def _categorical_traces(coords, idx, codes, legend, is_3d, size, symbol,
                        outline, hover_lines=(), customdata=None, opacity=None,
                        recede_residual=False):
    """One trace per display category, coloured from the shared legend mapping.

    ``codes`` holds each point's legend slot, so selecting a category is one
    vectorized integer compare rather than 942,563 Python string comparisons.

    Residual categories are emitted FIRST so they sit underneath. Plotly paints
    traces in the order they are added, and with the residual bucket last its
    ~308,000 grey glyphs were drawn on top of every coloured category - the map
    read as grey even where it was not.
    """
    rows_for = (lambda sel: None) if customdata is None else (
        lambda sel: [customdata[i] for i in np.where(sel)[0]])

    ordered = sorted(range(len(legend)),
                     key=lambda s: not colorby.is_residual(legend[s]["label"]))
    traces = []
    for slot in ordered:
        row = legend[slot]
        sel = codes == slot
        if not sel.any():
            continue
        residual = recede_residual and colorby.is_residual(row["label"])
        # A residual category recedes to RESIDUAL_OPACITY - but when the whole
        # corpus is already dimmed behind a retrieval, taking the min keeps it
        # receded rather than letting 0.26 make "Other" the *brightest* thing
        # on a 0.16 map.
        if residual:
            point_opacity = min(RESIDUAL_OPACITY, opacity) if opacity is not None \
                else RESIDUAL_OPACITY
        else:
            point_opacity = opacity
        traces.append(_scatter(
            coords, idx[sel], row["color"], is_3d,
            size * (RESIDUAL_SIZE_SCALE if residual else 1.0), symbol, outline,
            name=row["label"], hover_lines=hover_lines, customdata=rows_for(sel),
            opacity=point_opacity))
    return traces


def _osdr_customdata(codes: np.ndarray, legend: list[dict]) -> list[list]:
    """Rows of [sample_key, category] for the OSDR overlay hover.

    A slot of NOT_COVERED_CODE means this field says nothing about the sample,
    which the hover shows as "-" rather than inventing a category for it.
    """
    meta = data.osdr_metadata()
    keys = (meta["sample_key"].astype(str).to_numpy()
            if "sample_key" in meta.columns
            else np.array([f"OSDR {i}" for i in range(len(meta))]))
    return [[str(k), legend[c]["label"] if c >= 0 else "-"]
            for k, c in zip(keys, codes.tolist())]


def _retrieval_traces(coords, is_3d, retrieval) -> list:
    """The query and its hits, drawn where they actually sit in the space.

    `retrieval` is the payload the retrieval view stores: a `query_point` index
    into the global point order and a list of `hit_points`, which are ARCHS4
    memmap rows and therefore already point indices - ARCHS4 occupies rows
    0..n_archs4-1. No lookup, no join.

    **No lines are drawn between the query and its hits, deliberately.**
    Connecting them would be the obvious and the most striking choice, and it
    would assert something false. The retrieval ranks by cosine distance in
    512 dimensions; this map is a 2-D projection that does not preserve those
    distances, so a drawn edge would invite reading its length as similarity
    when a rank-1 hit can easily land further away on screen than a rank-5 one.
    Where the hits fall is worth seeing precisely because it is *not* the
    ranking - it is what the projection did with it.

    **Two cohorts can be drawn at once**, when a comparison was run. Which
    cohort a *member* belongs to is carried by fill hue; which cohort retrieved
    a *hit* is carried by ring shape, never by hue. That split is measured, not
    stylistic: no hue clears 3:1 against the worst categorical tissue bucket
    (the comparison network's own three measure 1.00 to 1.07), which is the
    same finding that made the ring white in the first place, while a member is
    a filled mark whose white outline already guarantees its contrast. A hit
    retrieved by both cohorts receives both traces and therefore draws as a
    ring inscribed in a square, so the shared set is emergent rather than
    computed and cannot disagree with the number the status banner quotes.
    """
    n = len(coords)
    cohorts = retrieval.get("cohorts")
    if not cohorts:
        # A payload written before cohorts existed, or by a caller assembling
        # the flat keys directly. Treat it as the single query it describes.
        cohorts = [{"role": "a", "label": retrieval.get("query_label") or "",
                    "hit_points": retrieval.get("hit_points") or [],
                    "hit_labels": retrieval.get("hit_labels") or [],
                    "hit_scores": retrieval.get("hit_scores") or [],
                    "query_points": (retrieval.get("query_points")
                                     or ([retrieval["query_point"]]
                                         if retrieval.get("query_point") is not None
                                         else []))}]
    comparing = len(cohorts) > 1
    traces = []

    # Non-gl traces: at most k + 2 points, and Scattergl does not centre
    # `markers+text` reliably.
    Scatter = go.Scatter3d if is_3d else go.Scatter

    # Scatter3d takes a much smaller symbol set than Scatter and rejects the
    # rest outright rather than falling back - `star` raises, which took the
    # whole figure callback down with a 500 the first time 3-D was opened with
    # a retrieval showing. `diamond` is the closest available mark that is
    # still not a plain circle, so the query stays distinguishable from a hit.
    # Both hit symbols below are in Scatter3d's vocabulary, so a hit encodes
    # identically in 2-D and 3-D.
    query_symbol = "diamond" if is_3d else "star"
    # `cliponaxis` is a 2-D-only property and is likewise a hard error in 3-D.
    text_extras = {} if is_3d else {"cliponaxis": False}
    # Scatter3d draws a marker of a given `size` considerably larger than
    # Scattergl does, which is why the corpus layers already halve theirs. The
    # overlay needs the same treatment: at full size the query halo rendered as
    # a teal disc that dominated the scene instead of marking a point in it.
    scale = 0.5 if is_3d else 1.0

    def _xyz(points):
        arr = np.asarray(points)
        out = dict(x=coords[arr, 0], y=coords[arr, 1])
        if is_3d:
            out["z"] = coords[arr, 2]
        return out

    def _style(role):
        if role == "b":
            return (theme.RETRIEVAL_QUERY_B, theme.RETRIEVAL_QUERY_B_RGB,
                    theme.RETRIEVAL_HIT_SYMBOL_B, theme.RETRIEVAL_HIT_SIZE_B)
        return (theme.RETRIEVAL_QUERY, theme.RETRIEVAL_QUERY_RGB,
                theme.RETRIEVAL_HIT_SYMBOL, theme.RETRIEVAL_HIT_SIZE)

    prepared = []
    for cohort in cohorts:
        hits = [int(i) for i in cohort.get("hit_points") or [] if 0 <= int(i) < n]
        members = [int(i) for i in cohort.get("query_points") or []
                   if 0 <= int(i) < n]
        prepared.append((cohort, hits, members, _style(cohort.get("role", "a"))))

    # --- Layer 1: halos, behind everything -----------------------------------
    for _cohort, _hits, members, (_colour, rgb, _sym, _size) in prepared:
        if not members:
            continue
        # A wide, faint ring so the query is findable in 942,563 points without
        # a glyph big enough to misrepresent where the sample actually is. It
        # narrows and fades as the cohort grows, so a 38-animal cohort reads as
        # a constellation rather than compositing into one disc.
        pooled = len(members) > 1
        halo = (theme.RETRIEVAL_QUERY_HALO_SIZE_POOLED if pooled
                else theme.RETRIEVAL_QUERY_HALO_SIZE)
        rgba = theme.halo_rgba(rgb, len(members))
        traces.append(Scatter(
            **_xyz(members), mode="markers", name="query halo",
            marker=dict(size=halo * scale, symbol="circle-open", color=rgba,
                        line=dict(width=1.5, color=rgba)),
            hoverinfo="skip", showlegend=False))

    # --- Layer 2: hit rings, larger symbol last so it cannot occlude ---------
    for cohort, hits, members, (_colour, _rgb, symbol, size) in prepared:
        if not hits:
            continue
        labels = cohort.get("hit_labels") or []
        scores = cohort.get("hit_scores") or []
        name = str(cohort.get("label") or "")
        ranks = _map_ranks(coords, hits, members)
        rows = []
        for i, point in enumerate(hits):
            gsm = str(labels[i]) if i < len(labels) else ""
            score = f"{float(scores[i]):.4f}" if i < len(scores) else "-"
            mr = ranks[i]
            who = f"{name}  ·  " if comparing and name else ""
            rows.append([
                gsm,
                f"{who}512-d rank {i + 1} of {len(hits)} retrieved"
                f"  ·  cosine {score}",
                (f"map rank {mr:,} of {n:,} from the nearest pooled member"
                 if mr is not None else ""),
            ])
        # Numerals are dropped in a comparison: two competing rank sets over the
        # same few hundred pixels is illegible, and the hover says strictly more
        # (a shared hit names both cohorts, both ranks and both cosines).
        marker_mode = "markers" if comparing else "markers+text"
        numerals = ([] if comparing else
                    [str(i + 1) if i < theme.RETRIEVAL_MAX_NUMERALS else ""
                     for i in range(len(hits))])
        traces.append(Scatter(
            **_xyz(hits), mode=marker_mode, name="retrieved hit",
            text=numerals, textposition="top center",
            textfont=dict(size=9, color=theme.RETRIEVAL_HIT_RING,
                          family="JetBrains Mono, SF Mono, monospace"),
            marker=dict(size=size * scale, symbol=symbol,
                        color=theme.RETRIEVAL_HIT_RING,
                        line=dict(width=theme.RETRIEVAL_HIT_LINE,
                                  color=theme.RETRIEVAL_HIT_RING)),
            customdata=rows,
            hovertemplate=("<b>%{customdata[0]}</b><br>%{customdata[1]}"
                           "<br>%{customdata[2]}<extra></extra>"),
            showlegend=False, **text_extras))

    # --- Layer 3: the members themselves, on top ----------------------------
    for cohort, _hits, members, (colour, _rgb, _sym, _size) in prepared:
        if not members:
            continue
        label = str(cohort.get("label") or "OSDR query")
        pooled = len(members) > 1
        # Members shrink a little when there are several, so a 38-animal cohort
        # reads as a constellation rather than as a blot.
        size = theme.RETRIEVAL_QUERY_SIZE * scale * (0.7 if pooled else 1.0)
        traces.append(Scatter(
            **_xyz(members), mode="markers", name="query",
            marker=dict(size=size, symbol=query_symbol, color=colour,
                        line=dict(width=2, color="#ffffff")),
            customdata=[[label]] * len(members),
            hovertemplate=("<b>%{customdata[0]}</b><br>"
                           + ("one of the pooled cohort samples"
                              if pooled else "the query sample")
                           + "<extra></extra>"),
            showlegend=False))
    return traces


def _map_ranks(coords, hits: list[int], members: list[int]) -> list:
    """Where each hit sits in the map's *own* ordering, per hit.

    This is the number that keeps the picture honest: the retrieval ranks by
    cosine in 512 dimensions and the map is a 2-D shadow of that space, so the
    two orderings disagree, often wildly. Showing both ranks side by side in the
    hover states the disagreement instead of leaving a reader to infer rank from
    what is nearest.

    Each hit is ranked from **the nearest member of the cohort that retrieved
    it**. The previous version measured every hit from `query_points[0]`, which
    for a pooled cohort is whichever animal came first in metadata order - an
    arbitrary choice that was merely unprincipled for one cohort and would be
    plainly wrong for the second arm of a comparison.

    Cost is one full-corpus pass per *winning* member rather than per member:
    the nearest member is found first from a `len(hits) x len(members)` block,
    which is tiny, and only the members that actually win are then swept.
    """
    if not hits or not members:
        return [None] * len(hits)
    h = coords[np.asarray(hits), :]
    m = coords[np.asarray(members), :]
    # (hits, members) squared distances, then the winning member per hit.
    d = ((h[:, None, :] - m[None, :, :]) ** 2).sum(axis=2)
    nearest = np.asarray(members)[d.argmin(axis=1)]

    ranks: list[int | None] = [None] * len(hits)
    sweep: dict[int, np.ndarray] = {}
    for i, point in enumerate(hits):
        origin = int(nearest[i])
        d2 = sweep.get(origin)
        if d2 is None:
            delta = coords - coords[origin]
            d2 = np.einsum("ij,ij->i", delta, delta)
            sweep[origin] = d2
        ranks[i] = int((d2 < d2[point]).sum())
    return ranks


def build_figure(method, dims, color_by, layers, budget, viewport,
                 retrieval=None):
    is_3d = dims == "3d"
    coords = data.coords(method, dims)
    n_archs4, n_osdr, total = data.counts()
    fig = go.Figure()
    spec = colorby.get(color_by)
    legend_data = {"title": spec.label, "items": []}
    badges: list[str] = []

    # A retrieval is being shown, so the corpus becomes the backdrop it is for
    # that question. Dimming the whole map rather than enlarging the twelve
    # points that matter keeps every glyph at a size that still means "one
    # sample sits here".
    showing_retrieval = bool(retrieval and (retrieval.get("hit_points")
                                            or retrieval.get("query_point") is not None))
    dim = theme.RETRIEVAL_DIM_ARCHS4 if showing_retrieval else None
    # OSDR stays brighter than ARCHS4: 2,108 diamonds at the cloud's opacity
    # vanish entirely, and losing the spaceflight corpus is the one thing this
    # map may not do.
    dim_osdr = theme.RETRIEVAL_DIM_OSDR if showing_retrieval else None

    if coords.shape[0] == 0:
        fig.update_layout(**theme.base_figure_layout(is_3d))
        fig.add_annotation(text=f"{method.upper()} coordinates not built yet",
                           showarrow=False, font=dict(color=theme.PLOT_TEXT, size=15))
        return fig, legend_data, [f"{method.upper()} not available"]

    coords_xy = coords[:, :2]
    codes, legend = _colour_plan(spec.key)

    covers_archs4 = colorby.covers_corpus(spec.key, colorby.ARCHS4)

    # Slot codes of the points actually drawn into a legend category, gathered
    # per layer so the legend can report what is on screen rather than what the
    # whole corpus holds. Context and highlight points carry no legend row, so
    # they are deliberately not gathered.
    drawn: list[np.ndarray] = []

    # --- Layer 1: ARCHS4 background ----------------------------------------
    if "archs4" in layers:
        idx = _archs4_sample_indices(coords_xy, int(budget), viewport)
        if is_3d and len(idx) > SCATTER3D_ARCHS4_CAP:
            idx = np.random.default_rng(1).choice(idx, SCATTER3D_ARCHS4_CAP,
                                                  replace=False)
        if covers_archs4:
            archs4_codes = codes[idx]
            for trace in _categorical_traces(coords, idx, archs4_codes, legend,
                                             is_3d, ARCHS4_SIZE, "circle", None,
                                             recede_residual=True, opacity=dim):
                fig.add_trace(trace)
            drawn.append(archs4_codes)
            badges.append(f"ARCHS4 live: <b>{len(idx):,}</b>")
        else:
            # These points have no value under this field, so they are drawn as
            # scenery: one faint colour, no legend row, nothing that could be
            # read as a category. A uniform grey glyph *in the palette* is what
            # made 99.8% of the map look like measured-and-empty.
            fig.add_trace(_scatter(coords, idx, theme.ARCHS4_CONTEXT, is_3d,
                                   ARCHS4_CONTEXT_SIZE, "circle", None,
                                   name="ARCHS4 (context)",
                                   opacity=min(0.35, dim) if dim else 0.35))
            badges.append(f"ARCHS4: <b>context only</b> · {spec.label} is OSDR-only")

    # --- Layer 2: OSDR overlay ---------------------------------------------
    if "osdr" in layers and n_osdr > 0:
        osdr_global = np.arange(n_archs4, n_archs4 + n_osdr)
        osdr_codes = codes[osdr_global]
        rows = _osdr_customdata(osdr_codes, legend)
        if colorby.covers_corpus(spec.key, colorby.OSDR):
            for trace in _categorical_traces(
                    coords, osdr_global, osdr_codes, legend, is_3d, OSDR_SIZE,
                    theme.OSDR_SYMBOL, theme.OSDR_OUTLINE,
                    hover_lines=OSDR_HOVER, customdata=rows, opacity=dim_osdr):
                fig.add_trace(trace)
            drawn.append(osdr_codes)
        else:
            # An ARCHS4-only field. OSDR keeps its distinct glyph in a single
            # warm highlight so the spaceflight corpus stays locatable without
            # borrowing a colour that means something else in the legend.
            fig.add_trace(_scatter(coords, osdr_global, theme.OSDR_HIGHLIGHT,
                                   is_3d, OSDR_SIZE, theme.OSDR_SYMBOL,
                                   theme.OSDR_OUTLINE, name="OSDR",
                                   hover_lines=OSDR_HOVER, customdata=rows,
                                   opacity=dim_osdr))
        badges.append(f"OSDR: <b>{n_osdr:,}</b>")

    # The legend reports the points that were actually plotted above, not the
    # whole-corpus tallies the colour plan ranked with.
    legend_data["items"] = _legend_with_drawn_counts(legend, drawn)

    # --- Layer 3: the retrieval, on top of everything ----------------------
    if showing_retrieval:
        for trace in _retrieval_traces(coords, is_3d, retrieval):
            fig.add_trace(trace)
        n_hits = len(retrieval.get("hit_points", []))
        cohorts = retrieval.get("cohorts") or []
        if len(cohorts) > 1:
            # The badge reports what is drawn right now, so it has to count both
            # arms and say how many marks carry both rings - the number the
            # whole comparison is about.
            n_shared = len(retrieval.get("shared_points") or [])
            badges.append(
                f"Showing <b>2</b> cohorts · <b>{n_hits}</b> hits · "
                f"<b>{n_shared}</b> retrieved by both")
        else:
            badges.append(
                f"Showing retrieval: <b>{n_hits}</b> hit{'s' if n_hits != 1 else ''}")

    fig.update_layout(**theme.base_figure_layout(is_3d))
    return fig, legend_data, badges
