"""The Dash layout: header, left control rail, and the plot.

The chrome matches Bridge RNA (light scientific instrument); the plot canvas is
dark navy. Controls are declarative here; behaviour lives in callbacks.py.

The shell is two columns. It used to be three, with a right-hand panel reporting
statistics for a lasso selection; that feature is gone and the plot took the
space back.
"""

from __future__ import annotations

from dash import dcc, html

from . import colorby, data, render, theme


def color_by_label(value: str) -> str:
    """The human-facing name for a color-by, for the legend heading."""
    return colorby.get(value).label


# The Projection control, in the order it offers them: the two neighbour-graph
# methods first, then the linear one. Every entry is derived from this - the
# options, their disabled state, and which one a fresh view opens on - so a
# fourth projection is one line here rather than four edits that can disagree.
METHOD_LABELS = (("umap", "UMAP"), ("tsne", "t-SNE"), ("pca", "PCA"))


def method_options() -> list[dict]:
    """The Projection pills, each disabled when its coordinates are not built.

    Disabled-and-visible rather than hidden, for the reason `colorby` gives for
    an unavailable color-by: hiding it makes the app look like it never had the
    feature. t-SNE is the one most likely to be missing, because it is the most
    expensive stage in the build and the only one that can be skipped without
    skipping the rest.
    """
    return [{"label": label, "value": key,
             "disabled": not data.method_available(key)}
            for key, label in METHOD_LABELS]


def default_method() -> str:
    """The first projection that is actually built.

    PCA is the floor rather than a fallback that might also be missing:
    `preflight.APP_REQUIRED` refuses to start the app without
    coords_pca2.parquet, so the `next(...)` default can never be reached on a
    machine that got this far.
    """
    return next((k for k, _ in METHOD_LABELS if data.method_available(k)), "pca")


def projection_params(method: str, dims: str) -> list[tuple[str, str, str]]:
    """(prefix, value, suffix) triples stating how the active projection was fit.

    ``value`` is the payload and is the only part set in mono tabular figures;
    it is empty for chips that are a bare word rather than a measurement, since
    setting "cosine" in a numeral font is just noise.

    Read off `projection_stats.json`, never from constants duplicated here: the
    rail describes the coordinates on screen, and a rail that recites what the
    build *would* have done is worse than one that says nothing, because it
    stays confident while the cache goes stale. A key the record does not carry
    drops its pair rather than rendering blank, so an older cache shows fewer
    parameters instead of a row of empty slots.

    `dims` is a real input, not decoration. t-SNE's negative-gradient method
    differs between the two: openTSNE's interpolation accelerator refuses more
    than two output dimensions, so the 3-D map is a Barnes-Hut layout and the
    2-D one is not. Naming one method for both would be the same class of lie
    the retrieval banner told when it announced every cached result as
    subprocess output.
    """
    # Say nothing about coordinates that are not there. The build record is
    # written before the fit it describes finishes, so an interrupted run leaves
    # a complete `tsne_*` record beside a missing coords_tsne3.parquet - and the
    # rail would then assert a Barnes-Hut layout over all 942,563 points next to
    # a plot reading "coordinates not built yet". Reading the record instead of
    # the code's constants does not help if the record outlives the artifact.
    if not data.coords_available(method, dims):
        return []

    s = data.projection_stats()
    out: list[tuple[str, str, str]] = []

    def measure(prefix: str, value, suffix: str = "") -> None:
        """A key=value parameter; skipped entirely when the record lacks it."""
        if value is not None and value != "":
            out.append((prefix, str(value), suffix))

    def word(text) -> None:
        """A bare descriptive chip, with no numeral to set apart."""
        if text:
            out.append((str(text), "", ""))

    def init_chip(record_value) -> None:
        """Name the starting layout from the record, not from a constant.

        The build can start UMAP from a spectral layout or a PCA one, and which
        it was is the difference between the map separating the corpora and not,
        so the chip has to follow the record rather than assume. The stored
        string is verbose ("spectral (normalized Laplacian, ...)"); the chip
        wants its first word.
        """
        if record_value:
            head = str(record_value).split()[0].lower()
            word("spectral init" if head.startswith("spectral") else "PCA init")

    if method == "umap":
        measure("n_neighbors=", s.get("umap_neighbors"))
        measure("min_dist=", s.get("umap_min_dist"))
        word(s.get("umap_metric"))
        init_chip(s.get("umap_init"))
    elif method == "tsne":
        measure("perplexity=", s.get("tsne_perplexity"))
        measure("exaggeration=", s.get("tsne_early_exaggeration"))
        word(s.get("tsne_metric"))
        init_chip(s.get("tsne_init"))
        word(s.get(f"tsne{dims[0]}_negative_gradient"))
    elif method == "pca":
        if s.get("pca_fit"):
            word("exact eigendecomposition")
        pc1 = s.get("pca_pc1_pct")
        if pc1 is not None:
            measure("PC1 ", f"{float(pc1):.1f}%")

    # The shared closer. All three are fit on the whole corpus, and that is the
    # property most worth stating: it is what they stopped approximating, and
    # the count is the one number that proves it.
    total = s.get("total")
    if total:
        measure("fit on all ", f"{int(total):,}", " points")
    return out


def projection_params_children(method: str, dims: str):
    """The parameter readout's spans, one per parameter.

    One span each so a 236 px rail wraps *between* parameters: an
    "n_neighbors=" stranded above its "30" reads as two facts rather than one.
    The separator between them is drawn by CSS for the same reason - a literal
    "·" in the text is one more place the line is allowed to break.
    """
    children = []
    for prefix, value, suffix in projection_params(method, dims):
        parts: list = [prefix] if prefix else []
        if value:
            parts.append(html.B(value))
        if suffix:
            parts.append(suffix)
        children.append(html.Span(parts, className="bm-param"))
    return children


def budget_options(dims: str) -> list[dict]:
    """The ARCHS4 point-budget tiers, which depend on the dimensionality.

    In 2-D the glyph cloud can carry the whole corpus, so the tiers climb to
    "All". In 3-D the cloud is capped at ``render.SCATTER3D_ARCHS4_CAP`` so
    rotation stays smooth, so the tiers stop there rather than offering a "500k"
    or "All" pill the 3-D view would silently redraw as 40,000 - a control that
    lies about what it does is worse than one that offers less.
    """
    n_archs4, _, _ = data.counts()
    if dims == "3d":
        cap = render.SCATTER3D_ARCHS4_CAP
        step = cap // 4
        vals = [step, step * 2, step * 3, cap]
        return [{"label": f"{v // 1000}k", "value": str(v)} for v in vals]
    return [
        {"label": "100k", "value": "100000"},
        {"label": "250k", "value": "250000"},
        {"label": "500k", "value": "500000"},
        {"label": "All", "value": str(n_archs4)},
    ]


def default_budget(dims: str) -> str:
    """The tier a fresh view of ``dims`` starts on: everything in 2-D, the cap
    in 3-D."""
    n_archs4, _, _ = data.counts()
    return str(render.SCATTER3D_ARCHS4_CAP) if dims == "3d" else str(n_archs4)


def resolve_budget(dims: str, current: str | None) -> str:
    """The budget value to use for ``dims``, preserving ``current`` if it is
    still one of that dimensionality's tiers and falling back to the default
    otherwise. This is what a dimensionality switch runs so a 3-D "All" cannot
    survive into a view that cannot honour it."""
    valid = {o["value"] for o in budget_options(dims)}
    return current if current in valid else default_budget(dims)


def _segmented(id_: str, options: list[dict], value: str):
    """A radio group styled as a segmented pill control.

    Only the container carries a class. Dash renders each option as a label with
    its own structural classes and marks the chosen one `.selected`, and
    `labelClassName` lands on an inner text span rather than the label itself -
    so the stylesheet targets Dash's classes scoped under `.bm-seg` instead.
    """
    return dcc.RadioItems(id=id_, options=options, value=value, className="bm-seg")


def control_rail() -> html.Div:
    n_archs4, n_osdr, _ = data.counts()

    return html.Div(
        className="bm-rail",
        children=[
            # The parameter readout sits directly under the control it
            # describes, which is the same rule the color-by coverage readout
            # below follows: the fact that qualifies a control belongs against
            # that control, not in a tooltip and not in the plot badges, which
            # report what is drawn right now and change on every zoom.
            html.Div(className="bm-group", children=[
                html.Div("Projection", className="bm-group-label"),
                _segmented("method", method_options(), default_method()),
                html.Div(id="method-params", className="bm-params"),
            ]),
            html.Div(className="bm-group", children=[
                html.Div("Dimensions", className="bm-group-label"),
                _segmented("dims", [
                    {"label": "2D", "value": "2d"},
                    {"label": "3D", "value": "3d"},
                ], "2d"),
            ]),
            # The color-by group carries its own coverage readout, so the answer
            # to "how much of this map is this field actually coloring?" sits
            # next to the control that decides it rather than being inferred
            # from how grey the plot looks.
            html.Div(className="bm-group", children=[
                html.Div("Color by", className="bm-group-label"),
                dcc.Dropdown(
                    id="color-by",
                    options=colorby.menu_options(),
                    value=colorby.default_key(),
                    clearable=False,
                    className="bm-dropdown",
                ),
                html.Div(id="coverage", className="bm-coverage"),
                html.Div(id="color-by-hint", className="bm-hint"),
            ]),
            html.Div(className="bm-group", children=[
                html.Div("Layers", className="bm-group-label"),
                dcc.Checklist(
                    id="layers",
                    options=[
                        {"label": f"ARCHS4 cloud ({n_archs4:,})", "value": "archs4"},
                        {"label": f"OSDR overlay ({n_osdr:,})", "value": "osdr"},
                    ],
                    value=["archs4", "osdr"],
                    className="bm-checklist",
                ),
            ]),
            # In 2-D the glyph sample can carry the whole corpus (the density
            # raster it used to sit above is gone, so the sample is the only
            # thing drawing those points). In 3-D the tiers stop at the rotation
            # cap; budget_options() decides, and callbacks.sync_budget_to_dims
            # swaps the tiers when the dimensionality changes.
            html.Div(className="bm-group", children=[
                html.Div("ARCHS4 point budget", className="bm-group-label"),
                _segmented("budget", budget_options("2d"), default_budget("2d")),
            ]),
            # Clicking an OSDR point offers a retrieval for it. Hidden until
            # something is clicked; the map is read rather than driven, so this
            # is an offer that appears in response to interest, not a control
            # sitting on the rail waiting to be understood.
            html.Div(id="picked-group", className="bm-group",
                     style={"display": "none"}, children=[
                html.Div("Selected sample", className="bm-group-label"),
                html.Div(id="picked-label", className="bm-picked"),
                dcc.Link(id="picked-link", href="/", className="bm-button",
                         children="Retrieve its Earth analogs →"),
            ]),
            # Shown only when there is a retrieval to show. Declared here
            # rather than created by a callback so Dash can validate the
            # callback graph against it at startup: a component that exists
            # only as callback output fails silently at runtime if its id is
            # mistyped, which is the same reason the legend is built this way.
            html.Div(id="retrieval-group", className="bm-group",
                     style={"display": "none"}, children=[
                html.Div("Your retrieval", className="bm-group-label"),
                dcc.Checklist(
                    id="show-retrieval",
                    options=[{"label": " Show it on the map", "value": "on"}],
                    value=["on"],
                    className="bm-checklist",
                ),
                html.Div(id="retrieval-summary", className="bm-hint"),
                html.Button("Frame the retrieval", id="frame-retrieval",
                            n_clicks=0, className="bm-button"),
                # What each mark means now lives in the key on the plot, beside
                # the marks. What stays here is the one thing the picture cannot
                # show: that a hit carries two different orderings, and the
                # nearer of two hits on screen is not the better one.
                # Not "into two". This div is static and the group stays on
                # screen in 3-D, where that would be a false statement to a
                # reader looking at three axes - the same class of error as a
                # parameter chip naming one t-SNE gradient method for both
                # dimensionalities. The claim is true of any projection, so it
                # is made that way rather than made dims-aware.
                html.Div(
                    "Hover a hit for its rank in the search and its rank on "
                    "the map. The two disagree: a projection cannot preserve "
                    "512-dimensional distances.",
                    className="bm-hint",
                ),
            ]),
        ],
    )


# --- The key on the plot -----------------------------------------------------
#
# Every mark the map can draw is decoded here, in one panel, beside the marks.
# Before this the map explained exactly one of its four encodings: the floating
# legend keyed hue for the corpus, and everything else - that a diamond is an
# OSDR sample, that a white ring is a retrieved hit, that a round ring and a
# square ring are two different cohorts, that a ring inside a square is a hit
# both of them found - was either prose on a rail 800 px from the glyph it
# described, or nothing at all. A viewer who did not already know the scheme
# could only decode the picture by hovering it one point at a time.
#
# The hues below are read from `theme`, never mirrored into the stylesheet, so
# a change to RETRIEVAL_QUERY moves the plot and the key together. The
# stylesheet owns the *shapes*, because a shape has no Python constant to drift
# from - and the white it draws the hit rings in is the same white for the same
# measured reason (theme.RETRIEVAL_HIT_RING: no hue clears 3:1 against the worst
# categorical bucket on the navy canvas).

#: What the corpus glyph shapes mean. Neutral-filled, because these marks take
#: their color from the color-by and the shape is the whole message.
CORPUS_KEY = (("archs4", "ARCHS4"), ("osdr", "OSDR"))
CORPUS_KEY_COLOR = "#8ea6c9"


def _key_glyph(kind: str, color: str | None = None):
    """One mark, drawn as the mark rather than named in words.

    `kind` selects the shape from the stylesheet; `color` is the fill for the
    shapes that have one, applied inline so it can come straight from `theme`.
    """
    fill = ([html.Span(className="bm-key-glyph-fill",
                       style={"background": color})] if color else [])
    return html.Span(className=f"bm-key-glyph is-{kind}", children=fill)


def _key_row(kind: str, label: str, count=None, color: str | None = None,
             hidden: bool = False):
    """A glyph, what it means, and how many are on screen right now.

    The count follows the color legend's rule and reports what is *drawn*,
    which for a retrieval is the whole of it - nothing about the overlay is
    sampled or budgeted. It is omitted entirely rather than rendered as an
    empty slot for the rows that are a shape key and have nothing to count.
    """
    tail = ([html.Span(f"{count:,}" if isinstance(count, int) else str(count),
                       className="bm-key-count")]
            if count is not None else [])
    return html.Div(
        className="bm-key-row" + (" is-hidden" if hidden else ""),
        children=[_key_glyph(kind, color),
                  html.Span(label, className="bm-key-label"), *tail])


#: A member mark is a star in 2-D and a diamond in 3-D, because `Scatter3d`
#: rejects `star` outright. The key follows the plot rather than picking one
#: and being wrong in the other view.
def _member_shape(dims: str) -> str:
    return "diamond" if dims == "3d" else "star"


def retrieval_key_children(overlay: dict | None, roles: tuple[str, ...],
                           dims: str) -> list:
    """The retrieval section of the key: every overlay mark, decoded.

    `overlay` is the **full** payload, including an arm the user has unticked,
    and `roles` is what is actually ticked. A hidden arm keeps its rows, receded
    and marked "hidden", so the hue it owns is still named - dropping them would
    leave a gold glyph on screen a moment later with nothing to look it up in,
    and would make the key silently disagree with the ticks above it.

    **A comparison is grouped by role, not by cohort, and that is the whole
    design.** The encoding has two factors - which cohort, and member versus
    hit - and they do not use the same channel: hue names the cohort on a
    member, ring shape names it on a hit. Listing each cohort's two marks
    together buries that. Listing the two member rows adjacent and the two hit
    rows adjacent shows it: one pair differs only in hue, the next differs only
    in shape, and the reader sees each channel vary with the other held fixed.
    No sentence has to assert the rule, which is what makes this readable at a
    glance rather than after a paragraph.

    The single-query state must not pay for the comparison: a plain search gets
    two rows and no headings, because with one query on screen there is nothing
    for a name or a group to distinguish it from.
    """
    if not overlay or not overlay.get("cohorts"):
        return []

    cohorts = overlay["cohorts"]
    shape = _member_shape(dims)

    if len(cohorts) == 1:
        c = cohorts[0]
        # The single tick hides the whole overlay, so the rows recede exactly as
        # a comparison's do. Missing this was a real defect: unticking "Show it
        # on the map" left the plot with no star and no ring while the key went
        # on counting one member and five hits, which is the failure the count
        # rule exists to prevent - and it happened in the commonest state of all.
        hidden = c["role"] not in roles
        rows = []
        n_members = len(c["query_points"])
        if n_members:
            # An uploaded sample was never embedded into this map, so it has no
            # coordinate, draws no member mark, and gets no row. "pooled member"
            # is earned by k >= 2, not by the code path: for a single-sample
            # search the one member *is* the query.
            rows.append(_key_row(
                shape, "pooled member" if n_members > 1 else "the query sample",
                "hidden" if hidden else n_members, theme.RETRIEVAL_QUERY,
                hidden=hidden))
        rows.append(_key_row("hit-a", "retrieved hit",
                             "hidden" if hidden else len(c["hit_points"]),
                             hidden=hidden))
        return [html.Div(className="bm-key bm-key--retrieval", children=rows)]

    hues = {"a": theme.RETRIEVAL_QUERY, "b": theme.RETRIEVAL_QUERY_B}
    hit_shapes = {"a": "hit-a", "b": "hit-b"}

    def arm(c, kind, count, color=None):
        hidden = c["role"] not in roles
        return _key_row(kind, c["label"] or "cohort",
                        "hidden" if hidden else count, color, hidden=hidden)

    rows = [html.Div("Pooled members", className="bm-key-head")]
    rows += [arm(c, shape, len(c["query_points"]), hues[c["role"]])
             for c in cohorts]
    rows.append(html.Div("Retrieved hits", className="bm-key-head"))
    rows += [arm(c, hit_shapes[c["role"]], len(c["hit_points"]))
             for c in cohorts]

    if all(c["role"] in roles for c in cohorts):
        # Last row of the hits group, where it belongs: it is not a third
        # cohort, it is one point wearing both cohorts' rings. Drawn only when
        # both arms are, because with one hidden the doubled mark cannot exist.
        rows.append(_key_row("hit-both", "retrieved by both",
                             len(overlay.get("shared_points") or [])))
    return [html.Div(className="bm-key bm-key--retrieval", children=rows)]


def corpus_key_children(layers: list | None) -> list:
    """The shape key for the two corpora, under the color list.

    Hue is what the color legend keys; this is the other channel, and it was
    the one nothing on screen explained. A diamond has meant "one of the 2,108
    NASA spaceflight samples" since the map was built, stated in the render
    module's docstring and nowhere a viewer could read it.

    Only drawn layers get a row, because a key is what you are looking at.

    It deliberately does not take the color-by. A row here says "this shape is
    an ARCHS4 sample", which stays true whether that corpus is drawn in eleven
    tissue hues or in one faint context color - and the context state is
    already stated twice, by the plot badge and by the coverage readout. Taking
    the color-by would imply this row varies with it, and invariant 5 is
    specifically that the context cloud gets no *category* treatment: a swatch,
    a hue from the palette, a count. A shape, with none of those, is not one.
    """
    drawn = set(layers or [])
    rows = [_key_row(f"corpus-{key}", label, color=CORPUS_KEY_COLOR)
            for key, label in CORPUS_KEY if key in drawn]
    if not rows:
        return []
    return [html.Div(className="bm-key bm-key--corpus", children=rows)]


def legend_panel() -> html.Div:
    """The floating key: what is drawn on top, what the colors mean, what the
    shapes mean.

    Every part is declared statically and filled by callbacks, rather than the
    whole panel being rebuilt as callback output. Dash validates its callback
    graph against the initial layout, so components that only ever exist as
    callback output cannot be checked - a typo in one of their ids fails
    silently at runtime instead of loudly at startup.

    The three sections are ordered by how transient each is. The retrieval sits
    at the top because it is the thing the reader just asked for and the thing
    that will be gone next search; the color list is the standing state of the
    map and is the only part that scrolls; the corpus shapes are a footnote and
    never change. Only the middle one can grow, so only the middle one scrolls.
    """
    return html.Div(
        id="legend", className="bm-legend", style={"display": "none"},
        children=[
            html.Div(id="legend-retrieval"),
            html.Div(id="legend-color", className="bm-legend-section", children=[
                html.Div(id="legend-title", className="bm-legend-title"),
                dcc.Input(id="legend-search", className="bm-legend-search",
                          placeholder="filter categories…", type="text",
                          debounce=False, style={"display": "none"}),
                html.Div(id="legend-list", className="bm-legend-list"),
            ]),
            html.Div(id="legend-corpus"),
        ],
    )


def build_view() -> html.Div:
    """The map view, everything below the shared header.

    The map used to be its own app and drew its own header, with a corpus count
    strip reading "corpus 942,563 · ARCHS4 940,455 · OSDR 2,108". The shell
    draws the header now and that function is deleted rather than kept unused:
    both of its counts already appear on the control rail, in the Layers group,
    beside the toggle that decides whether each corpus is drawn at all.
    """
    return html.Div(className="bm-app", children=[
        html.Div(className="bm-body", children=[
            control_rail(),
            html.Div(className="bm-plot-wrap", children=[
                html.Div(id="plot-badges", className="bm-plot-badges"),
                legend_panel(),
                # dcc.Loading wraps its children in two nested divs of its own.
                # Both need an explicit full height or the chain from
                # .bm-plot-wrap collapses and the graph falls back to Plotly's
                # default 450 px, leaving half the canvas empty.
                #
                # delay_show keeps the map on screen during the short rebuilds
                # that follow a color-by change or a zoom step. Blanking a
                # 100k-point scatter behind a spinner for a few hundred
                # milliseconds on every interaction makes the whole map feel
                # like it is reloading; the spinner should only appear when the
                # wait is long enough to need explaining.
                dcc.Loading(
                    type="circle", color="#22c7bd",
                    delay_show=600,
                    overlay_style={"visibility": "visible", "opacity": 0.45},
                    parent_className="bm-plot-loading",
                    children=dcc.Graph(
                        id="manifold-graph",
                        className="dash-graph",
                        clear_on_unhover=True,
                        config={
                            "displaylogo": False,
                            "scrollZoom": True,
                            "displayModeBar": True,
                            # No selection feature exists, so neither selection
                            # tool is offered. Leaving lasso2d on the modebar
                            # would let a user draw a marquee that does nothing.
                            "modeBarButtonsToRemove": [
                                "select2d", "lasso2d", "autoScale2d"],
                        },
                        style={"height": "100%"},
                    ),
                ),
            ]),
        ]),
        dcc.Store(id="viewport-store"),
        dcc.Store(id="legend-store"),
    ])
