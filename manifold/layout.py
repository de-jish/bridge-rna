"""The Dash layout: header, left control rail, and the plot.

The chrome matches Bridge RNA (light scientific instrument); the plot canvas is
dark navy. Controls are declarative here; behaviour lives in callbacks.py.

The shell is two columns. It used to be three, with a right-hand panel reporting
statistics for a lasso selection; that feature is gone and the plot took the
space back.
"""

from __future__ import annotations

from urllib.parse import quote

from dash import dcc, html

from . import colorby, data, find, render, theme


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


def _group(label_id: str, label: str, *children):
    """A rail group and the heading that names its controls, tied together.

    The headings here are styled divs rather than `<label>`s, and the controls
    under them are a Radix radio group, a Dash dropdown named by its own value,
    and a checklist - none of which a `for` attribute can reach. So the group
    carries the name: a screen reader announces "Projection" on entry and
    "UMAP, selected, 1 of 3" after it, where before it announced only the
    second half. Same pattern as `bridge_rna.layout._labelled`, and it changes
    nothing visually.
    """
    return html.Div(
        className="bm-group", role="group",
        **{"aria-labelledby": label_id},
        children=[html.Div(label, id=label_id, className="bm-group-label"), *children],
    )


def _segmented(id_: str, options: list[dict], value: str):
    """A radio group styled as a segmented pill control.

    Only the container carries a class. Dash renders each option as a label with
    its own structural classes and marks the chosen one `.selected`, and
    `labelClassName` lands on an inner text span rather than the label itself -
    so the stylesheet targets Dash's classes scoped under `.bm-seg` instead.
    """
    return dcc.RadioItems(id=id_, options=options, value=value, className="bm-seg")


def control_rail() -> html.Aside:
    n_archs4, n_osdr, _ = data.counts()

    return html.Aside(
        className="bm-rail",
        **{"aria-labelledby": "map-controls-heading"},
        children=[
            # The map view had one heading (the shell's H1) and no landmark
            # below the header, so a screen reader arriving here found a page
            # with a title and no structure - where the retrieval view already
            # offered a nav, a controls aside, a main and an inspector aside.
            # The heading is hidden because the rail's own group labels already
            # say what each control is and a visible "Map controls" would only
            # repeat the page.
            html.H2("Map controls", id="map-controls-heading",
                    className="visually-hidden"),
            # The parameter readout sits directly under the control it
            # describes, which is the same rule the color-by coverage readout
            # below follows: the fact that qualifies a control belongs against
            # that control, not in a tooltip and not in the plot badges, which
            # report what is drawn right now and change on every zoom.
            _group(
                "projection-label", "Projection",
                _segmented("method", method_options(), default_method()),
                html.Details(className="bm-method-details", children=[
                    html.Summary("Projection details"),
                    html.Div(id="method-params", className="bm-params"),
                    html.P("Projected distances differ from the embedding cosine similarity used for retrieval.",
                           className="bm-hint"),
                ]),
            ),
            _group(
                "dims-label", "Dimensions",
                _segmented("dims", [
                    {"label": "2D", "value": "2d"},
                    {"label": "3D", "value": "3d"},
                ], "2d"),
            ),
            # The color-by group carries its own coverage readout, so the answer
            # to "how much of this map is this field actually coloring?" sits
            # next to the control that decides it rather than being inferred
            # from how grey the plot looks.
            _group(
                "color-by-label", "Color by",
                dcc.Dropdown(
                    id="color-by",
                    options=colorby.menu_options(),
                    value=colorby.default_key(),
                    clearable=False,
                    className="bm-dropdown",
                ),
                html.Div(id="coverage", className="bm-coverage"),
                html.Div(id="color-by-hint", className="bm-hint"),
            ),
            _group(
                "layers-label", "Layers",
                dcc.Checklist(
                    id="layers",
                    options=[
                        {"label": f"ARCHS4 cloud ({n_archs4:,})", "value": "archs4"},
                        {"label": f"OSDR overlay ({n_osdr:,})", "value": "osdr"},
                    ],
                    value=["archs4", "osdr"],
                    className="bm-checklist",
                ),
            ),
            # In 2-D the glyph sample can carry the whole corpus (the density
            # raster it used to sit above is gone, so the sample is the only
            # thing drawing those points). In 3-D the tiers stop at the rotation
            # cap; budget_options() decides, and callbacks.sync_budget_to_dims
            # swaps the tiers when the dimensionality changes.
            # The label names the quantity. It used to name the mechanism - the
            # word was "budget" - which was the rail's only piece of
            # implementation vocabulary, on a control whose four pills already
            # read 100k / 250k / 500k / All. The old wording is pinned in
            # `tests/test_app.py::REMOVED_COPY`.
            #
            # The component id stays `budget`. It is internal, three callbacks
            # and two test modules name it, and renaming it would be churn with
            # no reader on the other end.
            _group(
                "budget-label", "Number of ARCHS4 points",
                _segmented("budget", budget_options("2d"), default_budget("2d")),
            ),
            # "Where did this study land?" - the one question the map could not
            # be asked until now. It sits here, after the controls that decide how
            # the map is drawn and immediately above the panel that describes
            # whatever is currently selected, because the control and its result
            # belong together: that is the same rule that puts the coverage
            # readout under the color-by and the parameter chips under the
            # projection pills.
            #
            # A real `<label for>`, unlike every other group on this rail. Dash
            # renders a Dropdown as a button and a Slider as a Radix thumb, and
            # neither is a labelable element, which is why they need the
            # `role="group"` wrapper `_group` builds. `dcc.Input` puts the id on
            # the actual `<input>` - verified in the DOM, and it is the one
            # place Dash's id and className land on different elements - so this
            # control can be named the ordinary way.
            html.Div(
                id="find-group", className="bm-group", role="group",
                **{"aria-labelledby": "find-label"},
                children=[
                    html.Label("Find a study", id="find-label",
                               className="bm-group-label", htmlFor="find-input"),
                    dcc.Input(
                        id="find-input", type="text",
                        className="bm-find-input",
                        # `debounce=False`, which is a change and is the whole
                        # reason the suggestions can exist: Dash's `debounce`
                        # means "publish the value on Enter or blur", so with it
                        # on there is no per-keystroke value to complete. The
                        # commit is not lost, it is made explicit - `n_submit`
                        # and `n_blur` are exactly the two events `debounce` was
                        # firing on, and they are Inputs to `resolve_find` now.
                        # That split is what keeps a 942,563-point figure from
                        # being rebuilt on every letter of "GSM4256053".
                        debounce=False,
                        # The browser's own history dropdown would sit on top of
                        # ours, offering whatever this user last typed into a
                        # box called `find-input` on any page.
                        autoComplete="off",
                        # The placeholder still carries the grammar, because it
                        # is what an empty box says and the list has nothing to
                        # offer until something is typed.
                        #
                        # Its length is measured, not judged. The field is
                        # 215 px wide inside its padding on the 268 px rail, and
                        # an earlier four-grammar wording rendered at 237 px, so
                        # it taught three formats and then trailed off in the
                        # middle of the fourth. Two grammars leave room to spell
                        # both out.
                        # `test_the_find_placeholder_fits_its_box` in
                        # `tests/e2e_check.py` re-measures it in the browser,
                        # because no Python assertion can.
                        placeholder="GSE143281 or OSD-100",
                        # The combobox semantics - role, aria-autocomplete,
                        # aria-controls, aria-expanded, aria-activedescendant -
                        # are set by `assets/find-suggest.js` and not here.
                        # `dcc.Input` declares no `aria-*` or `role` wildcard in
                        # Dash 4, so passing them raises at layout time rather
                        # than being ignored. That is the right place for them
                        # anyway: three of the five are live state that only the
                        # code operating the list can keep true, and the feature
                        # does not exist without that file.
                    ),
                    # The suggestion list. Inline rather than floating: the rail
                    # scrolls, so an absolutely-positioned popup would either be
                    # clipped by that scroll container or detach from the box it
                    # belongs to. Inline, it is bounded by its own max-height
                    # and pushes the panels below it down for as long as it is
                    # open, which is a change the reader caused and can undo.
                    #
                    # Empty when there is nothing to offer, and hidden by
                    # `:empty` rather than by a style this or any callback
                    # writes. Whether it is *open* is the browser's business -
                    # Escape, a click outside, a chosen row - and lives in one
                    # class on the group above. Two independent facts, one
                    # owner each, composed by CSS into one `display`.
                    html.Div(
                        id="find-suggestions", className="bm-suggest",
                        role="listbox",
                        **{"aria-label": "Matching studies"},
                    ),
                    html.Div(id="find-status", className="bm-hint"),
                    html.Button("Fit study", id="frame-find", n_clicks=0,
                                className="bm-button",
                                style={"display": "none"}),
                ],
            ),
            # Clicking an OSDR point offers a retrieval for it. Hidden until
            # something is clicked; the map is read rather than driven, so this
            # is an offer that appears in response to interest, not a control
            # sitting on the rail waiting to be understood.
            # Its children are callback output because they differ by corpus:
            # an OSDR sample offers a retrieval and a GEO sample offers its NCBI
            # record, and those are a `dcc.Link` and an `<a>` respectively - not
            # one component with a swapped href. The group itself stays declared
            # so Dash can still validate the callback graph against it.
            html.Div(id="picked-group", className="bm-group",
                     style={"display": "none"}),
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
                html.Button("Fit results", id="frame-retrieval",
                            n_clicks=0, className="bm-button"),
                html.Button("Explore neighborhood", id="explore-neighborhood",
                            n_clicks=0, className="bm-button"),
                # What each mark means lives in the key on the plot, beside the
                # marks. Nothing else belongs here.
                #
                # A standing paragraph used to, telling the reader to hover a hit
                # for its two ranks and warning that a projection cannot
                # preserve 512-dimensional distances. Removed on 2026-08-11. The
                # disagreement it described is real and is still shown -
                # `render._map_ranks` puts both ranks in every hit's hover, which
                # is where a reader meets the fact at the moment it applies - so
                # the rail was spending three permanent lines telling the reader
                # to go and read a tooltip. The sentence is pinned in
                # `tests/test_app.py::REMOVED_COPY`, hence described not quoted.
            ]),
            # Every viewport action on one rail, in the order they are reached:
            # frame a find, frame a retrieval, then undo whichever of them - or
            # the reader's own scroll-zoom - narrowed the map.
            #
            # **Reset view used to sit on the plot**, on the argument that it
            # qualifies the viewport and the viewport belongs to the plot. That
            # argument was true and still split one feature across two surfaces:
            # the button that framed the map was on the rail and the button that
            # unframed it was most of a screen away on the canvas, so the two
            # could not be read as a pair and the reader who framed something had
            # to go looking for the way back. Undoing an action belongs beside
            # the action.
            #
            # There is still exactly one of it, for the reason there always was:
            # framing a find, framing a retrieval and a plain scroll-zoom all
            # write the same `viewport-store` and are the same state by the time
            # they arrive, so one control reverses all three. A button per frame
            # button would be two writers for one piece of state and neither
            # would answer a scroll-zoom.
            #
            # Declared here and hidden rather than created by a callback, for the
            # reason the legend's parts are: Dash validates its callback graph
            # against the initial layout, and an id that only ever exists as
            # callback output fails silently in the browser rather than loudly at
            # startup.
            html.Div(id="view-group", className="bm-group",
                     style={"display": "none"}, children=[
                html.Div("View", className="bm-group-label"),
                html.Button("Reset view", id="reset-view", n_clicks=0,
                            className="bm-button",
                            title="Return to the whole map"),
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
    if kind == "neighborhood":
        # A distinct open dot, with its hue sourced from the same theme token
        # as Plotly rather than mirrored into the stylesheet.
        return html.Span(
            className="bm-key-glyph is-neighborhood",
            style={"border": f"2px solid {color}", "borderRadius": "50%",
                   "boxSizing": "border-box"})
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
                           dims: str, neighborhood: dict | None = None) -> list:
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
    n_evidence = int((neighborhood or {}).get("locatable") or 0)
    evidence_row = (_key_row(
        "neighborhood", "512-D evidence neighbor", n_evidence,
        theme.NEIGHBORHOOD_COLOR) if n_evidence else None)
    if not overlay or not overlay.get("cohorts"):
        rows = [evidence_row] if evidence_row is not None else []
        return ([html.Div(className="bm-key bm-key--retrieval", children=rows)]
                if rows else [])

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
        if evidence_row is not None:
            rows.append(evidence_row)
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
    if evidence_row is not None:
        rows.append(evidence_row)
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


def suggestion_children(suggestions: dict | None) -> list:
    """The rows inside the suggestion listbox.

    Each row is a real option in a real listbox: `role="option"` under the
    `role="listbox"` the layout declares, so a screen reader announces "3 of 10"
    rather than reading ten unlabelled divs. The row is a `<div>` and not a
    `<button>` deliberately - a button inside a listbox is announced as a button
    and takes its own place in the tab order, and an ARIA combobox keeps focus
    in the text field and moves a *virtual* cursor instead.

    The identifier its `id` carries is what the click callback reads, so the
    value a row commits is the value the row was built from and there is no
    index into a list that a re-render could invalidate.

    A miss gets one flat row rather than an empty box, and only when the query
    was the right *shape* to have matched: "no sample matches GSM99999999" is
    worth saying, where a dropdown volunteering "no matches" under the word
    "liver" would contradict the sentence below the box, which correctly says
    that free text is not what this control takes.
    """
    suggestions = suggestions or {}
    items = suggestions.get("items") or []
    if not items:
        if suggestions.get("reason") == find.ABSENT:
            return [html.Div("Nothing on this map matches that.",
                             className="bm-suggest-empty")]
        return []
    return [
        html.Div(
            id={"type": "find-suggestion", "value": str(item["value"])},
            className="bm-suggest-row",
            role="option",
            n_clicks=0,
            **{"aria-selected": "false"},
            children=[
                html.Span(str(item["label"]), className="bm-suggest-label"),
                html.Span(str(item["detail"]), className="bm-suggest-detail",
                          title=str(item["detail"])),
            ],
        )
        for item in items
    ]


def selected_panel_children(record: dict | None) -> list:
    """The rail panel describing whatever is currently selected.

    One panel for both ways of selecting: clicking an OSDR diamond, and finding
    anything. They are the same question - what is this sample - so two panels
    answering it would be the clutter this feature exists to avoid.

    The action at the bottom differs by corpus, and the difference is not
    cosmetic. An OSDR sample offers a retrieval, which is an in-app navigation
    and therefore a `dcc.Link`. A GEO sample offers its NCBI record, which is
    an external URL and therefore an `<a>`: a `dcc.Link` pointing off-site hands
    the href to Dash's client-side router, which tries to resolve it as an
    application route instead of leaving the page.
    """
    if not record:
        return []
    rows = [
        html.Div(className="bm-picked-row", children=[
            html.Span(k, className="bm-picked-key"),
            html.Span(v, className="bm-picked-value"),
        ])
        for k, v in record.get("rows") or []
    ]
    children = [
        html.Div("Selected sample", className="bm-group-label"),
        html.Div(str(record.get("title") or ""), className="bm-picked"),
        *rows,
    ]
    if record.get("sample_key"):
        children.append(dcc.Link(
            "Find related ARCHS4 samples", className="bm-button",
            href=f"/?q={quote(str(record['sample_key']))}"))
    elif record.get("geo"):
        children.append(html.A(
            "Open the GEO record →", className="bm-button",
            href=f"{find.GEO_ACC_URL}{quote(str(record['geo']))}",
            target="_blank", rel="noopener noreferrer"))
    return children


def found_key_children(found: dict | None) -> list:
    """The key row for a found identifier.

    A mark with no row is a glyph a reader can only decode by hovering it, which
    is the whole finding `docs/design-notes.md#map-key` records - and this one is the fifth
    encoding the map can carry at once, so it earns a row rather than being the
    exception.

    The count is what is *drawn*, following the same rule as every other row
    here, so a series past `FIND_MAX_MARKS` reports the cap rather than the
    total. The plot badge carries "500 of 8,764"; a key row saying 8,764 beside
    500 visible marks would be the count rule broken in the one place it is most
    obviously checkable.

    Unlike the member mark, this one needs no dimensionality argument. A star
    becomes a diamond in 3-D because `Scatter3d` has no star and a diamond is a
    genuinely different shape; here both spellings - `x-open` in 2-D and `x` in
    3-D, since `Scatter3d` rejects the first - draw the same X, so the key is
    the same row in both views and says so with one class.

    White comes from the stylesheet rather than from `theme` through an inline
    style, which is the hit rings' arrangement and for the hit rings' reason:
    white here is the shape channel's ink, not a hue identifying anything, so
    there is no per-cohort value that could drift out of step with the plot.
    """
    points = (found or {}).get("points") or []
    if not points:
        return []
    drawn = min(len(points), theme.FIND_MAX_MARKS)
    return [html.Div(className="bm-key bm-key--found", children=[
        _key_row("found", str(found.get("label") or "found"), drawn),
    ])]


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

    The four sections are ordered by how transient each is. A found sample sits
    above even the retrieval, because it is the most fleeting mark on the map -
    it is replaced by the next thing typed into the box, where a retrieval
    survives until the next search on the other view; the color list is the
    standing state of the map and is the only part that scrolls; the corpus
    shapes are a footnote and never change. Only the color list can grow, so
    only the color list scrolls.
    """
    return html.Div(
        id="legend", className="bm-legend", style={"display": "none"},
        children=[
            html.Div(id="legend-found"),
            html.Div(id="legend-retrieval"),
            html.Div(id="legend-color", className="bm-legend-section", children=[
                html.Div(id="legend-title", className="bm-legend-title"),
                # The label and the field are hidden and shown as one unit, so
                # a screen reader never meets an orphan label describing a
                # field that is not there. dcc.Input puts the id on the input
                # element, so this is a real label association rather than the
                # group wrapper the dropdowns need; it is hidden from the eye
                # because the heading above already says what the list is, and
                # the placeholder that was doing this job stops being a name
                # the moment anything is typed into it.
                html.Div(id="legend-search-group", style={"display": "none"}, children=[
                    html.Label("Filter the color categories", htmlFor="legend-search",
                               className="visually-hidden"),
                    dcc.Input(id="legend-search", className="bm-legend-search",
                              placeholder="filter categories…", type="text",
                              debounce=False),
                ]),
                html.Div(id="legend-list", className="bm-legend-list"),
            ]),
            html.Div(id="legend-corpus"),
        ],
    )


def _neighborhood_empty(message: str) -> list:
    return [html.Div(message, className="bm-neighborhood-empty", role="status")]


# A pattern-matching id cannot use the empty string as an actionable identity:
# the focus callback correctly treats an empty value as no selection. Keep the
# display copy human-readable while giving the one unassigned-study group a
# stable identity shared with the map overlay.
NO_GSE_FOCUS_VALUE = "__neighborhood_no_gse__"


def _neighborhood_row(kind: str, value: str, primary: str,
                      secondary: str, trailing: str, selected: bool,
                      disabled: bool = False) -> html.Button:
    """One keyboard-native evidence row shared by Studies and Samples."""
    return html.Button(
        id={"type": f"neighborhood-{kind}", "value": value},
        n_clicks=0,
        disabled=disabled,
        **{"aria-pressed": "true" if selected else "false"},
        className=("bm-neighborhood-row "
                   f"bm-neighborhood-{kind}"
                   + (" is-selected" if selected else "")),
        children=[
            html.Span(primary, className="bm-neighborhood-row-primary"),
            html.Span(secondary, className="bm-neighborhood-row-secondary"),
            html.Span(trailing, className="bm-neighborhood-row-trailing"),
        ],
    )


def _neighborhood_category_rows(category: dict) -> list:
    """The five leading recorded categories plus one honest remainder."""
    items = list(category.get("items") or [])
    shown = items[:5]
    if len(items) > 5:
        count = sum(int(item.get("count") or 0) for item in items[5:])
        covered = int(category.get("covered") or 0)
        shown.append({
            "label": "Other recorded categories",
            "count": count,
            "share": count / covered if covered else 0,
        })
    return [
        html.Div(className="bm-neighborhood-bar-row", children=[
            html.Div(className="bm-neighborhood-bar-label", children=[
                html.Span(str(item.get("label") or "Not recorded")),
                html.Span(f"{int(item.get('count') or 0):,}"),
            ]),
            html.Div(className="bm-neighborhood-bar-track", children=[
                html.Div(
                    className="bm-neighborhood-bar-fill",
                    style={
                        "width": (
                            f"{max(0, min(1, float(item.get('share') or 0))) * 100:.1f}%"
                        ),
                    },
                ),
            ]),
        ])
        for item in shown
    ]


def neighborhood_overview_children(summary: dict) -> list:
    """Render the deterministic, denominator-explicit evidence summary."""
    depth = int(summary.get("depth") or 0)
    if not depth:
        return _neighborhood_empty("No samples are available in this neighborhood.")

    tissue = summary.get("tissue") or {}
    species = summary.get("species") or {}
    score = summary.get("score") or {}
    median = score.get("median")
    minimum = score.get("minimum")
    maximum = score.get("maximum")
    score_value = "Unavailable" if median is None else f"{float(median):.3f}"
    score_range = ("No numeric scores" if minimum is None or maximum is None else
                   f"Range {float(minimum):.3f}–{float(maximum):.3f}")
    top_three = int(summary.get("top_three_study_samples") or 0)

    def composition(title: str, category: dict) -> html.Div:
        rows = _neighborhood_category_rows(category)
        return html.Div(className="bm-neighborhood-composition", children=[
            html.H3(title, className="bm-neighborhood-section-title"),
            *(rows or [html.Div("No recorded metadata.",
                               className="bm-neighborhood-muted")]),
        ])

    leading = list(summary.get("leading_studies") or [])[:3]
    leading_children = [
        html.Div(className="bm-neighborhood-leading-row", children=[
            html.Span(str(group.get("display_gse") or "No GSE recorded"),
                      className="bm-neighborhood-leading-name"),
            html.Span(
                f"{int(group.get('sample_count') or 0):,} sample"
                f"{'s' if int(group.get('sample_count') or 0) != 1 else ''}",
                className="bm-neighborhood-leading-count",
            ),
        ])
        for group in leading
    ]

    return [
        html.P(str(summary.get("sentence") or ""),
               className="bm-neighborhood-summary"),
        html.Div(className="bm-neighborhood-coverage", children=[
            html.Span(
                f"Tissue categories {int(tissue.get('covered') or 0):,} of {depth:,}"),
            html.Span(
                f"Species metadata {int(species.get('covered') or 0):,} of {depth:,}"),
        ]),
        html.Div(className="bm-neighborhood-metrics", children=[
            html.Div(className="bm-neighborhood-metric", children=[
                html.Strong(f"{int(summary.get('study_count') or 0):,}"),
                html.Span("GEO studies"),
            ]),
            html.Div(className="bm-neighborhood-metric", children=[
                html.Strong(f"{top_three:,} of {depth:,}"),
                html.Span("in top 3 studies"),
            ]),
            html.Div(className="bm-neighborhood-metric", children=[
                html.Strong(score_value),
                html.Span("median cosine"),
                html.Small(score_range),
            ]),
        ]),
        composition("Tissue categories", tissue),
        composition("Species composition", species),
        html.Div(className="bm-neighborhood-leading", children=[
            html.H3("Leading studies", className="bm-neighborhood-section-title"),
            *(leading_children or [html.Div(
                "No GEO studies are represented.",
                className="bm-neighborhood-muted")]),
        ]),
    ]


def _geo_link(accession: str, label: str) -> html.A:
    return html.A(
        label,
        href=("https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc="
              f"{quote(accession, safe='')}"),
        target="_blank",
        rel="noopener noreferrer",
        className="bm-neighborhood-geo",
    )


def neighborhood_studies_children(groups: list[dict],
                                    focus: dict | None) -> list:
    """Render grouped evidence studies without nesting links in buttons."""
    if not groups:
        return _neighborhood_empty("No studies are represented in this neighborhood.")
    selected = focus or {}
    children = []
    for group in groups:
        gse = str(group.get("gse") or "")
        display_gse = str(group.get("display_gse") or "No GSE recorded")
        focus_value = gse or NO_GSE_FOCUS_VALUE
        count = int(group.get("sample_count") or 0)
        rank = int(group.get("best_rank") or 0)
        title = (f"Example sample: {group['title']}" if group.get("title")
                 else "Sample title not recorded")
        tissue = str(group.get("dominant_tissue") or "Tissue not recorded")
        tissue_count = int(group.get("dominant_tissue_count") or 0)
        tissue_covered = int(group.get("tissue_covered") or 0)
        button = _neighborhood_row(
            "study", focus_value, display_gse,
            f"{title} · {tissue} · {tissue_count:,} of {tissue_covered:,} "
            "with tissue metadata",
            f"{count:,} sample{'s' if count != 1 else ''} · best rank {rank:,}",
            selected.get("kind") == "study"
            and str(selected.get("value") or "") == focus_value,
        )
        row_children = [button]
        if gse:
            row_children.append(_geo_link(gse, f"Open {gse} in GEO"))
        children.append(html.Div(
            row_children, className="bm-neighborhood-study-wrap"))
    return children


def _sample_detail(row: dict) -> html.Div:
    gsm = str(row.get("gsm") or "GSM not recorded")
    score = row.get("score")
    score_text = "Not recorded" if score is None else f"{float(score):.3f}"
    fields = [
        ("Study", str(row.get("gse") or "Not recorded")),
        ("Tissue", str(row.get("tissue") or "Not recorded")),
        ("Species", str(row.get("species") or "Not recorded")),
        ("Cosine", score_text),
    ]
    return html.Div(className="bm-neighborhood-detail", children=[
        html.Div(className="bm-neighborhood-detail-head", children=[
            html.H3(gsm),
            _geo_link(gsm, f"Open {gsm} in GEO") if row.get("gsm") else None,
        ]),
        html.P(str(row.get("title") or "Title not recorded")),
        html.Dl([
            child
            for label, value in fields
            for child in (html.Dt(label), html.Dd(value))
        ]),
    ])


def neighborhood_samples_children(rows: list[dict],
                                   focus: dict | None, *,
                                   detail_row: dict | None = None) -> list:
    """Render every filtered evidence sample and its cached selected detail."""
    selected = focus or {}
    selected_gsm = (str(selected.get("value") or "")
                    if selected.get("kind") == "sample" else "")
    detail = (detail_row if detail_row is not None
              and str(detail_row.get("gsm") or "") == selected_gsm
              else next((row for row in rows
                         if str(row.get("gsm") or "") == selected_gsm), None))
    if not rows and detail is None:
        return _neighborhood_empty("No samples match this search.")
    children = [_sample_detail(detail)] if detail is not None else []
    if not rows:
        children.extend(_neighborhood_empty("No samples match this search."))
    for row in rows:
        gsm = str(row.get("gsm") or "GSM not recorded")
        gse = str(row.get("gse") or "No GSE")
        tissue = str(row.get("tissue") or "Tissue not recorded")
        species = str(row.get("species") or "Species not recorded")
        title = str(row.get("title") or "Title not recorded")
        rank = int(row.get("rank") or 0)
        score = row.get("score")
        score_text = "Unavailable" if score is None else f"{float(score):.3f}"
        children.append(_neighborhood_row(
            "sample", gsm, gsm, f"{gse} · {tissue} · {species} · {title}",
            f"#{rank:,} · {score_text}", gsm == selected_gsm,
        ))
    return children


def neighborhood_unavailable_children(reason: str) -> list:
    """Render a stale/legacy result without pretending evidence exists."""
    return _neighborhood_empty(
        reason or "This retrieval has no evidence neighborhood available.")


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
            html.Div(id="map-workspace", className="bm-map-workspace", children=[
            html.Main(className="bm-plot-wrap", **{"aria-label": "Embedding map"}, children=[
                # The strip across the top of the canvas: what is drawn right
                # now. Nothing else. "Reset view" used to share it and moved to
                # the rail on 2026-08-13, next to the two buttons that frame -
                # see the `view-group` comment in `control_rail` for why.
                html.Div(className="bm-plot-badges", children=[
                    html.Div(id="plot-badges", className="bm-badges"),
                ]),
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
                            # Plotly sizes its canvas once and keeps it unless
                            # told otherwise. The plot's container changes size
                            # for reasons that have nothing to do with a
                            # callback - the stacked breakpoint below 900 px, a
                            # phone rotating, a window being dragged - and
                            # without this the scatter kept the geometry it was
                            # first laid out with and drew a quadrant of the
                            # corpus into the whole canvas. The retrieval
                            # view's graph has always set it.
                            "responsive": True,
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
            html.Aside(
                id="neighborhood-drawer",
                className="bm-neighborhood",
                style={"display": "none"},
                **{"aria-labelledby": "neighborhood-heading"},
                children=[
                    html.Div(className="bm-neighborhood-head", children=[
                        html.Div("Neighborhood explorer",
                                 className="bm-group-label"),
                        html.Button(
                            "Close", id="neighborhood-close", n_clicks=0,
                            className="bm-neighborhood-close",
                            **{"aria-label": "Close neighborhood explorer"},
                        ),
                        html.H2(id="neighborhood-heading", tabIndex=-1),
                        html.Div(id="neighborhood-meta", className="bm-hint"),
                        dcc.RadioItems(
                            id="neighborhood-arm", value="a",
                            className="bm-seg bm-neighborhood-arm",
                        ),
                        dcc.RadioItems(
                            id="neighborhood-tab", value="overview",
                            options=[
                                {"label": "Overview", "value": "overview"},
                                {"label": "Studies", "value": "studies"},
                                {"label": "Samples", "value": "samples"},
                            ],
                            className="bm-seg bm-neighborhood-tabs",
                        ),
                        html.Label(
                            "Search neighborhood samples",
                            htmlFor="neighborhood-search",
                            className="visually-hidden",
                        ),
                        dcc.Input(
                            id="neighborhood-search", type="search",
                            placeholder="Search GSM, GSE, tissue…",
                            className="bm-neighborhood-search",
                            style={"display": "none"},
                        ),
                    ]),
                    html.Div(
                        id="neighborhood-body",
                        className="bm-neighborhood-body",
                    ),
                    html.Div(
                        "Ranked by cosine similarity in 512-D. Map position does not determine rank.",
                        className="bm-neighborhood-foot",
                    ),
                ],
            ),
            ]),
        ]),
        dcc.Store(id="viewport-store"),
        dcc.Store(id="legend-store"),
        # Declared here rather than created by a callback, for the reason the
        # legend's parts are: Dash validates its callback graph against the
        # initial layout, so an id that only ever exists as callback output
        # fails silently at runtime instead of loudly at startup.
        dcc.Store(id="find-store"),
        # One chosen suggestion, and the only reason it is a store rather than
        # the click itself: a pattern-matching `ALL` input fires whenever its
        # family is re-rendered, which for the suggestion rows is every
        # keystroke. Dash discards the answer to a request that a newer request
        # for the same callback supersedes, so with that input on the callback
        # that searches, the keystroke-driven no-op overtook the Enter that had
        # already been answered - the first find of every session was lost, and
        # the rest survived on timing. This store changes only when a row is
        # actually clicked, so the search callback is never woken with nothing
        # to do. See `callbacks.choose_suggestion`.
        dcc.Store(id="find-chosen"),
        dcc.Store(
            id="neighborhood-open-store",
            data={"open": False, "opener": "explore-neighborhood",
                  "focus_target": None},
        ),
        dcc.Store(id="neighborhood-focus-store"),
        dcc.Store(id="neighborhood-focus-sink"),
    ])
