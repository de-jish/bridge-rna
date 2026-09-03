"""Wiring: controls -> figure, zoom -> level-of-detail, legend filter."""

from __future__ import annotations

import numpy as np
from dash import ALL, Input, Output, State, ctx, html, no_update

from bridge_rna import neighborhoods

from . import colorby, data, find, layout, render, theme

# The legend search only earns its space once the list is long enough to be
# hard to scan; below that it is chrome.
LEGEND_SEARCH_MIN_ITEMS = 8


#: The roles a drawn query can have. "a" is every single-sample, uploaded and
#: single-cohort search as well as the first arm of a comparison; "b" exists
#: only when a comparison was run.
ROLE_A, ROLE_B = "a", "b"


def _locate_hits(hits: list, n_archs4: int):
    """ARCHS4 hits to point indices, dropping any that cannot be located.

    An ARCHS4 hit carries `archs4_index`, its row in the embedding memmap, and
    ARCHS4 occupies rows `0..n_archs4-1` of the map's global point order - so
    the row *is* the point. A hit that predates that column is not drawn at all
    rather than drawn in the wrong place.
    """
    points, labels, scores = [], [], []
    for hit in hits or []:
        idx = hit.get("archs4_index")
        if idx is None:
            continue
        idx = int(idx)
        if 0 <= idx < n_archs4:
            points.append(idx)
            labels.append(str(hit.get("gsm") or ""))
            scores.append(float(hit.get("score") or 0.0))
    return points, labels, scores


def _locate_members(keys: list, n_archs4: int) -> list[int]:
    """OSDR sample keys to point indices, by the join the whole app is built on.

    A pooled cohort has no single position in the space - its query vector is a
    mean, and no projection was fit on it - so what gets drawn is every member.
    """
    keys = [str(k) for k in keys if str(k)]
    if not keys:
        return []
    meta = data.osdr_metadata()
    if "sample_key" not in meta.columns:
        return []
    row_of = {k: i for i, k in enumerate(meta["sample_key"].astype(str))}
    return [n_archs4 + int(row_of[k]) for k in keys if k in row_of]


def _retrieval_overlay(hits_payload: dict | None,
                       roles: tuple[str, ...] = (ROLE_A, ROLE_B)) -> dict | None:
    """Turn a stored retrieval into point indices on this map.

    Returns one entry in `cohorts` per drawn query, plus flat `hit_points` /
    `query_points` / `query_point` / `query_label` keys holding the **union**
    across those entries. The flat keys are what `_frame_for` and the retrieval
    summary read, so keeping them a union rather than cohort A's own values is
    what makes a comparison frame and describe both arms without either of them
    learning what a comparison is.

    `roles` is which arms the user has ticked on the map rail. An untickable
    single-query payload always produces the one `a` entry, so every existing
    behaviour falls out of the general path unchanged.

    Returns None when there is nothing to draw.
    """
    if not hits_payload:
        return None
    n_archs4, _n_osdr, _ = data.counts()
    comparison = hits_payload.get("comparison") or {}

    # A cohort's `sample_id` is a cohort id and matches no sample_key, so the
    # single-sample fallback below can never collide with a cohort's members.
    sources = [
        (ROLE_A,
         hits_payload.get("hits"),
         (hits_payload.get("member_ids")
          or [hits_payload.get("sample_id") or ""]),
         _cohort_label(hits_payload.get("query"))),
    ]
    if comparison:
        sources.append(
            (ROLE_B,
             comparison.get("hits_b"),
             comparison.get("member_ids") or [],
             _cohort_label(comparison.get("query_b"))))

    cohorts = []
    for role, hits, keys, label in sources:
        if role not in roles:
            continue
        points, labels, scores = _locate_hits(hits, n_archs4)
        members = _locate_members(keys, n_archs4)
        if not points and not members:
            continue
        cohorts.append({"role": role,
                        # The cohort's own name for a pooled query; for a single
                        # sample there is none, and its sample key is the name.
                        "label": label or (str(keys[0]) if keys else ""),
                        "hit_points": points, "hit_labels": labels,
                        "hit_scores": scores, "query_points": members})

    if not cohorts:
        return None

    hit_points = [p for c in cohorts for p in c["hit_points"]]
    query_points = [p for c in cohorts for p in c["query_points"]]
    first = set(cohorts[0]["hit_points"])
    shared_points = sorted(first.intersection(
        p for c in cohorts[1:] for p in c["hit_points"]))

    if len(cohorts) > 1:
        query_label = " vs ".join(c["label"] or "cohort" for c in cohorts)
    elif not query_points:
        # Nothing of the query is on this map: an uploaded sample has no
        # `sample_key`, so it has no coordinate. The label stays empty and the
        # rail says "an OSDR sample" rather than announcing "0 pooled cohort
        # samples", which would be false twice over.
        query_label = ""
    elif len(query_points) == 1:
        query_label = cohorts[0]["label"]
    else:
        query_label = f"{len(query_points)} pooled cohort samples"

    return {"cohorts": cohorts,
            "hit_points": hit_points,
            "hit_labels": [lb for c in cohorts for lb in c["hit_labels"]],
            "hit_scores": [s for c in cohorts for s in c["hit_scores"]],
            "shared_points": shared_points,
            # `query_point` stays for the single-sample case: `_frame_for` needs
            # one origin, and any drawn member serves.
            "query_point": query_points[0] if query_points else None,
            "query_points": query_points,
            "query_label": query_label}


def _neighborhood_payload_for_arm(hits_payload: dict | None,
                                  arm: str) -> dict | None:
    """Return the stored evidence payload for one comparison arm."""
    if not hits_payload:
        return None
    if arm == ROLE_B:
        return (hits_payload.get("comparison") or {}).get("neighborhood_b")
    return hits_payload.get("neighborhood")


def next_neighborhood_open_state(trigger: str | None, previous: dict | None,
                                 has_retrieval: bool) -> dict:
    """Apply one open/close event without giving the store another writer."""
    old = previous if isinstance(previous, dict) else {}
    opened = bool(old.get("open"))
    opener = str(old.get("opener") or "explore-neighborhood")
    if opener not in ("frame-retrieval", "explore-neighborhood"):
        opener = "explore-neighborhood"
    if trigger == "hits-store":
        return {"open": False, "opener": opener, "focus_target": None}
    if trigger == "neighborhood-close":
        return {
            "open": False,
            "opener": opener,
            "focus_target": opener if opened else None,
        }
    if trigger in ("frame-retrieval", "explore-neighborhood") and has_retrieval:
        return {
            "open": True,
            "opener": trigger,
            "focus_target": "neighborhood-heading",
        }
    return {"open": opened, "opener": opener, "focus_target": None}


def _neighborhood_arm_label(hits_payload: dict | None, arm: str) -> str:
    payload = hits_payload or {}
    if arm == ROLE_B:
        comparison = payload.get("comparison") or {}
        neighborhood = comparison.get("neighborhood_b") or {}
        query = comparison.get("query_b") or {}
        fallback = "Cohort B"
    else:
        neighborhood = payload.get("neighborhood") or {}
        query = payload.get("query") or {}
        fallback = "Evidence neighborhood"
    return (str(neighborhood.get("label") or "")
            or str(query.get("cohort_label") or query.get("sample_name") or "")
            or fallback)


def neighborhood_arm_options(hits_payload: dict | None) -> tuple[list[dict], dict]:
    """Name comparison arms independently and hide the redundant single arm."""
    payload = hits_payload or {}
    comparison = payload.get("comparison") or {}
    label_a = _neighborhood_arm_label(payload, ROLE_A)
    if not comparison:
        return [{"label": label_a, "value": ROLE_A}], {"display": "none"}

    label_b = _neighborhood_arm_label(payload, ROLE_B)
    return [
        {"label": f"A · {label_a}", "value": ROLE_A},
        {"label": f"B · {label_b}", "value": ROLE_B},
    ], {}


def next_neighborhood_focus(trigger, value):
    """Translate an explicit row/map click into the explorer's focus shape."""
    if trigger == "hits-store":
        return None
    if isinstance(trigger, dict):
        kind = {"neighborhood-study": "study",
                "neighborhood-sample": "sample"}.get(trigger.get("type"))
        selected = str(trigger.get("value") or "")
        if kind and selected and value:
            return {"kind": kind, "value": selected}
        return no_update
    if trigger != "manifold-graph" or not value:
        return no_update
    points = value.get("points") or []
    custom = points[0].get("customdata") if points else None
    if (not isinstance(custom, (list, tuple)) or len(custom) < 3
            or custom[0] != "neighborhood" or not custom[2]):
        return no_update
    return {"kind": "sample", "value": str(custom[2])}


def next_neighborhood_interaction(trigger, value):
    """Return the single-owner focus/tab/arm/search transition."""
    focus = next_neighborhood_focus(trigger, value)
    if trigger == "hits-store":
        return focus, "overview", ROLE_A, ""
    if trigger == "manifold-graph" and focus is not no_update:
        return focus, "samples", no_update, no_update
    return focus, no_update, no_update, no_update


def neighborhood_drawer_state(hits_payload: dict | None, arm: str,
                              tab: str, search: str | None,
                              focus: dict | None) -> dict:
    """Build every drawer output from one selected, cached evidence arm."""
    payload = _neighborhood_payload_for_arm(hits_payload, arm or ROLE_A) or {}
    options, arm_style = neighborhood_arm_options(hits_payload)
    available = bool(payload.get("available"))
    summary = neighborhoods.summarize(payload)
    groups = neighborhoods.study_groups(payload)
    all_rows = neighborhoods.sample_rows(payload)
    rows = neighborhoods.sample_rows(payload, search or "")
    selected_gsm = (str(focus.get("value") or "")
                    if (focus or {}).get("kind") == "sample" else "")
    detail_row = next((row for row in all_rows
                       if str(row.get("gsm") or "") == selected_gsm), None)
    summary["leading_studies"] = groups[:3]
    depth = int(summary.get("depth") or 0)
    study_count = int(summary.get("study_count") or 0)
    tabs = [
        {"label": "Overview", "value": "overview"},
        {"label": f"Studies {study_count:,}", "value": "studies"},
        {"label": f"Samples {depth:,}", "value": "samples"},
    ]
    if available:
        builders = {
            "overview": lambda: layout.neighborhood_overview_children(summary),
            "studies": lambda: layout.neighborhood_studies_children(groups, focus),
            "samples": lambda: layout.neighborhood_samples_children(
                rows, focus, detail_row=detail_row),
        }
        body = builders.get(tab, builders["overview"])()
        meta = (f"{depth:,} nearest sample{'s' if depth != 1 else ''} · "
                "cosine in 512-D")
    else:
        body = layout.neighborhood_unavailable_children(
            str(payload.get("reason") or
                "Run this retrieval again to build its evidence neighborhood."))
        meta = "Evidence neighborhood unavailable"
    return {
        "heading": _neighborhood_arm_label(hits_payload, arm or ROLE_A),
        "meta": meta,
        "arm_options": options,
        "arm_style": arm_style,
        "tabs": tabs,
        "search_style": ({} if available and tab == "samples"
                         else {"display": "none"}),
        "body": body,
    }


def _neighborhood_overlay(hits_payload: dict | None, arm: str,
                          focus: dict | None) -> dict | None:
    """Normalize one stored evidence neighborhood for Plotly.

    The stored payload remains the complete drawer record. This map view keeps
    only rows with an integer ARCHS4 index, in their ranked order, and reports
    both the returned and locatable totals so omitted metadata never becomes a
    silent change in the evidence count.
    """
    payload = _neighborhood_payload_for_arm(hits_payload, arm)
    if not isinstance(payload, dict) or not payload.get("available"):
        return None

    selected = focus or {}
    focus_field = {"study": "gse", "sample": "gsm"}.get(selected.get("kind"))
    focus_value = str(selected.get("value") or "")
    n_archs4, _n_osdr, _total = data.counts()
    points: list[int] = []
    rows: list[dict] = []
    focus_points: list[int] = []
    for row in payload.get("hits") or []:
        if not isinstance(row, dict):
            continue
        point = row.get("archs4_index")
        # JSON has one integer number type for these indices. Numeric strings,
        # floats, and booleans are malformed rather than coordinates to coerce.
        if (not isinstance(point, int) or isinstance(point, bool)
                or not 0 <= point < n_archs4):
            continue
        points.append(point)
        rows.append(row)
        row_focus_value = str(row.get(focus_field) or "")
        if focus_field == "gse" and not row_focus_value:
            row_focus_value = layout.NO_GSE_FOCUS_VALUE
        if (focus_field and focus_value
                and row_focus_value == focus_value):
            focus_points.append(point)

    returned = payload.get("depth_returned")
    if not isinstance(returned, int) or isinstance(returned, bool):
        returned = len(payload.get("hits") or [])
    return {
        "label": str(payload.get("label") or ""),
        "points": points,
        "rows": rows,
        "focus_points": focus_points,
        "returned": returned,
        "locatable": len(points),
    }


def _cohort_label(query: dict | None) -> str:
    """The cohort's own name, for the map key. Empty for a non-cohort query."""
    return str((query or {}).get("cohort_label") or "")


def _roles_from_checklist(value: list | None) -> tuple[str, ...]:
    """Which cohorts the rail's ticks ask for.

    `"on"` is cohort A rather than `"a"` so that a single-query retrieval keeps
    the exact value the control has always held, and so a session store written
    before comparisons could be drawn still resolves.
    """
    ticked = set(value or ())
    return tuple(role for role, key in ((ROLE_A, "on"), (ROLE_B, ROLE_B))
                 if key in ticked)


def _viewport_from_relayout(relayout: dict | None):
    """Interpret a Plotly relayout event as a viewport change.

    Three outcomes, and the distinction matters: a concrete window (re-stratify
    the sample inside it), ``None`` (reset to the full corpus), or the sentinel
    ``"unchanged"`` for events that are not zooms at all - a hover, a legend
    click, a drag-mode switch - which must leave the current sample alone rather
    than triggering a full resample.
    """
    if not relayout:
        return None
    keys = ["xaxis.range[0]", "xaxis.range[1]", "yaxis.range[0]", "yaxis.range[1]"]
    if all(k in relayout for k in keys):
        x0, x1, y0, y1 = (relayout[k] for k in keys)
        return (min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1))
    if relayout.get("xaxis.autorange") or relayout.get("autosize"):
        return None
    return "unchanged"


def frame_points(points, method: str, dims: str):
    """A viewport containing every point given, with room to breathe.

    The geometry behind both framing actions - the retrieval's and the find's -
    so the two cannot drift into padding the same map differently.

    At full-corpus zoom a retrieval's points are a few pixels apart - the hits
    really are that close, which is the finding - so the rings overlap into one
    illegible mark. Framing is what makes them separately readable, and in both
    callers it is offered as an action rather than done automatically: arriving
    already zoomed in would hide the thing worth seeing first, which is *where
    in the whole corpus* the points landed.

    Returns None (the whole map) if there is nothing to frame, or in 3-D, where
    the viewport store does not drive the camera.
    """
    if dims != "2d":
        return None
    coords = data.coords(method, "2d")
    if coords.shape[0] == 0:
        return None
    inside = [int(p) for p in (points or []) if 0 <= int(p) < coords.shape[0]]
    if not inside:
        return None

    xy = coords[np.asarray(inside), :2]
    x0, y0 = xy.min(axis=0)
    x1, y1 = xy.max(axis=0)
    # A tight cluster would otherwise frame to a zero-width window. The pad is
    # a share of the span with an absolute floor, so a single point still gets
    # a sensible window.
    pad_x = max((x1 - x0) * 0.6, 0.35)
    pad_y = max((y1 - y0) * 0.6, 0.35)
    return _clamped_to_corpus(
        [float(x0 - pad_x), float(x1 + pad_x),
         float(y0 - pad_y), float(y1 + pad_y)], coords)


def _clamped_to_corpus(window, coords):
    """Never frame wider than the corpus itself.

    A pad that is a share of the span is right for a retrieval's handful of
    neighbouring points and wrong for a set that spans the map: 60% either side
    of a spread set asks for a window larger than everything there is, so the
    "frame" zooms *out* and the user ends up seeing less than they started
    with. It is reachable from both callers - an OSDR study is drawn from one
    experiment but not from one region.

    Measured on the real corpus before this guard existed: OSD-457's 192
    samples framed to **1.22x** the corpus width and GSE228590's 8,764 to
    **1.03x**, against 0.02x for a typical study and 0.06x for a typical
    series. Clamping each edge to the corpus keeps a spread set framed at
    exactly the whole map, which is the honest answer for a set that really is
    everywhere, and leaves every compact set untouched.
    """
    x0, x1, y0, y1 = window
    return [float(max(x0, coords[:, 0].min())), float(min(x1, coords[:, 0].max())),
            float(max(y0, coords[:, 1].min())), float(min(y1, coords[:, 1].max()))]


def _frame_for(hits_payload, method: str, dims: str,
               roles: tuple[str, ...] = (ROLE_A, ROLE_B)):
    """The retrieval's framing: the query and every hit that is ticked.

    Frames what is actually drawn, so a comparison with one arm hidden frames
    the arm on screen rather than a window sized for both.
    """
    overlay = _retrieval_overlay(hits_payload, roles)
    if overlay is None:
        return None
    points = list(overlay["hit_points"])
    if overlay["query_point"] is not None:
        points.append(overlay["query_point"])
    return frame_points(points, method, dims)


def find_status_children(found: dict | None):
    """What the find box says under itself, and it is four different sentences.

    A miss is not one outcome. Answering "liver" and "GSM999999999" with the
    same "not found" tells the first user their search is broken when the truth
    is that they wanted a different control, and tells the second nothing about
    whether this machine could have looked it up at all. Each reason
    `manifold/find.py` distinguishes gets its own answer, and the two that are
    fixable say what to do.

    The no-metadata sentence is `colorby.ARCHS4_META_HINT` rather than a second
    copy of it, so the two controls that depend on that optional join cannot
    end up naming two different commands.
    """
    if not found:
        return ""
    reason = found.get("reason") or ""
    label = str(found.get("label") or "")

    if not reason:
        n = len(found.get("points") or [])
        shown = min(n, theme.FIND_MAX_MARKS)
        head = [html.B(label), f" · {n:,} sample{'s' if n != 1 else ''}"]
        if shown < n:
            # The cap is stated wherever the number is, not only on the badge.
            head.append(f", marking the first {shown:,}")
        return head + ["."]

    if reason == find.EMPTY:
        return ""
    if reason == find.NO_GEO_METADATA:
        return [html.B(label), " is a GEO accession, and this machine has no "
                "GEO metadata to look it up in. ", colorby.ARCHS4_META_HINT]
    if reason == find.ABSENT:
        n_archs4, n_osdr, _ = data.counts()
        return [html.B(label), " is not on this map. It holds "
                f"{n_archs4:,} GEO samples and the {n_osdr:,} OSDR samples "
                "that were embedded."]
    # find.SHAPE
    return ["Not a study identifier. Type a GEO series (GSE143281) or an OSDR "
            "study (OSD-100). To find a tissue or a condition instead, use ",
            html.B("Color by"), " above."]


def chosen_suggestion(clicks, values) -> dict | None:
    """The click payload a suggestion row commits, or None if none was clicked.

    `values` is every identifier currently on offer, in the order Dash reports
    their clicks, so the chosen one is read positionally from the family that
    actually fired rather than from a separately-stored list that a re-render
    could have replaced.

    The `seq` is not decoration. This becomes a store, and Dash fires a callback
    on a store only when its value *changes* - so choosing GSM4256053, clearing
    the box, and choosing GSM4256053 again would write an identical payload the
    second time and commit nothing. The running click total makes every choice a
    distinct value, including a repeat of the last one.
    """
    triggered = ctx.triggered_id
    if not isinstance(triggered, dict) or triggered.get("type") != "find-suggestion":
        return None
    total = sum(n or 0 for n in (clicks or []))
    if not total:
        return None
    return {"value": str(triggered.get("value") or ""), "seq": total}


def _fired(prop: str) -> bool:
    """Did `prop` actually change in the request that woke this callback?

    `ctx.triggered_id` names one trigger and says nothing about *why* the
    callback ran, and "why" is the whole question for a callback with a
    per-keystroke input among its dependencies. Dash wakes a callback whenever
    any of its inputs is re-rendered, appears, or disappears - measured here: a
    `fill("")` that empties the suggestion family woke `resolve_find` with no
    click and no submit, and it committed the empty box, wiping the marks from
    a search the reader had not touched.

    So the commit is gated on the prop having a value in `ctx.triggered` rather
    than on the absence of other explanations. `value is not None` is what
    separates a real event from a component Dash is merely reporting as new.
    """
    return any(t.get("prop_id") == prop and t.get("value") is not None
               for t in (ctx.triggered or []))


def viewport_axes(vp, dims: str, *, preserve: bool = False) -> dict:
    """The axis layout that puts `vp` on screen, or the whole corpus back.

    **The `autorange` branch is not a no-op and must not be dropped.**
    `theme.base_figure_layout` sets `uirevision="keep"`, which asks Plotly to
    preserve the user's own pan and zoom across a figure update unless the
    incoming figure *changes* the attribute in question. Framing works on that
    rule: it sets a range where the previous figure had none. Resetting has to
    go the other way, and an absent key is not a changed value - leaving the
    range off and expecting Plotly to autorange leans on the one thing
    `uirevision` exists to prevent. Saying `autorange` outright is the only
    honest spelling of "back to the whole map", and it is what makes the reset
    button, a projection change and the modebar's own reset agree, since all
    three reach `update_figure` as `viewport = None`.

    Explorer-only figure updates are different: an absent stored viewport means
    "keep the whole-map range already on screen", not "calculate a new one".
    For those, ``preserve`` omits both axes so ``uirevision`` can do its job.

    3-D contributes nothing: the viewport store does not drive a `Scatter3d`
    camera, so there is no axis to pin and none to release.
    """
    if dims != "2d":
        return {}
    if vp:
        return {"xaxis": dict(range=[vp[0], vp[1]]),
                "yaxis": dict(range=[vp[2], vp[3]])}
    if preserve:
        return {}
    return {"xaxis": dict(autorange=True), "yaxis": dict(autorange=True)}


def is_framed(viewport, dims: str) -> bool:
    """Is the map showing a window rather than the whole corpus?

    The one definition of "framed", so the button that undoes it and the
    callback that produced it cannot disagree about when there is anything to
    undo. It is a property of `viewport-store` alone, deliberately: framing a
    find, framing a retrieval and the user's own scroll-zoom all end up there
    and are the same state by the time they arrive, which is why one control
    can reverse all three.

    3-D is never framed. `frame_points` declines outright there and
    `update_figure` passes `None` for the viewport, because pinning 2-D axis
    ranges does nothing to a `Scatter3d` camera - the same reason both frame
    buttons hide in 3-D. Offering a reset for a state the view cannot be in
    would be a click with no visible effect, which is what the lasso was
    removed for.
    """
    return dims == "2d" and bool(viewport)


def coverage_children(color_by: str):
    """The coverage readout under the color-by control.

    States the exact number of points the selected field colors. This is the
    control that answers "why is so much of my map not colored?" before the
    user has to ask it, and it is why an OSDR-only field is no longer
    indistinguishable from a broken one.

    A whole-map field says nothing. The full bar already is the answer, and the
    sentence that used to sit under it - "Colors all 942,563 points." - only
    restated the bar in words. The readout exists for the partial case, which is
    the one that raises a question.
    """
    spec = colorby.get(color_by)
    covered, total = colorby.coverage(color_by)
    pct = covered / total * 100 if total else 0.0
    whole_map = covered >= total

    bar = html.Div(className="bm-coverage-bar", children=html.Div(
        className="bm-coverage-fill" + ("" if whole_map else " partial"),
        style={"width": f"{max(pct, 1.5):.1f}%"}))

    if whole_map:
        text = ""
    elif covered:
        text = (f"Colors {covered:,} of {total:,} points ({pct:.1f}%). "
                "ARCHS4 is drawn as faint context.")
    else:
        text = "No data for this field on this machine."

    children = [bar]
    if text:
        children.append(html.Div(text, className="bm-coverage-text"))
    missing = spec.missing_hint()
    if missing:
        children.append(html.Div(missing, className="bm-coverage-fix"))
    return children


def register(app):
    @app.callback(
        Output("manifold-graph", "figure"),
        Output("plot-badges", "children"),
        Output("legend", "style"),
        Output("legend-title", "children"),
        Output("legend-search-group", "style"),
        Output("legend-store", "data"),
        Output("legend-retrieval", "children"),
        Output("legend-color", "style"),
        Output("legend-corpus", "children"),
        Output("legend-found", "children"),
        Output("coverage", "children"),
        Output("color-by-hint", "children"),
        Input("method", "value"),
        Input("dims", "value"),
        Input("color-by", "value"),
        Input("layers", "value"),
        Input("budget", "value"),
        Input("viewport-store", "data"),
        # The retrieval lives on the shell, so the map can draw a search that
        # was run before the user walked over here. An Input rather than a
        # State: running a new retrieval should redraw the map that is showing
        # the old one.
        Input("hits-store", "data"),
        Input("show-retrieval", "value"),
        Input("find-store", "data"),
        Input("neighborhood-open-store", "data"),
        Input("neighborhood-arm", "value"),
        Input("neighborhood-focus-store", "data"),
    )
    def update_figure(method, dims, color_by, layers, budget, viewport,
                      hits_payload, show_retrieval, found, open_state, arm,
                      focus):
        vp = tuple(viewport) if viewport else None
        roles = _roles_from_checklist(show_retrieval)
        retrieval = _retrieval_overlay(hits_payload, roles) if roles else None
        state = open_state or {}
        neighborhood = (_neighborhood_overlay(hits_payload, arm or ROLE_A, focus)
                        if state.get("open") else None)
        fig, legend_data, badges = render.build_figure(
            method, dims, color_by, layers or [], budget,
            vp if dims == "2d" else None, retrieval=retrieval,
            neighborhood=neighborhood, found=found)
        legend_data["title"] = layout.color_by_label(color_by)
        preserve_view = ctx.triggered_id in {
            "neighborhood-open-store", "neighborhood-arm",
            "neighborhood-focus-store",
        }
        fig.update_layout(**viewport_axes(vp, dims, preserve=preserve_view))
        badge_children = [_badge(b) for b in badges]
        items = legend_data.get("items", [])

        # The key reads the *full* overlay and is told separately which arms are
        # ticked, so an unticked cohort can keep its row, receded. `retrieval`
        # above is the drawn subset and is what the figure gets.
        key = layout.retrieval_key_children(
            _retrieval_overlay(hits_payload), roles, dims,
            neighborhood=neighborhood)
        corpus = layout.corpus_key_children(layers)
        found_key = layout.found_key_children(found)

        # The panel is shown when any section has content. It used to hang off
        # the color items alone, so an OSDR-only field with the OSDR layer
        # unticked took the whole key off screen - and would now take a drawn
        # retrieval's key with it.
        legend_style = ({} if (items or key or corpus or found_key)
                        else {"display": "none"})
        color_style = {} if items else {"display": "none"}
        search_style = ({} if len(items) >= LEGEND_SEARCH_MIN_ITEMS
                        else {"display": "none"})
        title = f"Color · {legend_data['title']}"
        return (fig, badge_children, legend_style, title, search_style,
                legend_data, key, color_style, corpus, found_key,
                coverage_children(color_by), colorby.get(color_by).hint)

    @app.callback(
        Output("budget", "options"),
        Output("budget", "value"),
        Input("dims", "value"),
        State("budget", "value"),
        prevent_initial_call=True,
    )
    def sync_budget_to_dims(dims, current):
        """Re-fit the point-budget tiers to the dimensionality.

        3-D caps the ARCHS4 cloud for smooth rotation, so its tiers stop at the
        cap instead of offering "All" or "500k" pills the 3-D view would draw
        as 40,000. The value is preserved if it is still one of this
        dimensionality's tiers and reset to the default otherwise, so a 2-D
        "All" cannot linger on the control after a switch to 3-D.
        """
        return layout.budget_options(dims), layout.resolve_budget(dims, current)

    @app.callback(
        Output("method-params", "children"),
        Input("method", "value"),
        Input("dims", "value"),
    )
    def show_projection_params(method, dims):
        """State how the projection on screen was actually fit.

        Its own callback rather than a ninth Output on `update_figure`, which
        fires on eight Inputs: these parameters are a build record, so
        recomputing them on every zoom step would be waste, and worse, the
        readout would go stale or blank whenever the figure path failed. The
        two Inputs it does take are the two that can change the answer.
        """
        return layout.projection_params_children(method, dims)

    @app.callback(
        Output("find-input", "value"),
        Output("find-chosen", "data"),
        Input({"type": "find-suggestion", "value": ALL}, "n_clicks"),
        State({"type": "find-suggestion", "value": ALL}, "id"),
        prevent_initial_call=True,
    )
    def choose_suggestion(clicks, ids):
        """A row was clicked: put it in the box, and publish it as a commit.

        **This callback exists so that the commit does not.** A pattern-matching
        `ALL` input fires whenever its family is *re-rendered*, and this family
        re-renders on every keystroke - so a callback carrying both this input
        and the search would be woken constantly with nothing to do. That is not
        merely wasteful, it is a lost search: Dash discards the response to a
        request that a newer request for the same callback has superseded, and
        the keystroke-driven no-op reliably overtook the Enter that had already
        been answered correctly on the server. Measured on the real app - the
        first find of every session vanished, the rest survived on timing alone.

        So the family's input lives here, where being woken spuriously costs a
        `no_update` on a store nobody redraws, and `find-store` is written by a
        callback with no per-keystroke input at all.
        """
        chosen = chosen_suggestion(clicks, ids)
        if chosen is None:
            return no_update, no_update
        return chosen["value"], chosen

    @app.callback(
        Output("find-store", "data"),
        # The two events `debounce=True` used to fire on, now named outright
        # because the input publishes its value on every keystroke and only
        # these mean "I have finished". Committing here rather than on `value`
        # is what keeps a 942,563-point figure from being rebuilt per letter.
        Input("find-input", "n_submit"),
        Input("find-input", "n_blur"),
        # The third way to finish, arriving as a store rather than as the click
        # itself - see `choose_suggestion` for why that indirection is the whole
        # point. Every input here changes only when someone has actually
        # finished, so this callback is never woken with nothing to do and can
        # never be superseded by its own no-op.
        Input("find-chosen", "data"),
        State("find-input", "value"),
        prevent_initial_call=True,
    )
    def resolve_find(_submits, _blurs, chosen, typed):
        """Turn what was committed into the points it names.

        One writer for `find-store`, three ways to reach it, and the three
        cannot disagree: each hands the same `find.find` the same kind of
        string, so a chosen suggestion resolves exactly as if it had been typed.

        A chosen row commits its own identifier rather than the box's current
        text, because the text is put there by a different callback and this
        must not depend on having won a race with it.

        The default is `no_update`: a commit is positively identified rather
        than inferred from the absence of other explanations, because Dash also
        wakes a callback when an input component newly appears - which is how
        clearing the box once wiped the marks from a search nobody had touched.
        """
        if _fired("find-chosen.data") and chosen:
            return find.find(str(chosen.get("value") or ""))
        if _fired("find-input.n_submit") or _fired("find-input.n_blur"):
            return find.find(typed)
        return no_update

    @app.callback(
        Output("find-suggestions", "children"),
        Input("find-input", "value"),
    )
    def offer_suggestions(query):
        """Complete the identifier being typed, on every keystroke.

        Cheap enough to run per keystroke and measured rather than assumed: a
        warm accession lookup is a handful of `searchsorted` calls over the
        int32 index, about 2 ms, and the OSDR side is a scan of 2,108 strings.
        The one expensive call is the first, which builds the 460 ms index -
        the same build the first search has always paid for, now paid a few
        keystrokes earlier.
        """
        return layout.suggestion_children(find.suggest(query))

    @app.callback(
        Output("find-status", "children"),
        Output("frame-find", "style"),
        Input("find-store", "data"),
        Input("dims", "value"),
    )
    def describe_the_find(found, dims):
        """The sentence under the box, and whether framing is on offer.

        Framing is a button and never automatic, which is the rule
        `frame_points` was already written to and the frame-retrieval button
        already follows. Two reasons, and the second is the stronger. A found
        set can span the map - an OSDR study's 192 samples framed to 1.22x the
        corpus before the clamp went in - so "framing" it would zoom the user
        *out*, silently, as the result of typing. And a 2-D neighbourhood is not
        a similarity neighbourhood on this map: its 20 nearest points overlap
        the true cosine top-20 by a median of 0, so dropping someone into a
        zoomed view of their sample's surroundings invites reading those
        surroundings as related when they are not. Marking answers "where is
        it"; zooming asserts something further, so it waits to be asked for.

        Hidden in 3-D for the reason the retrieval's frame button is: framing
        pins the 2-D axis ranges and the 3-D camera ignores them, so the button
        would be a click with no visible effect.
        """
        offer = bool(found and found.get("points")) and dims == "2d"
        return find_status_children(found), ({} if offer else {"display": "none"})

    @app.callback(
        Output("picked-group", "style"),
        Output("picked-group", "children"),
        Input("manifold-graph", "clickData"),
        Input("find-store", "data"),
    )
    def select_point(click_data, found):
        """Describe whatever is selected, however it was selected.

        One writer, two entry points. Clicking an OSDR diamond and finding a
        sample are the same question - what is this - so they share a panel
        rather than growing a second one beside it.

        A click and a find end in the same record builder: the overlay's
        customdata already carries the sample key, and `find.osdr_sample(key)`
        resolves exactly that to exactly that point, in the shape `find.find`
        returns. So there is one record builder and one panel instead of two of
        each drifting apart.

        It is `osdr_sample` rather than `find` because the box takes studies
        only. A diamond on screen is an unambiguous reference to one sample, so
        clicking it is not the free-text problem the typed grammar refuses - but
        it is not something the box offers either, and the two entry points
        should not pretend otherwise.

        Only the OSDR overlay carries customdata - the ARCHS4 cloud has none,
        deliberately, because 940,455 rows of it cost about 600 KB of dead
        payload per figure. A click returning none is a click on the cloud, and
        the *find* result stays on screen rather than being cleared by it: a
        stray click on empty space should not wipe the panel describing what the
        user just searched for.
        """
        if ctx.triggered_id == "manifold-graph":
            points = (click_data or {}).get("points") or []
            custom = points[0].get("customdata") if points else None
            key = str(custom[0]) if custom else ""
            # An OSDR key is "<accession>|<sample name>". Anything else is not
            # one, and a click on the cloud is not a selection at all.
            found = find.osdr_sample(key) if "|" in key else None
            if found is None:
                return no_update, no_update

        record = find.describe(found)
        if not record:
            return {"display": "none"}, []
        return {}, layout.selected_panel_children(record)

    @app.callback(
        Output("retrieval-group", "style"),
        Output("frame-retrieval", "style"),
        Output("explore-neighborhood", "style"),
        Output("show-retrieval", "options"),
        Output("show-retrieval", "value"),
        Input("hits-store", "data"),
        Input("dims", "value"),
    )
    def show_retrieval_group(hits_payload, dims):
        """Reveal the retrieval control only when there is a retrieval.

        An always-visible control that does nothing until you have searched
        somewhere else is worse than no control: it reads as broken.

        The same argument hides the frame button in 3-D. Framing works by
        pinning the 2-D axis ranges, which the 3-D camera ignores, so the
        button would have been a click with no visible effect - the thing this
        map removed the lasso for.

        The ticks are only *reset* when the retrieval itself changes. This
        callback also fires on a dimensionality switch, because the frame button
        is 2-D only, and reasserting the value there would silently re-tick an
        arm the user had hidden the moment they looked at it in 3-D.

        A comparison turns the single "Show it on the map" tick into one tick
        per cohort, each carrying that cohort's own name. It is the same control
        rather than a new one, so nothing else on the rail moves, and unticking
        one is the escape hatch when two large cohorts crowd the same region.

        The key *under* the ticks is a separate callback, because it describes
        what is drawn and therefore has to depend on the ticks - which this one
        writes, so it cannot also read them.
        """
        freeze = ctx.triggered_id == "dims"
        single = [{"label": " Show it on the map", "value": "on"}]

        def ticks(options, value):
            return ((no_update, no_update) if freeze else (options, value))

        overlay = _retrieval_overlay(hits_payload)
        if overlay is None:
            return ({"display": "none"}, {"display": "none"},
                    {"display": "none"},
                    *ticks(single, ["on"]))

        frame_style = {} if dims == "2d" else {"display": "none"}
        if len(overlay["cohorts"]) < 2:
            return {}, frame_style, {}, *ticks(single, ["on"])
        options = [
            {"label": f"  {c['label'] or 'cohort'}", "value": key}
            for c, key in zip(overlay["cohorts"], ("on", ROLE_B))
        ]
        return {}, frame_style, {}, *ticks(options, ["on", ROLE_B])

    @app.callback(
        Output("neighborhood-open-store", "data"),
        Input("frame-retrieval", "n_clicks"),
        Input("explore-neighborhood", "n_clicks"),
        Input("neighborhood-close", "n_clicks"),
        Input("hits-store", "data"),
        State("neighborhood-open-store", "data"),
        prevent_initial_call=True,
    )
    def set_neighborhood_open(_frame, _explore, _close, hits_payload, previous):
        has_retrieval = _retrieval_overlay(hits_payload) is not None
        return next_neighborhood_open_state(
            ctx.triggered_id, previous, has_retrieval)

    @app.callback(
        Output("neighborhood-drawer", "style"),
        Output("map-workspace", "className"),
        Input("neighborhood-open-store", "data"),
    )
    def show_neighborhood(open_state):
        opened = bool((open_state or {}).get("open"))
        return ({} if opened else {"display": "none"},
                "bm-map-workspace" + (" is-neighborhood-open" if opened else ""))

    app.clientside_callback(
        """
        function(state) {
            const opened = !!(state && state.open);
            const targetId = state && state.focus_target;
            if (!targetId) {
                return window.dash_clientside.no_update;
            }
            let attempts = 0;
            const focusWhenReady = function() {
                const target = document.getElementById(targetId);
                if (target && target.offsetParent !== null) {
                    target.focus();
                    return;
                }
                if (!opened) {
                    const fallback = document.getElementById("explore-neighborhood");
                    if (fallback && fallback.offsetParent !== null) {
                        fallback.focus();
                        return;
                    }
                }
                attempts += 1;
                if (attempts < 30) {
                    window.requestAnimationFrame(focusWhenReady);
                }
            };
            window.requestAnimationFrame(focusWhenReady);
            return {open: opened, opener: targetId, sequence: Date.now()};
        }
        """,
        Output("neighborhood-focus-sink", "data"),
        Input("neighborhood-open-store", "data"),
        prevent_initial_call=True,
    )

    @app.callback(
        Output("neighborhood-heading", "children"),
        Output("neighborhood-meta", "children"),
        Output("neighborhood-arm", "options"),
        Output("neighborhood-arm", "style"),
        Output("neighborhood-tab", "options"),
        Output("neighborhood-search", "style"),
        Output("neighborhood-body", "children"),
        Input("hits-store", "data"),
        Input("neighborhood-arm", "value"),
        Input("neighborhood-tab", "value"),
        Input("neighborhood-search", "value"),
        Input("neighborhood-focus-store", "data"),
    )
    def render_neighborhood(hits_payload, arm, tab, search, focus):
        state = neighborhood_drawer_state(
            hits_payload, arm or ROLE_A, tab or "overview", search, focus)
        return (state["heading"], state["meta"], state["arm_options"],
                state["arm_style"], state["tabs"], state["search_style"],
                state["body"])

    @app.callback(
        Output("neighborhood-focus-store", "data"),
        Output("neighborhood-tab", "value"),
        Output("neighborhood-arm", "value"),
        Output("neighborhood-search", "value"),
        Input({"type": "neighborhood-study", "value": ALL}, "n_clicks"),
        Input({"type": "neighborhood-sample", "value": ALL}, "n_clicks"),
        Input("manifold-graph", "clickData"),
        Input("hits-store", "data"),
        prevent_initial_call=True,
    )
    def select_neighborhood(_study_clicks, _sample_clicks, click_data,
                            hits_payload):
        trigger = ctx.triggered_id
        if trigger == "hits-store":
            changed = hits_payload
        elif trigger == "manifold-graph":
            changed = click_data
        else:
            changed = next((item.get("value") for item in (ctx.triggered or [])
                            if item.get("value") is not None), None)
        return next_neighborhood_interaction(trigger, changed)

    @app.callback(
        Output("retrieval-summary", "children"),
        Input("hits-store", "data"),
        Input("show-retrieval", "value"),
    )
    def describe_the_retrieval(hits_payload, show_retrieval):
        """One line saying what the ticks are currently drawing.

        This used to be the map's only key for the retrieval: a swatch per
        cohort and a paragraph explaining that a ring inside a square is a hit
        both arms found. All of that moved onto the plot, into
        `layout.retrieval_key_children`, next to the marks it decodes - the
        ticks directly above already carry each cohort's name, so the rail was
        naming them twice and explaining glyphs nowhere near them.

        What is left is the control's own feedback, and it reads the ticks
        rather than the payload so a hidden arm is reported as hidden.
        """
        full = _retrieval_overlay(hits_payload)
        if full is None:
            return ""

        roles = _roles_from_checklist(show_retrieval)
        shown = [c for c in full["cohorts"] if c["role"] in roles]
        if not shown:
            # The tick hides the overlay outright, and saying so is the whole
            # job of this line. It used to describe the retrieval regardless, so
            # unticking left the rail describing marks over a map with none of
            # them on it.
            return ("Not drawn. Tick it above to put it back on the map."
                    if len(full["cohorts"]) < 2 else
                    "Neither arm is drawn. Tick one to put it back on the map.")

        if len(full["cohorts"]) < 2:
            n = len(set(full["hit_points"]))
            query = full["query_label"] or "an OSDR sample"
            # The sentence used to carry a trailing clause about the marks
            # sitting where they do in the space, which narrated the picture the
            # reader is already looking at rather than telling them anything
            # about it. Removed on 2026-08-11; the count is the part that
            # carries information, so the sentence stops after it. The phrase is
            # pinned in `tests/test_app.py::REMOVED_COPY`, which is why it is
            # described here rather than quoted.
            return [
                html.B(query.split("|")[-1]),
                f" and its {n} nearest ARCHS4 neighbor{'s' if n != 1 else ''}.",
            ]

        if len(shown) < 2:
            return [html.B(shown[0]["label"] or "One arm"),
                    " only. Tick both to see which samples they share."]
        # Distinct samples, not drawn marks. `hit_points` is a concatenation
        # across the arms, so a hit both retrieved appears in it twice - and
        # quoting its length made the rail say "10 hits, 2 of them retrieved by
        # both" for the same search whose banner on the other view said "share
        # 2 of 8 retrieved samples". Two surfaces, one comparison, two numbers.
        n_shared = len(full["shared_points"])
        n_hits = len(set(full["hit_points"]))
        return [
            "Both arms drawn, ",
            html.B(f"{n_hits}"), " samples retrieved, ",
            html.B(f"{n_shared}"), " of them by both.",
        ]

    @app.callback(
        Output("viewport-store", "data"),
        Input("manifold-graph", "relayoutData"),
        Input("method", "value"),
        Input("dims", "value"),
        # Framing the retrieval is a viewport change, so it is an Input to the
        # callback that already owns the viewport rather than a second writer.
        # Two callbacks writing one Output is a race Dash only sometimes
        # rejects, and `test_every_output_has_exactly_one_writer` pins it.
        Input("frame-retrieval", "n_clicks"),
        # Framing a find is the same kind of event as framing a retrieval, so it
        # arrives at the callback that already owns the viewport rather than
        # becoming a second writer to it.
        Input("frame-find", "n_clicks"),
        # And so does undoing any of it. One store holds the framing, whoever
        # narrowed it, so one Input clears it - rather than a reset handler per
        # frame button, which would be two ways to write one value and would
        # still leave a plain scroll-zoom with no way back.
        Input("reset-view", "n_clicks"),
        State("hits-store", "data"),
        State("find-store", "data"),
        State("method", "value"),
        State("dims", "value"),
        # Frame what is actually on screen: with one arm of a comparison
        # unticked, framing both would zoom out past everything drawn.
        State("show-retrieval", "value"),
        prevent_initial_call=True,
    )
    def update_viewport(relayout, method, dims, _frame_clicks, _find_clicks,
                        _reset_clicks, hits_payload, found, method_state,
                        dims_state, show_retrieval):
        from dash import ctx
        if ctx.triggered_id in ("method", "dims", "reset-view"):
            # None is this callback's word for the whole corpus, so the reset
            # button lands in the same branch a projection change does. It
            # clears the viewport and nothing else: the find, the retrieval, the
            # selected sample and the ticks above them are all other stores, so
            # the query a user framed is still there to be framed again.
            return None
        if ctx.triggered_id == "frame-retrieval":
            roles = _roles_from_checklist(show_retrieval) or (ROLE_A, ROLE_B)
            return _frame_for(hits_payload, method_state, dims_state, roles)
        if ctx.triggered_id == "frame-find":
            # `no_update`, never None. None is this callback's "reset to the
            # whole corpus", so a frame click with nothing found - or in 3-D,
            # where frame_points declines - would zoom the user out instead of
            # doing nothing.
            window = frame_points((found or {}).get("points"),
                                  method_state, dims_state)
            return window if window else no_update
        vp = _viewport_from_relayout(relayout)
        if vp == "unchanged":
            return no_update
        if vp is None:
            return None
        return list(vp)

    @app.callback(
        Output("view-group", "style"),
        Input("viewport-store", "data"),
        Input("dims", "value"),
    )
    def show_reset_view(viewport, dims):
        """Offer the reset only while there is something to reset.

        A permanently visible "Reset view" over an unframed map is a control
        that does nothing, which is the standard this map already applies to the
        two frame buttons and to the selection tools it removed from the
        modebar.
        """
        return {} if is_framed(viewport, dims) else {"display": "none"}

    @app.callback(
        Output("legend-list", "children"),
        Input("legend-store", "data"),
        Input("legend-search", "value"),
    )
    def render_legend(legend_data, query):
        """Render the legend rows, filtered by the search box.

        One callback owns this output: the figure callback publishes categories
        to legend-store and this renders them, so a search term survives a
        re-render and there is no second writer to race with.
        """
        return filtered_legend_rows(legend_data, query)


def _badge(html_text: str):
    # badges may carry <b> tags; render via a small parser.
    return html.Div(className="bm-badge", children=_html_with_bold(html_text))


def _html_with_bold(text: str):
    parts = []
    rest = text
    while "<b>" in rest and "</b>" in rest:
        pre, rest = rest.split("<b>", 1)
        bold, rest = rest.split("</b>", 1)
        if pre:
            parts.append(pre)
        parts.append(html.B(bold))
    if rest:
        parts.append(rest)
    return parts


def _legend_rows(items):
    return [
        html.Div(className="bm-legend-item", children=[
            html.Span(className="bm-legend-swatch", style={"background": it["color"]}),
            html.Span(it["label"], title=it["label"],
                      style={"overflow": "hidden", "textOverflow": "ellipsis"}),
            html.Span(f"{it['count']:,}", className="bm-legend-count"),
        ])
        for it in items
    ]


def filtered_legend_rows(legend_data, query):
    """Legend rows matching ``query`` (case-insensitive substring on the label)."""
    items = (legend_data or {}).get("items", [])
    if query:
        needle = query.strip().lower()
        items = [i for i in items if needle in str(i["label"]).lower()]
        if not items:
            return html.Div("no matching categories", className="bm-legend-empty")
    return _legend_rows(items)
