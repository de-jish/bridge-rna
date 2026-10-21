"""The results half of the retrieval view: the match table, its filters, the plot.

The search callbacks in `callbacks.py` write `hits-store` and stop there. This
module turns that payload into everything a reader looks at afterwards, so the
two halves meet in one store and nothing here needs to know which query source
ran. Every rule about which rows are in view lives in `results.py`, and every
callback here calls the same `visible_rows`, so the table, the plot, the export
and the AI summary cannot disagree about it. Design and measurements:
docs/retrieval-results.md.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any

import pandas as pd
from dash import ALL, ClientsideFunction, Input, Output, State, ctx, dcc, html, no_update

from . import result_plots as P
from . import results as R
from .figures import _empty_network_figure, build_comparison_figure, build_network_figure
from .similarity import identical_groups
from .util import _format_count, _safe_str

# --- Layout ------------------------------------------------------------------

# One list drives the plot switch, the caption and the callback, so a fifth
# view is one entry here. The captions say what a mark *is*; what the reader
# should conclude is theirs.
PLOT_KINDS = (
    {"value": "rank", "label": "Score by rank"},
    {"value": "similarity", "label": "Similarity"},
    {"value": "series", "label": "Series"},
    {"value": "network", "label": "Network"},
)
DEFAULT_PLOT_KIND = PLOT_KINDS[0]["value"]

MATCH_COUNT_MIN = 3
MATCH_COUNT_MAX = 1000
MATCH_COUNT_DEFAULT = 5
# The slider moves in log10(k). Linear spacing from 3 to 1,000 would put every
# value anyone used before this change (3 to 30) inside the first 3% of the
# track, so the track is logarithmic and the box beside it takes an exact k.
_MATCH_MARKS = (3, 10, 30, 100, 300, 1000)

DEFAULT_SORT = {"column": "rank_a", "descending": False}

# A filter control's id and the Filters key it fills, in one place for the
# layout, the callbacks and the Clear button.
FILTER_INPUTS = (
    ("filter-series", "series"),
    ("filter-series-mode", "series_mode"),
    ("filter-tissue", "tissue"),
    ("filter-species", "species"),
    ("filter-min-score", "min_score"),
    ("filter-text", "text"),
    ("filter-merge", "merge_identical"),
    ("filter-retrieved-by", "retrieved_by"),
)


def _log_k(k: int) -> float:
    return round(math.log10(k), 4)


def clamp_match_count(value: Any) -> int:
    """A usable k from whatever the box or a stale caller sent."""
    try:
        k = int(round(float(value)))
    except (TypeError, ValueError):
        return MATCH_COUNT_DEFAULT
    return max(MATCH_COUNT_MIN, min(MATCH_COUNT_MAX, k))


def requested_match_count(typed: Any, position: Any) -> int:
    """The k a search should use: the box when it holds a number, else the slider.

    The two agree whenever `sync_match_count` has run; this is for the moment
    they cannot, such as a box cleared by hand, where falling back to a fixed
    default would quietly search for a different k than the slider shows.
    """
    try:
        float(typed)
    except (TypeError, ValueError):
        return k_from_slider(position)
    return clamp_match_count(typed)


def k_from_slider(position: Any) -> int:
    """k for a slider position, rounded to two significant figures above 20.

    A drag lands on log positions, so 10**2.531 would offer "339 matches".
    Two significant figures keep a dragged value readable; the box still takes
    any exact k.
    """
    try:
        raw = 10 ** float(position)
    except (TypeError, ValueError, OverflowError):
        return MATCH_COUNT_DEFAULT
    if raw > 20:
        magnitude = 10 ** (int(math.floor(math.log10(raw))) - 1)
        raw = round(raw / magnitude) * magnitude
    return clamp_match_count(raw)


def build_topk_control() -> Any:
    """The number of matches: a log slider for range and a box for an exact k."""
    return html.Div(
        className="control topk-control",
        role="group",
        **{"aria-labelledby": "topk-label"},
        children=[
            html.Div(className="topk-head", children=[
                html.Label("Number of matches", id="topk-label",
                           htmlFor="topk-input", className="control-label"),
                # No min/max here, deliberately: Dash 4 reports a number outside
                # them as None, so "5000" reached the callback as nothing to
                # clamp and the next search ran with the default 5. The box
                # passes what was typed and `sync_match_count` writes back the
                # clamped value, which the reader then sees.
                dcc.Input(id="topk-input", type="number", step=1,
                          value=MATCH_COUNT_DEFAULT,
                          debounce=True, className="topk-input",
                          inputMode="numeric"),
            ]),
            html.Div(className="control-slider", children=[
                dcc.Slider(
                    id="topk-slider",
                    min=_log_k(MATCH_COUNT_MIN), max=_log_k(MATCH_COUNT_MAX),
                    step=0.001, value=_log_k(MATCH_COUNT_DEFAULT),
                    marks={_log_k(k): f"{k:,}" for k in _MATCH_MARKS},
                    allow_direct_input=False,
                    updatemode="drag",
                ),
            ]),
        ],
    )


def _filter_group(label_id: str, label: str, *children: Any,
                  extra_class: str = "", **attrs: Any) -> Any:
    return html.Div(
        className="results-filter" + (f" {extra_class}" if extra_class else ""),
        role="group",
        **{"aria-labelledby": label_id},
        **attrs,
        children=[html.Label(label, id=label_id, className="control-label"), *children],
    )


def build_results_panel() -> Any:
    """The match table with its filters, count line, pager and export."""
    return html.Section(
        id="results-panel",
        className="panel results-panel",
        **{"aria-labelledby": "results-title"},
        children=[
            html.Div(className="results-head", children=[
                html.Div(className="results-heading", children=[
                    html.H2("Matches", id="results-title", className="panel-title"),
                    html.P(id="results-summary", className="panel-subtitle",
                           **{"aria-live": "polite"}),
                ]),
                html.Div(className="results-actions", children=[
                    dcc.Checklist(
                        id="filter-merge", value=[],
                        options=[{"label": " Merge identical profiles",
                                  "value": "on"}],
                        className="dash-checklist results-merge"),
                    html.Button("Clear filters", id="filter-clear", n_clicks=0,
                                className="btn-ghost btn-compact",
                                style={"display": "none"}),
                    html.Button("Download CSV", id="results-download-button",
                                n_clicks=0, className="btn-secondary btn-compact",
                                disabled=True),
                    dcc.Download(id="results-download"),
                ]),
            ]),
            html.Div(id="results-filters", className="results-filters", children=[
                html.Div(
                    className="results-filter results-filter--wide",
                    role="group",
                    **{"aria-labelledby": "filter-series-label"},
                    children=[
                        html.Div(className="results-filter-head", children=[
                            html.Label("Series", id="filter-series-label",
                                       className="control-label"),
                            dcc.RadioItems(
                                id="filter-series-mode", value="only", inline=True,
                                options=[{"label": "Show only", "value": "only"},
                                         {"label": "Hide", "value": "hide"}],
                                className="results-seg results-seg--mini"),
                        ]),
                        dcc.Dropdown(id="filter-series", multi=True, options=[],
                                     value=[], placeholder="All series",
                                     className="results-dropdown"),
                    ],
                ),
                _filter_group(
                    "filter-tissue-label", "Tissue category",
                    dcc.Dropdown(id="filter-tissue", multi=True, options=[],
                                 value=[], placeholder="All",
                                 className="results-dropdown"),
                ),
                _filter_group(
                    "filter-species-label", "Species",
                    dcc.Dropdown(id="filter-species", multi=True, options=[],
                                 value=[], placeholder="All",
                                 className="results-dropdown"),
                ),
                html.Div(className="results-filter", children=[
                    html.Label("Minimum score", htmlFor="filter-min-score",
                               className="control-label"),
                    dcc.Input(id="filter-min-score", type="text", inputMode="decimal",
                              placeholder="Any", debounce=0.4, autoComplete="off",
                              className="results-input results-input--score"),
                ]),
                html.Div(className="results-filter results-filter--grow", children=[
                    html.Label("Text contains", htmlFor="filter-text",
                               className="control-label"),
                    dcc.Input(id="filter-text", type="search", debounce=0.3,
                              placeholder="GSM, series, title or source",
                              className="results-input"),
                ]),
                _filter_group(
                    "filter-retrieved-by-label", "Retrieved by",
                    dcc.RadioItems(id="filter-retrieved-by", value="either",
                                   inline=True, options=[],
                                   className="results-seg"),
                    id="filter-retrieved-by-group",
                    style={"display": "none"},
                    extra_class="results-filter--wide",
                ),
            ]),
            html.Div(id="results-table-wrap", className="results-table-wrap",
                     tabIndex="-1",
                     children=empty_table_message(
                         "Run a search to list its matches here.")),
            html.Nav(className="results-pager", **{"aria-label": "Match pages"},
                     children=[
                         html.Button("Previous", id="results-prev", n_clicks=0,
                                     disabled=True, className="btn-ghost btn-compact"),
                         html.Span(id="results-page-label", role="status",
                                   className="results-page-label"),
                         html.Button("Next", id="results-next", n_clicks=0,
                                     disabled=True, className="btn-ghost btn-compact"),
                     ]),
            dcc.Store(id="results-store"),
            dcc.Store(id="results-sort-store", data=DEFAULT_SORT),
            dcc.Store(id="results-page-store", data=0),
            dcc.Store(id="inspector-store"),
            # GSMs in table order across all pages: what the AI summary and the
            # clientside request builder read instead of re-filtering.
            dcc.Store(id="results-visible-store"),
            dcc.Store(id="ai-request-store"),
            # A marker the clientside row highlighter writes; nothing reads it.
            dcc.Store(id="results-selection-painted"),
        ],
    )


def build_plot_panel(graph_legend: Any) -> Any:
    """One plot at a time, drawn from the rows the table shows."""
    return html.Section(
        id="plot-panel",
        className="panel panel--canvas plot-panel",
        **{"aria-labelledby": "plot-title"},
        children=[
            html.Div(className="plot-head", children=[
                html.H2("Plot", id="plot-title", className="visually-hidden"),
                dcc.RadioItems(
                    id="results-plot-kind", value=DEFAULT_PLOT_KIND, inline=True,
                    options=[{"label": k["label"], "value": k["value"]}
                             for k in PLOT_KINDS],
                    className="results-seg results-seg--plot",
                ),
                html.P(id="plot-caption", className="panel-subtitle plot-caption"),
            ]),
            html.Div(id="graph-legend", className="graph-legend",
                     style={"display": "none"}, children=graph_legend),
            html.Div(
                className="graph-wrap",
                children=[
                    dcc.Graph(
                        id="results-plot",
                        className="dash-graph",
                        figure=_empty_network_figure(
                            "Select an OSDR sample, then run a search."),
                        config={"displaylogo": False, "responsive": True,
                                "modeBarButtonsToRemove": ["select2d", "lasso2d",
                                                           "autoScale2d"],
                                "toImageButtonOptions": {"filename": "bridge-rna-plot"}},
                        style={"height": "100%"},
                    ),
                ],
            ),
        ],
    )


def empty_table_message(text: str, *, action: Any = None) -> Any:
    return html.Div(className="results-empty", children=[
        html.P(text, className="results-empty-text"),
        *([action] if action is not None else []),
    ])


# --- The table ---------------------------------------------------------------

_SINGLE_COLUMNS = (
    ("rank_a", "Rank", "num"),
    ("gsm", "GSM", "id"),
    ("score_a", "Score", "num"),
    ("gse", "Series", "id"),
    ("title", "Title", "text"),
    ("tissue", "Tissue category", "text"),
    ("species", "Species", "species"),
)


def _columns(results: dict) -> tuple[tuple[str, str, str], ...]:
    if not results.get("comparison"):
        return _SINGLE_COLUMNS
    return (
        ("rank_a", "Rank A", "num"),
        ("score_a", "Score A", "num"),
        ("rank_b", "Rank B", "num"),
        ("score_b", "Score B", "num"),
    ) + tuple(c for c in _SINGLE_COLUMNS if c[0] not in ("rank_a", "score_a"))


def _cell(row: dict, key: str, kind: str, k: int) -> Any:
    value = row.get(key)
    if key == "gsm":
        children: list[Any] = [html.Button(
            _safe_str(value), id={"type": "result-row", "gsm": _safe_str(value)},
            n_clicks=0, className="result-select",
            **{"aria-label": f"Inspect {_safe_str(value)}"})]
        twins = [g for g in row.get("identical") or [] if g]
        if twins:
            children.append(html.Span(
                "Same profile: " + ", ".join(twins), className="result-twin",
                title="An identical expression profile is also deposited under "
                      "this accession."))
        return html.Td(children, className="col-gsm")
    cls = f"col-{key}"
    if kind == "num":
        if value is None:
            return html.Td(html.Span(f"Not in top {k:,}", className="visually-hidden"),
                           className=f"{cls} col-num is-missing")
        text = f"{int(value):,}" if key.startswith("rank") else f"{float(value):.4f}"
        return html.Td(text, className=f"{cls} col-num")
    text = _safe_str(value)
    if kind == "text":
        return html.Td(text, className=f"{cls} col-text", title=text or None)
    return html.Td(text, className=cls)


def _header(key: str, label: str, kind: str, sort: dict) -> Any:
    active = sort.get("column") == key
    direction = "descending" if sort.get("descending") else "ascending"
    return html.Th(
        html.Button(label, id={"type": "results-sort", "column": key}, n_clicks=0,
                    className="sort-button" + (f" is-{direction}" if active else "")),
        scope="col",
        className=f"col-{key}" + (" col-num" if kind == "num" else ""),
        **({"aria-sort": direction} if active else {}),
    )


def render_table(rows: list[dict], results: dict, sort: dict,
                 selected_gsm: str = "") -> Any:
    columns = _columns(results)
    k = int(results.get("k") or 0)
    body = [
        html.Tr(
            [_cell(row, key, kind, k) for key, _label, kind in columns],
            className="result-row" + (" is-selected" if row["gsm"] == selected_gsm else ""),
            **{"data-gsm": row["gsm"]},
        )
        for row in rows
    ]
    return html.Table(className="results-table", children=[
        html.Caption("Matches, ranked by embedding cosine similarity",
                     className="visually-hidden"),
        html.Thead(html.Tr([_header(key, label, kind, sort)
                            for key, label, kind in columns])),
        html.Tbody(body),
    ])


# --- Shared derivations ------------------------------------------------------

def _reference_count() -> int:
    from .retrieval import _archs4_sample_count  # noqa: PLC0415

    return _archs4_sample_count() or 0


def filters_from_inputs(values: list[Any]) -> dict:
    raw = {key: value for (_id, key), value in zip(FILTER_INPUTS, values)}
    merge = raw.get("merge_identical")
    raw["merge_identical"] = bool(merge and "on" in merge)
    return R.normalize_filters(raw)


def visible_rows(results: dict | None, filters: dict,
                 sort: dict | None) -> tuple[list[dict], int]:
    """The rows the table shows, in table order, and how many merging hid."""
    if not results or not results.get("rows"):
        return [], 0
    kept, hidden = R.apply_filters(results["rows"], filters)
    return R.sort_rows(kept, sort or DEFAULT_SORT), hidden


def _identical_gsm_groups(payload: dict) -> list[list[str]]:
    """Groups of retrieved accessions whose expression profiles are identical."""
    hits = list(payload.get("hits") or [])
    hits += list((payload.get("comparison") or {}).get("hits_b") or [])
    seen: dict[str, int] = {}
    for h in hits:
        idx = h.get("archs4_index")
        gsm = _safe_str(h.get("gsm"))
        if gsm and idx is not None and gsm not in seen:
            try:
                seen[gsm] = int(idx)
            except (TypeError, ValueError):
                continue
    if len(seen) < 2:
        return []
    gsms = list(seen)
    try:
        from .retrieval import reference_vectors  # noqa: PLC0415

        unit = reference_vectors([seen[g] for g in gsms])
    except Exception as exc:  # the table works without the marks
        print(f"[results] identical-profile check skipped: {exc}", flush=True)
        return []
    return [[gsms[i] for i in group] for group in identical_groups(unit)]


def _summary(results: dict, shown: int, hidden: int, filtered: bool) -> str:
    total = len(results.get("rows") or [])
    across = " across both cohorts" if results.get("comparison") else ""
    if filtered or hidden:
        text = f"{shown:,} of {total:,} matches shown{across}."
        if hidden:
            noun = "profile" if hidden == 1 else "profiles"
            text += f" {hidden:,} identical {noun} merged."
        ref = int(results.get("reference_count") or 0)
        if ref:
            text += f" Ranks are positions among all {ref:,} reference samples."
        return text
    label = _safe_str(results.get("label_a"))
    ranked = "ranked by embedding cosine similarity, not biological equivalence."
    if results.get("comparison"):
        return f"{total:,} matches across both cohorts, {ranked}"
    return f"{total:,} matches" + (f" for {label}, " if label else ", ") + ranked


def _option_list(pairs: list[tuple[str, int]]) -> list[dict]:
    return [{"label": f"{value} ({count:,})", "value": value} for value, count in pairs]


def _hits_frame(rows: list[dict], side: str) -> pd.DataFrame:
    """Rows retrieved by one side, as the hit frame the network figures read."""
    rank, score = f"rank_{side}", f"score_{side}"
    mine = sorted((r for r in rows if r.get(rank) is not None), key=lambda r: r[rank])
    return pd.DataFrame([{
        "gsm": r["gsm"], "score": r[score], "gse": r.get("gse", ""),
        "title": r.get("title", ""), "source_name": r.get("source_name", ""),
        "characteristics": "",
    } for r in mine[:P.NETWORK_MAX_ROWS]],
        columns=["gsm", "score", "gse", "title", "source_name", "characteristics"])


_EMPTY_PLOT = {
    "sample": "Select an OSDR sample, then run a search.",
    "cohort": "Define a cohort, then pool and search it.",
    "upload": "Upload a counts file, then embed and search it.",
}


def build_plot(kind: str, results: dict | None, rows: list[dict],
               selected: dict | None, mode: str = "sample") -> tuple[Any, str, str]:
    """(figure, caption, legend kind or "") for one plot view."""
    if not results or not results.get("rows"):
        return _empty_network_figure(_EMPTY_PLOT.get(mode, _EMPTY_PLOT["sample"])), "", ""
    if not rows:
        return _empty_network_figure("No matches pass these filters."), "", ""
    kind_sel = _safe_str((selected or {}).get("kind"))
    node = _safe_str((selected or {}).get("node_id"))
    selected_gsm = node if kind_sel == "gsm" else None
    selected_gse = node if kind_sel == "gse" else None
    n = len(rows)

    if kind == "similarity":
        locatable = [r for r in rows if r.get("archs4_index") is not None]
        take = locatable[:P.SIMILARITY_MAX_ROWS]
        if len(take) < 2:
            return (_empty_network_figure(
                "Similarity needs at least two matches from the reference index."),
                "", "")
        from .retrieval import reference_vectors  # noqa: PLC0415

        unit = reference_vectors([r["archs4_index"] for r in take])
        caption = (f"Pairwise cosine among the first {len(take):,} of {n:,} matches "
                   "in table order, clustered by average linkage."
                   if n > len(take) else
                   f"Pairwise cosine among {len(take):,} matches, clustered by "
                   "average linkage.")
        return P.similarity_figure(take, unit, selected_gsm), caption, ""

    if kind == "series":
        counts = P.series_counts(results, rows)
        shown = min(len(counts), P.SERIES_MAX_BARS)
        caption = (f"Matches per GEO series. Showing {shown:,} of {len(counts):,} series."
                   if len(counts) > shown else "Matches per GEO series.")
        return (P.series_figure(results, rows, selected_gsm, selected_gse),
                caption, "")

    if kind == "network":
        query_a = pd.Series(results.get("query_a") or {})
        if results.get("comparison"):
            fig = build_comparison_figure(
                query_a=query_a, hits_a=_hits_frame(rows, "a"),
                query_b=pd.Series(results.get("query_b") or {}),
                hits_b=_hits_frame(rows, "b"))
            legend = "comparison"
        else:
            fig = build_network_figure(query=query_a, hits_df=_hits_frame(rows, "a"))
            legend = "cohort" if results.get("mode") == "cohort" else "sample"
        # The network once had the whole canvas to itself. Below it now sits a
        # table, so it asks for the height its column of nodes needs: at 30
        # matches, squeezed into the plot panel's floor, the series labels on
        # the right ran into each other.
        drawn = sum(1 for t in fig.data
                    for c in (t.customdata if t.customdata is not None else [])
                    if len(c) > 1 and c[0] == "gsm")
        fig.update_layout(height=max(420, 26 * drawn + 80))
        cap = P.NETWORK_MAX_ROWS
        caption = (f"Showing the first {cap:,} of {n:,} matches"
                   + (" per cohort." if results.get("comparison") else ".")
                   if n > cap else "")
        return fig, caption, legend

    k = int(results.get("k") or 0)
    depth = len(results.get("profile_a") or [])
    caption = (f"Cosine similarity by rank to depth {depth:,}. "
               f"Points are the {k:,} retrieved matches"
               + ("; filtered-out matches are faded." if n < len(results["rows"]) else "."))
    return P.score_rank_figure(results, rows, selected_gsm), caption, ""


# --- Callbacks ---------------------------------------------------------------

def register(app) -> None:  # noqa: C901 - one callback per concern, registered together
    filter_inputs = [Input(cid, "value") for cid, _ in FILTER_INPUTS]
    filter_states = [State(cid, "value") for cid, _ in FILTER_INPUTS]

    @app.callback(
        Output("topk-slider", "value"),
        Output("topk-input", "value"),
        Input("topk-slider", "value"),
        Input("topk-input", "value"),
        prevent_initial_call=True,
    )
    def sync_match_count(position, typed):
        """Keep the log slider and the exact box saying the same k."""
        if ctx.triggered_id == "topk-slider":
            return no_update, k_from_slider(position)
        if typed in (None, ""):
            return no_update, no_update
        k = clamp_match_count(typed)
        return _log_k(k), (k if k != typed else no_update)

    @app.callback(
        Output("results-store", "data"),
        Output("filter-series", "options"),
        Output("filter-tissue", "options"),
        Output("filter-species", "options"),
        Output("filter-retrieved-by", "options"),
        Output("filter-retrieved-by-group", "style"),
        *[Output(cid, "value", allow_duplicate=True) for cid, _ in FILTER_INPUTS],
        Output("selected-node-store", "data", allow_duplicate=True),
        Output("see-on-map", "style"),
        Output("see-on-map", "children"),
        Input("hits-store", "data"),
        # The view's own panel, as a trigger. After a client-side route change
        # Dash reruns a callback only when one of its *inputs* is newly
        # mounted, and hits-store lives on the shell, which never remounts - so
        # with hits-store alone, coming back from the map showed an empty table
        # beside 472 KB of results still sitting in session storage. Measured
        # in the browser before this input was added.
        Input("results-panel", "id"),
        prevent_initial_call="initial_duplicate",
    )
    def derive_results(payload, _mounted):
        defaults = R.default_filters()
        reset = [
            defaults["series"], defaults["series_mode"], defaults["tissue"],
            defaults["species"], defaults["min_score"], defaults["text"],
            ["on"] if defaults["merge_identical"] else [], defaults["retrieved_by"],
        ]
        results = R.build_results(
            payload, reference_count=_reference_count(),
            identical_groups=_identical_gsm_groups(payload) if payload else ())
        if not results:
            return (None, [], [], [], [], {"display": "none"}, *reset, None,
                    {"display": "none"}, "")
        query = (payload or {}).get("query") or {}
        results["query_a"] = {"sample_id": _safe_str(query.get("sample_id")),
                              "sample_name": _safe_str(query.get("sample_name"))}
        comparison = (payload or {}).get("comparison") or {}
        query_b = comparison.get("query_b") or {}
        results["query_b"] = {"sample_id": _safe_str(query_b.get("sample_id")),
                              "sample_name": _safe_str(query_b.get("sample_name"))}
        options = R.filter_options(results["rows"])
        by_options, by_style = [], {"display": "none"}
        if results.get("comparison"):
            a, b = results.get("label_a") or "A", results.get("label_b") or "B"
            by_options = [
                {"label": "Either", "value": "either"},
                {"label": "Both", "value": "both"},
                {"label": f"Only {a}", "value": "a"},
                {"label": f"Only {b}", "value": "b"},
            ]
            by_style = {}
        locatable = any(r.get("archs4_index") is not None for r in results["rows"])
        return (
            results,
            _option_list(options["series"]),
            _option_list(options["tissue"]),
            _option_list(options["species"]),
            by_options, by_style,
            *reset,
            None,
            {} if locatable else {"display": "none"},
            "View results on map" if locatable else "",
        )

    @app.callback(
        Output("results-table-wrap", "children"),
        Output("results-summary", "children"),
        Output("results-page-label", "children"),
        Output("results-prev", "disabled"),
        Output("results-next", "disabled"),
        Output("filter-clear", "style"),
        Output("results-download-button", "disabled"),
        Output("results-page-store", "data"),
        Output("results-visible-store", "data"),
        Input("results-store", "data"),
        *filter_inputs,
        Input("results-sort-store", "data"),
        Input("results-page-store", "data"),
        State("selected-node-store", "data"),
    )
    def update_table(results, *args):
        *filter_values, sort, page, selected = args
        filters = filters_from_inputs(filter_values)
        active = R.filters_active(filters, results)
        clear_style = {} if active else {"display": "none"}
        if not results or not results.get("rows"):
            return (empty_table_message("Run a search to list its matches here."),
                    "", "", True, True, clear_style, True, 0, [])
        rows, hidden = visible_rows(results, filters, sort)
        # Anything but the pager asks for the first page: a new result, a
        # filter or a sort changes which rows page 3 would hold.
        requested = page if ctx.triggered_id == "results-page-store" else 0
        info = R.page_rows(rows, requested or 0)
        summary = _summary(results, len(rows), hidden, active)
        if not rows:
            return (empty_table_message("No matches pass these filters."),
                    summary, "", True, True, clear_style, True, info["page"], [])
        selected_gsm = (_safe_str((selected or {}).get("node_id"))
                        if _safe_str((selected or {}).get("kind")) == "gsm" else "")
        label = (f"Rows {info['start']:,}-{info['end']:,} of {info['total']:,}"
                 if info["pages"] > 1 else "")
        return (
            render_table(info["rows"], results, sort or DEFAULT_SORT, selected_gsm),
            summary, label,
            info["page"] <= 0, info["page"] >= info["pages"] - 1,
            clear_style, False, info["page"],
            [r["gsm"] for r in rows],
        )

    @app.callback(
        Output("results-page-store", "data", allow_duplicate=True),
        Input("results-prev", "n_clicks"),
        Input("results-next", "n_clicks"),
        State("results-page-store", "data"),
        prevent_initial_call=True,
    )
    def turn_page(_prev, _next, page):
        step = -1 if ctx.triggered_id == "results-prev" else 1
        return max(0, int(page or 0) + step)

    @app.callback(
        Output("results-sort-store", "data"),
        Input({"type": "results-sort", "column": ALL}, "n_clicks"),
        State("results-sort-store", "data"),
        prevent_initial_call=True,
    )
    def sort_by(clicks, sort):
        # A pattern-matching input also fires when its family is re-rendered,
        # which every page turn does; only a real click carries a count.
        if not ctx.triggered or not any(t.get("value") for t in ctx.triggered):
            return no_update
        column = _safe_str((ctx.triggered_id or {}).get("column"))
        sort = sort or DEFAULT_SORT
        if sort.get("column") == column:
            return {"column": column, "descending": not sort.get("descending")}
        # Scores read best first; ranks and text read from the top.
        return {"column": column, "descending": column.startswith("score")}

    @app.callback(
        *[Output(cid, "value", allow_duplicate=True) for cid, _ in FILTER_INPUTS],
        Input("filter-clear", "n_clicks"),
        prevent_initial_call=True,
    )
    def clear_filters(_n):
        d = R.default_filters()
        return (d["series"], d["series_mode"], d["tissue"], d["species"],
                d["min_score"], d["text"], ["on"] if d["merge_identical"] else [],
                d["retrieved_by"])

    @app.callback(
        Output("selected-node-store", "data", allow_duplicate=True),
        Input({"type": "result-row", "gsm": ALL}, "n_clicks"),
        prevent_initial_call=True,
    )
    def select_row(_clicks):
        if not ctx.triggered or not any(t.get("value") for t in ctx.triggered):
            return no_update
        return {"kind": "gsm", "node_id": _safe_str((ctx.triggered_id or {}).get("gsm"))}

    @app.callback(
        Output("selected-node-store", "data", allow_duplicate=True),
        Input("results-plot", "clickData"),
        prevent_initial_call=True,
    )
    def select_mark(click_data):
        points = (click_data or {}).get("points") or []
        custom = points[0].get("customdata") if points else None
        if not custom or len(custom) < 2 or not custom[1]:
            return no_update
        return {"kind": _safe_str(custom[0]), "node_id": _safe_str(custom[1])}

    # The inspector needs the selected hit's full GEO record, which lives in
    # hits-store. Picking it out in the browser keeps a row click from posting
    # the whole payload (385 KB at k=1,000) back to the server.
    app.clientside_callback(
        ClientsideFunction(namespace="results", function_name="inspectorContext"),
        Output("inspector-store", "data"),
        Input("selected-node-store", "data"),
        Input("hits-store", "data"),
    )
    app.clientside_callback(
        ClientsideFunction(namespace="results", function_name="markSelected"),
        Output("results-selection-painted", "data"),
        Input("selected-node-store", "data"),
        Input("results-table-wrap", "children"),
    )

    @app.callback(
        Output("results-plot", "figure"),
        Output("plot-caption", "children"),
        Output("graph-legend", "style"),
        Output("graph-legend", "children"),
        Input("results-store", "data"),
        *filter_inputs,
        Input("results-sort-store", "data"),
        Input("results-plot-kind", "value"),
        Input("selected-node-store", "data"),
        Input("query-mode-store", "data"),
    )
    def update_plot(results, *args):
        *filter_values, sort, kind, selected, mode = args
        # The network draws no selection of its own: assets/network-focus.js
        # emphasises the clicked node in the SVG, and a rebuilt figure is a new
        # graph to it, so redrawing on a click dropped that emphasis the moment
        # the pointer left (e2e_network_style.py caught it). The other three
        # views mark the selection themselves and do redraw.
        if (ctx.triggered_id == "selected-node-store"
                and (_safe_str(kind) or DEFAULT_PLOT_KIND) == "network"):
            return no_update, no_update, no_update, no_update
        rows, _hidden = visible_rows(results, filters_from_inputs(filter_values), sort)
        try:
            fig, caption, legend = build_plot(_safe_str(kind) or DEFAULT_PLOT_KIND,
                                              results, rows, selected,
                                              _safe_str(mode) or "sample")
        except Exception as exc:  # a plot must never take the table down with it
            print(f"[results] plot failed: {exc}", flush=True)
            return (_empty_network_figure("This plot could not be drawn."),
                    _safe_str(exc), {"display": "none"}, no_update)
        if not legend:
            return fig, caption, {"display": "none"}, no_update
        from .layout import build_graph_legend  # noqa: PLC0415

        return fig, caption, {}, build_graph_legend(legend)

    @app.callback(
        Output("results-download", "data"),
        Input("results-download-button", "n_clicks"),
        State("results-store", "data"),
        *filter_states,
        State("results-sort-store", "data"),
        State("hits-store", "data"),
        prevent_initial_call=True,
    )
    def download_rows(n_clicks, results, *args):
        *filter_values, sort, payload = args
        if not n_clicks or not results or not results.get("rows"):
            return no_update
        filters = filters_from_inputs(filter_values)
        rows, _hidden = visible_rows(results, filters, sort)
        hits = list((payload or {}).get("hits") or [])
        hits += list(((payload or {}).get("comparison") or {}).get("hits_b") or [])
        lookup: dict[str, dict] = {}
        for h in hits:
            lookup.setdefault(_safe_str(h.get("gsm")), h)
        now = datetime.now(timezone.utc)
        text = R.to_csv(rows, results, filters, sort or DEFAULT_SORT, lookup,
                        now.strftime("%Y-%m-%d %H:%M:%S UTC"))
        stem = "".join(c if c.isalnum() or c in "-_." else "_"
                       for c in _safe_str(results.get("label_a")) or "retrieval")
        return dcc.send_string(text, f"bridge-rna_{stem}_{now:%Y%m%d-%H%M%S}.csv")
