"""Retrieval callbacks, registered onto the shell's Dash app.

These were module-level `@app.callback` decorators against a module-level app
object, which is exactly what a router cannot mount. They are the same bodies,
one indent level deeper, inside a `register(app)` the shell calls once.
"""

from __future__ import annotations

import atexit
import base64
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs

import pandas as pd
from dash import ALL, Input, Output, State, ctx, html, no_update

from .ai import (
    _call_ai_summary,
    _format_geo_context_text,
    _format_hits_table_text,
    _format_osdr_query_text,
    _load_ai_prompt_template,
    unavailable_reason,
)
from . import cohorts as C
from .config import GENERIC_ENTREZ_EMAIL, MAX_UPLOAD_BYTES
from .figures import (
    _empty_network_figure,
    build_comparison_figure,
    build_network_figure,
)
from .geo import _enrich_hits_from_ncbi_eutils
from .layout import (
    ARCHS4_SAMPLE_COUNT,
    CANVAS_SUBTITLES,
    QUERY_MODES,
    build_graph_legend,
    samples_df,
)
from .neighborhoods import (
    build_payload,
    evidence_unavailable_reason,
    unavailable_payload,
)
from .panels import (
    build_cohort_card,
    build_details_panel,
    build_stability_panel,
    build_status_banner,
)
from .retrieval import (
    COHORT_MODE,
    TIER_CACHED,
    TIER_SUBPROCESS,
    TIER_UNAVAILABLE,
    UPLOAD_MODE,
    cached_query_vectors,
    run_cohort_retrieval_with_neighborhood,
    run_uploaded_retrieval_with_neighborhood,
    sample_tier,
    search_hits_with_neighborhood,
)
from .util import _format_count, _last_nonempty_line, _safe_str


def _evidence_payload(frame: pd.DataFrame | None, label: str = "", *,
                      mode: str = "") -> dict:
    if frame is None:
        return unavailable_payload(evidence_unavailable_reason(mode))
    return build_payload(frame, label=label)


def _retrieval_phrase(mode: str) -> str:
    """The clause the status banner uses to name the path that answered.

    Kept in one place so the dropdown path and the uploaded path describe
    themselves consistently, and so a new path names itself here rather than in a
    callback body. The invariant is that the interface always says which ran.
    """
    return {
        "cached": "from the precomputed OSDR embedding, scored against all "
                  f"{_format_count(ARCHS4_SAMPLE_COUNT)} ARCHS4 samples",
        "precomputed": "from a supplied query-embedding table",
        "demo": "by embedding the counts matrix from scratch",
        UPLOAD_MODE: "by embedding the uploaded counts matrix live, scored against "
                     f"all {_format_count(ARCHS4_SAMPLE_COUNT)} ARCHS4 samples",
        COHORT_MODE: "from the pooled mean of the cohort's precomputed "
                     "embeddings, scored against all "
                     f"{_format_count(ARCHS4_SAMPLE_COUNT)} ARCHS4 samples",
    }.get(mode, mode)


# --- Cohort mode -------------------------------------------------------------


def _cohort_query_series(cohort, geometry, excluded: list[str]) -> pd.Series:
    """A query identity for a pooled cohort, standing where a sample row does.

    A cohort is not in `samples_df` and has no single sample name, so the figure
    and the inspector need an identity synthesized from the group. It carries
    `is_cohort` so the inspector renders it as a group rather than as one sample
    with a blank name, and the member lists travel as newline-joined strings
    because the store is JSON and this keeps the payload flat.

    `sample_id` is deliberately *not* a member's key. The banner and the query
    node must never label a pooled result with one animal's name.
    """
    grouped = ", ".join(C.FACETS_BY_KEY[k].label for k in cohort.facets)
    outliers = [m for m, flag in zip(geometry.members, geometry.outliers) if flag]
    return pd.Series({
        "sample_id": cohort.cohort_id,
        "sample_name": cohort.describe(),
        "cohort_label": cohort.label,
        "study_id": cohort.study,
        "tissue": cohort.values.get("tissue", ""),
        "condition": cohort.values.get("spaceflight", ""),
        "is_cohort": "1",
        "grouped_by": grouped,
        # No stability field. It used to carry `expected_stability(k)`, a curve
        # looked up by cohort size; it is measured during the search now and
        # travels in the payload's own `stability` block, which the inspector's
        # stability panel reads. See docs/design-notes.md#live-stability.
        "members": "\n".join(geometry.members),
        "excluded": "\n".join(excluded),
        "outliers": "\n".join(outliers),
    })


def _geometry_for(members: list[str]) -> tuple[Any, list[str]]:
    """Cohort geometry over the members that have a cached vector.

    Returns `(geometry, missing)`. A member with no cached embedding cannot be
    pooled, and silently averaging the rest would make the interface state a
    size it did not use.
    """
    rows, missing = cached_query_vectors(members)
    usable = [m for m in members if m not in set(missing)]
    if rows.shape[0] == 0:
        return None, missing
    return C.cohort_geometry(usable, rows), missing


def _cohort_options(study: str, facets: list[str]) -> list[dict[str, Any]]:
    """This study's cohorts, with size, and singletons disabled with the reason.

    Same treatment the sample picker gives an unretrievable sample: shown,
    disabled, and saying why, rather than hidden. A cohort that vanishes looks
    like a bug in the grouping; a disabled one is a fact about the study.
    """
    opts = []
    for c in C.build_cohorts(facets=facets, study=study):
        singleton = c.size < C.MIN_COHORT_SIZE
        suffix = ("  ·  1 sample, nothing to pool" if singleton
                  else f"  ·  {c.size} samples"
                       + ("  ·  low N" if c.tier == C.TIER_LOW_N else ""))
        opts.append({"label": f"{c.label}{suffix}", "value": c.cohort_id,
                     "disabled": singleton})
    return opts


def _uploaded_query_series(filename: str, column: str) -> pd.Series:
    """A minimal query row standing in for an uploaded sample.

    An uploaded sample is not in `samples_df`, so the figure and inspector need a
    query identity synthesized from the file. OSDR spaceflight data is Mus
    musculus - which `_build_query_details` already hardcodes - and the finer
    biology fields are genuinely unknown, so they are left blank rather than
    guessed.
    """
    return pd.Series({
        "sample_id": f"UPLOAD|{filename}::{column}",
        "sample_name": column or filename,
        "study_id": "Uploaded file",
        "tissue": "",
        "condition": "",
        "strain": "",
        "sex": "",
        "duration": "",
    })


def _query_series(hits_payload: dict[str, Any] | None) -> pd.Series | None:
    """Resolve the query row for a stored retrieval, uploaded or catalog.

    Prefers the `query` dict carried in the payload (the only source an uploaded
    sample has) and falls back to the OSDR catalog by `sample_id` so retrievals
    stored before this field existed still resolve.
    """
    q = (hits_payload or {}).get("query")
    if isinstance(q, dict) and q:
        return pd.Series(q)
    sid = _safe_str((hits_payload or {}).get("sample_id"))
    match = samples_df.loc[samples_df["sample_id"] == sid]
    return match.iloc[0] if not match.empty else None


def _query_dict(query: pd.Series) -> dict[str, str]:
    """A JSON-safe, NaN-free view of a query row for the hits-store payload."""
    return {str(k): _safe_str(v) for k, v in query.to_dict().items()}


# --- staging an uploaded counts file ----------------------------------------
#
# An upload has to reach disk, because the embedder is a subprocess and takes a
# path. The first version wrote a `NamedTemporaryFile(delete=False)` per upload
# and never removed any of them, so a long-lived instance accumulated one
# whole counts matrix (about 1.3 MB for a two-sample OSDR file, up to the
# 200 MB cap) per upload, forever, surviving process exit. A single looped E2E
# run left 32 files and 29 MB behind.
#
# Three bounds fix that without changing the flow: every staged file lives in
# one process-owned directory removed at exit, each session's previous file is
# unlinked when its next upload arrives, and a directory abandoned by a killed
# process is reaped by the next run. Steady state is one file per active
# session, and nothing outlives the process for longer than one restart.
_UPLOAD_DIR: Path | None = None
_UPLOAD_DIR_PREFIX = "bridge_rna_uploads_"


def _sweep_abandoned_upload_dirs() -> list[Path]:
    """Reap staging directories whose owning process is gone.

    Exit-time cleanup cannot carry this on its own. `atexit` does not run on
    SIGTERM, which is how a supervisor stops a server, nor on SIGKILL or a
    crash - and a signal handler is not an option either, because uploads
    arrive on Dash's request threads and `signal.signal` may only be called
    from the main one. So each staging directory is tagged with its owner's
    PID, and the next run reaps the ones whose owner is no longer running.

    PID reuse can only make a dead directory look alive, which delays its
    cleanup by one run; it can never remove a live server's staged file.
    """
    reaped: list[Path] = []
    for d in Path(tempfile.gettempdir()).glob(f"{_UPLOAD_DIR_PREFIX}*"):
        m = re.match(rf"{_UPLOAD_DIR_PREFIX}(\d+)_", d.name)
        if not d.is_dir() or not m:
            continue
        pid = int(m.group(1))
        # `os.kill(0, ...)` addresses the whole process group rather than a
        # process, so PID 0 must never reach the liveness probe.
        if pid <= 0 or pid == os.getpid():
            continue
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            shutil.rmtree(d, ignore_errors=True)
            reaped.append(d)
        except OSError:
            pass  # Alive but owned by another user: not ours to remove.
    return reaped


def _upload_dir() -> Path:
    """The one directory staged uploads live in, created on first use."""
    global _UPLOAD_DIR
    if _UPLOAD_DIR is None:
        _sweep_abandoned_upload_dirs()
        _UPLOAD_DIR = Path(tempfile.mkdtemp(
            prefix=f"{_UPLOAD_DIR_PREFIX}{os.getpid()}_"))
        atexit.register(shutil.rmtree, _UPLOAD_DIR, True)
    return _UPLOAD_DIR


def _stage_upload(raw: bytes, suffix: str) -> str:
    fh = tempfile.NamedTemporaryFile(
        prefix="upload_", suffix=suffix, dir=str(_upload_dir()), delete=False)
    try:
        fh.write(raw)
    finally:
        fh.close()
    return fh.name


def _discard_upload(previous: dict[str, Any] | None) -> None:
    """Remove the file a previous upload in this session staged, if any.

    Confined to `_upload_dir()` so a malformed or stale store cannot point this
    at anything the app did not itself write.
    """
    path = _safe_str((previous or {}).get("path"))
    if not path:
        return
    try:
        p = Path(path).resolve()
        if p.is_file() and p.parent == _upload_dir().resolve():
            p.unlink()
    except OSError:
        pass


def _format_bytes(n: int) -> str:
    size = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{n} B"


def _requested_sample(search: str | None) -> str:
    """The `q` parameter of a `/?q=<sample_id>` URL, or "".

    This is how the map hands a sample to the retrieval view. It is a URL
    parameter rather than a shared store so the link is a real link: it can be
    opened in a new tab, bookmarked, or pasted to a colleague.
    """
    if not search:
        return ""
    return _safe_str(parse_qs(search.lstrip("?")).get("q", [""])[0])


def register(app) -> None:
    """Attach every retrieval callback to `app`. Called once by the shell."""

    @app.callback(
        Output("study-dropdown", "value"),
        Input("url", "search"),
        prevent_initial_call=True,
    )
    def study_from_query_string(search: str | None):
        """Follow a `/?q=<sample_id>` link to the right study when the URL changes.

        This handles *live* navigation - clicking the map's "Retrieve its Earth
        analogs" link changes `url.search` without a page load, and this switches
        the study to match. It carries `prevent_initial_call` because firing on
        the initial load fought the study dropdown's own initialization and left
        both dropdowns empty. The *cold-load* case - a pasted or bookmarked
        `/?q=` - is handled instead by `layout._initial_study`, which reads the
        request and picks the right study before any callback runs.
        """
        sample_id = _requested_sample(search)
        if not sample_id:
            return no_update
        match = samples_df.loc[samples_df["sample_id"].astype(str) == sample_id]
        return _safe_str(match.iloc[0]["study_id"]) if not match.empty else no_update

    @app.callback(
        Output("sample-dropdown", "options"),
        Output("sample-dropdown", "value"),
        Input("study-dropdown", "value"),
        State("url", "search"),
    )
    def update_sample_options(study_id: str, search: str | None = None):
        """List a study's samples, and say up front which ones can be answered.

        788 of the 2,896 samples cannot be retrieved by any path: 733 have no
        recorded spaceflight value, which the embedding filter drops before it
        looks for the sample, and 55 pass that filter but match no column in
        their counts matrix. They stay in the list and are disabled rather than
        hidden: a sample that silently vanishes looks like a bug in the
        catalogue, while a disabled one with a reason is a fact about the data.
        This is the treatment the map's color-by menu gives a field whose
        artifact is not built.
        """
        filtered = samples_df[samples_df["study_id"] == study_id].copy()
        suffix = {
            TIER_CACHED: "",
            TIER_SUBPROCESS: "  ·  slow, not on the map",
            TIER_UNAVAILABLE: "  ·  unavailable",
        }
        opts = []
        for _, r in filtered.iterrows():
            tier = sample_tier(_safe_str(r["sample_id"]), _safe_str(r["sample_name"]),
                               _safe_str(r.get("counts_path")),
                               _safe_str(r.get("condition")))
            label = (f"{_safe_str(r['sample_name'])} | {_safe_str(r['condition'])}"
                     f" | {_safe_str(r['tissue'])}{suffix[tier]}")
            opts.append({"label": label, "value": _safe_str(r["sample_id"]),
                         "disabled": tier == TIER_UNAVAILABLE})
        # Never select a disabled option. Two studies - OSD-462 (54 samples)
        # and OSD-374 (16) - have nothing selectable at all, and choosing
        # opts[0] there would arm Search with a query that cannot succeed.
        # Selecting nothing is the honest state, and the preview explains it.
        enabled = [o for o in opts if not o["disabled"]]
        # A `?q=` link from the map wins, provided that sample is in this study
        # and is actually retrievable. Otherwise the first enabled option.
        requested = _requested_sample(search)
        if requested and any(o["value"] == requested for o in enabled):
            return opts, requested
        return opts, (enabled[0]["value"] if enabled else None)


    @app.callback(
        Output("sample-preview", "children"),
        Input("sample-dropdown", "value"),
        State("study-dropdown", "value"),
    )
    def update_sample_preview(sample_id: str, study_id: str | None = None):
        """Instant local summary of the selected OSDR sample, shown before any search."""
        empty = html.P("Select a sample to preview its metadata.", className="sample-preview-empty")
        if not sample_id:
            # No sample is selected. Usually that just means the page just
            # loaded - but for the 24 studies where every sample is
            # unavailable, the picker deliberately selects nothing, and a
            # generic "select a sample" hides why nothing is selectable. Say so.
            if study_id is not None:
                in_study = samples_df[samples_df["study_id"] == study_id]
                if len(in_study) and all(
                    sample_tier(_safe_str(r["sample_id"]), _safe_str(r["sample_name"]),
                                _safe_str(r.get("counts_path")), _safe_str(r.get("condition")))
                    == TIER_UNAVAILABLE
                    for _, r in in_study.iterrows()
                ):
                    return html.P(
                        f"No sample in {study_id} can be retrieved: none has a "
                        "spaceflight condition recorded, or the counts matrix "
                        "does not contain it. Choose another study.",
                        className="sample-preview-empty")
            return empty
        # Belt and braces: the picker never selects a disabled option, but if a
        # sample reaches here that cannot be answered, say so before the click
        # rather than after a wait.
        match_any = samples_df.loc[samples_df["sample_id"].astype(str) == str(sample_id)]
        if not match_any.empty:
            r0 = match_any.iloc[0]
            if sample_tier(_safe_str(r0["sample_id"]), _safe_str(r0["sample_name"]),
                           _safe_str(r0.get("counts_path")),
                           _safe_str(r0.get("condition"))) == TIER_UNAVAILABLE:
                reason = ("no spaceflight condition is recorded for it"
                          if not _safe_str(r0.get("condition"))
                          else "its name matches no column in its study's counts matrix")
                return html.P(
                    f"This sample cannot be retrieved: {reason}, so there is "
                    "nothing to embed.",
                    className="sample-preview-empty")
        match = samples_df.loc[samples_df["sample_id"].astype(str) == str(sample_id)]
        if match.empty:
            return empty
        row = match.iloc[0]

        def _tidy(value: Any) -> str:
            # Unwrap ISA-Tab unit annotations, e.g. "37 {day}" -> "37 day".
            return re.sub(r"\s*\{([^}]*)\}", r" \1", _safe_str(value)).strip()

        fields = [
            ("Study", _tidy(row.get("study_id"))),
            ("Tissue", _tidy(row.get("tissue"))),
            ("Spaceflight", _tidy(row.get("condition"))),
            ("Strain", _tidy(row.get("strain"))),
            ("Sex", _tidy(row.get("sex"))),
            ("Duration", _tidy(row.get("duration"))),
        ]
        detail_rows = [
            html.Div(
                className="sample-preview-row",
                children=[
                    html.Span(label, className="sample-preview-key"),
                    html.Span(value, className="sample-preview-val"),
                ],
            )
            for label, value in fields
            if value
        ]
        return html.Div(
            className="sample-preview-card",
            children=[
                html.Div(_safe_str(row.get("sample_name")), className="sample-preview-name"),
                html.Div(className="sample-preview-grid", children=detail_rows),
            ],
        )


    @app.callback(
        Output("network-graph", "figure"),
        Output("hits-store", "data"),
        Output("search-status", "children"),
        Input("search-button", "n_clicks"),
        State("sample-dropdown", "value"),
        State("topk-slider", "value"),
        State("entrez-email-input", "value"),
        State("biopython-toggle", "value"),
        running=[
            (Output("search-button", "disabled"), True, False),
            (Output("query-running-indicator", "children"), "Finding related ARCHS4 samples...", ""),
        ],
    )
    def run_search(
        _: int,
        sample_id: str,
        topk: int,
        entrez_email: str | None,
        biopython_toggle: list[str] | None,
    ):
        if not sample_id:
            return (
                _empty_network_figure("Select an OSDR sample, then run a search."),
                None,
                None,
            )

        q_row = samples_df.loc[samples_df["sample_id"] == sample_id].iloc[0]
        enable_biopython = bool(biopython_toggle and "on" in biopython_toggle)
        email_value = _safe_str(entrez_email) or GENERIC_ENTREZ_EMAIL
        try:
            hits_df, neighborhood_df, mode = search_hits_with_neighborhood(
                samples_df=samples_df,
                sample_id=sample_id,
                topk=int(topk),
                entrez_email=email_value,
                enable_biopython_metadata=enable_biopython,
            )
        except Exception as exc:
            detail = getattr(exc, "detail", "") or _safe_str(exc)
            return (
                _empty_network_figure("Retrieval failed - see status for details."),
                None,
                build_status_banner(
                    _last_nonempty_line(_safe_str(exc)) or "Retrieval failed.",
                    kind="error",
                    detail=detail,
                ),
            )

        network = build_network_figure(query=q_row, hits_df=hits_df)

        # Name the path that actually ran. This used to special-case only
        # "precomputed", so when the cached path was added every one of its
        # results was announced as "real demo script output" - the interface
        # asserting something that was not true about how the answer was made.
        how = _retrieval_phrase(mode)
        # The precomputed path (a supplied query-embedding table) does not
        # enrich even when the box is ticked, so it must not claim to. Only the
        # cached and demo paths run the NCBI fetch.
        enriched = (enable_biopython and _safe_str(entrez_email)
                    and mode in ("cached", "demo"))
        status_message = (
            f"Retrieved {len(hits_df)} hits {how}"
            + (", plus GEO and PubMed enrichment." if enriched else ".")
        )
        status = build_status_banner(status_message, kind="good")
        payload = {
            "sample_id": sample_id,
            "entrez_email": email_value,
            "biopython_enabled": bool(enable_biopython),
            "mode": mode,
            "hits": hits_df.to_dict(orient="records"),
            "query": _query_dict(q_row),
            "neighborhood": _evidence_payload(
                neighborhood_df, label=_safe_str(q_row.get("sample_name")),
                mode=mode,
            ),
        }
        return network, payload, status


    # --- Cohort mode ------------------------------------------------------

    @app.callback(
        Output("query-mode-store", "data"),
        *[Output(f"mode-tab-{m['key']}", "className") for m in QUERY_MODES],
        *[Output(f"mode-tab-{m['key']}", "aria-selected") for m in QUERY_MODES],
        *[Output(f"mode-panel-{m['key']}", "style") for m in QUERY_MODES],
        *[Output(f"action-slot-{m['key']}", "style") for m in QUERY_MODES],
        Output("study-group", "style"),
        Output("network-graph", "figure", allow_duplicate=True),
        *[Input(f"mode-tab-{m['key']}", "n_clicks") for m in QUERY_MODES],
        State("hits-store", "data"),
        # Which mode we are already in, so a call this callback did not ask for
        # can restate it instead of guessing. See the fallback below.
        State("query-mode-store", "data"),
        # The layout already renders Sample selected and the other two hidden,
        # so there is nothing for the initial call to do - and doing it anyway
        # was a real bug rather than a wasted round trip. Restyling the action
        # slots on load remounted the buttons inside them, and a Dash callback
        # fires when an input component newly appears, so both the cohort and
        # the upload search callbacks ran at page load with `n_clicks: 0`.
        # The canvas greeted every visitor with "Cohort retrieval failed".
        prevent_initial_call=True,
    )
    def switch_query_mode(*args):
        """Show one query source at a time, and only that one.

        The rail used to stack the sample picker and the upload dropzone, which
        left the reader to work out which one the Search button was armed with.
        Only one query can run, so only one source is on screen, and the button
        that runs it is the only button in the action slot.

        Everything is derived from `QUERY_MODES` rather than hand-listed, so a
        fifth source is one entry there and no edits here.
        """
        hits_payload, stored_mode = args[-2], args[-1]
        triggered = _safe_str(ctx.triggered_id)
        active = next((m["key"] for m in QUERY_MODES
                       if f"mode-tab-{m['key']}" == triggered), "")

        if not active:
            # Not a tab click, so this is a call nobody asked for - Dash fires a
            # callback when an input component newly appears, which is the same
            # mechanism that once ran both searches at `n_clicks: 0`.
            #
            # Restate the mode we are already in. Falling back to QUERY_MODES[0]
            # would silently drop the rail back to Sample, which is a guess
            # dressed as a decision: this callback has no evidence the user
            # wants Sample, only that something remounted a tab.
            #
            # This is belt and braces rather than the fix for anything observed:
            # the bug that *looked* like this - clicking Cohort on arrival and
            # watching the panel close itself - was the router repainting the
            # whole view on top of a correct answer, and it is fixed in
            # `app.render_page`.
            keys = {m["key"] for m in QUERY_MODES}
            candidate = _safe_str(stored_mode)
            active = candidate if candidate in keys else QUERY_MODES[0]["key"]
        shown, hidden = {}, {"display": "none"}

        # The empty canvas has to invite the thing the active mode can actually
        # do. Left alone it kept saying "Select an OSDR sample" while Cohort mode
        # was open, which is the smallest version of the same fault as a status
        # banner naming the wrong path. A drawn network is never replaced -
        # switching modes to look at the controls must not throw away a result.
        canvas = no_update
        if not hits_payload:
            canvas = _empty_network_figure({
                "sample": "Select an OSDR sample, then run a search.",
                "cohort": "Define a cohort, then pool and search it.",
                "upload": "Upload a counts file, then embed and search it.",
            }[active])

        return (
            active,
            *["mode-tab" + (" is-active" if m["key"] == active else "")
              for m in QUERY_MODES],
            *["true" if m["key"] == active else "false" for m in QUERY_MODES],
            *[shown if m["key"] == active else hidden for m in QUERY_MODES],
            *[shown if m["key"] == active else hidden for m in QUERY_MODES],
            # Upload has no study; Sample and Cohort share one.
            hidden if active == "upload" else shown,
            canvas,
        )

    @app.callback(
        Output("cohort-facets-store", "data"),
        Output({"type": "facet-chip", "key": ALL}, "className"),
        Output({"type": "facet-chip", "key": ALL}, "aria-pressed"),
        Input({"type": "facet-chip", "key": ALL}, "n_clicks"),
        State("cohort-facets-store", "data"),
    )
    def toggle_facet(_clicks, current: list[str] | None):
        """Add or remove one facet from the cohort definition.

        `normalize_facets` is what actually decides the set, so the pinned study
        cannot be removed even by a crafted request, and the order is always the
        registry's rather than click order.
        """
        chosen = set(C.normalize_facets(current))
        clicked = ctx.triggered_id
        if isinstance(clicked, dict) and clicked.get("type") == "facet-chip":
            key = clicked.get("key")
            if key in C.FACETS_BY_KEY and key not in C.PINNED_FACETS:
                chosen.symmetric_difference_update({key})
        active = C.normalize_facets(chosen)
        return (
            list(active),
            ["facet-chip" + (" is-on" if f.key in active else "")
             + (" is-pinned" if f.pinned else "") for f in C.FACETS],
            ["true" if f.key in active else "false" for f in C.FACETS],
        )

    @app.callback(
        Output("cohort-facet-summary", "children"),
        Input("cohort-facets-store", "data"),
    )
    def describe_cohort_definition(facets: list[str] | None):
        """What the current definition produces, over the whole corpus.

        Corpus-wide rather than per-study on purpose: this line reports the
        consequence of the *definition*, and a definition is not a property of
        whichever study happens to be selected. The per-study consequence is
        already visible in the cohort dropdown right below it.
        """
        if not C.cohorts_available():
            return html.Span(
                "Cohorts need cache/osdr_metadata.parquet. Run "
                "precompute/embed_osdr.py.", className="facet-summary-warn")
        groups = [c for c in C.build_cohorts(facets=facets)]
        poolable = [c for c in groups if c.size >= C.MIN_COHORT_SIZE]
        if not poolable:
            return html.Span("This definition produces no cohort with two or "
                             "more samples.", className="facet-summary-warn")
        sizes = [c.size for c in poolable]
        pooled = sum(sizes)
        return [
            html.Strong(f"{len(poolable)} cohorts"),
            f" across {len({c.study for c in poolable})} studies · "
            f"{min(sizes)} to {max(sizes)} samples each · "
            f"{pooled:,} of {len(C.cohort_metadata()):,} samples grouped",
        ]

    @app.callback(
        Output("cohort-dropdown", "options"),
        Output("cohort-dropdown", "value"),
        Input("study-dropdown", "value"),
        Input("cohort-facets-store", "data"),
        State("cohort-dropdown", "value"),
    )
    def update_cohort_options(study: str, facets: list[str] | None,
                              current: str | None):
        """List this study's cohorts, keeping the selection where it survives.

        Reticking a facet redefines every cohort, so most selections cannot
        survive it. The one that can is an unchanged id, and holding onto it
        means widening from tissue+arm to arm alone does not throw away the
        cohort you were looking at when it happens to be the same group.
        """
        opts = _cohort_options(_safe_str(study), facets or list(C.DEFAULT_FACETS))
        enabled = [o for o in opts if not o["disabled"]]
        if current and any(o["value"] == current for o in enabled):
            return opts, current
        return opts, (enabled[0]["value"] if enabled else None)

    @app.callback(
        Output("cohort-members", "options"),
        Output("cohort-members", "value"),
        Input("cohort-dropdown", "value"),
        State("cohort-facets-store", "data"),
    )
    def update_cohort_members(cohort_id: str | None, facets: list[str] | None):
        """List the cohort's members, every one of them ticked to begin with.

        Each row carries its leave-one-out cosine: the member's agreement with
        the centroid of *the others*, which is the statistic that can actually
        surface an outlier. Scoring against the full centroid would let an
        outlier drag the reference towards itself and hide inside it.
        """
        cohort = C.find_cohort(_safe_str(cohort_id), facets=facets)
        if cohort is None:
            return [], []
        geometry, _missing = _geometry_for(list(cohort.members))
        if geometry is None:
            return [], []
        options = []
        for name, loo, is_outlier in zip(geometry.members, geometry.loo_cosines,
                                         geometry.outliers):
            short = name.split("|", 1)[-1]
            tag = "  ·  furthest from the rest" if is_outlier else ""
            options.append({
                "label": f"{short}  ·  {loo:.4f}{tag}",
                "value": name,
            })
        return options, list(geometry.members)

    @app.callback(
        Output("cohort-card", "children"),
        Output("cohort-compare-card", "children"),
        Output("cohort-members-summary", "children"),
        Output("cohort-search-button", "disabled"),
        Input("cohort-members", "value"),
        Input("cohort-dropdown", "value"),
        Input("cohort-compare-dropdown", "value"),
        State("cohort-facets-store", "data"),
    )
    def update_cohort_card(included: list[str] | None, cohort_id: str | None,
                           compare_id: str | None, facets: list[str] | None):
        """Restate the pooled size of every query that is about to run.

        Size, and nothing else. This used to restate a stability figure too,
        looked up from a curve by that size; the number is measured during the
        search now and `build_stability_panel` reports it in the inspector
        afterwards, so what is left here is the one fact the rail can state
        before anything has been scored.

        It reads the *included* members rather than the cohort's full
        membership, because excluding two of six changes the size the card
        states. A card that kept quoting the cohort's nominal size while the
        pool shrank underneath it would be the same failure as the status banner
        that announced cached results as subprocess output.

        The same argument reaches one step further and is why this writes two
        cards. Arming a comparison runs a *second* independent pooled query, and
        the rail used to say nothing about its size at all while both the
        network figure and the map gave it a color and drew it. A pooled query
        that is described nowhere is one the reader cannot weigh: cohort B can
        easily be the smaller and shakier of the two, which is exactly what
        decides how much of the overlap number to believe.

        B's membership is not editable - the member ticks belong to the
        selected cohort - so its card always describes the whole sibling.
        """
        empty = ""
        cohort = C.find_cohort(_safe_str(cohort_id), facets=facets)
        if cohort is None:
            # It promised the trust readout this card used to carry. Nothing
            # here can deliver that any more, and in the only state that shows
            # this line the study has no poolable cohort to select either.
            return (html.Div("No cohort in this study has two or more samples "
                             "to pool.",
                             className="cohort-card cohort-card--empty"),
                    empty, "Members", True)
        members = [m for m in (included or []) if m in set(cohort.members)]
        excluded = [m for m in cohort.members if m not in set(members)]

        # The second card is resolved first so it can be reported even while the
        # first is in a state that cannot pool. Its absence is the common case
        # and costs one dict lookup.
        other = C.find_cohort(_safe_str(compare_id), facets=facets)
        other_card, role = empty, ""
        if other is not None:
            other_geometry, _ = _geometry_for(list(other.members))
            if other_geometry is not None:
                facet = C.contrast_facet(cohort, other)
                contrast = (C.FACETS_BY_KEY[facet].label.lower()
                            if facet else "")
                role = "a"
                other_card = build_cohort_card(other, other_geometry, role="b",
                                               contrast=contrast)

        if len(members) < C.MIN_COHORT_SIZE:
            note = ("Tick at least two samples to pool."
                    if excluded else
                    "Pooling needs at least two samples.")
            return (html.Div(note, className="cohort-card cohort-card--empty"),
                    other_card, f"Members ({len(members)} of {cohort.size})",
                    True)
        geometry, _missing = _geometry_for(members)
        if geometry is None:
            return (html.Div("These samples have no precomputed embeddings.",
                             className="cohort-card cohort-card--empty"),
                    other_card, "Members", True)
        summary = (f"Members ({len(members)})" if not excluded
                   else f"Members ({len(members)} of {cohort.size}, "
                        f"{len(excluded)} excluded)")
        return (build_cohort_card(cohort, geometry, role=role), other_card,
                summary, False)

    @app.callback(
        Output("cohort-compare-dropdown", "options"),
        Output("cohort-compare-dropdown", "value"),
        Output("cohort-compare-hint", "children"),
        Input("cohort-dropdown", "value"),
        State("cohort-facets-store", "data"),
    )
    def update_compare_options(cohort_id: str | None, facets: list[str] | None):
        """Offer only siblings that differ in exactly one facet.

        That restriction is what makes the comparison mean anything. Two cohorts
        differing only in the spaceflight arm produce an overlap number that is
        about the arm; two differing in tissue *and* arm produce a number nobody
        can attribute to either.
        """
        cohort = C.find_cohort(_safe_str(cohort_id), facets=facets)
        if cohort is None:
            return [], None, ""
        siblings = C.sibling_cohorts(cohort)
        if not siblings:
            return ([], None,
                    "No sibling cohort in this study differs by exactly one "
                    "facet, so there is nothing to compare against.")
        opts = []
        for s in siblings:
            facet = C.contrast_facet(cohort, s)
            label = C.FACETS_BY_KEY[facet].label if facet else "differs"
            opts.append({"label": f"{s.label}  ·  {s.size} samples  ·  differs "
                                  f"by {label.lower()}",
                         "value": s.cohort_id})
        return (opts, None,
                "Runs the second cohort as its own pooled query and reports how "
                "much of the top-k the two share.")

    @app.callback(
        Output("network-graph", "figure", allow_duplicate=True),
        Output("hits-store", "data", allow_duplicate=True),
        Output("search-status", "children", allow_duplicate=True),
        Input("cohort-search-button", "n_clicks"),
        State("cohort-dropdown", "value"),
        State("cohort-members", "value"),
        State("cohort-compare-dropdown", "value"),
        State("cohort-facets-store", "data"),
        State("topk-slider", "value"),
        State("entrez-email-input", "value"),
        State("biopython-toggle", "value"),
        running=[
            (Output("cohort-search-button", "disabled"), True, False),
            (Output("cohort-running-indicator", "children"),
             "Pooling the cohort and retrieving neighbors...", ""),
        ],
        prevent_initial_call=True,
    )
    def run_cohort_search(
        n_clicks: int,
        cohort_id: str | None,
        included: list[str] | None,
        compare_id: str | None,
        facets: list[str] | None,
        topk: int,
        entrez_email: str | None,
        biopython_toggle: list[str] | None,
    ):
        # Belt and braces against the remount case above: a search runs when
        # somebody clicks Search, and a click means n_clicks is at least one.
        if not n_clicks:
            return no_update, no_update, no_update

        cohort = C.find_cohort(_safe_str(cohort_id), facets=facets)
        if cohort is None:
            return (_empty_network_figure("Select a cohort, then run a search."),
                    no_update,
                    build_status_banner("Select a cohort to start.", kind="info"))

        members = [m for m in (included or []) if m in set(cohort.members)]
        excluded = [m for m in cohort.members if m not in set(members)]
        enable_biopython = bool(biopython_toggle and "on" in biopython_toggle)
        email_value = _safe_str(entrez_email) or GENERIC_ENTREZ_EMAIL

        try:
            hits_df, neighborhood_df, rows, stability = (
                run_cohort_retrieval_with_neighborhood(members, topk=int(topk))
            )
            geometry = C.cohort_geometry(members, rows)
            if enable_biopython and _safe_str(entrez_email):
                hits_df = _enrich_hits_from_ncbi_eutils(hits_df, email_value)

            other = C.find_cohort(_safe_str(compare_id), facets=facets)
            other_hits, other_neighborhood, other_geometry, other_stability = (
                None, None, None, None
            )
            if other is not None:
                other_hits, other_neighborhood, other_rows, other_stability = (
                    run_cohort_retrieval_with_neighborhood(
                        list(other.members), topk=int(topk)
                    )
                )
                other_geometry = C.cohort_geometry(list(other.members), other_rows)
        except Exception as exc:
            detail = getattr(exc, "detail", "") or _safe_str(exc)
            return (
                _empty_network_figure("Cohort retrieval failed - see status."),
                no_update,
                build_status_banner(
                    _last_nonempty_line(_safe_str(exc)) or "Cohort retrieval failed.",
                    kind="error", detail=detail),
            )

        query = _cohort_query_series(cohort, geometry, excluded)
        payload: dict[str, Any] = {
            "sample_id": cohort.cohort_id,
            "entrez_email": email_value,
            "biopython_enabled": bool(enable_biopython),
            "mode": COHORT_MODE,
            "hits": hits_df.to_dict(orient="records"),
            "query": _query_dict(query),
            # The map draws every pooled member rather than one query point,
            # because a cohort has no single position in the space.
            "member_ids": list(geometry.members),
            # Measured during the scan that produced these hits, over this
            # cohort's own leave-one-out pools at this depth. It travels with the
            # hits rather than being recomputed, because recomputing it means a
            # second memmap pass and it is a property of *these* hits.
            "stability": stability.as_dict() if stability is not None else None,
            "neighborhood": _evidence_payload(
                neighborhood_df, label=cohort.label
            ),
        }

        how = _retrieval_phrase(COHORT_MODE)
        enriched = bool(enable_biopython and _safe_str(entrez_email))
        excluded_note = (f" {len(excluded)} excluded." if excluded else "")

        if other_hits is None:
            network = build_network_figure(query=query, hits_df=hits_df)
            message = (
                f"Retrieved {len(hits_df)} hits for {geometry.size} pooled "
                f"samples ({cohort.describe()}) {how}."
                + excluded_note
                + (" Plus GEO and PubMed enrichment." if enriched else "")
            )
            return network, payload, build_status_banner(message, kind="good")

        # Two cohorts, two independent pooled queries, one network.
        other_query = _cohort_query_series(other, other_geometry, [])
        shared = set(hits_df["gsm"]) & set(other_hits["gsm"])
        union = set(hits_df["gsm"]) | set(other_hits["gsm"])
        overlap = len(shared) / len(union) if union else 0.0
        facet = C.contrast_facet(cohort, other)
        facet_label = C.FACETS_BY_KEY[facet].label.lower() if facet else "one facet"

        network = build_comparison_figure(
            query_a=query, hits_a=hits_df, query_b=other_query, hits_b=other_hits)
        payload["comparison"] = {
            "query_b": _query_dict(other_query),
            "hits_b": other_hits.to_dict(orient="records"),
            "shared": sorted(shared),
            "overlap": overlap,
            "facet": facet_label,
            # The map draws the second cohort's members too, for the same reason
            # it draws the first cohort's: a pooled query has no position, so
            # what can be drawn is who went into it. This is a real list rather
            # than the newline-joined `members` string inside `query_b`, which
            # exists for the inspector to print.
            "member_ids": list(other_geometry.members),
            # Cohort B is measured on its own terms. It can easily be the
            # shakier arm, and that is exactly what decides how much of the
            # overlap number below to believe.
            "stability": (other_stability.as_dict()
                          if other_stability is not None else None),
            "neighborhood_b": _evidence_payload(
                other_neighborhood, label=other.label
            ),
        }
        message = (
            f"Two pooled queries differing by {facet_label}: "
            f"{cohort.label} ({geometry.size} samples) and {other.label} "
            f"({other_geometry.size}). They share {len(shared)} of "
            f"{len(union)} retrieved samples, a Jaccard overlap of "
            f"{overlap:.2f}." + excluded_note
        )
        return network, payload, build_status_banner(message, kind="good")

    @app.callback(
        Output("upload-store", "data"),
        Output("upload-preview", "children"),
        Output("upload-sample-column", "options"),
        Output("upload-sample-column", "value"),
        Output("upload-column-control", "style"),
        Output("upload-search-button", "disabled"),
        Input("upload-counts", "contents"),
        State("upload-counts", "filename"),
        State("upload-store", "data"),
        prevent_initial_call=True,
    )
    def handle_upload(contents: str | None, filename: str | None,
                      previous: dict[str, Any] | None = None):
        """Decode an uploaded counts file, stage it, and expose its sample columns.

        The file is decoded once, written under `_upload_dir()`, and only its
        *header* is read here so the column picker fills without parsing the
        whole matrix. The embedding itself happens later, on the Embed & search
        click, which is why the file has to persist past this callback rather
        than living in a `with` block.

        A new upload supersedes this session's previous one, so that one is
        removed first, and a file that fails to parse is removed rather than
        left behind for a search that can never use it.
        """
        hidden = {"display": "none"}
        if not contents:
            return None, "", [], None, hidden, True

        _discard_upload(previous)

        try:
            _, b64 = contents.split(",", 1)
            raw = base64.b64decode(b64)
        except Exception:
            return (None, build_status_banner("Could not decode the uploaded file.", kind="error"),
                    [], None, hidden, True)

        if len(raw) > MAX_UPLOAD_BYTES:
            msg = (f"That file is {_format_bytes(len(raw))}; the limit is "
                   f"{_format_bytes(MAX_UPLOAD_BYTES)}.")
            return None, build_status_banner(msg, kind="error"), [], None, hidden, True

        name = _safe_str(filename) or "uploaded_counts.csv"
        suffix = Path(name).suffix or ".csv"
        staged = _stage_upload(raw, suffix)

        def _reject(message: str):
            """Report the reason and take the unusable file back off disk."""
            _discard_upload({"path": staged})
            return (None, build_status_banner(message, kind="error"),
                    [], None, hidden, True)

        sep = "\t" if suffix.lower() in (".tsv", ".txt") else ","
        try:
            cols = list(map(str, pd.read_csv(staged, sep=sep, index_col=0, nrows=0).columns))
        except Exception as exc:
            return _reject(f"Could not read the counts file: {exc}")
        if not cols:
            return _reject(
                "No sample columns found. Expected mouse Ensembl gene IDs in the "
                "first column and one or more sample columns.")

        options = [{"label": c, "value": c} for c in cols]
        preview = html.Div(className="upload-loaded", children=[
            html.Div(name, className="upload-filename"),
            html.Div(
                f"{len(cols)} sample column{'s' if len(cols) != 1 else ''} detected.",
                className="upload-meta"),
        ])
        store = {"path": staged, "filename": name, "columns": cols}
        # Show the column picker only when there is a choice to make.
        col_style = {} if len(cols) > 1 else hidden
        return store, preview, options, cols[0], col_style, False

    @app.callback(
        Output("network-graph", "figure", allow_duplicate=True),
        Output("hits-store", "data", allow_duplicate=True),
        Output("search-status", "children", allow_duplicate=True),
        Input("upload-search-button", "n_clicks"),
        State("upload-store", "data"),
        State("upload-sample-column", "value"),
        State("topk-slider", "value"),
        State("entrez-email-input", "value"),
        State("biopython-toggle", "value"),
        running=[
            (Output("upload-search-button", "disabled"), True, False),
            (Output("upload-running-indicator", "children"),
             "Embedding the uploaded sample and retrieving neighbors...", ""),
        ],
        prevent_initial_call=True,
    )
    def run_uploaded_search(
        n_clicks: int,
        upload_store: dict[str, Any] | None,
        sample_column: str | None,
        topk: int,
        entrez_email: str | None,
        biopython_toggle: list[str] | None,
    ):
        # See run_cohort_search: this one was firing at page load too, and
        # announcing "Upload a counts file first" before anyone had done
        # anything at all.
        if not n_clicks:
            return no_update, no_update, no_update

        if not upload_store or not upload_store.get("path"):
            return no_update, no_update, build_status_banner(
                "Upload a counts file first.", kind="info")

        filename = _safe_str(upload_store.get("filename")) or "uploaded file"
        columns = upload_store.get("columns") or [""]
        column = _safe_str(sample_column) or _safe_str(columns[0])
        enable_biopython = bool(biopython_toggle and "on" in biopython_toggle)
        email_value = _safe_str(entrez_email) or GENERIC_ENTREZ_EMAIL

        try:
            hits_df, neighborhood_df = run_uploaded_retrieval_with_neighborhood(
                counts_path=upload_store["path"],
                sample_column=column,
                topk=int(topk),
                entrez_email=email_value,
                enable_biopython_metadata=enable_biopython,
            )
        except Exception as exc:
            detail = getattr(exc, "detail", "") or _safe_str(exc)
            return (
                _empty_network_figure("Embedding failed - see status for details."),
                no_update,
                build_status_banner(
                    _last_nonempty_line(_safe_str(exc))
                    or "Embedding the uploaded sample failed.",
                    kind="error", detail=detail,
                ),
            )

        query = _uploaded_query_series(filename, column)
        network = build_network_figure(query=query, hits_df=hits_df)
        enriched = bool(enable_biopython and _safe_str(entrez_email))
        status_message = (
            f"Retrieved {len(hits_df)} hits {_retrieval_phrase(UPLOAD_MODE)}"
            + (", plus GEO and PubMed enrichment." if enriched else ".")
        )
        status = build_status_banner(status_message, kind="good")
        payload = {
            "sample_id": _safe_str(query["sample_id"]),
            "entrez_email": email_value,
            "biopython_enabled": bool(enable_biopython),
            "mode": UPLOAD_MODE,
            "hits": hits_df.to_dict(orient="records"),
            "query": _query_dict(query),
            "neighborhood": _evidence_payload(neighborhood_df, label=column),
        }
        return network, payload, status

    @app.callback(
        Output("ai-summary-output", "children"),
        Output("ai-summary-status", "children"),
        Input("ai-summary-button", "n_clicks"),
        Input("hits-store", "data"),
        running=[
            (Output("ai-summary-button", "disabled"), True, False),
            (Output("ai-summary-status", "children"), "Generating summary...", ""),
            (Output("ai-summary-status", "className"), "ai-status ai-status--loading", "ai-status"),
        ],
        prevent_initial_call=True,
    )
    def generate_ai_summary(_: int, hits_payload: dict[str, Any] | None):
        # One callback owns both generation and invalidation. Dash discards an
        # older response when a result change schedules this newer invocation.
        if ctx.triggered_id == "hits-store":
            return "", ""
        if not hits_payload:
            return "", "Run a search before generating a summary."

        query_row = _query_series(hits_payload)
        if query_row is None:
            return "", "Selected query sample is missing from local metadata."

        # Before the expensive part, not after it. Assembling the prompt fetches
        # a study abstract per accession over the network, so a machine with no
        # model installed - which is what a fresh clone is - used to spend up to
        # a minute gathering text for a request that could never be sent, and
        # only then say "install Ollama". That answer is knowable immediately.
        blocked = unavailable_reason()
        if blocked:
            return blocked, ""

        hits_df = pd.DataFrame(hits_payload.get("hits", []))
        email = _safe_str(hits_payload.get("entrez_email")) or GENERIC_ENTREZ_EMAIL

        # The study abstracts and overall-design text are the substance of what
        # the model reasons over, and the local cache does not carry them. With
        # search-time enrichment now off by default, this is where they get
        # fetched: once, for the hits about to be summarized, rather than on
        # every search whether or not anyone asks for a hypothesis.
        if not hits_df.empty and "geo_summary" in hits_df.columns:
            if not hits_df["geo_summary"].astype(str).str.strip().any():
                try:
                    hits_df = _enrich_hits_from_ncbi_eutils(hits_df, email)
                except Exception as exc:  # never let enrichment block the summary
                    print(f"[ai] GEO enrichment failed, summarizing without it: {exc}",
                          flush=True)

        prompt_template = _load_ai_prompt_template()
        prompt = prompt_template.format(
            osdr_metadata=_format_osdr_query_text(query_row),
            retrieved_hits_table=_format_hits_table_text(hits_df),
            geo_summaries=_format_geo_context_text(hits_df),
        )

        summary = _call_ai_summary(prompt)
        return summary, ""


    @app.callback(
        Output("graph-legend", "children"),
        Output("canvas-subtitle", "children"),
        Input("hits-store", "data"),
    )
    def describe_the_canvas(hits_payload: dict[str, Any] | None):
        """Keep the legend and the subtitle describing what is actually drawn.

        A comparison draws no GSE nodes and two query stars whose colors mean
        something; a single-sample network draws one star and a GSE column. The
        strip above the plot said the second thing in both cases, which is the
        same class of error as the status banner that announced every cached
        result as subprocess output: the interface asserting something about
        itself that is not true.
        """
        payload = hits_payload or {}
        if payload.get("comparison"):
            kind = "comparison"
        elif payload.get("mode") == COHORT_MODE:
            kind = "cohort"
        else:
            kind = "sample"
        return build_graph_legend(kind), CANVAS_SUBTITLES[kind]

    @app.callback(
        Output("stability-panel", "children"),
        Output("stability-panel", "style"),
        Input("hits-store", "data"),
    )
    def report_measured_stability(hits_payload: dict[str, Any] | None):
        """Show how far the result on screen survived dropping one of its own.

        Driven by `hits-store` rather than written directly by the search
        callback, for the same reason the legend and the map link are: the
        router destroys this view when you walk to the map, so a panel painted
        by the search would come back blank while the hits it describes are
        still on the canvas.

        It is also why the measurement travels *in* the payload. Recomputing it
        here would mean a second pass over the 963 MB memmap on every remount,
        for a number that is a property of hits already retrieved.
        """
        return build_stability_panel(hits_payload)

    @app.callback(
        Output("see-on-map", "style"),
        Output("see-on-map", "children"),
        Input("hits-store", "data"),
    )
    def offer_the_map(hits_payload: dict[str, Any] | None):
        """Offer the map only once there is a retrieval to show on it.

        Hits retrieved before `archs4_index` existed, or by the demo path,
        cannot be located on the map. Offering the link anyway would send
        someone to a map that draws nothing and looks broken.
        """
        payload = hits_payload or {}
        # A comparison retrieved two hit lists and the map now draws both, so
        # the offer counts both. Counting cohort A's alone would send someone to
        # a map showing more than the link promised.
        hits = list(payload.get("hits") or [])
        hits += list((payload.get("comparison") or {}).get("hits_b") or [])
        locatable = [h for h in hits if h.get("archs4_index") is not None]
        if not locatable:
            return {"display": "none"}, ""
        n = len(locatable)
        return {}, f"View {n} match{'es' if n != 1 else ''} on map"

    @app.callback(
        Output("selected-node-store", "data"),
        Input("network-graph", "clickData"),
    )
    def select_node(click_data: dict[str, Any] | None):
        if not click_data:
            return None
        points = click_data.get("points", [])
        if not points:
            return None
        custom = points[0].get("customdata")
        if not custom or len(custom) < 2:
            return None
        return {"kind": custom[0], "node_id": custom[1]}


    @app.callback(
        Output("details-panel", "children"),
        Input("hits-store", "data"),
        Input("selected-node-store", "data"),
    )
    def render_details(hits_payload: dict[str, Any] | None, selected_node: dict[str, Any] | None):
        if not hits_payload:
            return [
                html.H2("Details", className="panel-title"),
                html.P("Select a query, sample, or study in the network to inspect its metadata.",
                       className="details-empty"),
            ]

        entrez_email = _safe_str(hits_payload.get("entrez_email")) or GENERIC_ENTREZ_EMAIL
        q_row = _query_series(hits_payload)
        if q_row is None:
            q_row = pd.Series({"sample_id": _safe_str(hits_payload.get("sample_id"))})

        # A comparison draws two query stars and three bands of hits, and every
        # one of those nodes is clickable. Looking node ids up in cohort A's
        # hits alone meant a B-only hit, and the second star, both opened "No
        # metadata found" - the figure offering something the inspector could
        # not answer.
        comparison = hits_payload.get("comparison") or {}
        rows = list(hits_payload.get("hits") or []) + list(comparison.get("hits_b") or [])
        hits_df = pd.DataFrame(rows)
        if not hits_df.empty and "gsm" in hits_df.columns:
            # A shared hit is in both lists with the same GEO metadata and a
            # different score; cohort A's row is kept, which is the one the
            # single-cohort inspector would have shown.
            hits_df = hits_df.drop_duplicates(subset="gsm", keep="first").reset_index(drop=True)
        q_row_b = (pd.Series(comparison["query_b"])
                   if isinstance(comparison.get("query_b"), dict) else None)

        # Open a GSM whose study context was never fetched, and fetch it now -
        # one accession, for the hit actually being read. This is what makes
        # search-time enrichment safe to leave off.
        #
        # The condition tests the *study context* fields specifically. It used
        # to ask whether any of gse/title/geo_summary/pubmed_ids had content,
        # which was a reasonable proxy while a hit arrived either fully
        # enriched or completely bare. The cached path always fills gse and
        # title from the local join, so that test now passes for every hit and
        # the abstract would never be fetched at all.
        if (selected_node and _safe_str(selected_node.get("kind")) == "gsm"
                and not hits_df.empty and entrez_email):
            gsm = _safe_str(selected_node.get("node_id"))
            one = hits_df[hits_df["gsm"] == gsm]
            if not one.empty:
                r = one.iloc[0]
                has_context = any(
                    _safe_str(r.get(c))
                    for c in ["geo_summary", "geo_design", "pubmed_ids"]
                )
                if not has_context:
                    enriched_one = _enrich_hits_from_ncbi_eutils(one.copy(), entrez_email)
                    row_mask = hits_df["gsm"] == gsm
                    for col in enriched_one.columns:
                        # Add the column first if the cached path never created
                        # it. Without this, the `_biopython` fields the NCBI
                        # fetch returns - platform, entry type, GDS type,
                        # release date, FTP link, and the whole Publication
                        # section - were dropped for a cached hit, because the
                        # cached schema has none of them and the copy only kept
                        # columns that already existed. Ten fetched fields, ten
                        # blank rows in the inspector.
                        if col not in hits_df.columns:
                            hits_df[col] = ""
                        hits_df.loc[row_mask, col] = enriched_one.iloc[0][col]

        return build_details_panel(query=q_row, selected_payload=selected_node,
                                   hits_df=hits_df, query_b=q_row_b)
