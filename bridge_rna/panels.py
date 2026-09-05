"""The inspector: detail rows, sections, banners, and the details panel.

Everything here returns Dash components and reads no data of its own; the
callbacks hand it the query row and the hits frame.
"""

from __future__ import annotations

import re
from typing import Any

import pandas as pd
from dash import html

from .config import ROOT
from .osdr import _fetch_osdr_study_summary
from .preflight import (
    _canonical_gene_order_is_authoritative,
    preflight_retrieval_requirements,
)
from .util import _first_non_empty, _safe_str



def _detail_row(label: str, value: Any, mono: bool = False) -> Any:
    """A single label / value row for the details panel."""
    text = _safe_str(value)
    cls = "value" + (" mono" if mono else "")
    val = html.Span(text, className=cls) if text else html.Span("Not recorded", className=cls + " empty")
    return html.Div(className="detail-row", children=[html.Span(label, className="label"), val])


def _accession_row(label: str, accession: str) -> Any:
    """Link recognized source identifiers, retaining their recorded spelling."""
    osdr = re.fullmatch(r"(?:OSD-|GLDS-)(\d+)", accession)
    if osdr:
        url = f"https://osdr.nasa.gov/bio/repo/data/studies/OSD-{osdr.group(1)}"
        source = "OSDR"
    elif re.fullmatch(r"G(?:SM|SE)\d+", accession):
        url = f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={accession}"
        source = "GEO"
    else:
        return _detail_row(label, accession, mono=True)
    return html.Div(className="detail-row", children=[
        html.Span(label, className="label"),
        html.Span(className="value mono", children=html.A(
            accession, href=url, target="_blank", rel="noopener noreferrer",
            title=f"Open {accession} in {source}")),
    ])


def _detail_link_row(label: str, url: str) -> Any:
    url = _safe_str(url)
    if not url:
        return _detail_row(label, "")
    href = url if re.match(r"^(https?|ftp)://", url) else f"ftp://{url}"
    return html.Div(
        className="detail-row",
        children=[
            html.Span(label, className="label"),
            html.Span(className="value", children=html.A(url, href=href, target="_blank")),
        ],
    )


def _detail_section(title: str, rows: list[Any]) -> Any | None:
    rows = [r for r in rows if r is not None]
    if not rows:
        return None
    return html.Div(className="details-section", children=[html.Div(title, className="details-section-title"), *rows])


def _detail_text_block(title: str, text: str, collapsible: bool = False, placeholder: str = "Not available.") -> Any:
    """Full-width long-form text block; collapsible for multi-paragraph fields."""
    text = _safe_str(text)
    if collapsible and text:
        return html.Details(
            className="detail-collapse",
            children=[html.Summary(title), html.Div(text, className="detail-block-body")],
        )
    return html.Div(
        className="detail-block",
        children=[
            html.Div(title, className="detail-block-title"),
            html.Div(text or placeholder, className="detail-block-body"),
        ],
    )


def _details_head(kicker: str, heading: str, score: float | None = None) -> Any:
    children: list[Any] = [
        html.Div(
            children=[
                html.Div(kicker, className="details-kicker"),
                html.H3(heading, className="details-heading"),
            ]
        )
    ]
    if score is not None:
        children.append(html.Span(f"Cosine {score:.4f}", className="score-badge",
                                  title="Cosine similarity between query and sample embeddings"))
    return html.Div(className="details-head", children=children)


AUTHORITATIVE_GENE_LIST = ROOT / "data" / "archs4" / "train_orthologs" / "canonical_genes.csv"


def build_gene_list_banner() -> Any:
    """Persistent banner shown when retrieval is running on a stand-in gene list.

    demo_osdr_top5.py prints this warning, but the app captures the subprocess
    output and only reads it when the process fails, so on a successful run the
    warning is discarded and never reaches the person looking at the results.

    The test is on the gene *ordering*, not on whether a file exists. An
    existence check would clear the banner for any file sitting at the
    authoritative path, including a wrong-order one -- the same failure the
    banner exists to announce.
    """
    if _canonical_gene_order_is_authoritative(AUTHORITATIVE_GENE_LIST):
        return None

    detail = (
        "The authoritative gene list is missing, so retrieval is running on a stand-in."
        if not AUTHORITATIVE_GENE_LIST.exists()
        else "The gene list in place does not match the ordering the ARCHS4 index was built with."
    )

    return html.Div(
        className="invalid-banner",
        children=[
            html.Span("Results are not scientifically valid", className="invalid-banner-title"),
            html.Span(
                f"{detail} A list that reproduces the model's gene count but not its "
                "training gene order builds query vectors in a different gene space "
                "than the ARCHS4 index, so similarity scores look plausible but are "
                "not meaningful and must not be interpreted biologically.",
                className="invalid-banner-body",
            ),
        ],
    )


def build_setup_banner() -> Any:
    """Persistent banner listing unmet prerequisites, shown before any search.

    preflight_retrieval_requirements() is otherwise consulted only inside
    run_real_retrieval, which raises on failure. A fresh clone whose Git LFS
    payload never arrived therefore looks completely healthy: the app serves,
    the sample dropdowns populate from ordinary Git files, and nothing hints
    at a problem until someone picks a sample, clicks Search, and waits for
    the error. Surfacing it at layout time costs one preflight call at import.
    """
    try:
        missing, _ = preflight_retrieval_requirements()
    except Exception as exc:  # never let a diagnostic stop the app from serving
        missing = [f"preflight check failed: {exc}"]

    if not missing:
        return None

    return html.Div(
        className="setup-banner",
        children=[
            html.Span("Setup incomplete, so retrieval cannot run", className="setup-banner-title"),
            html.Span(
                "The interface loaded, but the files retrieval depends on are not ready. "
                "Everything else on this page works; searching will fail until these are resolved.",
                className="setup-banner-body",
            ),
            html.Ul(className="setup-banner-list", children=[html.Li(m) for m in missing]),
        ],
    )


def build_status_banner(message: str, kind: str = "info", detail: str | None = None) -> Any:
    """One-line status banner. ``kind`` is info | good | error.

    When ``detail`` is provided (e.g. a full error blob), a collapsed
    "Show details" disclosure is appended so debugging text stays out of the
    primary viewport but remains reachable.
    """
    children: list[Any] = [html.Span(message, className="status-banner-text")]
    if detail and _safe_str(detail) and _safe_str(detail) != _safe_str(message):
        children.append(
            html.Details(
                className="status-details",
                children=[
                    html.Summary("Show details"),
                    html.Pre(_safe_str(detail), className="status-details-pre"),
                ],
            )
        )
    return html.Div(children, className=f"status-banner status-{kind}")


# --- The cohort card: how much to trust this pooled query --------------------


#: What each role is called on the rail. A comparison pools two cohorts and
#: draws both, in the network figure and on the map, and the only thing binding
#: a card to those glyphs is the color of the dot beside this name - so
#: `.cohort-role-dot` in assets/retrieve.css mirrors GRAPH_THEME's `cohort_a`
#: and `cohort_b` exactly, and a test pins the pair.
COHORT_ROLES = {"a": "Cohort A", "b": "Cohort B"}


def _cohort_role_head(cohort_label: str, role: str, contrast: str) -> list[Any]:
    """The `● COHORT B · differs by spaceflight arm` line, and the cohort's name.

    Shown only when there are two pooled queries, because a lone one has nothing
    to be distinguished from. The name is beside the mark rather than left to
    the color, because cohort B's hex cannot agree across the retrieval network
    and the map (`docs/design-notes.md#map-key`), so the binding is the name.
    """
    if role not in COHORT_ROLES:
        return []
    return [
        html.Div(
            className="cohort-role",
            children=[
                html.Span(className="cohort-role-dot"),
                html.Span(COHORT_ROLES[role], className="cohort-role-name"),
                (html.Span(f"differs by {contrast}",
                           className="cohort-role-contrast")
                 if contrast else None),
            ],
        ),
        html.Div(cohort_label, className="cohort-card-title"),
    ]


def build_cohort_card(cohort, geometry, role: str = "",
                      contrast: str = "") -> Any:
    """What is true about a pooled cohort *before* it is searched: its size.

    That is the whole card, and the narrowing is the point. It used to carry a
    result-stability figure read out of `STABILITY_BY_K`, a curve of mean
    leave-one-out agreement against cohort size measured over all 212 cohorts.
    The number was real and the label was accurate, and it was still misleading
    where it sat: a population average printed beside one cohort's name is read
    as a property of *that* cohort, and the spread inside a size bucket is most
    of the range. Measured live, a cohort of 7 scoring 0.316 and a cohort of 6
    scoring 0.849 were both told 0.72.

    So stability moved to where it can be measured instead of predicted: it is
    computed during the search, over this cohort's own leave-one-out pools at
    the depth on screen, and it is reported by `build_stability_panel` on the
    right afterwards. A live number cannot sit under the picker, because it does
    not exist until the query runs - showing the previous cohort's figure there
    would be worse than the curve was. `docs/design-notes.md#live-stability` has the
    evidence and the rejected alternatives.

    ``role`` names which arm of a comparison this is, and is empty when only one
    cohort is being pooled. When it is set the card gains a colored role line
    and the cohort's own name, because a comparison runs **two** pooled queries
    and describing only the selected one left the second with a color on two
    canvases and a size nowhere. ``contrast`` is the one facet the two differ
    in, stated on the second card because it is a property of the pair.
    """
    from .cohorts import TIER_SINGLETON

    role_class = f" is-{role}" if role in COHORT_ROLES else ""
    k = geometry.size
    if geometry.tier == TIER_SINGLETON:
        return html.Div(
            className="cohort-card cohort-card--empty" + role_class,
            children=html.Div(
                "Pooling needs at least two samples. Select a larger cohort, or "
                "use Sample mode for a single one.",
                className="cohort-card-note"),
        )

    return html.Div(
        className="cohort-card" + role_class,
        children=[
            *_cohort_role_head(cohort.label, role, contrast),
            html.Div(
                className="cohort-card-head",
                children=[
                    html.Span(f"{k}", className="cohort-card-k"),
                    html.Span("samples pooled into one query",
                              className="cohort-card-k-label"),
                ],
            ),
        ],
    )


# --- The stability panel: how far this result survived being poked ------------


def _share(x: float) -> str:
    """A share in [0, 1], at two decimals, unless that would round it to zero.

    The baseline and the gain it produced sit in one sentence, so rounding the
    baseline to `0.00` while printing `340.0x` beside it makes the sentence
    contradict itself: nothing divided by nothing does not give 340. A cohort
    whose members share almost no hits alone is exactly the case where the gain
    is most worth stating, so the fix is to print the small number honestly
    rather than to suppress the ratio.

    Real 0.0 keeps two decimals; the caller says "no two of them agree on a hit
    alone" for that case and never reaches the ratio at all.
    """
    return f"{x:.2f}" if x <= 0.0 or x >= 0.005 else f"{x:.3f}"


def _stability_block(measurement: dict[str, Any], label: str = "",
                     role: str = "") -> Any:
    """One measured cohort: the number, the meter, and what was measured.

    ``measurement`` is `cohorts.StabilityMeasurement.as_dict()` as it came back
    through `hits-store`, so this renders JSON rather than the dataclass and
    survives the round trip the router forces when someone walks to the map and
    back.

    The four parts are wrapped rather than emitted flat, and each wrapper is one
    row of the pair grid: the name, the number with its meter and note, the
    member that moves it most, and the low-stability flag. `.stability-pair`
    aligns those rows across the two arms with `grid-template-rows: subgrid`,
    which is what puts cohort A's 0.89 on the same baseline as cohort B's 0.94.
    Either of the last two rows can be absent from either arm - a cohort whose
    members all matter equally names none, and only a shaky arm is flagged - so
    the rows are assigned by class rather than by counting children. Assigning
    them by position would let cohort B's flag land in the row holding cohort
    A's member name.
    """
    from .cohorts import STABILITY_FLOOR

    pooled = float(measurement.get("pooled") or 0.0)
    single = float(measurement.get("single_sample") or 0.0)
    depth = int(measurement.get("depth") or 0)
    size = int(measurement.get("size") or 0)
    gain = measurement.get("gain")
    low = bool(measurement.get("is_low"))
    weakest = _safe_str(measurement.get("weakest_member"))
    weakest_value = measurement.get("weakest_value")

    # The scale, measured on this cohort in the same pass at the same depth. A
    # bare 0.62 has none, and the constant that used to supply one was fixed at
    # top-5 while this number follows the slider.
    # Kept to one short sentence. A comparison renders this block twice, and
    # every line spent here is a line of cohort B's measurement pushed off the
    # bottom of the panel.
    if gain is None:
        scale = "no two of them agree on a hit alone."
    else:
        scale = (f"one alone overlaps another by {_share(single)}, "
                 f"a {float(gain):.1f}x gain.")

    # The role line and the cohort's name are one grid row, so the two names sit
    # on one baseline however many lines either of them wraps to. A lone cohort
    # gets no head at all - there is nothing for a letter to distinguish it from
    # - and the row simply collapses.
    head = _cohort_role_head(label, role, contrast="")
    children: list[Any] = [
        *([html.Div(className="stability-name", children=head)] if head else []),
        html.Div(
            className="cohort-stat",
            children=[
                # No "Result stability" label here: the panel heading above is
                # that label, and printing it again per block cost a line each
                # and pushed cohort B's whole measurement below the fold on a
                # comparison - which is the one thing this panel exists to show.
                html.Div(
                    className="stability-head",
                    children=[
                        html.Span(f"{pooled:.2f}", className="stability-value"),
                        html.Span(f"of {depth}", className="stability-value-unit"),
                    ],
                ),
                html.Div(
                    className="cohort-meter",
                    children=html.Div(
                        className="cohort-meter-fill" + (" is-low" if low else ""),
                        style={"width": f"{max(0.0, min(1.0, pooled)) * 100:.1f}%"},
                    ),
                ),
                html.Div(f"{size} pooled, and {scale}",
                         className="cohort-stat-note"),
            ],
        ),
    ]

    # The row is always drawn, and when there is no such member it says so.
    # `cohorts.weakest_member` returns None when every member's absence moves the
    # list equally far, which is a real answer and not a missing one - but an
    # absent row and a clipped row look identical on screen, and a clipped row on
    # exactly this line is the defect this layout was built to fix. Saying it in
    # words is what keeps the two columns the same shape and keeps silence from
    # having two meanings.
    named = bool(weakest) and weakest_value is not None
    children.append(html.Div(
        className="stability-weakest",
        children=[
            # Label and score on one line, the name on its own beneath it. All
            # three cannot share a baseline row in a 155px column: the label and
            # the value are fixed width, which left about 29px for a
            # 27-character sample key and wrapped it one character per line.
            html.Div(
                className="stability-weakest-row",
                children=[
                    html.Span("Moves it most",
                              className="stability-weakest-label"),
                    (html.Span(f"{float(weakest_value):.2f}",
                               className="stability-weakest-value")
                     if named else None),
                ],
            ),
            # The accession is dropped from the name: every member of a cohort
            # shares it, the details panel below states it once, and carrying it
            # here wrapped the name onto a second line for nothing.
            html.Span(weakest.split("|", 1)[-1] if named
                      else "every member equally",
                      className="stability-weakest-name"
                      + ("" if named else " is-none")),
        ],
    ))

    if low:
        # Amber, not red, for the reason the map's coverage bar is amber: a
        # result that moves when an animal is dropped is reporting correctly,
        # not failing. What changed is that the trigger is now the measurement
        # itself rather than cohort size standing in for it.
        # One line, not two. The three-line version said the same thing twice
        # over on a comparison and cost cohort B its place on screen; the
        # threshold is still read from the constant that triggered the flag, so
        # the sentence cannot drift away from the rule.
        children.append(html.Div(
            className="cohort-flag",
            children=html.Span(
                f"Under {round(STABILITY_FLOOR * 100)}%: read these as a "
                "neighbourhood, not a ranking.",
                className="cohort-flag-title"),
        ))

    role_class = f" is-{role}" if role in COHORT_ROLES else ""
    return html.Div(className="stability-cohort" + role_class, children=children)


def build_stability_panel(payload: dict[str, Any] | None) -> tuple[list[Any], dict[str, Any]]:
    """The inspector's stability panel, and whether to show it at all.

    Returns `(children, style)`. The panel is hidden outright unless the
    retrieval on screen is a pooled cohort that carries a measurement, because
    there is no such thing as leave-one-out stability for a single sample or an
    uploaded file: there is nothing to leave out. Offering an empty panel for
    those modes would be the same empty promise as the map link before a search.

    It lives here, on the right, rather than on the control rail, because it is
    a property of the **result** and not of the selection. It cannot exist until
    the query has run, and a live number under the picker would either turn every
    dropdown change into a memmap pass or show the previous cohort's figure.

    A comparison ran two independent pooled queries and gets two blocks, each
    under its own role dot and its own name, for the reason the rail carries two
    cards: stability is a property of each arm separately, and an overlap of 0.25
    between two arms measuring 0.86 means something quite different from the same
    0.25 between one at 0.86 and one at 0.31.

    **The two blocks sit side by side, in even columns, with their rows aligned.**
    They were stacked until 2026-08-06, and stacking gave the two arms visibly
    unequal treatment: cohort A rendered complete and cohort B's last row was
    clipped by the panel's own fold at every viewport measured, from 7.8 px at
    1680x1050 to 65.6 px at 1280x800. The row that went was the one naming the
    animal whose absence moves the result furthest, which is the only actionable
    line in the block. Side by side the panel holds 354 px of content where it
    held 456, so nothing is clipped, the two arms are the same size by
    construction rather than by coincidence, and 0.89 sits on the same baseline
    as 0.94 instead of 160 px below it - which is the comparison this panel
    exists to support. `docs/design-notes.md#stability-panel-even-split` has the measurements.
    """
    payload = payload or {}
    primary = payload.get("stability")
    if not isinstance(primary, dict) or not primary:
        return [], {"display": "none"}

    comparison = payload.get("comparison") or {}
    second = comparison.get("stability")
    paired = isinstance(second, dict) and bool(second)

    query = payload.get("query") or {}
    label_a = _safe_str(query.get("cohort_label"))
    contrast = _safe_str(comparison.get("facet"))

    blocks = [_stability_block(primary, label=label_a,
                               role="a" if paired else "")]
    if paired:
        query_b = comparison.get("query_b") or {}
        blocks.append(_stability_block(
            second, label=_safe_str(query_b.get("cohort_label")), role="b"))

    # Both arms are retrieved at the same depth, so what was measured is said
    # once, in the subtitle, rather than in every block. That is what keeps two
    # measurements on screen together, which is the whole point of measuring two.
    #
    # The facet the two differ in is said here for the same reason, and it moved
    # here from cohort B's role line when the blocks went side by side. It is a
    # fact about the pair, not about either arm - the one thing in the panel that
    # is neither shared nor differing - and hanging it under B's letter made B's
    # name start a line lower than A's, which is exactly the ragged edge that
    # side-by-side columns exist to remove.
    depth = int(primary.get("depth") or 0)
    return (
        [
            html.Div(
                className="panel-header",
                children=[
                    html.Span(className="panel-dot panel-dot--teal"),
                    html.Div(children=[
                        html.H2("Result stability", className="panel-title"),
                        html.P(
                            f"Measured on this query: how much of these {depth} "
                            f"hits comes back when any one pooled sample is "
                            f"dropped, averaged over all of them.",
                            className="panel-subtitle"),
                        (html.P(f"The two arms differ by {contrast}.",
                                className="stability-contrast")
                         if paired and contrast else None),
                    ]),
                ],
            ),
            html.Div(
                className="stability-pair" + (" is-pair" if paired else ""),
                children=blocks,
            ),
        ],
        {},
    )


def build_cohort_details(query: pd.Series, role: str = "") -> list[Any]:
    """Inspector view of a pooled cohort query node.

    The single-sample panel would render this as one sample with a blank name,
    so a cohort gets its own: what defined it, how many went in, what came back
    out about its spread, and which members were excluded.

    ``role`` names which arm of a comparison this is, and only a comparison
    passes one. Without it the panel opened on cohort A and read as *the* pooled
    query rather than as one of two, with nothing saying the other star on the
    canvas leads to its twin. It is the same letter and the same dot the rail's
    cards carry, so the card, the star and this heading name one thing.
    """
    label = _safe_str(query.get("cohort_label")) or "Cohort"
    study = _safe_str(query.get("study_id"))
    members = [m for m in _safe_str(query.get("members")).split("\n") if m]
    excluded = [m for m in _safe_str(query.get("excluded")).split("\n") if m]
    outliers = {m for m in _safe_str(query.get("outliers")).split("\n") if m}
    kicker = ("Pooled OSDR cohort" if role not in COHORT_ROLES
              else f"Pooled OSDR cohort · {COHORT_ROLES[role]}")

    parts: list[Any] = [
        _details_head(kicker, label),
        _detail_section(
            "Definition",
            [
                _accession_row("Study", study),
                _detail_row("Grouped by", _safe_str(query.get("grouped_by"))),
                _detail_row("Samples pooled", str(len(members))),
                # Result stability used to be a fourth row here. It is measured
                # now rather than looked up, and `build_stability_panel` reports
                # it directly above this panel on the same column - two copies of
                # one number stacked vertically is noise, and this one would go
                # off screen the moment a hit node was clicked.
            ],
        ),
    ]

    if members:
        parts.append(html.Div(
            className="details-section",
            children=[
                html.Div(f"Pooled members ({len(members)})",
                         className="details-section-title"),
                html.Ul(
                    className="cohort-member-list",
                    children=[
                        html.Li(
                            className="cohort-member" + (" is-outlier" if m in outliers else ""),
                            children=[
                                html.Span(m, className="cohort-member-name"),
                                (html.Span("furthest from the rest",
                                           className="cohort-member-tag")
                                 if m in outliers else None),
                            ],
                        )
                        for m in members
                    ],
                ),
            ],
        ))

    if excluded:
        parts.append(html.Div(
            className="details-section",
            children=[
                html.Div(f"Excluded by you ({len(excluded)})",
                         className="details-section-title"),
                html.Ul(className="cohort-member-list cohort-member-list--excluded",
                        children=[html.Li(m, className="cohort-member") for m in excluded]),
            ],
        ))

    parts.append(_detail_text_block(
        "How this query was built",
        "Each member's cached 512-d embedding was L2-normalized and averaged, "
        "then the mean was normalized again - the maximum-likelihood mean "
        "direction for vectors compared by cosine, so every animal gets one "
        "vote regardless of how concentrated its transcriptome is. The pooled "
        "vector was then scored against every ARCHS4 sample by exactly the scan "
        "a single-sample search uses.",
        collapsible=True))

    return [p for p in parts if p is not None]


OSDR_ACCESSION_RE = re.compile(r"^(OSD|GLDS)-\d+$")


def _build_osdr_query_metadata_block(query: pd.Series) -> list[Any]:
    """Appendable OSDR metadata section for the right panel.

    Only an OSDR accession has an OSDR study behind it. An uploaded sample
    carries the synthesized study_id "Uploaded file", which the Identity
    section already shows, so rendering an "OSDR study" section for it repeats
    that row, adds a "Study title —" that can never fill, and sends a lookup
    for a study that does not exist. Say nothing instead.
    """
    study_id = _safe_str(query.get("study_id", ""))
    if not OSDR_ACCESSION_RE.match(study_id):
        return []
    summary = _fetch_osdr_study_summary(study_id)
    study_title = _safe_str(summary.get("study_title", ""))
    study_description = _safe_str(summary.get("study_description", ""))
    study_publication_title = _safe_str(summary.get("study_publication_title", ""))
    protocol = _safe_str(summary.get("study_protocol_description", ""))
    section = _detail_section(
        "OSDR study",
        [
            _accession_row("Study ID", study_id),
            _detail_row("Study title", study_title),
        ],
    )
    blocks: list[Any] = [section] if section else []
    if study_description:
        blocks.append(_detail_text_block("Study description", study_description, collapsible=True))
    if study_publication_title:
        blocks.append(_detail_text_block("Publication title", study_publication_title, collapsible=True))
    if protocol:
        blocks.append(_detail_text_block("Protocol description", protocol, collapsible=True))
    return blocks


def _build_query_details(query: pd.Series, compact: bool,
                         role: str = "") -> list[Any]:
    """Details for the OSDR query node. ``compact`` omits the finer biology rows."""
    # A pooled cohort is a query too, and rendering it through the single-sample
    # path would show one blank sample name where a group belongs.
    if _safe_str(query.get("is_cohort")) == "1":
        return build_cohort_details(query, role=role)
    heading = _safe_str(query.get("sample_name")) or _safe_str(query.get("sample_id")) or "Query sample"
    biology_rows = [
        _detail_row("Species", "Mus musculus"),
        _detail_row("Tissue", _safe_str(query.get("tissue"))),
        _detail_row("Condition", _safe_str(query.get("condition"))),
    ]
    if not compact:
        biology_rows += [
            _detail_row("Strain", _safe_str(query.get("strain"))),
            _detail_row("Sex", _safe_str(query.get("sex"))),
            _detail_row("Duration", _safe_str(query.get("duration"))),
        ]
    parts: list[Any] = [
        _details_head("Query sample", heading),
        _detail_section(
            "Identity",
            [
                _detail_row("Sample ID", _safe_str(query.get("sample_id")), mono=True),
                _accession_row("Study ID", _safe_str(query.get("study_id"))),
            ],
        ),
        _detail_section("Biology", biology_rows),
    ]
    parts += _build_osdr_query_metadata_block(query)
    return [p for p in parts if p is not None]


def build_details_panel(query: pd.Series, selected_payload: dict[str, Any] | None,
                        hits_df: pd.DataFrame,
                        query_b: pd.Series | None = None) -> list[Any]:
    node_kind = _safe_str(selected_payload.get("kind")) if selected_payload else ""
    node_id = _safe_str(selected_payload.get("node_id")) if selected_payload else ""

    if not selected_payload or node_kind == "query":
        # A comparison is on screen exactly when there is a second query, so
        # the letters appear only when there are two arms to tell apart.
        return _build_query_details(query, compact=not selected_payload,
                                    role="a" if query_b is not None else "")

    # The comparison figure draws a second query star, tagged `query2`. Without
    # this it fell through to the GSM lookup, found nothing, and reported "No
    # metadata found" for a node the figure had just drawn.
    if node_kind == "query2":
        if query_b is None:
            return [
                _details_head("Details", "No metadata"),
                html.P("This retrieval carries no second cohort.",
                       className="details-empty"),
            ]
        # `compact` has no default, and False is what the `query` branch above
        # resolves to whenever a node was actually clicked.
        return _build_query_details(query_b, compact=False, role="b")

    if node_kind == "gse":
        df = hits_df[hits_df["gse"] == node_id]
        examples = ", ".join(df["gsm"].head(8).astype(str).tolist())
        return [
            _details_head("GSE study", node_id),
            _accession_row("Study", node_id),
            _detail_section(
                "Overview",
                [
                    _detail_row("Connected GSM hits", str(len(df))),
                    _detail_row("Example GSMs", examples),
                ],
            ),
            html.P("Click an individual GSM node for full GEO fields.", className="details-empty-hint"),
        ]

    df = hits_df[hits_df["gsm"] == node_id]
    if df.empty:
        return [
            _details_head("Details", "No metadata"),
            html.P("No metadata found for the selected node.", className="details-empty"),
        ]

    r = df.iloc[0]
    species = _first_non_empty(r, ["species", "geo_taxon_biopython"])
    source_name = _first_non_empty(r, ["source_name", "source_name_ch1"])
    characteristics = _first_non_empty(r, ["characteristics", "characteristics_ch1"])
    gse = _first_non_empty(r, ["gse", "series_id", "geo_gse_biopython"])
    platform = _first_non_empty(r, ["geo_platform_biopython", "platform_ncbi"])
    entry_type = _first_non_empty(r, ["geo_entry_type_biopython", "entry_type_ncbi"])
    gds_type = _first_non_empty(r, ["geo_gds_type_biopython", "gds_type_ncbi"])
    pdat = _first_non_empty(r, ["geo_pdat_biopython", "pdat_ncbi"])
    n_samples = _first_non_empty(r, ["geo_n_samples_biopython", "n_samples_ncbi"])
    ftp_link = _first_non_empty(r, ["geo_ftp_link_biopython", "ftp_link_ncbi"])

    title = _first_non_empty(r, ["title", "geo_title_biopython"])
    geo_summary = _first_non_empty(r, ["geo_summary", "geo_summary_biopython", "geo_abstract_biopython"])
    geo_design = _first_non_empty(r, ["geo_design", "geo_overall_design_biopython", "design_ncbi"])
    pubmed_ids = _first_non_empty(r, ["pubmed_ids", "geo_pubmed_ids_biopython", "pubmed_id"])
    pubmed_title = _first_non_empty(r, ["pubmed_title_biopython", "pubmed_title_ncbi"])
    pubmed_journal = _first_non_empty(r, ["pubmed_journal_biopython", "pubmed_journal_ncbi"])
    pubmed_date = _first_non_empty(r, ["pubmed_pub_date_biopython", "pubmed_pub_date_ncbi"])
    pubmed_doi = _first_non_empty(r, ["pubmed_doi_biopython", "pubmed_doi_ncbi"])

    parts: list[Any] = [
        _details_head("ARCHS4 hit · GSM", _safe_str(r.get("gsm")), score=float(r.get("score", 0.0))),
        _detail_section(
            "Identity",
            [
                _accession_row("GSM", _safe_str(r.get("gsm"))),
                _accession_row("GSE", gse),
                _detail_row("Title", title),
            ],
        ),
        _detail_section(
            "Biology",
            [
                _detail_row("Species", species),
                _detail_row("Source name", source_name),
                _detail_row("Characteristics", characteristics),
            ],
        ),
        _detail_section(
            "Platform & series",
            [
                _detail_row("Platform", platform),
                _detail_row("Entry type", entry_type),
                _detail_row("GDS type", gds_type),
                _detail_row("Release date", pdat),
                _detail_row("Series sample count", n_samples),
                _detail_link_row("FTP link", ftp_link),
            ],
        ),
    ]

    if _safe_str(geo_summary) or _safe_str(geo_design):
        context = html.Div(className="details-section", children=[html.Div("Study context", className="details-section-title")])
        blocks = [c for c in [
            _detail_text_block("GEO summary", geo_summary, collapsible=True) if _safe_str(geo_summary) else None,
            _detail_text_block("Overall design", geo_design, collapsible=True) if _safe_str(geo_design) else None,
        ] if c is not None]
        context.children = context.children + blocks
        parts.append(context)

    pub_rows = [
        _detail_row("PubMed IDs", pubmed_ids, mono=True),
        _detail_row("Title", pubmed_title),
        _detail_row("Journal / date", " ".join(x for x in [pubmed_journal, pubmed_date] if x)),
        _detail_row("DOI", pubmed_doi),
    ]
    if any(_safe_str(v) for v in [pubmed_ids, pubmed_title, pubmed_journal, pubmed_date, pubmed_doi]):
        parts.append(_detail_section("Publication", pub_rows))

    return [p for p in parts if p is not None]
