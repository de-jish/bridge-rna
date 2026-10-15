"""Documented comparisons and their paired evidence in the retrieval workspace."""
import os
from urllib.parse import quote

import pandas as pd
from dash import Input, Output, State, ctx, dcc, html, no_update

from . import cohorts as C
from .comparison import EVIDENCE_PAGE_SIZE, evidence_groups, evidence_page, paired_rankings, study_context
from .reviewed import COMPARISONS, get_comparison, resolve_comparison

DEFAULT_COMPARISON = "osd-194-retina"


def _link(accession):
    return html.A(accession, href="https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc="
                  + quote(accession, safe=""), target="_blank", rel="noopener noreferrer")


def comparison_notes(key):
    preset = get_comparison(key)
    return [
        html.P(f"Space Flight ({len(preset.flight)}) versus Ground Control ({len(preset.ground)}). "
               f"Female {preset.strain} mice; {', '.join(preset.durations)}.",
               className="comparison-note"),
        html.P(preset.note, className="comparison-note"),
        html.Details(children=[
            html.Summary("Definition and source"),
            html.P("Source membership and local metadata checked on 2026-10-15. "
                   "This is a documented comparison, not independent biological validation."),
            html.A(f"Open {preset.study} in OSDR", href=preset.source_url,
                   target="_blank", rel="noopener noreferrer"),
            html.P(f"Diet: {preset.diet}. Hardware: Rodent Habitat. "
                   "Genotype is not recorded in the local metadata. "
                   "Study design and unmeasured confounding still require review."),
            *[html.Div([html.H4(arm), html.Ul([html.Li(m.split("|", 1)[1]) for m in members])])
              for arm, members in (("Flight members", preset.flight), ("Ground members", preset.ground))],
            html.P("OSD-238 is not offered: its coarse flight group combines onboard gravity "
                   "and diet conditions. Basal and vivarium controls are not substituted for ground controls."),
        ]),
    ]


def comparison_panel():
    return html.Div(id="mode-panel-comparison", className="mode-panel",
                    role="tabpanel", style={"display": "none"}, children=[
        html.Div(role="group", **{"aria-labelledby": "reviewed-comparison-label"}, children=[
            html.Label("Documented comparison", id="reviewed-comparison-label", className="control-label"),
            dcc.Dropdown(id="reviewed-comparison", clearable=False, value=DEFAULT_COMPARISON,
                         options=[dict(label=c.label, value=c.key) for c in COMPARISONS]),
        ]),
        html.Div(id="reviewed-comparison-notes", className="comparison-definition",
                 children=comparison_notes(DEFAULT_COMPARISON)),
    ])


def evidence_children(payload, depth, page=0):
    paired = (payload or {}).get("paired_evidence")
    if not paired:
        return []
    preset = get_comparison(payload["reviewed_comparison"])
    all_groups = evidence_groups(paired["rows"], depth)
    groups, count = evidence_page(paired["rows"], depth, page)
    context = paired["context"]
    content = [
        html.H2(f"{preset.study} · {preset.tissue}", className="panel-title"),
        html.P(f"{len(preset.flight)} flight mice and {len(preset.ground)} ground mice. "
               f"{count} reference samples in {len(all_groups)} GSE groups from the union "
               f"of each arm’s top {depth}.", className="panel-subtitle"),
        html.P("A match can be close to both groups. Compare both ranks and absolute "
               "cosines before interpreting a study title.", className="comparison-note"),
        html.Details(children=[html.Summary("Methods and reference provenance"),
            html.P(f"Both arms search the same {paired['reference_count']:,} ARCHS4/GEO samples "
                   "using the spherical mean of individually normalized 512-D embeddings. "
                   "Ranks use the full reference collection, not just the displayed union. "
                   "Exact score ties are ordered by reference row index."),
            html.P("GSE groups are not necessarily independent experiments. Reference samples "
                   "have not been screened for spaceflight origin, cross-deposition, shared animals, "
                   "or related study families. Similarity and rank differences do not establish "
                   "a flight-specific effect, a shared mechanism, or statistical significance."),
            html.P("Study titles, summaries and designs below are source records from the "
                   "optional local metadata database. Tissue categories are keyword-derived; "
                   "raw sample annotations are shown separately."
                   if context["available"] else context["reason"]),
            html.P(preset.note),
            html.P("OSD-100 and OSD-104 are RR-1 tissue datasets that share animal IDs; "
                   "they are not independent animal cohorts."),
            html.A("OSDR comparison source", href=preset.source_url,
                   target="_blank", rel="noopener noreferrer"),
        ]),
    ]
    for i, group in enumerate(groups):
        gse = group["gse"]
        study = context.get("studies", {}).get(gse, {})
        body = []
        if gse:
            body.append(html.P([_link(gse), " · GEO source record"]))
        if study:
            body.append(html.Details(children=[html.Summary("Study context"),
                html.H4("Source summary"), html.P(study["summary"] or "Not recorded"),
                html.H4("Source experimental design"), html.P(study["design"] or "Not recorded")]))
        table_rows = []
        for r in group["rows"]:
            memberships = context.get("memberships", {}).get(r["gsm"], [])
            annotations = [html.P(f"{label}: {r.get(field) or 'Not recorded'}")
                           for label, field in (("Source name", "source_name"),
                                                ("Source characteristics", "characteristics"),
                                                ("Derived tissue category", "tissue"),
                                                ("Species", "species"))]
            annotations.append(html.P(["Recorded GSE memberships: ",
                *[html.Span([_link(g), " "]) for g in memberships]]
                if memberships else "Additional GSE memberships are not available."))
            table_rows.append(html.Tr([
                html.Td([_link(r["gsm"]), html.Div(r.get("title") or "Sample title not recorded"),
                         html.Details(children=[html.Summary("Sample annotations"), *annotations])]),
                html.Td(f"{r['rank_a']:,}", className="comparison-number"),
                html.Td(f"{r['rank_b']:,}", className="comparison-number"),
                html.Td(f"{r['score_a']:.6f}", className="comparison-number"),
                html.Td(f"{r['score_b']:.6f}", className="comparison-number"),
            ]))
        body.append(html.Div(className="comparison-table-wrap", role="region", tabIndex=0,
                             **{"aria-label": f"Paired evidence for {gse or 'samples without a GSE'}"}, children=[
            html.Table(className="comparison-table", children=[
                html.Caption("Ranks and cosine similarities to the two pooled groups"),
                html.Thead(html.Tr([html.Th(label, scope="col") for label in
                                   ("Reference sample", "Flight rank", "Ground rank", "Flight cosine", "Ground cosine")])),
                html.Tbody(table_rows),
            ])]))
        content.append(html.Details(className="comparison-study", open=i < 2, children=[
            html.Summary([html.Strong(gse or "No GSE recorded"),
                          f" · {len(group['rows'])} samples", html.Div(study.get("title") or "Study title not available")]),
            *body,
        ]))
    return content


def run_documented_comparison(key, topk):
    """One execution result feeds the existing network, inspector, and map."""
    from .callbacks import _cohort_query_series, _query_dict, _evidence_payload
    from .retrieval import cached_query_vectors, _load_archs4_index, _annotate_from_cache

    a, b = resolve_comparison(key)
    vectors = []
    geometries = []
    for cohort in (a, b):
        rows, missing = cached_query_vectors(cohort.members)
        if missing:
            raise ValueError("Comparison members are missing cached embeddings.")
        vectors.append(C.cohort_query_vector(rows))
        geometries.append(C.cohort_geometry(cohort.members, rows))
    index, _, _ = _load_archs4_index()
    result = paired_rankings(index, *vectors)
    evidence = result["rows"]
    annotated = _annotate_from_cache(
        pd.Series([r["archs4_index"] for r in evidence]).to_numpy(),
        pd.Series([r["score_a"] for r in evidence]).to_numpy())
    records = [{**ann, **r} for ann, r in zip(annotated.to_dict("records"), evidence)]
    by_index = {r["archs4_index"]: r for r in records}
    frames = []
    for arm, order in enumerate((result["order_a"], result["order_b"])):
        score_field = "score_a" if arm == 0 else "score_b"
        frames.append(pd.DataFrame([{**by_index[int(i)], "score": by_index[int(i)][score_field]} for i in order]))
    qa, qb = [_cohort_query_series(c, g, []) for c, g in zip((a, b), geometries)]
    ha, hb = [f.iloc[:int(topk)].copy() for f in frames]
    shared = set(ha.gsm) & set(hb.gsm)
    union = set(ha.gsm) | set(hb.gsm)
    payload = dict(sample_id=a.cohort_id, mode="cohort", query=_query_dict(qa),
                   hits=ha.to_dict("records"), member_ids=list(a.members),
                   neighborhood=_evidence_payload(frames[0], a.label),
                   reviewed_comparison=key,
                   paired_evidence=dict(rows=records, reference_count=result["reference_count"],
                       context=study_context(records, os.environ.get("BRIDGE_RNA_STUDY_CONTEXT_DB"))),
                   comparison=dict(query_b=_query_dict(qb), hits_b=hb.to_dict("records"),
                       shared=sorted(shared), overlap=len(shared) / len(union) if union else 0.,
                       facet="spaceflight arm", member_ids=list(b.members),
                       neighborhood_b=_evidence_payload(frames[1], b.label)))
    return payload, qa, qb, ha, hb


def register(app):
    @app.callback(Output("reviewed-comparison-notes", "children"), Input("reviewed-comparison", "value"))
    def show_definition(key):
        try:
            return comparison_notes(key)
        except ValueError as exc:
            return str(exc)

    @app.callback(Output("network-graph", "figure", allow_duplicate=True),
                  Output("hits-store", "data", allow_duplicate=True),
                  Output("search-status", "children", allow_duplicate=True),
                  Input("comparison-search-button", "n_clicks"),
                  State("reviewed-comparison", "value"), State("topk-slider", "value"),
                  running=[(Output("comparison-search-button", "disabled"), True, False),
                           (Output("comparison-running-indicator", "children"), "Comparing both groups...", "")],
                  prevent_initial_call=True)
    def search(n_clicks, key, topk):
        from .figures import build_comparison_figure
        from .panels import build_status_banner
        if not n_clicks:
            return no_update, no_update, no_update
        try:
            payload, qa, qb, ha, hb = run_documented_comparison(key, topk)
            return (build_comparison_figure(qa, ha, qb, hb), payload,
                    build_status_banner("Compared the documented flight and ground groups "
                                        "against the same ARCHS4/GEO reference collection.", kind="good"))
        except Exception as exc:
            # Clear prior scientific results rather than leaving stale evidence under an error.
            from .figures import _empty_network_figure
            return (_empty_network_figure("Comparison unavailable."), None,
                    build_status_banner(str(exc), kind="error"))

    @app.callback(Output("comparison-results-switch", "style"), Output("results-view", "value"),
                  Input("hits-store", "data"))
    def choose_view(payload):
        available = bool((payload or {}).get("paired_evidence"))
        return ({}, "evidence") if available else ({"display": "none"}, "network")

    @app.callback(Output("network-panel", "style"), Output("comparison-evidence-panel", "style"),
                  Output("comparison-evidence-body", "children"),
                  Input("results-view", "value"), Input("hits-store", "data"),
                  Input("comparison-depth", "value"), Input("comparison-page", "data"))
    def show_evidence(view, payload, depth, page):
        children = evidence_children(payload, int(depth or 30), int(page or 0))
        show = view == "evidence" and bool(children)
        return ({"display": "none"} if show else {},
                {} if show else {"display": "none"}, children)

    @app.callback(Output("comparison-page", "data"),
                  Output("comparison-page-label", "children"),
                  Output("comparison-previous", "disabled"), Output("comparison-next", "disabled"),
                  Input("comparison-previous", "n_clicks"), Input("comparison-next", "n_clicks"),
                  Input("comparison-depth", "value"), Input("hits-store", "data"),
                  State("comparison-page", "data"))
    def page_evidence(previous, next_clicks, depth, payload, page):
        rows = (payload or {}).get("paired_evidence", {}).get("rows", [])
        _, total = evidence_page(rows, int(depth or 30), 0)
        last = max(0, (total - 1) // EVIDENCE_PAGE_SIZE)
        if ctx.triggered_id == "comparison-next" and next_clicks:
            page = min(last, int(page or 0) + 1)
        elif ctx.triggered_id == "comparison-previous" and previous:
            page = max(0, int(page or 0) - 1)
        else:
            page = 0
        page = min(last, page)
        start = page * EVIDENCE_PAGE_SIZE
        label = f"Samples {start + 1 if total else 0}–{min(total, start + EVIDENCE_PAGE_SIZE)} of {total}"
        return page, label, page == 0, page == last
