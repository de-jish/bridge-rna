"""Small results controls and source metadata disclosures, using existing styles."""
from dash import Input, Output, State, ctx, dcc, html, no_update
import pandas as pd

from . import result_metadata as metadata
from . import results
from .figures import _empty_network_figure, build_network_figure, build_comparison_figure


def controls():
    return html.Div(id='results-controls', className='results-controls',
                    style={'display': 'none'}, children=[
        dcc.Store(id='results-view-store'),
        dcc.Store(id='results-rows-store'),
        dcc.Download(id='results-csv'),
        html.Div(className='results-actions', children=[
            html.Span(id='results-count', role='status'),
            html.Button('Export CSV', id='results-export', className='btn-secondary'),
        ]),
        html.Div(id='results-metadata-status', className='control-hint', role='status'),
        html.Details(className='advanced-group', children=[
            html.Summary('Filter retrieved matches', className='advanced-summary'),
            html.P('Filters apply only to the matches already retrieved. '
                   'The map, similarity profile and AI summary retain the full search.',
                   className='details-empty-hint'),
            html.Div(className='results-filters', children=[
                html.Div(role='group', **{'aria-labelledby': f'results-{field}-label'}, children=[
                    html.Label(label, id=f'results-{field}-label', className='control-label'),
                    dcc.Dropdown(id=f'results-{field}', options=[], value=None,
                                 placeholder='All', clearable=True),
                ]) for field, label in results.FILTERS.items()
            ]),
            html.Button('Reset filters', id='results-reset', className='btn-ghost'),
        ]),
        html.Div(role='group', **{'aria-labelledby': 'results-sample-label'}, children=[
            html.Label('Ranked matches', id='results-sample-label', className='control-label'),
            dcc.Dropdown(id='results-sample', options=[], value=None,
                         placeholder='Select a match to inspect', clearable=True),
        ]),
        html.Div(id='results-order-note', className='control-hint'),
    ])


def metadata_details(meta):
    from .panels import _detail_row, _detail_text_block, _accession_row
    if not meta:
        return None
    rows = []
    for field, label in metadata.FIELDS.items():
        rows.append(_detail_row(label, meta[field]))
    sources = [html.P('Selected source annotations; separate from derived tissue and AI interpretation.',
                      className='details-empty-hint')]
    for field, label in metadata.FIELDS.items():
        sources += [_detail_row(label + ' source', meta['sources'][field]),
                    _detail_row(label + ' stored value', metadata.display_value(meta['raw'][field])
                                if meta['raw'][field] is None else str(meta['raw'][field]))]
    if meta.get('assertions'):
        sources.append(html.Details(children=[html.Summary('Source evidence'), *[
            _detail_text_block(f"{a['field']} · {a['source']}",
                              f"{a['raw_value'] or 'Unknown'}; method: {a['annotation_method']}; "
                              f"evidence: {a['evidence_source'] or 'Unknown'}; "
                              f"ontology: {a['ontology_id'] or 'Unknown'}")
            for a in meta['assertions']]]))
    for conflict in meta['conflicts']:
        sources.append(_detail_text_block(
            f"{conflict['field']} conflict ({conflict['status']})",
            f"{conflict['source_a']}: {conflict['value_a']}; "
            f"{conflict['source_b']}: {conflict['value_b']}. {conflict['notes'] or ''}"))
    for study in meta['studies']:
        sources.append(html.Details(children=[
            html.Summary(study['gse'] + ': ' + metadata.display_value(study['title'])),
            _accession_row('GEO study', study['gse']),
            _detail_text_block('Summary', study['summary'] or 'Unknown'),
            _detail_text_block('Experimental design', study['overall_design'] or 'Unknown'),
            _detail_row('PubMed IDs', metadata.display_value(study['pubmed_ids'])),
            _detail_row('Retrieved', metadata.display_value(study['retrieved_at'])),
            _detail_row('Source SHA-256', metadata.display_value(study['source_sha256'])),
        ]))
    if not meta['studies']:
        sources.append(_detail_row('Study context', 'Unknown'))
    extra = meta['sample_context']
    for key, label in [('treatment_protocol','GEO treatment protocol'),
                       ('growth_protocol','GEO growth protocol'), ('description','GEO description')]:
        sources.append(_detail_text_block(label, extra.get(key) or 'Unknown', collapsible=True))
    if extra.get('source_gse'):
        sources.append(_accession_row('Sample context source', extra['source_gse']))
        sources.append(_detail_row('Sample context retrieved', metadata.display_value(extra.get('retrieved_at'))))
    sources.append(html.A('Metadata database in Box', href=f'https://app.box.com/file/{metadata.BOX_FILE_ID}',
                          target='_blank', rel='noopener noreferrer'))
    sources.append(_detail_row('Box version', meta['box_version']))
    return html.Div(className='details-section metadata-details', children=[
        html.Div('Source annotations', className='details-section-title'),
        *rows,
        html.Details(children=[html.Summary('Metadata sources and study context'), *sources]),
    ])


def prepare(payload, filters):
    """Enrich only the small retrieved set, then select it in its original order."""
    all_hits = list(payload.get('hits') or [])
    b_hits = list((payload.get('comparison') or {}).get('hits_b') or [])
    annotated, status = metadata.enrich(all_hits + b_hits)
    enriched = dict(payload, hits=annotated[:len(all_hits)])
    if payload.get('comparison'):
        enriched['comparison'] = dict(payload['comparison'], hits_b=annotated[len(all_hits):])
    rows = results.result_rows(enriched)
    visible = results.filter_rows(rows, {} if status else filters)
    return results.filtered_payload(enriched, visible), rows, visible, status


def register(app):
    fields = list(results.FILTERS)

    @app.callback(
        Output('results-view-store', 'data'), Output('results-rows-store', 'data'),
        Output('results-controls', 'style'), Output('results-count', 'children'),
        Output('results-metadata-status', 'children'),
        Output('results-sample', 'options'), Output('results-sample', 'value'),
        Output('results-order-note', 'children'), Output('results-export', 'disabled'),
        *[Output(f'results-{field}', 'options') for field in fields],
        *[Output(f'results-{field}', 'value') for field in fields],
        *[Output(f'results-{field}', 'disabled') for field in fields],
        Input('hits-store', 'data'), Input('results-reset', 'n_clicks'),
        *[Input(f'results-{field}', 'value') for field in fields],
        State('results-sample', 'value'),
    )
    def show_results(payload, _reset, *values):
        selected = values[-1]
        filters = dict(zip(fields, values[:-1]))
        if ctx.triggered_id in (None, 'hits-store', 'results-reset'):
            filters = dict.fromkeys(fields)
        view, rows, visible, status = prepare(payload or {}, filters)
        # Searches already draw their network. Only filters/reset or a restored
        # route need another draw; sending the same figure twice races Plotly.
        view['_redraw_results'] = ctx.triggered_id != 'hits-store'
        if status:
            filters = dict.fromkeys(fields)
            visible = rows
            view = results.filtered_payload(view, rows)
        options = []
        comparison = bool((payload or {}).get('comparison'))
        for row in visible:
            rank_score = (f"{row['rank']}. {row['gsm']} · {row['score']:.6f}"
                          if row['rank'] is not None else row['gsm'])
            if comparison:
                rank_score = row['gsm'] + ' · ' + ' · '.join(
                    f"{arm} #{row[r]}: {row[s]:.6f}" for arm, r, s in
                    [('A','rank','score'),('B','rank_b','score_b')] if row[r] is not None)
            options.append({'label':rank_score, 'value':row['gsm']})
        if selected not in {row['gsm'] for row in visible} or ctx.triggered_id == 'hits-store':
            selected = None
        note = ('A rank order, then B-only matches in B rank order. CSV uses the same order.'
                if comparison else 'Original similarity rank order. CSV uses the same order.')
        return (view if payload else None, visible,
                {} if payload else {'display':'none'},
                f'{len(visible)} of {len(rows)} matches shown', status,
                options, selected, note, not bool(visible),
                *[results.filter_options(rows, field) for field in fields],
                *[filters[field] for field in fields],
                *[bool(status) or not rows for _ in fields])

    @app.callback(Output('network-graph', 'figure', allow_duplicate=True),
                  Input('results-view-store', 'data'), prevent_initial_call=True)
    def draw_filtered_network(payload):
        if not payload or not payload.get('_redraw_results', True):
            return no_update
        if not payload.get('hits') and not (payload.get('comparison') or {}).get('hits_b'):
            return _empty_network_figure('No matches meet these filters. Reset filters to show all retrieved matches.')
        query = pd.Series(payload['query'])
        a = pd.DataFrame(payload['hits'], columns=None if payload['hits'] else ['gsm','score','gse'])
        if payload.get('comparison'):
            comparison = payload['comparison']
            b = pd.DataFrame(comparison['hits_b'], columns=None if comparison['hits_b'] else ['gsm','score','gse'])
            figure = build_comparison_figure(query, a, pd.Series(comparison['query_b']), b)
        else:
            figure = build_network_figure(query, a)
        # An empty filtered state carries [0, 1] axes; explicitly release its
        # y range when matches return. No projection/retrieval coordinates change.
        figure.update_yaxes(autorange=True)
        return figure

    @app.callback(Output('selected-node-store', 'data', allow_duplicate=True),
                  Input('results-sample', 'value'), prevent_initial_call=True)
    def select_result(gsm):
        return {'kind':'gsm', 'node_id':gsm} if gsm else None

    @app.callback(Output('results-csv', 'data'), Input('results-export', 'n_clicks'),
                  State('results-rows-store', 'data'), prevent_initial_call=True)
    def download_results(clicks, rows):
        if not clicks or not rows:
            return no_update
        return dcc.send_string(results.export_csv(rows), 'bridge-rna-matches.csv')
