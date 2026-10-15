"""Optional experimental single-sample matching controls and baseline evidence."""
import json
from dash import html, dcc, Input, Output
from .matching import METHODS


def matching_control():
    return html.Div(className='control matching-control', role='group',
                    **{'aria-labelledby': 'matching-method-label'}, children=[
        html.Label('Matching method', id='matching-method-label', className='control-label'),
        dcc.Dropdown(id='matching-method', value='cosine', clearable=False,
                     options=[{'label': v, 'value': k} for k, v in METHODS.items()]),
        html.Details(children=[html.Summary('About experimental matching'),
            html.P('Background removal subtracts the reference mean and one dominant component. '
                   'Whitening subtracts the mean and rebalances all directions, retaining the leading component. Both change the ranking, '
                   'not just the displayed numbers. Original cosine remains the default.'),
            html.P('These methods are exploratory. Greater score separation does not establish '
                   'better biological matches. Single-sample searches only; other query modes use original cosine.'),
            html.P('Map coordinates remain unchanged. Projected map distance does not determine adjusted rank.'),
        ]),
    ])


def matching_evidence(payload):
    evidence = (payload or {}).get('matching')
    if not evidence:
        return None
    rows = (payload or {}).get('hits', [])
    table_rows = []
    for rank, r in enumerate(rows, 1):
        gsm = r['gsm']
        table_rows.append(html.Tr([
            html.Td([html.A(gsm, href=f'https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={gsm}',
                           target='_blank', rel='noopener noreferrer'),
                     html.Div(r.get('title', ''), className='matching-source')]),
            *[html.Td(v, className='comparison-number') for v in (
                str(rank), f"{r['raw_rank']:,}", f"{r['score']:.6f}", f"{r['raw_cosine']:.6f}")],
        ]))
    raw, adj = evidence['raw_top_range'], evidence['adjusted_top_range']
    return html.Section(className='matching-evidence panel', **{'aria-label': 'Matching comparison'}, children=[
        html.H2('Matching comparison', className='panel-title'),
        html.P(evidence['label'], className='matching-method-name'),
        html.P(f"{evidence['overlap']} of {evidence['topk']} matches also appear in the original top "
               f"{evidence['topk']}. All {evidence['reference_count']:,} reference samples were searched.",
               className='comparison-note'),
        html.Div(className='comparison-table-wrap', children=html.Table(className='comparison-table', children=[
            html.Caption('Same query and reference collection; scores measure different geometries.'),
            html.Thead(html.Tr([html.Th(t, scope='col') for t in
                ['Reference sample', 'New rank', 'Original rank', 'Adjusted cosine', 'Original cosine']])),
            html.Tbody(table_rows),
        ])),
        html.Details(className='matching-distribution', children=[
            html.Summary('Score distributions and reproduction settings'),
            html.P(f"Original top-{evidence['topk']} range: {raw[0]:.6f} to {raw[1]:.6f}. "
                   f"Adjusted top-{evidence['topk']} range: {adj[0]:.6f} to {adj[1]:.6f}.",
                   className='comparison-note'),
            html.Div(className='comparison-table-wrap', children=html.Table(className='comparison-table', children=[
                html.Caption('Percentiles across the complete reference collection, not just the returned hits'),
                html.Thead(html.Tr([html.Th(t, scope='col') for t in ['Percentile', 'Original cosine', 'Adjusted cosine']])),
                html.Tbody([html.Tr([html.Td(f'{p}%'), html.Td(f'{a:.6f}'), html.Td(f'{b:.6f}')])
                            for p, a, b in zip(evidence['percentiles'], evidence['raw_percentiles'], evidence['adjusted_percentiles'])]),
            ])),
            html.P('A wider range is a numerical observation, not biological validation. '
                   'Adjusted cosine is not a probability or a percentage of biological similarity.', className='comparison-note'),
            html.Pre(json.dumps(evidence['fit'], indent=2), className='matching-settings'),
        ]),
    ])


def register(app):
    @app.callback(Output('matching-evidence', 'children'), Input('hits-store', 'data'),
                  Input('query-mode-store', 'data'))
    def update(payload, mode):
        return matching_evidence(payload) if mode == 'sample' else None
