"""Network presentation must retain identity, geometry and inspection data."""
import pandas as pd

from bridge_rna.figures import GRAPH_THEME, build_network_figure
from bridge_rna.layout import build_graph_legend


def test_network_roles_and_relationships_remain_identifiable():
    query = pd.Series({'sample_id': 'OSD-1|query', 'sample_name': 'query'})
    hits = pd.DataFrame({'gsm': ['GSM1', 'GSM2'], 'gse': ['GSE1', 'GSE1'],
                         'score': [.99, .98]})
    fig = build_network_figure(query, hits)
    nodes = fig.data[-1]
    by_id = {row[1]: i for i, row in enumerate(nodes.customdata)}
    assert set(by_id) == {'OSD-1|query', 'GSM1', 'GSM2', 'GSE1'}
    assert [(nodes.x[by_id[n]], nodes.y[by_id[n]]) for n in ('OSD-1|query', 'GSM1', 'GSM2', 'GSE1')] == [(0, 0), (1, .7), (1, -.7), (2.1, 0)]
    assert nodes.marker.symbol[by_id['OSD-1|query']] == 'circle'
    assert nodes.marker.color[by_id['OSD-1|query']] == '#ffffff'
    assert nodes.marker.line.color[by_id['OSD-1|query']] == '#d83933'
    assert nodes.marker.size[by_id['OSD-1|query']] > nodes.marker.size[by_id['GSM1']]
    assert nodes.marker.color[by_id['GSM1']] == '#0b3d91'
    assert nodes.marker.symbol[by_id['GSE1']] == 'diamond'
    assert nodes.marker.color[by_id['GSE1']] == '#58585b'
    assert 'Score: 0.990' in nodes.customdata[by_id['GSM1']][2]
    assert [tuple(t.meta['network_edge']) for t in fig.data[:-1]] == [
        ('OSD-1|query', 'GSM1'), ('GSM1', 'GSE1'),
        ('OSD-1|query', 'GSM2'), ('GSM2', 'GSE1')]
    assert len({t.line.color for t in fig.data[:-1]}) == 1
    assert fig.layout.clickmode == 'event'
    assert 'legend-swatch--query' in str(build_graph_legend())
    assert 'legend-swatch--star' in str(build_graph_legend('comparison'))


def test_network_marks_and_edges_have_nontext_contrast():
    for role in ('query', 'gsm', 'gse', 'edge', 'edge_gse'):
        color = GRAPH_THEME[role].lstrip('#')
        rgb = [int(color[i:i+2], 16) / 255 for i in (0, 2, 4)]
        luminance = sum(w * (v / 12.92 if v <= .04045 else ((v + .055) / 1.055) ** 2.4)
                        for v, w in zip(rgb, (.2126, .7152, .0722)))
        assert 1.05 / (luminance + .05) >= 3, role
