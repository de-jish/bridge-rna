"""Presentation-only filtering and CSV. The input is an already retrieved set."""
import csv
import io
import json

from .result_metadata import BOX_FILE_ID, FIELDS, UNKNOWN, display_value

FILTERS = {k: v for k, v in FIELDS.items() if k != 'perturbation'} | {'study': 'Study'}


def result_rows(payload):
    """One displayed/exported row per GSM; preserve both comparison-arm ranks.

    Ordering is A's original ranking followed by B-only matches in B's ranking.
    Each score belongs to its own query, never a combined similarity measure.
    """
    rows = {}
    for arm, hits in [('a', payload.get('hits') or []),
                      ('b', (payload.get('comparison') or {}).get('hits_b') or [])]:
        for rank, hit in enumerate(hits, 1):
            gsm = hit['gsm']
            if gsm not in rows:
                rows[gsm] = dict(hit, rank=None, score=None, rank_b=None, score_b=None)
            rows[gsm]['rank' if arm == 'a' else 'rank_b'] = rank
            rows[gsm]['score' if arm == 'a' else 'score_b'] = hit['score']
    return list(rows.values())


def values_for(row, field):
    meta = row.get('metadata') or {}
    if field == 'study':
        return [s['gse'] for s in meta.get('studies', [])] or [UNKNOWN]
    return [display_value(meta.get(field))]


def filter_options(rows, field):
    values = {value for row in rows for value in values_for(row, field)}
    return [{'label': value, 'value': value}
            for value in sorted(values, key=lambda v: (v == UNKNOWN, v.casefold()))]


def filter_rows(rows, filters):
    return [row for row in rows if all(not value or value in values_for(row, field)
                                      for field, value in filters.items())]


def filtered_payload(payload, rows):
    """Retain arm ordering and query context without modifying the raw search."""
    keep = {row['gsm'] for row in rows}
    result = dict(payload, hits=[r for r in payload.get('hits', []) if r['gsm'] in keep])
    if payload.get('comparison'):
        result['comparison'] = dict(payload['comparison'], hits_b=[
            r for r in payload['comparison'].get('hits_b', []) if r['gsm'] in keep])
    return result


def export_csv(rows):
    columns = ['rank', 'gsm', 'score', 'rank_b', 'score_b', 'archs4_index',
               'species', 'tissue', 'gse', 'title']
    columns += [key for field in FIELDS for key in (field, field+'_source', field+'_raw')]
    columns += ['studies_json', 'sample_context_json', 'conflicts_json', 'assertions_json',
                'metadata_box_file', 'metadata_box_version']
    output = io.StringIO(newline='')
    writer = csv.DictWriter(output, fieldnames=columns)
    writer.writeheader()
    for row in rows:
        meta = row.get('metadata') or {}
        record = {key: row.get(key, '') for key in columns[:10]}
        for field in FIELDS:
            record[field] = display_value(meta.get(field))
            record[field+'_source'] = display_value(meta.get('sources', {}).get(field))
            record[field+'_raw'] = meta.get('raw', {}).get(field)
        for name, default in [('studies', []), ('sample_context', {}), ('conflicts', []), ('assertions', [])]:
            record[name+'_json'] = json.dumps(meta.get(name, default), ensure_ascii=False)
        record['metadata_box_file'] = BOX_FILE_ID if meta else ''
        record['metadata_box_version'] = meta.get('box_version', '')
        # Source annotations are untrusted spreadsheet cells. Preserve numbers
        # (including negative cosine values), escape formula-leading text only.
        writer.writerow({k: "'"+v if isinstance(v, str) and
                         v.lstrip().startswith(('=', '+', '-', '@', '\t', '\r', '\n'))
                         else v for k, v in record.items()})
    return output.getvalue()
