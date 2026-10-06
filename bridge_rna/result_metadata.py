"""Read-only metadata sidecar. Never participates in retrieval or row selection.

Queries are bounded by the displayed hits, use GSM primary keys, and check any
supplied embedding index. Study membership is collected separately so its
many-to-many relationship cannot multiply result rows. No full-corpus cache.
"""
from contextlib import closing
import json
import logging
import os
from pathlib import Path
import sqlite3

from .config import ROOT

FIELDS = {
    'organ': 'Organ', 'disease': 'Disease', 'treatment': 'Treatment',
    'experimental_setting': 'Experimental setting', 'perturbation': 'Perturbation',
}
BOX_FILE_ID = '2499909479489'
BOX_VERSION = '2773286356289'
BOX_SIZE = 4371132416
BOX_SHA1 = 'b207d54b15a3a21f662cd78407794731fe338c85'
UNKNOWN = 'Unknown'
# Exact placeholders only. "No", "Normal", "untreated" and compound source
# terms have biological meaning and must never be inferred to be missing.
MISSING = frozenset(('', 'na', 'n/a', 'nan', 'null', 'unknown', 'not available'))
log = logging.getLogger(__name__)


def display_value(value):
    text = '' if value is None else str(value).strip()
    return UNKNOWN if text.casefold() in MISSING else text


def database_path():
    return Path(os.environ.get('BRIDGE_METADATA_DB') or
                Path(os.environ.get('MANIFOLD_CACHE_DIR', ROOT / 'cache')) /
                'bridge_metadata.sqlite').expanduser().resolve()


def open_readonly(path):
    connection = sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro', uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute('PRAGMA query_only=ON')
    connection.execute('PRAGMA cache_size=-2048')
    return connection


def file_fingerprint(path):
    stat = Path(path).stat()
    return {key: getattr(stat, key) for key in
            ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')}


def validated_version(path):
    """Consume the offline integrity/join check without rereading 4 GB on search.

    A replaced or modified file requires a fresh validation receipt. This is
    operator provenance, not a signature against malicious local modification.
    """
    receipt = Path(os.environ.get('BRIDGE_METADATA_VALIDATION') or
                   ROOT / 'reports/result-metadata-validation.json')
    report = json.loads(receipt.read_text())
    if not isinstance(report, dict):
        raise OSError('Metadata validation receipt must be a JSON object.')
    if (report.get('fingerprint') != file_fingerprint(path) or
            report.get('box_file_id') != BOX_FILE_ID or
            report.get('box_version') != BOX_VERSION or
            report.get('sha1') != BOX_SHA1 or report.get('bytes') != BOX_SIZE or
            report.get('sqlite_integrity') != 'ok' or
            report.get('foreign_key_violations') != 0 or
            report.get('all_sample_mappings_equal') is not True):
        raise OSError('Metadata validation receipt is missing, stale or inconsistent.')
    return report['box_version']


def enrich(hits, *, database=None):
    """Return copies in input order plus an actionable unavailable-state message.

    On failure keep every original result. A missing sidecar is distinct from a
    null source annotation. Fail closed on an identity mismatch across the batch.
    """
    rows = [dict(row) for row in hits]
    if not rows:
        return rows, ''
    path = Path(database) if database is not None else database_path()
    if not path.is_file():
        return rows, 'Additional metadata unavailable. Configure the validated metadata database.'
    try:
        version = validated_version(path)
        with closing(open_readonly(path)) as db:
            records = {}
            for hit in rows:
                gsm = str(hit['gsm'])
                if gsm in records:
                    # A comparison can request the same identity in both arms.
                    expected = records[gsm]['global_index']
                    if hit.get('archs4_index') is not None and int(hit['archs4_index']) != expected:
                        raise ValueError('Metadata sample identity does not match the embedding index.')
                    continue
                identity = db.execute('SELECT global_index FROM samples WHERE gsm=?', (gsm,)).fetchone()
                if identity is None or (hit.get('archs4_index') is not None and
                                       int(hit['archs4_index']) != identity['global_index']):
                    raise ValueError('Metadata sample identity does not match the embedding index.')
                canonical = db.execute('SELECT * FROM canonical_metadata WHERE gsm=?', (gsm,)).fetchone()
                canonical = dict(canonical) if canonical else {}
                record = {field: display_value(canonical.get(field)) for field in FIELDS}
                record['raw'] = {field: canonical.get(field) for field in FIELDS}
                record['sources'] = {field: display_value(canonical.get(field + '_source'))
                                     for field in FIELDS}
                record['global_index'] = identity['global_index']
                record['assertions'] = [dict(r) for r in db.execute('''
                    SELECT field, raw_value, canonical_value, source, annotation_method,
                           evidence_source, ontology_id, status
                    FROM metadata_assertions WHERE gsm=? AND status='active'
                    AND field IN ('organ','disease','treatment','experimental_setting','perturbation')
                    ORDER BY field, source, raw_value
                ''', (gsm,))]
                record['studies'] = [dict(r) for r in db.execute('''
                    SELECT ss.gse, s.title, s.summary, s.overall_design, s.pubmed_ids,
                           s.source_url, s.retrieved_at, s.source_sha256
                    FROM sample_studies ss LEFT JOIN studies s USING(gse)
                    WHERE ss.gsm=? ORDER BY ss.gse
                ''', (gsm,))]
                record['conflicts'] = [dict(r) for r in db.execute('''
                    SELECT field, source_a, value_a, source_b, value_b, status, notes
                    FROM metadata_conflicts WHERE gsm=? ORDER BY field, source_a, source_b
                ''', (gsm,))]
                extra = db.execute('''SELECT treatment_protocol, growth_protocol, description,
                    source_url, source_gse, retrieved_at FROM geo_sample_extras WHERE gsm=?''', (gsm,)).fetchone()
                record['sample_context'] = dict(extra) if extra else {}
                record['box_version'] = version
                records[gsm] = record
            for row in rows:
                row['metadata'] = records[str(row['gsm'])]
        return rows, ''
    except (sqlite3.Error, ValueError, KeyError, OSError) as exc:
        log.warning('Additional metadata unavailable: %s', exc)
        detail = ' Sample identity mismatch.' if 'sample identity' in str(exc).lower() else ''
        return [dict(row) for row in hits], (
            'Additional metadata unavailable.' + detail + ' Revalidate the configured database.')
