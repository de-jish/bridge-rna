#!/usr/bin/env python3
"""Validate the pinned Box snapshot in place, read-only. Write only a JSON report.

Run before enabling BRIDGE_METADATA_DB, locally and again on staging/production
by its operator. No inference, source writes, database copies or index changes.
"""
from contextlib import closing
from datetime import datetime, timezone
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bridge_rna.result_metadata import (BOX_FILE_ID, BOX_VERSION, BOX_SIZE, BOX_SHA1,
                                        FIELDS, MISSING, open_readonly, file_fingerprint)


def validate(database, locations, report):
    database = Path(database).expanduser().resolve()
    before = database.stat()
    fingerprint = file_fingerprint(database)
    if before.st_size != BOX_SIZE:
        raise ValueError(f'Unexpected size: {before.st_size}; expected {BOX_SIZE}. Check the Box version.')
    started = time.monotonic()
    with database.open('rb') as source:
        sha1 = hashlib.file_digest(source, 'sha1').hexdigest()
    if sha1 != BOX_SHA1:
        raise ValueError('SHA-1 differs from the pinned Box version. Reconcile the version before use.')
    with closing(open_readonly(database)) as db:
        integrity = [row[0] for row in db.execute('PRAGMA integrity_check')]
        if integrity != ['ok']:
            raise ValueError(f'SQLite integrity check failed: {integrity[:5]}')
        if db.execute('PRAGMA foreign_key_check').fetchone() is not None:
            raise ValueError('SQLite foreign key violations')
        versions = {r[0] for r in db.execute('SELECT DISTINCT schema_version FROM metadata_assertions')}
        if versions != {'bridge-metadata-v1'}:
            raise ValueError(f'Unexpected schema versions: {versions}')
        identity = pd.read_sql_query('SELECT * FROM samples ORDER BY global_index', db)
        expected = pd.read_parquet(locations)
        if len(identity) != len(expected) or not identity.gsm.is_unique or not identity.global_index.is_unique:
            raise ValueError('Sample count or uniqueness mismatch')
        for actual, reference in [('gsm','geo_accession'), ('global_index','global_index'),
                                  ('species_id','species_id'), ('shard_idx','shard_idx'),
                                  ('shard_file','shard_file'), ('row_in_shard','row_in_shard')]:
            if not np.array_equal(identity[actual], expected[reference]):
                raise ValueError(f'Sample mapping mismatch: {actual}')
        if not np.array_equal(identity.global_index, np.arange(len(identity))):
            raise ValueError('Embedding indices are not contiguous in reference order')
        count = len(identity)
        canonical = db.execute('SELECT COUNT(*), COUNT(DISTINCT gsm) FROM canonical_metadata').fetchone()
        if tuple(canonical) != (count, count):
            raise ValueError('Canonical annotations are not one per reference sample')
        placeholders = ','.join('?' for _ in MISSING)
        coverage = {}
        for field in FIELDS:
            available = f"{field} IS NOT NULL AND lower(trim({field})) NOT IN ({placeholders})"
            n = db.execute(f'SELECT COUNT(*) FROM canonical_metadata WHERE {available}', tuple(MISSING)).fetchone()[0]
            coverage[field] = {'available': n, 'total':count, 'pct':round(n/count*100, 3),
                               'source_non_null':db.execute(f'SELECT COUNT({field}) FROM canonical_metadata').fetchone()[0]}
        context = db.execute('''SELECT COUNT(DISTINCT ss.gsm) FROM sample_studies ss
            JOIN studies s USING(gse) WHERE length(trim(s.summary)) > 0''').fetchone()[0]
        coverage['study_context'] = {'available':context, 'total':count, 'pct':round(context/count*100,3)}
        species_coverage = [dict(row) for row in db.execute('''SELECT s.species, count(*) total,
            count(c.organ) organ, count(c.disease) disease, count(c.treatment) treatment,
            count(c.experimental_setting) experimental_setting
            FROM samples s JOIN canonical_metadata c USING(gsm) GROUP BY s.species''')]
        conflict_count = db.execute('SELECT COUNT(*) FROM metadata_conflicts').fetchone()[0]
        multiple = db.execute('''SELECT COUNT(*) FROM (SELECT gsm FROM sample_studies
            GROUP BY gsm HAVING COUNT(*)>1)''').fetchone()[0]
        source_counts = {}
        for field in FIELDS:
            source_counts[field] = {str(r[0]):r[1] for r in db.execute(
                f'SELECT {field}_source,COUNT(*) FROM canonical_metadata GROUP BY {field}_source')}
        schema = [dict(r) for r in db.execute("SELECT name,sql FROM sqlite_master WHERE type='table'")]
    if fingerprint != file_fingerprint(database):
        raise ValueError('Database changed during validation')
    result = dict(validated_at=datetime.now(timezone.utc).isoformat(), database=str(database),
                  fingerprint=fingerprint,
                  box_file_id=BOX_FILE_ID, box_version=BOX_VERSION, bytes=before.st_size,
                  sha1=sha1, sqlite_integrity='ok', foreign_key_violations=0,
                  schema_versions=sorted(versions), sample_rows=count, all_sample_mappings_equal=True,
                  canonical_rows=count, coverage=coverage, species_non_null_coverage=species_coverage,
                  conflicts=conflict_count, multiple_study_samples=multiple, source_counts=source_counts,
                  schema=schema, elapsed_seconds=round(time.monotonic()-started,3))
    destination = Path(report)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database', type=Path, required=True)
    parser.add_argument('--locations', type=Path, default=ROOT/'archs4_sample_embeddings_full/sample_locations.parquet')
    parser.add_argument('--report', type=Path, default=ROOT/'reports/result-metadata-validation.json')
    args = parser.parse_args()
    result = validate(args.database, args.locations, args.report)
    print(json.dumps({k:v for k,v in result.items() if k not in ('schema', 'source_counts')}, indent=2))
