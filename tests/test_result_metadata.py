"""Metadata stays downstream of retrieval; rows, scores and source facts survive."""
import csv
import io
import sqlite3

import pytest


@pytest.fixture
def metadata_db(tmp_path, monkeypatch):
    # Small SQLite fixture exercises joins; receipt validation has separate
    # tests below. The running app always uses the validated real snapshot.
    monkeypatch.setattr('bridge_rna.result_metadata.validated_version', lambda path: 'test-version')
    path = tmp_path / 'metadata.sqlite'
    with sqlite3.connect(path) as db:
        db.executescript('''
            CREATE TABLE samples (gsm TEXT PRIMARY KEY, global_index INTEGER UNIQUE);
            CREATE TABLE canonical_metadata (
                gsm TEXT PRIMARY KEY, organ TEXT, disease TEXT, treatment TEXT,
                experimental_setting TEXT, perturbation TEXT,
                organ_source TEXT, disease_source TEXT, treatment_source TEXT,
                experimental_setting_source TEXT, perturbation_source TEXT);
            CREATE TABLE sample_studies (gsm TEXT, gse TEXT, PRIMARY KEY(gsm,gse));
            CREATE TABLE studies (gse TEXT PRIMARY KEY, title TEXT, summary TEXT,
                overall_design TEXT, pubmed_ids TEXT, source_url TEXT,
                retrieved_at TEXT, source_sha256 TEXT);
            CREATE TABLE metadata_conflicts (gsm TEXT, field TEXT, source_a TEXT,
                value_a TEXT, source_b TEXT, value_b TEXT, status TEXT, notes TEXT);
            CREATE TABLE metadata_assertions (gsm TEXT, field TEXT, raw_value TEXT,
                canonical_value TEXT, source TEXT, annotation_method TEXT,
                evidence_source TEXT, ontology_id TEXT, status TEXT);
            CREATE TABLE geo_sample_extras (gsm TEXT PRIMARY KEY,
                treatment_protocol TEXT, growth_protocol TEXT, description TEXT,
                source_url TEXT, source_gse TEXT, retrieved_at TEXT);
            INSERT INTO samples VALUES ('GSM1', 0), ('GSM2', 1), ('GSM3', 2);
            INSERT INTO canonical_metadata VALUES
                ('GSM1','Liver','Normal','NA','In Vivo','No','GEOMeta','GEOMeta','Gemma','GEOMeta','GEOMeta'),
                ('GSM2',NULL,NULL,'vehicle',NULL,NULL,NULL,NULL,'Gemma',NULL,NULL),
                ('GSM3','Lung','Cancer','=untrusted',NULL,NULL,'Gemma','GEOMeta','Gemma',NULL,NULL);
            INSERT INTO sample_studies VALUES ('GSM1','GSE2'),('GSM1','GSE1'),('GSM2','GSE3');
            INSERT INTO studies VALUES ('GSE1','Study one','Summary','Design','123','https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE1','2026-10-01','abc');
            INSERT INTO metadata_conflicts VALUES ('GSM1','organ','GEOMeta','Liver','Gemma','Lung','unresolved','specificity');
            INSERT INTO metadata_assertions VALUES
                ('GSM1','organ','left liver lobe','Liver','Gemma','ontology','source.tsv','UBERON:0002107','active'),
                ('GSM1','organ','Brain','Brain','Laya','model','pilot',NULL,'validation_only');
        ''')
    return path


def raw_hits():
    return [dict(gsm=f'GSM{i}', archs4_index=i-1, score=score, gse='GSE1',
                 tissue='Existing tissue', title='original')
            for i, score in [(3, .987654321), (1, .9), (2, .8)]]


def test_enrichment_is_a_stable_left_lookup_with_all_studies(metadata_db):
    from bridge_rna.result_metadata import enrich
    rows = raw_hits()
    result, status = enrich(rows, database=metadata_db)
    assert not status
    assert [{k: r[k] for k in rows[0]} for r in result] == rows
    assert all('metadata' not in r for r in rows)
    one = result[1]['metadata']
    assert one['organ'] == 'Liver'
    assert one['treatment'] == 'Unknown'
    assert one['raw']['treatment'] == 'NA'
    assert one['sources']['organ'] == 'GEOMeta'
    assert [s['gse'] for s in one['studies']] == ['GSE1', 'GSE2']
    assert one['studies'][1]['summary'] is None
    assert one['conflicts'][0]['value_b'] == 'Lung'
    assert len(one['assertions']) == 1
    assert one['assertions'][0]['raw_value'] == 'left liver lobe'
    assert metadata_db.read_bytes()[:16] == b'SQLite format 3\0'


def test_wrong_embedding_identity_refuses_metadata_without_losing_hits(metadata_db):
    from bridge_rna.result_metadata import enrich
    rows = raw_hits()
    rows[0]['archs4_index'] = 0
    result, status = enrich(rows, database=metadata_db)
    assert 'identity' in status.lower()
    assert all(r.get('metadata') is None for r in result)
    assert [r['score'] for r in result] == [r['score'] for r in rows]


def test_missing_database_is_distinct_from_missing_annotation(tmp_path):
    from bridge_rna.result_metadata import enrich
    result, status = enrich(raw_hits(), database=tmp_path/'absent.sqlite')
    assert 'unavailable' in status.lower()
    assert len(result) == 3
    assert not (tmp_path/'absent.sqlite').exists()


def test_filter_preserves_rank_order_unknown_and_empty_reset(metadata_db):
    from bridge_rna.result_metadata import enrich
    from bridge_rna.results import filter_rows, filter_options
    rows, _ = enrich(raw_hits(), database=metadata_db)
    assert filter_rows(rows, {}) == rows
    assert [r['gsm'] for r in filter_rows(rows, {'organ':'Unknown'})] == ['GSM2']
    assert [r['gsm'] for r in filter_rows(rows, {'study':'GSE2'})] == ['GSM1']
    assert filter_rows(rows, {'organ':'Liver','disease':'Cancer'}) == []
    assert filter_rows(rows, {'organ':None}) == rows
    assert {o['value'] for o in filter_options(rows, 'organ')} == {'Lung','Liver','Unknown'}


def test_csv_matches_visible_rows_and_keeps_full_scores_and_provenance(metadata_db):
    from bridge_rna.result_metadata import enrich
    from bridge_rna.results import export_csv, filter_rows, result_rows
    hits, _ = enrich(raw_hits(), database=metadata_db)
    rows = result_rows({'hits':hits})
    visible = filter_rows(rows, {'organ':'Liver'})
    output = list(csv.DictReader(io.StringIO(export_csv(visible))))
    assert len(output) == 1
    assert output[0]['gsm'] == 'GSM1'
    assert output[0]['rank'] == '2'
    assert float(output[0]['score']) == .9
    assert output[0]['organ_source'] == 'GEOMeta'
    assert output[0]['treatment_raw'] == 'NA'
    assert 'GSE2' in output[0]['studies_json']
    assert 'Lung' in output[0]['conflicts_json']
    assert list(csv.DictReader(io.StringIO(export_csv([])))) == []
    full = list(csv.DictReader(io.StringIO(export_csv(rows))))
    assert [r['gsm'] for r in full] == ['GSM3','GSM1','GSM2']
    assert float(full[0]['score']) == hits[0]['score']
    assert full[0]['treatment'].startswith("'")


def test_comparison_has_one_export_row_per_sample_with_both_original_ranks():
    from bridge_rna.results import result_rows
    rows = result_rows({'hits':raw_hits()[:2], 'comparison':{'hits_b':raw_hits()[1:]}})
    assert [r['gsm'] for r in rows] == ['GSM3','GSM1','GSM2']
    assert rows[1]['rank'] == 2 and rows[1]['rank_b'] == 1
    assert rows[1]['score'] == rows[1]['score_b'] == .9
    assert rows[2]['rank'] is None and rows[2]['rank_b'] == 2


def test_database_connection_cannot_write(metadata_db):
    from bridge_rna.result_metadata import open_readonly
    connection = open_readonly(metadata_db)
    try:
        with pytest.raises(sqlite3.OperationalError, match='readonly'):
            connection.execute("DELETE FROM samples")
    finally:
        connection.close()


def test_failed_metadata_resets_filters_without_erasing_the_network(tmp_path, monkeypatch):
    from bridge_rna import results_ui
    monkeypatch.setenv('BRIDGE_METADATA_DB', str(tmp_path/'missing.sqlite'))
    payload = {'hits':raw_hits()}
    view, rows, visible, status = results_ui.prepare(payload, {'organ':'Liver'})
    assert status
    assert view['hits'] == payload['hits']
    assert len(visible) == len(rows) == 3


def test_corrupt_database_does_not_break_retrieval(tmp_path):
    from bridge_rna.result_metadata import enrich
    path = tmp_path/'bad.sqlite'
    path.write_bytes(b'not sqlite')
    hits = raw_hits()
    rows, status = enrich(hits, database=path)
    assert rows == hits and 'Revalidate' in status


@pytest.mark.parametrize('kind', ['gsm', 'gse'])
def test_empty_filtered_inspector_retains_a_safe_state(kind):
    import pandas as pd
    from bridge_rna.panels import _build_selected_details
    result = _build_selected_details(pd.Series({'sample_id':'query'}),
                                    {'kind':kind, 'node_id':'GSM1'}, pd.DataFrame())
    assert 'Reset filters' in result[1].children


def test_snapshot_receipt_detects_file_replacement(tmp_path, monkeypatch):
    import json
    from bridge_rna import result_metadata as metadata
    database = tmp_path/'snapshot.sqlite'
    database.write_bytes(b'receipt test')
    receipt = tmp_path/'receipt.json'
    monkeypatch.setenv('BRIDGE_METADATA_VALIDATION', str(receipt))
    with pytest.raises(OSError):
        metadata.validated_version(database)
    for invalid in ('null', '[]'):
        receipt.write_text(invalid)
        rows, status = metadata.enrich(raw_hits(), database=database)
        assert rows == raw_hits() and 'Revalidate' in status
    report = dict(fingerprint=metadata.file_fingerprint(database),
                  box_file_id=metadata.BOX_FILE_ID, box_version=metadata.BOX_VERSION,
                  sha1=metadata.BOX_SHA1, bytes=metadata.BOX_SIZE,
                  sqlite_integrity='ok', foreign_key_violations=0,
                  all_sample_mappings_equal=True)
    receipt.write_text(json.dumps(report))
    assert metadata.validated_version(database) == metadata.BOX_VERSION
    database.write_bytes(b'replaced file')
    with pytest.raises(OSError, match='stale'):
        metadata.validated_version(database)
