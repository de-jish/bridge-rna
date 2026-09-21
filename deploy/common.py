"""Manifest and path checks shared by local shipping and user-run application."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import uuid

REFERENCES = {'data/ensembl/orthologs_one2one.txt',
              'data/gencode/gencode_v49_mouse_gene_exon_lengths.csv',
              'data/archs4/train_orthologs/canonical_genes.csv'}
DATA_DIRS = ('data/osdr', 'archs4_sample_embeddings_full', 'checkpoints_performer', 'cache')
HELPERS = {'common.py', 'apply.py', 'processes.py', 'preflight.py', 'requirements.in', 'requirements.lock'}
TOP = {'app.py','wsgi.py','osdr_metadata.py','demo_osdr_top5.py','generate_archs4_embeddings.py',
       'slim_performer_model.py','numerator_and_denominator.py','requirements.txt'}


def now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def valid_id(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', value):
        raise ValueError('invalid shipment ID')
    return value


def runtime_path(name):
    p = Path(name)
    if p.is_absolute() or '..' in p.parts or str(p) != name:
        return False
    if name in TOP | REFERENCES | {'prompts/ai_summary_prompt.txt', 'precompute/embed_upload.py'}:
        return True
    if len(p.parts) == 2:
        parent, leaf = p.parts
        if parent in ('bridge_rna', 'manifold'):
            return bool(re.fullmatch(r'[a-zA-Z_][a-zA-Z_0-9]*\.py', leaf))
        if parent == 'assets':
            return bool(re.fullmatch(r'[a-zA-Z0-9_-]+\.(css|js)', leaf))
        if parent == 'deploy':
            return leaf in HELPERS
    return False


def safe(root, relative):
    root = Path(root).absolute()
    p = Path(relative)
    if p.is_absolute() or '..' in p.parts:
        raise ValueError('unsafe relative path')
    current = root
    for part in ('', *p.parts):
        current = current / part
        if current.is_symlink():
            raise ValueError(f'symlink refused: {current}')
    return root / p


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    with tmp.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    os.replace(tmp, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def verify_bundle(root):
    root = Path(root)
    meta = json.loads(safe(root, 'ship.json').read_text())
    valid_id(meta['id'])
    for name, digest in meta['files'].items():
        if not runtime_path(name) or sha256(safe(root, name)) != digest:
            raise ValueError(f'unsafe path or content mismatch: {name}')
    actual = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() or p.is_symlink()}
    if actual != set(meta['files']) | {'ship.json'}:
        raise ValueError('unexpected or missing staged files')
    return meta


def changes(old, new):
    return {'added': sorted(new.keys()-old.keys()), 'removed': sorted(old.keys()-new.keys()),
            'changed': sorted(p for p in old.keys() & new.keys() if old[p] != new[p])}
