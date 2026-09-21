#!/usr/bin/env python3
"""Local build/audit/SFTP shipping. Never runs server commands."""
import argparse
import ast
import difflib
import hashlib
import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import uuid

from common import DATA_DIRS, REFERENCES, atomic_json, changes, now, runtime_path, sha256, verify_bundle
from sftp_client import SFTP, ROOT as REMOTE

REPO = Path(__file__).resolve().parents[1]
STORE = REPO/'.ships'


def git(*args):
    p = subprocess.run(['git', '-C', str(REPO), *args],capture_output=True,text=True)
    return p.stdout.strip() if p.returncode == 0 else None


def build(store=STORE):
    rid = now().split('+')[0].replace('-','').replace(':','')+'Z-'+uuid.uuid4().hex[:8]
    dest = store/'shipments'/rid/'payload';dest.mkdir(parents=True)
    files = json.loads((REPO/'deploy/runtime-files.json').read_text())
    for name in files:
        source=REPO/name
        if not runtime_path(name) or source.is_symlink() or source.resolve()!=source.absolute():
            raise ValueError(f'not an allowed runtime file: {name}')
        out=dest/name;out.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,out)
        if name.endswith('.py'):
            ast.parse(out.read_text(),filename=name)
    (dest/'requirements.txt').write_text('# Production runtime only; Python 3.14.\n-r deploy/requirements.lock\n')
    files.append('requirements.txt')
    serving=(REPO/'requirements.txt').read_text().split('# --- Manifold precompute')[0]
    pins=lambda text:dict(re.findall(r'^([a-zA-Z0-9_.-]+)==([^\s\\]+)',text,re.M))
    dependencies=pins((REPO/'deploy/requirements.in').read_text())
    if any(pins((REPO/'deploy/requirements.lock').read_text()).get(n)!=v for n,v in dependencies.items()):
        raise ValueError('production lock differs from its input pins')
    if pins(serving)!=dependencies:
        raise ValueError('serving definitions drifted from production dependency inputs')
    status=git('status','--porcelain','--untracked-files=all')
    meta={'id':rid,'built_utc':now(),'commit':git('rev-parse','HEAD'),'dirty':bool(status),
          'python':'3.14','dependencies':dependencies,'files':{n:sha256(dest/n) for n in sorted(files)},
          'manual_steps':['User runs apply, then the unchanged Gunicorn command. No data/schema migrations.'],
          'server_before':{},'removed':[], 'previous_manifest_sha256':None}
    atomic_json(dest/'ship.json',meta)
    verify_bundle(dest)
    atomic_json(store/'latest.json',{'id':rid})
    return dest


def check(dest, python):
    if not python.is_file():
        raise ValueError('create .ships/validation-env with Python 3.14 and the production lock first')
    subprocess.run([python,'-c','import sys; assert sys.version_info[:2] == (3,14)'],check=True)
    subprocess.run([python,'-m','pip','check'],check=True)
    from apply import environment_matches
    if not environment_matches(python.parent.parent,dest/'deploy/requirements.lock'):
        raise ValueError('local validation environment differs from production lock')
    with tempfile.TemporaryDirectory(prefix='bridge-ship-check-') as temporary:
        stage=Path(temporary)/'app';shutil.copytree(dest,stage)
        for name in DATA_DIRS:
            source=(REPO/name).resolve()
            if not source.is_dir():
                raise ValueError(f'local validation data missing: {source}')
            (stage/name).symlink_to(source,target_is_directory=True)
        env=os.environ.copy()
        for key in ('PYTHONPATH','PYTHONHOME','VIRTUAL_ENV'):
            env.pop(key,None)
        env.update(PYTHONDONTWRITEBYTECODE='1',TMPDIR=temporary,BRIDGE_RNA_ROOT=str(stage),
                   MANIFOLD_CACHE_DIR=str(stage/'cache'))
        subprocess.run([python,'-B',stage/'deploy/preflight.py','--report',dest.parent/'local-preflight.json'],
                       cwd=stage,env=env,check=True,timeout=1800)
    verify_bundle(dest)
    atomic_json(dest.parent/'validated.json',{'utc':now(),'files':json.loads((dest/'ship.json').read_text())['files']})


def read_optional(sftp, name):
    if sftp.checked(name,missing=True) is None:
        return None
    return sftp.get(name)


def audit(sftp, dest, store):
    meta=json.loads((dest/'ship.json').read_text())
    previous_bytes=read_optional(sftp,'ship.json')
    previous=json.loads(previous_bytes) if previous_bytes else {'files':{}}
    if any(not runtime_path(n) for n in previous['files']):
        raise ValueError('server manifest includes unmanaged paths')
    names=set(meta['files'])|set(previous['files'])
    backup=dest.parent/'server-before';backup.mkdir(exist_ok=True)
    if previous_bytes:
        (backup/'ship.json').write_bytes(previous_bytes)
    observed={};drift=[]
    for name in sorted(names):
        data=read_optional(sftp,name)
        observed[name]=hashlib.sha256(data).hexdigest() if data is not None else None
        if data is not None:
            out=backup/name;out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes(data)
        if name in previous['files'] and observed[name]!=previous['files'][name]:
            drift.append(name)
        if name in REFERENCES and data is not None and observed[name]!=meta['files'].get(name):
            raise ValueError(f'server reference differs; no data replacement authorized: {name}')
    removed=sorted(n for n in set(previous['files'])-set(meta['files']) if observed[n] and n not in REFERENCES)
    meta.update(server_before=observed,removed=removed,
                previous_manifest_sha256=hashlib.sha256(previous_bytes).hexdigest() if previous_bytes else None)
    baseline_path=store/'last-transferred.json'
    baseline=json.loads(baseline_path.read_text()) if baseline_path.exists() else {'files':{}}
    report={'id':meta['id'],'server_manifest_id':previous.get('id'),
            'last_transferred_id':baseline.get('id'),
            'since_last_transferred':changes(baseline['files'],meta['files']),
            'dependency_changes':changes(baseline.get('dependencies',{}),meta['dependencies']),
            'versus_server':changes({n:h for n,h in observed.items() if h},meta['files']),
            'server_drift':drift,'first_reconciliation':not bool(previous_bytes),
            'removed_on_apply':removed}
    for name in sorted(names & set(meta['files'])):
        before=backup/name;after=dest/name
        if before.is_file() and observed[name]!=meta['files'][name]:
            out=dest.parent/'diffs'/name;out.parent.mkdir(parents=True,exist_ok=True)
            if name.endswith('.woff2'):
                out.write_text(f'Binary font changed: {name}\n'
                               f'server SHA-256: {observed[name]}\n'
                               f'local SHA-256: {meta["files"][name]}\n')
            else:
                out.write_text(''.join(difflib.unified_diff(before.read_text().splitlines(True),after.read_text().splitlines(True),
                                                         fromfile='server/'+name,tofile='local/'+name)))
    atomic_json(dest/'ship.json',meta)
    atomic_json(dest.parent/'comparison.json',report)
    atomic_json(dest.parent/'server-before.json',observed)
    running=read_optional(sftp,'.ship/running.json')
    if running:
        receipt=json.loads(running)
        if previous_bytes and receipt.get('id')==previous.get('id') and receipt.get('manifest_sha256')==hashlib.sha256(previous_bytes).hexdigest():
            atomic_json(store/'last-confirmed-running.json',receipt)
    print(json.dumps(report,indent=2),flush=True)
    return meta,report


def transfer(sftp,dest,store,meta):
    validated=json.loads((dest.parent/'validated.json').read_text())
    if validated['files']!=meta['files']:
        raise ValueError('payload changed since local validation')
    verify_bundle(dest)
    remote='.ship/incoming/'+meta['id']
    if sftp.checked(remote,missing=True) is not None:
        raise ValueError('staging ID already exists; never overwrite it')
    sftp.mkdir(remote)
    dirs={remote}
    paths=sorted(meta['files'])+['ship.json']
    for name in paths:
        directory=remote+'/'+str(Path(name).parent)
        if directory not in dirs:
            sftp.mkdir(directory);dirs.add(directory)
        sftp.put(remote+'/'+name,(dest/name).read_bytes())
    for name in paths:
        if hashlib.sha256(sftp.get(remote+'/'+name)).hexdigest()!=sha256(dest/name):
            raise ValueError(f'transferred content mismatch: {name}')
    receipt={'id':meta['id'],'status':'transferred, not confirmed running','utc':now(),
             'files':meta['files'],'dependencies':meta['dependencies'],'manifest_sha256':sha256(dest/'ship.json')}
    atomic_json(dest.parent/'transferred.json',receipt)
    atomic_json(store/'last-transferred.json',receipt)
    print(f'TRANSFERRED AND READ-BACK VERIFIED: {REMOTE}/{remote}\nNOT APPLIED OR CONFIRMED RUNNING',flush=True)
    print(f'On NASA: python3 ~/fm_viz_new/{remote}/deploy/apply.py apply',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['build','audit','ship','confirm','status'])
    parser.add_argument('--reviewed-server-diff',action='store_true',help='The saved server/source differences were reviewed')
    args=parser.parse_args()
    cfg=json.loads((REPO/'deploy/ship.json').read_text())
    if cfg['remote_root']!=REMOTE:
        raise ValueError('this workflow is restricted to /home/ubuntu/fm_viz_new')
    if args.command=='status':
        for name in ('last-transferred.json','last-confirmed-running.json'):
            p=STORE/name
            print(name, p.read_text() if p.exists() else 'none')
        return
    if args.command=='confirm':
        with SFTP(cfg['host']) as sftp:
            data=read_optional(sftp,'.ship/running.json')
            canonical=read_optional(sftp,'ship.json')
        if not data or not canonical:
            raise ValueError('no server confirmation receipt yet; transfer alone is not startup verification')
        receipt=json.loads(data);meta=json.loads(canonical)
        if receipt.get('status')!='confirmed running' or receipt.get('id')!=meta['id'] or receipt.get('manifest_sha256')!=hashlib.sha256(canonical).hexdigest():
            raise ValueError('stale or mismatched running receipt')
        atomic_json(STORE/'last-confirmed-running.json',receipt)
        print('Recorded user-side verification at',receipt['utc'],'for',receipt['id'])
        return
    dest=build()
    print('BUILT',dest.parent.name,dest,flush=True)
    if args.command=='build':
        check(dest,REPO/cfg['validation_python']);return
    with SFTP(cfg['host']) as sftp:
        meta,report=audit(sftp,dest,STORE)
        if args.command=='audit':
            return
        if (report['first_reconciliation'] or report['server_drift']) and not args.reviewed_server_diff:
            raise ValueError(f'review {dest.parent}/diffs and rerun ship --reviewed-server-diff')
        check(dest,REPO/cfg['validation_python'])
        transfer(sftp,dest,STORE,meta)


if __name__=='__main__':
    main()
