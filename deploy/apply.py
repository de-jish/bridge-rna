#!/usr/bin/env python3
"""USER-RUN only. Apply staged files to fm_viz_new; never launches Gunicorn."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import subprocess
import sys
import time
import urllib.request

sys.dont_write_bytecode = True
from common import DATA_DIRS, REFERENCES, atomic_json, now, runtime_path, safe, sha256, verify_bundle
import processes

ARGS = ['--workers','2','--threads','4','--timeout','120','--bind','0.0.0.0:8000']


def root_path():
    root = Path.home() / 'fm_viz_new'
    if root.is_symlink() or not root.is_dir():
        raise ValueError('fm_viz_new must be the canonical, real application directory')
    return root


def phase(root, name, rid):
    atomic_json(safe(root, '.ship/status.json'), {'id': rid, 'phase': name, 'utc': now()})
    print(name, flush=True)


def call(argv, **kwargs):
    subprocess.run([str(x) for x in argv], check=True, timeout=1800, **kwargs)


def assert_port_available():
    import socket
    with socket.socket() as probe:
        probe.bind(('0.0.0.0',8000))


def target(root):
    return {'id':'legacy', 'release':str(root), 'venv':str(root/'fm_viz_env')}


def find_master(root):
    result = []
    for path in Path('/proc').iterdir():
        p = processes.process(int(path.name)) if path.name.isdigit() else None
        matched = processes.matches(p, target(root))
        if p and p['cwd'] == str(root) and p.get('comm') == 'gunicorn' and not matched:
            raise ValueError('unidentified Gunicorn in canonical directory; refusing file changes')
        if matched:
            try:
                record = processes.capture(p['pid'], target(root))
                result.append(record)
            except ValueError:
                continue
    if len(result) > 1:
        raise ValueError('multiple application masters; refusing to stop any')
    return result[0] if result else None


def expected_files(root, meta):
    manifest=safe(root,'ship.json')
    digest=sha256(manifest) if manifest.is_file() else None
    if digest!=meta.get('previous_manifest_sha256'):
        raise ValueError('production manifest changed since transfer')
    previous=json.loads(manifest.read_text())['files'] if digest else {}
    for name, digest in meta['server_before'].items():
        if not runtime_path(name):
            raise ValueError('unsafe baseline path')
        path = safe(root, name)
        actual = sha256(path) if path.is_file() else None
        if actual != digest:
            raise ValueError(f'production changed since transfer: {name}; re-audit and ship')
    for name in REFERENCES & meta['files'].keys():
        path = safe(root, name)
        if path.exists() and sha256(path) != meta['files'][name]:
            raise ValueError(f'reference data change requires separate review: {name}')
    for name in meta['removed']:
        if name not in previous or not runtime_path(name) or name in REFERENCES or not meta['server_before'].get(name):
            raise ValueError('removal is not a tracked production code file')


def preflight(root, code, python, report):
    # Only temporary links in the candidate; no shared/server data is copied.
    linked = []
    try:
        if code != root:
            for name in DATA_DIRS:
                source = safe(root, name)
                if not source.is_dir():
                    raise ValueError(f'required dataset directory: {source}')
                path = code/name
                path.symlink_to(source, target_is_directory=True)
                linked.append(path)
        env = os.environ.copy()
        for key in ('PYTHONPATH','PYTHONHOME','VIRTUAL_ENV'):
            env.pop(key, None)
        tmp = safe(root, '.ship/tmp'); tmp.mkdir(exist_ok=True)
        env.update(PYTHONDONTWRITEBYTECODE='1', BRIDGE_RNA_ROOT=str(code),
                   MANIFOLD_CACHE_DIR=str(code/'cache'), TMPDIR=str(tmp))
        call([python, '-B', code/'deploy/preflight.py', '--report', report], cwd=code, env=env)
    finally:
        for path in linked:
            path.unlink()


def environment_matches(env, lock):
    program = r'''
import importlib.metadata as md, re, sys
from packaging.requirements import Requirement
assert sys.version_info[:2] == (3,14), 'Python 3.14 required'
for line in open(sys.argv[1]):
    if not re.match(r'^[a-zA-Z0-9_.-]+==', line): continue
    requirement = Requirement(line.rstrip().rstrip('\\').strip())
    if requirement.marker and not requirement.marker.evaluate(): continue
    assert md.version(requirement.name) in requirement.specifier, requirement.name
'''
    result = subprocess.run([str(env/'bin/python'), '-c', program, str(lock)], capture_output=True)
    return result.returncode == 0


def prepare_environment(root, stage, work):
    env = root/'fm_viz_env'; lock = stage/'deploy/requirements.lock'
    if environment_matches(env, lock):
        call([env/'bin/python','-m','pip','check'])
        return env, False
    if sys.version_info[:2] != (3,14):
        raise ValueError('dependency update needs a Python 3.14 interpreter; old environment unchanged')
    candidate = work/'check-env'
    call([sys.executable,'-m','venv',candidate])
    wheels = work/'wheels'; wheels.mkdir()
    # Network installation occurs before any production process is stopped.
    call([candidate/'bin/python','-m','pip','--isolated','download','--no-cache-dir',
          '--require-hashes','--only-binary=:all:','--index-url','https://pypi.org/simple',
          '-r',lock,'-d',wheels])
    install(candidate, lock, wheels)
    return candidate, True


def install(env, lock, wheels):
    call([env/'bin/python','-m','pip','--isolated','install','--no-index','--find-links',wheels,
          '--require-hashes','--only-binary=:all:','-r',lock])
    call([env/'bin/python','-m','pip','check'])


def copy_file(source, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name+'.ship-tmp')
    if tmp.exists() or tmp.is_symlink():
        raise ValueError(f'temporary target already exists: {tmp}')
    shutil.copyfile(source, tmp)
    with tmp.open('rb') as stream:
        os.fsync(stream.fileno())
    os.replace(tmp, dest)


def restore(root, transaction):
    work = safe(root, transaction['work'])
    current = find_master(root)
    if current:
        raise ValueError('application was started during an incomplete apply; stop it explicitly before recovery')
    # No automatic Gunicorn restart: the user's original command starts it.
    old_env = work/'old-env'
    if old_env.is_dir():
        env = safe(root, 'fm_viz_env')
        if env.exists():
            shutil.rmtree(env)
        os.replace(old_env, env)
    for name, existed in transaction['backup'].items():
        dest = safe(root, name)
        if existed:
            copy_file(work/'old-code'/name, dest)
        elif dest.is_file():
            dest.unlink()
    safe(root, '.ship/transaction.json').unlink()
    phase(root, 'restored; ready for the usual Gunicorn command', transaction['id'])


def apply(root, stage):
    if platform.system() != 'Linux' or platform.machine() != 'x86_64':
        raise ValueError('apply requires Ubuntu/Linux x86_64')
    stage = Path(stage).absolute()
    if stage.parent != root/'.ship/incoming':
        raise ValueError('stage must be directly inside fm_viz_new/.ship/incoming')
    safe(root, str(stage.relative_to(root)))
    meta = verify_bundle(stage); rid = meta['id']
    if stage.name != rid:
        raise ValueError('staging directory ID mismatch')
    if safe(root, '.ship/transaction.json').exists():
        raise ValueError('interrupted application: run recover first')
    expected_files(root, meta)
    work = safe(root, '.ship/work-'+rid)
    work.mkdir()  # failed preparation is retained; never reuse it silently
    phase(root, 'preparing; production unchanged', rid)
    env, replace_env = prepare_environment(root, stage, work)
    preflight(root, stage, env/'bin/python', work/'preflight.json')
    expected_files(root, meta)
    master = find_master(root)
    # Capture first, then journal complete recovery BEFORE any stop or replacement.
    names = set(meta['files']) | set(meta['removed']) | {'ship.json'}
    backup = {}
    for name in names:
        old = safe(root, name); backup[name] = old.is_file()
        if old.is_file():
            path = work/'old-code'/name; path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(old,path)
    transaction = {'id':rid, 'work':str(work.relative_to(root)), 'backup':backup,
                   'master':master, 'replacing_environment':replace_env}
    atomic_json(root/'.ship/transaction.json',transaction)
    phase(root, 'applying; do not start Gunicorn until ready', rid)
    try:
        if master:
            processes.stop({}, dict(target(root), process=master))
        # If no verified master was found, refuse a occupied serving port rather
        # than guessing which process owns it. No process-name/group killing.
        assert_port_available()
        if replace_env:
            os.replace(root/'fm_viz_env',work/'old-env')
            # Venvs are not moved into place. Build at their FINAL path so the
            # user's activate script and gunicorn shebang remain correct.
            call([sys._base_executable,'-m','venv',root/'fm_viz_env'])
            install(root/'fm_viz_env',stage/'deploy/requirements.lock',work/'wheels')
        for name in meta['files']:
            copy_file(stage/name,safe(root,name))
        for name in meta['removed']:
            safe(root,name).unlink()
        copy_file(stage/'ship.json',root/'ship.json')
        preflight(root,root,root/'fm_viz_env/bin/python',work/'installed-preflight.json')
        atomic_json(root/'.ship/applied.json', {'id':rid,'status':'applied, not confirmed running',
                    'utc':now(),'manifest_sha256':sha256(root/'ship.json')})
        safe(root,'.ship/transaction.json').unlink()
    except BaseException:
        restore(root,transaction)
        raise
    shutil.rmtree(work)
    shutil.rmtree(stage)
    phase(root,'ready; run the usual Gunicorn command',rid)
    # Observes the user's subsequent manual start; never starts/restarts it.
    with (root/'.ship/verify.log').open('ab') as log:
        subprocess.Popen([sys.executable,'-B',root/'deploy/apply.py','confirm','--wait','600'],
                         stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True,cwd=root)


def confirm(root, wait):
    meta = json.loads(safe(root,'ship.json').read_text())
    deadline = time.monotonic()+wait
    while True:
        try:
            master = find_master(root)
            if not master:
                raise ValueError('no matching canonical Gunicorn master')
            for path in ('/','/map','/__release'):
                for _ in range(3):
                    req=urllib.request.Request('http://127.0.0.1:8000'+path,
                                               headers={'Cache-Control':'no-cache','Connection':'close'})
                    with urllib.request.urlopen(req, timeout=10) as response:
                        if response.status != 200 or response.headers.get('X-Bridge-Release') != meta['id']:
                            raise ValueError('wrong response/release identity')
                        if path == '/__release' and json.load(response)['release_id'] != meta['id']:
                            raise ValueError('release body mismatch')
            if not processes.same(master):
                raise ValueError('master changed during verification')
            atomic_json(root/'.ship/running.json',{'id':meta['id'],'status':'confirmed running',
                        'utc':now(),'manifest_sha256':sha256(root/'ship.json')})
            print('CONFIRMED RUNNING',meta['id']);return
        except (OSError,ValueError) as exc:
            if time.monotonic()>=deadline:
                raise RuntimeError(f'not confirmed running: {exc}')
            time.sleep(2)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['apply','run','recover','confirm','status'])
    parser.add_argument('--stage',type=Path)
    parser.add_argument('--wait',type=int,default=0)
    args=parser.parse_args();root=root_path()
    ship=safe(root,'.ship');ship.mkdir(exist_ok=True)
    if args.command == 'confirm':
        confirm(root,args.wait);return
    if args.command == 'status':
        print(safe(root,'.ship/status.json').read_text());return
    if args.command == 'apply':
        stage=Path(__file__).absolute().parents[1]
        with safe(root,'.ship/apply.log').open('ab') as log:
            child=subprocess.Popen([sys.executable,'-u','-B',str(Path(__file__).absolute()),'run',
                                    '--stage',str(stage)],stdin=subprocess.DEVNULL,stdout=log,stderr=log,
                                   cwd=root,start_new_session=True)
        print('Preparing/applying; persistent log: ~/fm_viz_new/.ship/apply.log',flush=True)
        if child.wait()!=0:
            raise RuntimeError('apply failed; inspect .ship/apply.log; production was not marked ready')
        print('READY. Start Gunicorn with your usual command.');return
    signal.signal(signal.SIGHUP,signal.SIG_IGN)
    def interrupted(signum,frame):
        raise KeyboardInterrupt('apply interrupted')
    signal.signal(signal.SIGTERM,interrupted)
    with safe(root,'.ship/apply.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.command=='recover':
            restore(root,json.loads(safe(root,'.ship/transaction.json').read_text()))
        else:
            try:
                apply(root,args.stage)
            except BaseException as exc:
                atomic_json(root/'.ship/status.json',{'phase':'failed; inspect apply.log before proceeding',
                            'error':str(exc),'utc':now()})
                raise


if __name__=='__main__':
    main()
