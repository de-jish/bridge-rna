"""Linux Gunicorn control. Explicit identities and pidfds; never signal a group."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from urllib.parse import urlsplit

IDENTITY = ('pid', 'start', 'boot', 'cwd', 'exe')


def process(pid):
    proc = Path('/proc') / str(pid)
    try:
        stat = (proc / 'stat').read_text().rsplit(')', 1)[1].split()
        if stat[0] == 'Z' or proc.stat().st_uid != os.getuid():
            return None
        return {'pid': int(pid), 'start': stat[19], 'ppid': int(stat[1]),
                'boot': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                'cwd': str((proc / 'cwd').resolve(strict=True)),
                'exe': str((proc / 'exe').resolve(strict=True)),
                'comm': (proc / 'comm').read_text().strip(),
                'argv': (proc / 'cmdline').read_bytes().decode().split('\0')[:-1]}
    except (OSError, ValueError):
        return None


def same(record):
    current = process(record['pid'])
    return current is not None and all(current.get(k) == record.get(k) for k in IDENTITY)


def matches(p, target):
    if not p or p['cwd'] != str(Path(target['release']).resolve()):
        return False
    python = Path(target['venv']) / 'bin/python'
    if p['exe'] != str(python.resolve()):
        return False
    module = 'wsgi:application'
    argv = p['argv']
    if len(argv) == 1:
        return False  # rewritten titles cannot establish the virtual environment
    if not argv or module not in argv:
        return False
    # Match actual exec argv, never search a shell's -c string for substrings.
    try:
        executable = Path(argv[0])
        if not executable.is_absolute():
            executable = Path(p['cwd']) / executable
        if executable.resolve() != python.resolve() or executable.parent.resolve() != python.parent.resolve():
            return False
        i = 1
        while i < len(argv) and argv[i] in ('-B', '-u'):
            i += 1
        if argv[i:i + 2] == ['-m', 'gunicorn']:
            return executable.parent.resolve() == python.parent.resolve()
        script = Path(argv[i])
        if not script.is_absolute():
            script = Path(p['cwd']) / script
        return script.resolve() == (Path(target['venv']) / 'bin/gunicorn').resolve()
    except (IndexError, OSError):
        return False


def capture(pid, target):
    p = process(pid)
    if not matches(p, target) or matches(process(p['ppid']), target):
        raise ValueError(f'PID {pid} is not the intended Gunicorn master')
    ancestor = process(os.getpid())
    while ancestor:
        if ancestor['pid'] == pid:
            raise ValueError('refusing the deployment process or one of its ancestors')
        ancestor = process(ancestor['ppid'])
    return {k: p[k] for k in IDENTITY}


def owns_listener(record, url):
    port = urlsplit(url).port
    if port is None:
        raise ValueError('verify_url must include the explicit Gunicorn TCP port')
    inodes = set()
    for name in ('tcp', 'tcp6'):
        for row in (Path('/proc/net') / name).read_text().splitlines()[1:]:
            fields = row.split()
            if fields[3] == '0A' and int(fields[1].split(':')[1], 16) == port:
                inodes.add(fields[9])
    try:
        sockets = {os.readlink(p) for p in (Path('/proc') / str(record['pid']) / 'fd').iterdir()}
    except OSError:
        return False
    return any(f'socket:[{inode}]' in sockets for inode in inodes)


def check(cfg, target):
    record = target.get('process')
    if not record or not same(record):
        raise ValueError('recorded Gunicorn master is absent or its identity changed')
    capture(record['pid'], target)
    if not owns_listener(record, cfg['verify_url']):
        raise ValueError('the intended Gunicorn master does not own the serving socket')


def locate_started(target):
    """Recover a child if the controller died between Popen and journalling PID."""
    record = target.get('process')
    if record:
        return record
    pidfile = target.get('pidfile')
    if not pidfile:
        return None
    # The unique --pid argument exists before Gunicorn writes its PID file.
    # Scan only for that exact argument plus the full target identity.
    found = []
    for path in Path('/proc').iterdir():
        p = process(int(path.name)) if path.name.isdigit() else None
        if matches(p, target) and pidfile in p['argv']:
            try:
                found.append(capture(p['pid'], target))
            except ValueError:
                pass  # worker, never a signal target
    if len(found) > 1:
        raise ValueError('ambiguous startup identity; refusing to signal')
    return found[0] if found else None


def stop(cfg, target):
    record = locate_started(target)
    if not record:
        return
    current = process(record['pid'])
    if current is None:
        return
    if not same(record):
        raise ValueError('PID identity changed; refusing to signal it')
    capture(record['pid'], target)
    # pidfd pins the kernel process, eliminating the check/kill PID-reuse race.
    if not hasattr(os, 'pidfd_open') or not hasattr(signal, 'pidfd_send_signal'):
        raise ValueError('Linux pidfd support is required; no unsafe signal fallback')
    workers = []
    for path in Path('/proc').iterdir():
        p = process(int(path.name)) if path.name.isdigit() else None
        if p and p['ppid'] == record['pid']:
            workers.append({k: p[k] for k in IDENTITY})
    try:
        fd = os.pidfd_open(record['pid'])
    except ProcessLookupError:
        return
    try:
        if not same(record):
            raise ValueError('master changed before termination; refusing to signal')
        signal.pidfd_send_signal(fd, signal.SIGTERM)
    finally:
        os.close(fd)
    deadline = time.monotonic() + 150
    while same(record) or any(same(p) for p in workers):
        if time.monotonic() > deadline:
            raise RuntimeError('Gunicorn did not stop gracefully; no forced kill was attempted')
        time.sleep(0.2)
