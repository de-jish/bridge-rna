"""Small SFTP v3 client over the existing OpenSSH authentication/configuration.

Only the SFTP subsystem is requested: no remote shell or exec channel.
The path checks implement this workflow's boundary, not account permissions.
"""
import os
from pathlib import PurePosixPath
import stat
import struct
import subprocess

ROOT = '/home/ubuntu/fm_viz_new'


def u32(n):
    return struct.pack('>I', n)


def string(value):
    value = value.encode() if isinstance(value, str) else value
    return u32(len(value)) + value


class Packet:
    def __init__(self, data):
        self.data = data
        self.pos = 0

    def take(self, n):
        result = self.data[self.pos:self.pos+n]
        if len(result) != n:
            raise ValueError('truncated SFTP packet')
        self.pos += n
        return result

    def int(self):
        return struct.unpack('>I', self.take(4))[0]

    def string(self):
        return self.take(self.int())

    def attrs(self):
        flags = self.int()
        result = {}
        if flags & 1:
            result['size'] = struct.unpack('>Q', self.take(8))[0]
        if flags & 2:
            self.take(8)
        if flags & 4:
            result['mode'] = self.int()
        if flags & 8:
            self.take(8)
        if flags & 0x80000000:
            for _ in range(self.int()):
                self.string(); self.string()
        return result


class SFTP:
    def __init__(self, host='nasa-server'):
        if not host or host.startswith('-') or any(c.isspace() for c in host):
            raise ValueError('invalid SSH alias')
        self.child = subprocess.Popen(['ssh', '-T', '-oBatchMode=yes', '-oConnectTimeout=10', '-oServerAliveInterval=15', '-oServerAliveCountMax=2',
                                       '-s', host, 'sftp'], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        self.seq = 0
        self.send(1, u32(3))
        kind, packet = self.recv()
        if kind != 2 or packet.int() != 3:
            raise ValueError('SFTP v3 is required')
        self.checked('')

    def send(self, kind, data):
        self.child.stdin.write(u32(len(data)+1) + bytes([kind]) + data)
        self.child.stdin.flush()

    def exact(self, n):
        data = bytearray()
        while len(data) < n:
            block = self.child.stdout.read(n-len(data))
            if not block:
                raise ConnectionError('SFTP connection closed')
            data.extend(block)
        return bytes(data)

    def recv(self):
        size = struct.unpack('>I', self.exact(4))[0]
        if size > 16*1024*1024:
            raise ValueError('oversized SFTP packet')
        data = self.exact(size)
        return data[0], Packet(data[1:])

    def call(self, kind, data):
        self.seq += 1
        self.send(kind, u32(self.seq) + data)
        kind, packet = self.recv()
        if packet.int() != self.seq:
            raise ValueError('SFTP response ID mismatch')
        if kind == 101:
            code = packet.int()
            message = packet.string().decode(errors='replace')
            if code == 2:
                raise FileNotFoundError(message)
            if code == 1:
                return None
            if code != 0:
                raise OSError(f'SFTP status {code}: {message}')
        return packet

    @staticmethod
    def path(relative):
        p = PurePosixPath(relative)
        if p.is_absolute() or '..' in p.parts or '\\' in relative or '\x00' in relative:
            raise ValueError('path outside authorized application directory')
        return ROOT + ('/' + str(p) if str(p) != '.' else '')

    def checked(self, relative, missing=False):
        full = self.path(relative)
        parts = PurePosixPath(relative).parts
        result = None
        for i in range(len(parts)+1):
            name = ROOT + ('/' + '/'.join(parts[:i]) if i else '')
            try:
                result = self.call(7, string(name)).attrs()  # LSTAT never follows links
            except FileNotFoundError:
                if missing:
                    return None
                raise
            if stat.S_ISLNK(result.get('mode', 0)):
                raise ValueError(f'symlink refused (target not accessed): {name}')
            if i < len(parts) and not stat.S_ISDIR(result.get('mode', 0)):
                raise ValueError(f'not a directory: {name}')
        return result

    def list(self, relative=''):
        info = self.checked(relative)
        if not stat.S_ISDIR(info['mode']):
            raise ValueError('directory required')
        handle = self.call(11, string(self.path(relative))).string()
        result = {}
        try:
            while True:
                packet = self.call(12, string(handle))
                if packet is None:
                    break
                for _ in range(packet.int()):
                    name = packet.string().decode()
                    packet.string()
                    attrs = packet.attrs()
                    if name not in ('.', '..'):
                        result[name] = attrs
        finally:
            self.call(4, string(handle))
        return result

    def get(self, relative, limit=16*1024*1024):
        try:
            from .common import runtime_path
        except ImportError:
            from common import runtime_path
        name = relative
        if name.startswith('.ship/incoming/'):
            name = '/'.join(name.split('/')[3:])
        if name not in ('ship.json', '.ship/running.json', '.ship/applied.json') and not runtime_path(name):
            raise ValueError(f'not an authorized application read: {relative}')
        info = self.checked(relative)
        if not stat.S_ISREG(info['mode']) or info['size'] > limit:
            raise ValueError(f'not an allowed small regular file: {relative}')
        handle = self.call(3, string(self.path(relative))+u32(1)+u32(0)).string()
        data = bytearray()
        try:
            while True:
                packet = self.call(5, string(handle)+struct.pack('>Q',len(data))+u32(32768))
                if packet is None:
                    break
                data.extend(packet.string())
                if len(data) > limit:
                    raise ValueError('remote file exceeded read limit')
        finally:
            self.call(4, string(handle))
        return bytes(data)

    def mkdir(self, relative):
        self.path(relative)
        if not relative.startswith('.ship/incoming/'):
            raise ValueError('writes limited to new .ship/incoming staging')
        parts = PurePosixPath(relative).parts
        for i in range(1, len(parts)+1):
            parent = '/'.join(parts[:i])
            if self.checked(parent, missing=True) is None:
                self.call(14, string(self.path(parent))+u32(0))
            elif not stat.S_ISDIR(self.checked(parent)['mode']):
                raise ValueError('staging parent is not a directory')

    def put(self, relative, data):
        self.path(relative)
        if not relative.startswith('.ship/incoming/'):
            raise ValueError('writes limited to new .ship/incoming staging')
        self.checked(str(PurePosixPath(relative).parent))
        if self.checked(relative, missing=True) is not None:
            raise ValueError('will not overwrite any remote file')
        # WRITE | CREAT | EXCL; never truncate existing files.
        handle = self.call(3, string(self.path(relative))+u32(2|8|32)+u32(0)).string()
        try:
            for offset in range(0, len(data), 32768):
                self.call(6, string(handle)+struct.pack('>Q',offset)+string(data[offset:offset+32768]))
        finally:
            self.call(4, string(handle))

    def close(self):
        self.child.stdin.close()
        self.child.wait(timeout=15)
        self.child.stdout.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
