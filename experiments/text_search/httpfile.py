"""A read-only, block-cached file object over HTTP range requests (enough for h5py)."""
import io
import requests


class HTTPRangeFile(io.RawIOBase):
    def __init__(self, url, block=1 << 22, size=None):
        self.url, self.block, self.pos = url, block, 0
        self.s = requests.Session()
        if size is None:
            r = self.s.get(url, headers={"Range": "bytes=0-0"}, timeout=60)
            size = int(r.headers["Content-Range"].split("/")[1])
        self.size, self.cache, self.fetched = size, {}, 0

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def _blk(self, i):
        if i not in self.cache:
            a = i * self.block
            b = min(self.size, a + self.block) - 1
            for _ in range(5):
                try:
                    r = self.s.get(self.url, headers={"Range": f"bytes={a}-{b}"}, timeout=120)
                    r.raise_for_status()
                    break
                except Exception:
                    continue
            self.cache[i] = r.content
            self.fetched += len(r.content)
            if len(self.cache) > 512:
                self.cache.pop(next(iter(self.cache)))
        return self.cache[i]

    def readinto(self, buf):
        n = min(len(buf), self.size - self.pos)
        out, got = memoryview(buf), 0
        while got < n:
            i, o = divmod(self.pos, self.block)
            d = self._blk(i)[o:o + n - got]
            out[got:got + len(d)] = d
            got += len(d)
            self.pos += len(d)
        return got
