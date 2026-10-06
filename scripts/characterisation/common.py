"""Shared helpers for the dataset-characterisation measurements."""

import json
import os
import struct
import subprocess
import time
import urllib.parse
from pathlib import Path

import cv2
import numpy as np

MAX_FRAMES = 200
TARGET_FPS = 2.0
TARGET_WIDTH = 1600
BLACK_LUM = 8.0  # mean luminance (0-255) below which a frame counts as black


def sample_indices(n_total: int, fps: float | None) -> tuple[list[int], int]:
    """Frames 0, k, 2k, ... with k = max(floor(N/200), round(fps/2), 1), at most 200 frames."""
    k = max(n_total // MAX_FRAMES, 1)
    if fps:
        k = max(k, round(fps / TARGET_FPS))
    idx = list(range(0, n_total, k))[:MAX_FRAMES]
    return idx, k


def resize_to_width(img: np.ndarray, width: int = TARGET_WIDTH) -> np.ndarray:
    """Downscale to `width` px wide, keeping aspect. Never upscales."""
    h, w = img.shape[:2]
    if w <= width:
        return img
    return cv2.resize(img, (width, round(h * width / w)), interpolation=cv2.INTER_AREA)


def mean_lum(img: np.ndarray) -> float:
    g = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(g.mean())


def download(url: str, dest: Path, retries: int = 4) -> Path:
    """curl download with resume; returns dest."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(retries):
        r = subprocess.run(["curl", "-s", "-S", "-L", "-f", "-C", "-", "-o", str(dest), url])
        if r.returncode == 0 and dest.exists() and dest.stat().st_size > 0:
            return dest
        time.sleep(5 + 10 * attempt)
    raise RuntimeError(f"download failed: {url}")


def zenodo_url(record: int, key: str) -> str:
    return f"https://zenodo.org/api/records/{record}/files/{urllib.parse.quote(key)}/content"


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=str))


# ---------------------------------------------------------------- ROS1 bag over HTTP Range
class RemoteBag:
    """Read a ROS1 bag on the public BLT Nextcloud share through HTTP Range requests."""

    BASE = "https://lcas.lincoln.ac.uk/nextcloud/public.php/webdav"
    # share token of the BLT Nextcloud link, which the dataset authors provide on registration
    TOKEN = os.environ.get("BLT_SHARE_TOKEN", "")

    def __init__(self, path: str):
        import requests

        if not self.TOKEN:
            raise RuntimeError("set BLT_SHARE_TOKEN to the token of the BLT Nextcloud share")
        self.path = path
        self.s = requests.Session()
        self.s.auth = (self.TOKEN, "")
        self.s.headers["X-Requested-With"] = "XMLHttpRequest"
        self.url = self.BASE + urllib.parse.quote(path)
        self.bytes_fetched = 0
        r = self.s.request("PROPFIND", self.url, headers={"Depth": "0"}, timeout=120)
        import re

        self.size = int(re.search(r"<d:getcontentlength>(\d+)<", r.text).group(1))
        self._load_index()

    def get(self, start: int, end: int) -> bytes:
        err = None
        for attempt in range(5):
            try:
                r = self.s.get(self.url, headers={"Range": f"bytes={start}-{end}"}, timeout=900)
                if r.status_code == 206 and len(r.content) == end - start + 1:
                    self.bytes_fetched += len(r.content)
                    return r.content
                err = f"HTTP {r.status_code}, {len(r.content)} bytes"
            except Exception as e:  # noqa: BLE001
                err = repr(e)
            time.sleep(3 + 5 * attempt)
        raise RuntimeError(f"range {start}-{end} failed: {err}")

    @staticmethod
    def _fields(buf) -> dict:
        out, i = {}, 0
        while i < len(buf):
            (n,) = struct.unpack_from("<I", buf, i)
            i += 4
            k, _, v = bytes(buf[i : i + n]).partition(b"=")
            i += n
            out[k.decode()] = v
        return out

    @classmethod
    def _record(cls, buf, pos):
        (hl,) = struct.unpack_from("<I", buf, pos)
        pos += 4
        h = cls._fields(buf[pos : pos + hl])
        pos += hl
        (dl,) = struct.unpack_from("<I", buf, pos)
        return h, pos + 4, dl

    @staticmethod
    def _t(b) -> float:
        s, ns = struct.unpack("<II", b)
        return s + ns * 1e-9

    def _load_index(self) -> None:
        head = self.get(0, 8191)
        h, _, _ = self._record(head, 13)
        (self.index_pos,) = struct.unpack("<Q", h["index_pos"])
        (nconn,) = struct.unpack("<I", h["conn_count"])
        (nchunk,) = struct.unpack("<I", h["chunk_count"])
        idx = self.get(self.index_pos, self.size - 1)
        pos, self.conns, self.chunks = 0, {}, []
        for _ in range(nconn):
            h, pos, dl = self._record(idx, pos)
            d = self._fields(idx[pos : pos + dl])
            pos += dl
            self.conns[struct.unpack("<I", h["conn"])[0]] = (h["topic"].decode(), d["type"].decode())
        for _ in range(nchunk):
            h, pos, dl = self._record(idx, pos)
            cnt = {struct.unpack_from("<I", idx, pos + 8 * j)[0] for j in range(dl // 8)}
            pos += dl
            self.chunks.append((struct.unpack("<Q", h["chunk_pos"])[0], self._t(h["start_time"]), self._t(h["end_time"]), cnt))
        self.chunks.sort()
        self.t0 = min(c[1] for c in self.chunks)
        self.t1 = max(c[2] for c in self.chunks)

    def conn_ids(self, topic: str) -> set:
        return {c for c, (t, _) in self.conns.items() if t == topic}

    def chunk_end(self, i: int) -> int:
        return (self.chunks[i + 1][0] if i + 1 < len(self.chunks) else self.index_pos) - 1

    def messages(self, raw: bytes, off: int = 0):
        """Yield (conn_id, bag_time, message_bytes) of the chunk record at raw[off]."""
        import bz2

        h, p, dl = self._record(raw, off)
        data = raw[p : p + dl]
        if h["compression"] == b"bz2":
            data = bz2.decompress(data)
        elif h["compression"] == b"lz4":
            import lz4.frame

            data = lz4.frame.decompress(data)
        q = 0
        while q < len(data):
            h, q, n = self._record(data, q)
            if h["op"] == b"\x02":
                yield struct.unpack("<I", h["conn"])[0], self._t(h["time"]), bytes(data[q : q + n])
            q += n

    def window(self, ts0: float, ts1: float, block: int = 36):
        """Yield (conn_id, seconds_from_bag_start, message_bytes) for bag time in [ts0, ts1]."""
        sel = [i for i, c in enumerate(self.chunks) if c[2] >= self.t0 + ts0 and c[1] <= self.t0 + ts1]
        if not sel:
            return
        for b in range(sel[0], sel[-1] + 1, block):
            e = min(b + block, sel[-1] + 1)
            start = self.chunks[b][0]
            raw = self.get(start, self.chunk_end(e - 1))
            for i in range(b, e):
                for cid, tm, msg in self.messages(raw, self.chunks[i][0] - start):
                    if ts0 <= tm - self.t0 <= ts1:
                        yield cid, tm - self.t0, msg

    def sample(self, topics: list[str], step: float):
        """One chunk every `step` seconds that holds any of `topics`; yield (topic, t, message_bytes)."""
        want = {c: t for t in topics for c in self.conn_ids(t)}
        i, t = 0, self.t0
        while t < self.t1:
            while i < len(self.chunks) and (self.chunks[i][1] < t or not (set(want) & self.chunks[i][3])):
                i += 1
            if i >= len(self.chunks):
                break
            raw = self.get(self.chunks[i][0], self.chunk_end(i))
            seen = set()
            for cid, tm, msg in self.messages(raw):
                if cid in want and want[cid] not in seen:
                    seen.add(want[cid])
                    yield want[cid], tm - self.t0, msg
            t = self.chunks[i][1] + step
            i += 1
