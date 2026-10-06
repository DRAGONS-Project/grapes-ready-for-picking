"""VineLiDAR headers read remotely (no point data downloaded): point count, bounding box, declared CRS.

usage: anchors_vld.py <results_dir> [gcp_lonlat.json]
Writes <results_dir>/per_sequence/vld.json. If a list of GCP lon/lat pairs is given (from the Botrytis
anchor file), also reports which GCPs fall inside each cloud's XY bounding box.
"""

import json
import os
import re
import struct
import sys
from pathlib import Path

import requests

from common import write_json, zenodo_url

RECORD = 8113105


def rng(url, a, b):
    r = requests.get(url, headers={"Range": f"bytes={a}-{b}"}, timeout=120)
    r.raise_for_status()
    return r.content


def read_header(url: str) -> dict:
    h = rng(url, 0, 374)
    assert h[:4] == b"LASF", h[:4]
    major, minor = h[24], h[25]
    hsize, off_points, n_vlr = struct.unpack_from("<HII", h, 94)
    fmt, rec_len, n_legacy = struct.unpack_from("<BHI", h, 104)
    sx, sy, sz, ox, oy, oz, maxx, minx, maxy, miny, maxz, minz = struct.unpack_from("<12d", h, 131)
    n = struct.unpack_from("<Q", h, 247)[0] if (major, minor) >= (1, 4) else n_legacy
    vl = rng(url, hsize, off_points - 1)
    pos, crs, vlrs = 0, None, []
    for _ in range(n_vlr):
        if pos + 54 > len(vl):
            break
        user = vl[pos + 2:pos + 18].split(b"\0")[0].decode(errors="replace")
        rid, length = struct.unpack_from("<HH", vl, pos + 18)
        data = vl[pos + 54:pos + 54 + length]
        vlrs.append(f"{user}:{rid}")
        if rid == 2112:  # OGC WKT
            wkt = data.split(b"\0")[0].decode(errors="replace")
            m = re.findall(r'AUTHORITY\["EPSG","(\d+)"\]', wkt)
            crs = dict(kind="WKT", epsg=m[-1] if m else None, wkt_head=wkt[:120])
        elif rid == 34735 and crs is None:  # GeoKeyDirectory
            keys = struct.unpack_from(f"<{len(data) // 2}H", data)
            epsg = None
            for i in range(4, len(keys) - 3, 4):
                if keys[i] in (3072, 2048) and keys[i + 1] == 0:  # ProjectedCSTypeGeoKey / GeographicTypeGeoKey
                    epsg = keys[i + 3] if keys[i] == 3072 or epsg is None else epsg
            crs = dict(kind="GeoTIFF keys", epsg=str(epsg) if epsg else None)
        pos += 54 + length
    if (major, minor) >= (1, 4):  # LAS 1.4 may keep the CRS in an extended VLR at the end of the file
        evlr_start, n_evlr = struct.unpack_from("<QI", h, 235)
        p = evlr_start
        for _ in range(n_evlr):
            eh = rng(url, p, p + 59)
            user = eh[2:18].split(b"\0")[0].decode(errors="replace")
            rid, length = struct.unpack_from("<HQ", eh, 18)
            vlrs.append(f"E:{user}:{rid}")
            if rid == 2112 and length < 20000:
                wkt = rng(url, p + 60, p + 59 + length).split(b"\0")[0].decode(errors="replace")
                m = re.findall(r'AUTHORITY\["EPSG","(\d+)"\]', wkt)
                crs = dict(kind="WKT (EVLR)", epsg=m[-1] if m else None, wkt_head=wkt[:120])
            p += 60 + length
    out = dict(las_version=f"{major}.{minor}", point_format=fmt & 0x3F, n_points=int(n), bbox_min=[minx, miny, minz], bbox_max=[maxx, maxy, maxz],
               bbox_source="header", scale=[sx, sy, sz], offset=[ox, oy, oz], crs=crs, vlrs=vlrs)
    if maxx == minx == 0.0 and maxy == miny == 0.0:  # empty header bbox: read the points once
        import tempfile

        import laspy
        import numpy as np

        with tempfile.NamedTemporaryFile(suffix=".laz", dir=os.environ.get("TMPDIR")) as tf:
            with requests.get(url, stream=True, timeout=600) as r:
                for chunk in r.iter_content(1 << 22):
                    tf.write(chunk)
            tf.flush()
            lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
            with laspy.open(tf.name) as f:
                for pts in f.chunk_iterator(5_000_000):
                    xyz = np.column_stack([pts.x, pts.y, pts.z])
                    lo, hi = np.minimum(lo, xyz.min(0)), np.maximum(hi, xyz.max(0))
        out.update(bbox_min=lo.tolist(), bbox_max=hi.tolist(), bbox_source="computed from points (header bbox is all zeros)")
    return out


def main() -> None:
    results = Path(sys.argv[1])
    gcps = json.loads(Path(sys.argv[2]).read_text()) if len(sys.argv) > 2 else None
    rec = requests.get(f"https://zenodo.org/api/records/{RECORD}", timeout=60).json()
    out = []
    utm = None
    if gcps:
        from pyproj import Transformer
        x, y = Transformer.from_crs("EPSG:4326", "EPSG:32629", always_xy=True).transform([g[0] for g in gcps], [g[1] for g in gcps])
        utm = list(zip(x, y))
    for f in sorted(rec["files"], key=lambda f: f["key"]):
        h = read_header(zenodo_url(RECORD, f["key"]))
        h.update(file=f["key"], size_bytes=f["size"])
        if utm:
            h["gcp_inside_xy_bbox"] = int(sum(h["bbox_min"][0] <= px <= h["bbox_max"][0] and h["bbox_min"][1] <= py <= h["bbox_max"][1] for px, py in utm))
            h["n_gcp"] = len(utm)
        out.append(h)
        print(json.dumps({k: v for k, v in h.items() if k != "vlrs"})[:400])
    write_json(results / "per_sequence" / "vld.json", out)


if __name__ == "__main__":
    main()
