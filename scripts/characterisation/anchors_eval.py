"""Scale validation: align a run's registered camera centres to its metric anchor (Sim(3),
Umeyama) and report the residual.

BLT (D2): the anchor is the RTK-GNSS track (/gps/fix), fetched over HTTP range for the row's
pass window and interpolated to each frame's header stamp in a local ENU frame. Botrytis
(D1): the anchor is the per-capture EXIF GPS recorded during characterisation.

The residual is the RMSE between aligned camera centres and anchor positions, in metres —
camera-position error after the best similarity alignment, so it validates scale and shape,
not absolute georeferencing. Rows with under 8 anchored cameras are reported but flagged.

usage: anchors_eval.py blt <run_poses.csv> <sequence id> [--out results/anchors/geometric.csv]
       anchors_eval.py bot <run_poses.csv> <sequence id> [--out ...]
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
REPO = HERE.parent.parent


def umeyama(src: np.ndarray, dst: np.ndarray):
    """Similarity transform (s, R, t) minimising ||s R src + t - dst||."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    cov = xd.T @ xs / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    var_s = (xs ** 2).sum() / len(src)
    s = float(np.trace(np.diag(D) @ S) / var_s)
    t = mu_d - s * R @ mu_s
    return s, R, t


def ate(src, dst):
    s, R, t = umeyama(src, dst)
    res = (s * (R @ src.T).T + t) - dst
    return float(np.sqrt((res ** 2).sum(axis=1).mean())), s


def enu(lat, lon, alt, lat0, lon0, alt0):
    re = 6378137.0
    x = math.radians(lon - lon0) * re * math.cos(math.radians(lat0))
    y = math.radians(lat - lat0) * re
    z = (alt - alt0) if alt is not None and alt0 is not None else 0.0
    return [x, y, z]


def read_poses(path: Path):
    rows = list(csv.DictReader(open(path)))
    return {r["image"]: np.array([float(r["cx"]), float(r["cy"]), float(r["cz"])])
            for r in rows}


def blt_anchor(seq_id: str, poses: dict):
    """RTK fixes over the pass window, interpolated to each frame's header stamp."""
    from common import RemoteBag
    from rosbags.typesys import Stores, get_typestore
    from sequences import SEQUENCES
    seq = next(s for s in SEQUENCES if s["id"] == seq_id)
    meta = json.loads((REPO / "results/characterisation/per_sequence" /
                       f"{seq_id}.meta.json").read_text())
    ta, tb = meta["pass_window_s"]
    stamps = {f"frame_{i:06d}.png": fm["header_stamp"]
              for i, fm in enumerate(meta["frames"])}
    ts = get_typestore(Stores.ROS1_NOETIC)
    bag = RemoteBag(seq["bag"])
    fix_ids = bag.conn_ids("/gps/fix")
    fixes = []
    # passes cut on header stamps come from bags that record fixes late: read past the window
    late = meta.get("odometry_record_lag_max_s", 0.0) + 10 if meta.get("clock") == "header" else 0.0
    for cid, t, msg in bag.window(max(ta - 2, 0), tb + 2 + late):
        if cid in fix_ids:
            m = ts.deserialize_ros1(msg, "sensor_msgs/msg/NavSatFix")
            hs = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
            fixes.append((hs, m.latitude, m.longitude, m.altitude, int(m.status.status)))
    if len(fixes) < 8:
        return None, dict(note=f"only {len(fixes)} fixes in window")
    fixes.sort()
    f = np.array(fixes)
    lat0, lon0, alt0 = f[0, 1], f[0, 2], f[0, 3]
    pts = np.array([enu(la, lo, al, lat0, lon0, alt0) for la, lo, al in f[:, 1:4]])
    src, dst = [], []
    for name, c in poses.items():
        hs = stamps.get(name)
        if hs is None or hs < f[0, 0] or hs > f[-1, 0]:
            continue
        p = np.array([np.interp(hs, f[:, 0], pts[:, k]) for k in range(3)])
        src.append(c)
        dst.append(p)
    info = dict(n_fixes=len(fixes), status_counts={int(k): int(v) for k, v in
                zip(*np.unique(f[:, 4].astype(int), return_counts=True), strict=True)})
    return (np.array(src), np.array(dst)), info


def _ratio(v):
    return float(v.num) / float(v.den)


def _exif_gps_stream(zf, member: str):
    """lat/lon/alt of one capture. The whole member is read: MicaSense TIFFs keep the GPS
    IFD near the end of the file, so a head read finds nothing."""
    import io as _io

    import exifread
    with zf.open(member) as f:
        head = f.read()
    tags = exifread.process_file(_io.BytesIO(head), details=False)
    lat, lon = tags.get("GPS GPSLatitude"), tags.get("GPS GPSLongitude")
    if not lat or not lon:
        return None
    la = sum(_ratio(v) / 60 ** i for i, v in enumerate(lat.values))
    lo = sum(_ratio(v) / 60 ** i for i, v in enumerate(lon.values))
    if str(tags.get("GPS GPSLatitudeRef")) == "S":
        la = -la
    if str(tags.get("GPS GPSLongitudeRef")) == "W":
        lo = -lo
    alt = tags.get("GPS GPSAltitude")
    return la, lo, _ratio(alt.values[0]) if alt else None


def bot_anchor(seq_id: str, poses: dict):
    """EXIF GPS per capture, streamed from the deposit zip (64 KB per member over HTTP)."""
    import io as _io
    import zipfile as _zf
    base = seq_id.removesuffix("_rgb")
    flight = base.removeprefix("bot_")
    flist = REPO / "results/characterisation/frame_lists" / f"{seq_id}.txt"
    stems = {}
    for ln in flist.read_text().splitlines():
        if ln.strip():
            name, lab = ln.split("	")
            stems[name] = lab if not lab.endswith(".tif") else lab.rsplit("_", 1)[0]
    cache = REPO / "results/anchors/_gps_cache" / f"bot_{flight}.json"
    gps = json.loads(cache.read_text()) if cache.exists() else {}
    todo = sorted({st for st in stems.values() if st not in gps})
    if todo:
        from region_metric import HttpFile
        url = f"https://zenodo.org/api/records/7383601/files/{flight}.zip/content"
        z = _zf.ZipFile(_io.BufferedReader(HttpFile(url), buffer_size=1 << 20))
        members = set(z.namelist())
        for st in todo:
            gps[st] = (_exif_gps_stream(z, f"{st}_2.tif")
                       if f"{st}_2.tif" in members else None)
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(gps))
    src, dst, origin = [], [], None
    for name, c in poses.items():
        g = gps.get(stems.get(name))
        if g is None:
            continue
        if origin is None:
            origin = g
        src.append(c)
        dst.append(enu(g[0], g[1], g[2], origin[0], origin[1], origin[2]))
    return ((np.array(src), np.array(dst)) if src else None, dict(n_gps=len(dst)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["blt", "bot"])
    ap.add_argument("poses", type=Path)
    ap.add_argument("seq_id")
    ap.add_argument("--run", default=None, help="label for the CSV row (default: poses path)")
    ap.add_argument("--out", type=Path, default=REPO / "results/anchors/geometric.csv")
    a = ap.parse_args()
    poses = read_poses(a.poses)
    pair, info = (blt_anchor if a.kind == "blt" else bot_anchor)(a.seq_id, poses)
    row = dict(run=a.run or str(a.poses), id=a.seq_id, kind=a.kind,
               n_cameras=len(poses), **{k: json.dumps(v) if isinstance(v, dict) else v
                                        for k, v in info.items()})
    if pair is not None and len(pair[0]) >= 4:
        rmse, scale = ate(pair[0], pair[1])
        row.update(n_matched=len(pair[0]), ate_rmse_m=round(rmse, 3),
                   scale_m_per_unit=round(scale, 5),
                   flagged="yes" if len(pair[0]) < 8 else "")
    else:
        row.update(n_matched=0 if pair is None else len(pair[0]), ate_rmse_m=None)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["run", "id", "kind", "n_cameras", "n_matched", "ate_rmse_m", "scale_m_per_unit",
            "flagged", "n_fixes", "status_counts", "n_gps", "note", "anchors"]
    old = []
    if a.out.exists():
        old = [r for r in csv.DictReader(a.out.open()) if r["run"] != row["run"]]
    with a.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(old + [row])
    print(json.dumps(row, default=str))


if __name__ == "__main__":
    main()
