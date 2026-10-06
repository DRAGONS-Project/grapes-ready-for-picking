"""Stationary frames in the BLT passes: consecutive-frame motion from kept artefacts.

Motion between consecutive sampled frames is measured three ways, none of which runs COLMAP:
  sfm    distance between consecutive camera centres of a registered model (the quantity behind
         the characterisation column baseline_frac), from the kept pose file, as a ratio to the
         sequence's own median step;
  rtk    distance between the RTK positions (/gps/fix) interpolated to the two frames' header
         stamps, in metres (needs rtk_<date>.json written by --fetch);
  odo    the odometry speed (/odometry/gps twist) at the frame, matched once by header stamp and
         once by bag record time (the pass window was selected on bag record time).
A step is stationary when it is below one tenth of the sequence's median step.

usage: blt_stationary.py --fetch <sequence id> [before_s after_s]   read fixes and odometry around
                                                                   the pass window (remote bag)
       blt_stationary.py                                            write the CSVs
outputs (results/ablations/blt_stationary/):
  blt_stationary_start.csv     one row per consecutive frame pair of the two passes with a
                               stationary segment (15 Sep and 13 Jul front)
  blt_stationary_summary.csv   one row per kept BLT pose file: leading / trailing stationary frames
"""

import csv
import json
import math
import re
import statistics as st
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RES = REPO / "results"
OUT = RES / "ablations/blt_stationary"
THR = 0.1


def fetch(seq_id, before=20.0, after=8.0):
    import time
    sys.path.insert(0, str(Path(__file__).parent))
    from common import RemoteBag
    from rosbags.typesys import Stores, get_typestore
    from sequences import SEQUENCES
    seq = next(s for s in SEQUENCES if s["id"] == seq_id)
    meta = json.loads((RES / "characterisation/per_sequence" / f"{seq_id}.meta.json").read_text())
    ta, tb = meta["pass_window_s"]
    ts = get_typestore(Stores.ROS1_NOETIC)
    bag = RemoteBag(seq["bag"])
    fix_ids, odo_ids = bag.conn_ids("/gps/fix"), bag.conn_ids("/odometry/gps")
    fixes, odos, t0 = [], [], time.time()
    for cid, t, msg in bag.window(max(ta - before, 0), tb + after):
        if cid in fix_ids:
            m = ts.deserialize_ros1(msg, "sensor_msgs/msg/NavSatFix")
            fixes.append((round(t, 4), m.header.stamp.sec + m.header.stamp.nanosec * 1e-9,
                          m.latitude, m.longitude, m.altitude, int(m.status.status)))
        elif cid in odo_ids:
            m = ts.deserialize_ros1(msg, "nav_msgs/msg/Odometry")
            odos.append((round(t, 4), m.header.stamp.sec + m.header.stamp.nanosec * 1e-9,
                         m.pose.pose.position.x, m.pose.pose.position.y, m.twist.twist.linear.x))
    OUT.mkdir(parents=True, exist_ok=True)
    date = re.search(r"(\d{8})", seq_id).group(1)
    (OUT / f"rtk_{date}.json").write_text(json.dumps(dict(
        sequence=seq_id, bag=seq["bag"], pass_window_s=[ta, tb], read_window_s=[max(ta - before, 0), tb + after],
        columns=dict(fixes=["bag_time_s", "header_stamp", "lat", "lon", "alt", "status"],
                     odometry=["bag_time_s", "header_stamp", "x", "y", "speed_mps"]),
        fixes=fixes, odometry=odos, bytes_fetched_gb=round(bag.bytes_fetched / 1e9, 2))))
    print(seq_id, len(fixes), "fixes", len(odos), "odometry messages", round(bag.bytes_fetched / 1e9, 2), "GB",
          round(time.time() - t0), "s")


def frame_index(name):
    return int(re.search(r"(\d+)\.\w+$", name).group(1))


def sfm_steps(path):
    rows = sorted(csv.DictReader(open(path)), key=lambda r: r["image"])
    pos = {frame_index(r["image"]): [float(r[k]) for k in ("cx", "cy", "cz")] for r in rows}
    return pos, {i: math.dist(pos[i], pos[i + 1]) for i in pos if i + 1 in pos}


def interp(x, xs, ys):
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    lo, hi = 0, len(xs) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if xs[mid] <= x:
            lo = mid
        else:
            hi = mid
    w = (x - xs[lo]) / (xs[hi] - xs[lo]) if xs[hi] > xs[lo] else 0.0
    return ys[lo] + w * (ys[hi] - ys[lo])


def frame_times(seq_id):
    out = []
    for ln in open(RES / "characterisation/frame_lists" / f"{seq_id}.txt"):
        m = re.search(r"^(\S+)\tbag_time=([\d.]+)s header_stamp=([\d.]+)", ln)
        out.append((m.group(1), float(m.group(2)), float(m.group(3))))
    return out


def edge_runs(flags):
    lead = 0
    while lead < len(flags) and flags[lead]:
        lead += 1
    trail = 0
    while trail < len(flags) - lead and flags[-1 - trail]:
        trail += 1
    return lead, trail


def detail(seq_id, pose_file):
    frames = frame_times(seq_id)
    _, steps = sfm_steps(pose_file)
    med = st.median(steps.values())
    date = re.search(r"(\d{8})", seq_id).group(1)
    rtk_file = OUT / f"rtk_{date}.json"
    rtk = json.loads(rtk_file.read_text()) if rtk_file.exists() else None
    rows = []
    if rtk:
        fx = sorted(rtk["fixes"], key=lambda f: f[1])
        lat0, lon0 = fx[0][2], fx[0][3]
        earth = 6378137.0
        east = [math.radians(f[3] - lon0) * earth * math.cos(math.radians(lat0)) for f in fx]
        north = [math.radians(f[2] - lat0) * earth for f in fx]
        hs = [f[1] for f in fx]
        px = [interp(f[2], hs, east) for f in frames]
        py = [interp(f[2], hs, north) for f in frames]
        rstep = [math.hypot(px[i + 1] - px[i], py[i + 1] - py[i]) for i in range(len(frames) - 1)]
        rmed = st.median(rstep)
        oh = sorted(rtk["odometry"], key=lambda o: o[1])
        ob = sorted(rtk["odometry"], key=lambda o: o[0])
        v_hdr = [interp(f[2], [o[1] for o in oh], [o[4] for o in oh]) for f in frames]
        v_bag = [interp(f[1], [o[0] for o in ob], [o[4] for o in ob]) for f in frames]
    for i in range(len(frames) - 1):
        s = steps.get(i)
        row = dict(sequence=seq_id, frame=i, frame_name=frames[i][0], bag_time_s=frames[i][1],
                   header_stamp=frames[i][2],
                   sfm_step_ratio=round(s / med, 4) if s is not None else "",
                   sfm_stationary=(s < THR * med) if s is not None else "")
        if rtk:
            row.update(rtk_step_m=round(rstep[i], 4), rtk_step_ratio=round(rstep[i] / rmed, 4),
                       rtk_stationary=rstep[i] < THR * rmed,
                       odometry_speed_by_header_mps=round(v_hdr[i], 3),
                       odometry_speed_by_bag_time_mps=round(v_bag[i], 3))
        rows.append(row)
    return rows


def summary():
    files = (sorted(RES.glob("characterisation/poses/blt_*.csv")) + sorted(RES.glob("full_program/B/blt_*/blt_*.csv"))
             + sorted(RES.glob("full_program/C/blt/*/raw/blt_*.csv")) + sorted(RES.glob("full_program/C/blt/*/*/poses.csv")))
    rows = []
    for f in files:
        pos, steps = sfm_steps(f)
        if len(steps) < 5:
            continue
        rel = f.relative_to(RES)
        seq = f.stem if f.name != "poses.csv" else rel.parts[3]
        n_frames = sum(1 for _ in open(RES / "characterisation/frame_lists" / f"{seq}.txt"))
        med = st.median(steps.values())
        full = all(i in steps for i in range(n_frames - 1))
        flags = [steps[i] < THR * med for i in sorted(steps)]
        lead, trail = edge_runs(flags) if full else ("", "")
        low = sorted(i for i, s in steps.items() if s < THR * med)
        # slow tail: from the first step after which no step reaches half the median again
        tail_step = ""
        if full:
            i = n_frames - 1
            while i > 0 and steps[i - 1] < 0.5 * med:
                i -= 1
            tail_step = i if i < n_frames - 1 else ""
        rows.append(dict(sequence=seq, pose_file=str(rel), n_frames=n_frames, n_registered=len(pos),
                         all_steps_measured=full, median_step_sfm_units=round(med, 4),
                         leading_stationary_steps=lead, trailing_stationary_steps=trail,
                         steps_below_tenth_median=len(low),
                         first_step_below_tenth_median=low[0] if low else "",
                         slow_tail_first_frame=(tail_step + 1) if tail_step != "" else "",
                         slow_tail_frames=(n_frames - tail_step - 1) if tail_step != "" else 0 if full else ""))
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for seq, pose in (("blt_gr_20220915_front", "full_program/C/blt/blt_gr_20220915_front/raw/blt_gr_20220915_front.csv"),
                      ("blt_gr_20220713_front", "full_program/C/blt/blt_gr_20220713_front/hand_both/poses.csv")):
        rows += detail(seq, RES / pose)
    cols = []
    for r in rows:
        cols += [c for c in r if c not in cols]
    with open(OUT / "blt_stationary_start.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    srows = summary()
    with open(OUT / "blt_stationary_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(srows[0].keys()))
        w.writeheader()
        w.writerows(srows)
    print(f"blt_stationary_start.csv: {len(rows)} rows; blt_stationary_summary.csv: {len(srows)} rows")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--fetch":
        fetch(sys.argv[2], *(float(a) for a in sys.argv[3:5]))
    else:
        main()
