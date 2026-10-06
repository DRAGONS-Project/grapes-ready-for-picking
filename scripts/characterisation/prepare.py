"""Fetch one sequence, apply the sampling rule, and write resized PNG frames + metadata.

usage: prepare.py <sequence index or id> <work_dir> <results_dir>

Writes <work_dir>/images/ (flat, or one sub-folder per camera group for mixed-size stills),
<work_dir>/meta.json, and <results_dir>/frame_lists/<id>.txt. Nothing under the dataset
sources is modified; downloads go to <work_dir>/src.
"""

import io
import json
import math
import re
import sys
import zipfile
from collections import Counter
from pathlib import Path

import cv2
import exifread
import numpy as np

from common import (BLACK_LUM, TARGET_WIDTH, RemoteBag, download, mean_lum, resize_to_width, sample_indices,
                    write_json, zenodo_url)
from sequences import SEQUENCES

REPO = Path(__file__).resolve().parents[2]
LOCAL_VIDEOS = {p.name: p for p in list((REPO / "data/mots").glob("*.mp4")) + list((REPO / "data/bodegas_terras_gauda").glob("*.mp4"))}
GCS_ZIP = Path.home() / "disk_checks_20260930/GrapeDatabase.zip"


# ------------------------------------------------------------------ EXIF helpers
def exif_tags(data: bytes) -> dict:
    try:
        return exifread.process_file(io.BytesIO(data), details=False)
    except Exception:  # noqa: BLE001
        return {}


def _ratio(v) -> float:
    v = v.values[0] if hasattr(v, "values") else v
    return float(v.num) / float(v.den) if hasattr(v, "num") else float(v)


def exif_gps(tags: dict):
    try:
        def dms(tag, ref):
            d, m, s = [float(x.num) / float(x.den) for x in tags[tag].values]
            val = d + m / 60 + s / 3600
            return -val if str(tags.get(ref, "")) in ("S", "W") else val
        lat, lon = dms("GPS GPSLatitude", "GPS GPSLatitudeRef"), dms("GPS GPSLongitude", "GPS GPSLongitudeRef")
        alt = _ratio(tags["GPS GPSAltitude"]) if "GPS GPSAltitude" in tags else None
        return lat, lon, alt
    except Exception:  # noqa: BLE001
        return None


def exif_focal_px(tags: dict, native_w: int, native_h: int):
    """Documented focal length in pixels at native resolution, with the EXIF fields it came from."""
    try:
        if "EXIF FocalLength" in tags and "EXIF FocalPlaneXResolution" in tags:
            f_mm = _ratio(tags["EXIF FocalLength"])
            res = _ratio(tags["EXIF FocalPlaneXResolution"])
            unit = int(str(tags.get("EXIF FocalPlaneResolutionUnit", "2")).split()[0]) if str(tags.get("EXIF FocalPlaneResolutionUnit", "2"))[0].isdigit() else 2
            per_mm = {2: res / 25.4, 3: res / 10.0, 4: res}.get(unit, res / 25.4)
            return f_mm * per_mm, f"EXIF FocalLength {f_mm:g} mm x FocalPlaneXResolution"
        if "EXIF FocalLengthIn35mmFilm" in tags:
            f35 = _ratio(tags["EXIF FocalLengthIn35mmFilm"])
            if f35 > 0:
                return f35 / 43.27 * math.hypot(native_w, native_h), f"EXIF 35 mm-equivalent focal {f35:g} mm (diagonal convention)"
    except Exception:  # noqa: BLE001
        pass
    return None, None


# ------------------------------------------------------------------ writers
def finish(seq, work, results, frames, meta):
    """frames: list of (label, bgr_or_gray image already resized, group_key or None)."""
    img_dir = work / "images"
    groups = Counter(g for _, _, g in frames)
    multi = len(groups) > 1
    names = []
    for i, (label, img, g) in enumerate(frames):
        sub = img_dir / g if multi else img_dir
        sub.mkdir(parents=True, exist_ok=True)
        name = f"frame_{i:06d}.png" if seq["population"] not in ("stills",) else f"{Path(label).stem}.png"
        cv2.imwrite(str(sub / name), img)
        names.append((f"{g}/{name}" if multi else name, label))
    (results / "frame_lists").mkdir(parents=True, exist_ok=True)
    fl, text = results / "frame_lists" / f"{seq['id']}.txt", "".join(f"{n}\t{lab}\n" for n, lab in names)
    if not fl.exists():
        fl.write_text(text)
    elif fl.read_text() != text:  # frame lists are fixed once written: never regenerate, but say so
        fl.with_suffix(".rerun.txt").write_text(text)
        meta["notes"].append("this run selected different frames from the committed frame list; the new list is in "
                             f"{fl.with_suffix('.rerun.txt').name} and the committed list was left unchanged")
    widths = Counter(img.shape[1] for _, img, _ in frames)
    meta.update(dataset=seq["dataset"], sequence_id=seq["sequence_id"], id=seq["id"], population=seq["population"],
                n_images=len(frames), width_used=max(widths), widths=dict(widths),
                camera_groups={str(k): v for k, v in groups.items()}, camera_mode="PER_FOLDER" if multi else "SINGLE")
    if max(widths) < TARGET_WIDTH:
        meta["notes"].append(f"native width {max(widths)} px < {TARGET_WIDTH}: not upscaled")
    if multi:
        meta["notes"].append(f"mixed image dimensions: intrinsics shared within each of {len(groups)} dimension groups (PER_FOLDER); "
                             "pycolmap 4.0.4 CameraMode.AUTO would give one camera per image")
    if seq.get("notes"):
        meta["notes"].append(seq["notes"])
    write_json(work / "meta.json", meta)
    write_json(results / "per_sequence" / f"{seq['id']}.meta.json", meta)
    print(json.dumps({k: v for k, v in meta.items() if k not in ("frames", "exif")}, default=str)[:1500])


# ------------------------------------------------------------------ video
def prep_video(seq, work, results):
    parts = []
    for key in seq["keys"]:
        p = LOCAL_VIDEOS.get(key) or download(zenodo_url(seq["record"], key), work / "src" / key)
        cap = cv2.VideoCapture(str(p))
        parts.append((key, p, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS),
                      int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
        cap.release()
    fps = parts[0][3]
    n_container = sum(p[2] for p in parts)

    def decode(sel):
        out, lums, g, joins, prev_small = {}, [], 0, [], None
        for key, p, _, _, _, _ in parts:
            cap = cv2.VideoCapture(str(p))
            local = 0
            while True:
                ok, fr = cap.read()
                if not ok:
                    break
                small = cv2.resize(fr, (160, 90), interpolation=cv2.INTER_AREA)
                lums.append(mean_lum(small))
                if local == 0 and prev_small is not None:
                    joins.append((g, float(np.abs(small.astype(np.int16) - prev_small.astype(np.int16)).mean())))
                prev_small = small
                if g in sel:
                    out[g] = (f"{key}:{local}", resize_to_width(fr))
                g += 1
                local += 1
            cap.release()
        return out, lums, g, joins

    idx, k = sample_indices(n_container, fps)
    out, lums, n_dec, joins = decode(set(idx))
    notes = []
    if n_dec != n_container:
        notes.append(f"decoded {n_dec} frames, container says {n_container}; sampling recomputed on decoded count")
        idx, k = sample_indices(n_dec, fps)
        out, lums, n_dec, joins = decode(set(idx))
    if joins:
        notes.append("part joins (global frame, mean abs diff of 160x90 thumbnails vs previous frame): "
                     + ", ".join(f"{g}:{d:.1f}" for g, d in joins))
    frames = [(f"{out[i][0]} (global {i})", out[i][1], None) for i in idx if i in out]
    meta = dict(source=f"zenodo {seq['record']}: " + ", ".join(seq["keys"]), n_frames_total=n_dec, fps_source=round(fps, 3), k=k,
                frame_spacing_s=round(k / fps, 4), native_size=f"{parts[0][4]}x{parts[0][5]}",
                black_frames_share=float(np.mean(np.array(lums) < BLACK_LUM)), black_share_basis="full sequence",
                focal_px_doc=None, focal_doc_source="undetermined: no intrinsics or EXIF in the video container", notes=notes,
                frames=[dict(global_index=i) for i in idx])
    finish(seq, work, results, frames, meta)


# ------------------------------------------------------------------ BLT (remote bag, side ZED, one corridor pass)
def prep_blt(seq, work, results):
    from rosbags.typesys import Stores, get_typestore

    ts = get_typestore(Stores.ROS1_NOETIC)
    bag = RemoteBag(seq["bag"])
    SIDE, INFO, ODO, FIX = ("/side/zed_node/rgb/image_rect_color/compressed", "/side/zed_node/rgb/camera_info", "/odometry/gps", "/gps/fix")
    notes = []
    # 1. trajectory, one odometry message every 3 s
    traj = []
    # clock="header": select the pass on message header stamps. Some bags record odometry seconds
    # after the images that carry the same stamp, so a bag-time window lands late.
    hdr = seq.get("clock") == "header"
    for _, t, msg in bag.sample([ODO], 3.0):
        m = ts.deserialize_ros1(msg, "nav_msgs/msg/Odometry")
        q = m.pose.pose.orientation
        yaw = math.degrees(math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z)))
        th = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
        traj.append((th if hdr else t, m.pose.pose.position.x, m.pose.pose.position.y, m.twist.twist.linear.x, yaw, t))
    if hdr:
        traj.sort()
    # 2. candidate passes: maximal runs with speed > 0.4 m/s and heading within 15 deg of the run's first heading
    segs, cur = [], None
    for i, (t, x, y, v, yaw, *_) in enumerate(traj):
        dyaw = abs((yaw - traj[cur[0]][4] + 180) % 360 - 180) if cur else 0
        if v > 0.4 and dyaw < 15:
            cur = cur or [i, i]
            cur[1] = i
        else:
            if cur:
                segs.append(tuple(cur))
            cur = [i, i] if v > 0.4 else None
    if cur:
        segs.append(tuple(cur))
    cands = []
    for a, b in segs:
        dur = traj[b][0] - traj[a][0]
        if dur < 10:
            continue
        disp = math.hypot(traj[b][1] - traj[a][1], traj[b][2] - traj[a][2])
        vmean = float(np.mean([p[3] for p in traj[a:b + 1]]))
        cands.append(dict(t0=traj[a][0], t1=traj[b][0], dur=dur, disp=disp, ratio=disp / (vmean * dur + 1e-9), vmean=vmean))
    cands.sort(key=lambda c: -c["dur"])
    same = seq.get("same_pass_as")
    if same and hdr:  # keep the row of an earlier cut: only the pass boundaries come from the header clock
        ref = json.loads((REPO / "results/characterisation/per_sequence" / f"{same}.meta.json").read_text())
        h0, h1 = ref["frames"][0]["header_stamp"], ref["frames"][-1]["header_stamp"]
        overlap = lambda c: min(c["t1"], h1) - max(c["t0"], h0)
        cands = sorted((c for c in cands if overlap(c) > 0), key=lambda c: -overlap(c))
        notes.append(f"same corridor pass as {same} (the candidate run overlapping its frames most)")
    side = bag.conn_ids(SIDE)

    def side_frame(t):  # (mean luminance, header stamp - bag time) of the first side frame at or after bag time t
        for cid, tb_, msg in bag.window(t, t + 1.0, block=12):
            if cid in side:
                m = ts.deserialize_ros1(msg, "sensor_msgs/msg/CompressedImage")
                hs = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
                return mean_lum(cv2.imdecode(np.frombuffer(bytes(m.data), np.uint8), cv2.IMREAD_REDUCED_GRAYSCALE_8)), hs - tb_
        return None, None

    c_img = 0.0  # header stamp minus bag time of the side images (constant within a bag)
    if hdr:
        c_img = next(c for c in (side_frame(p[5])[1] for p in traj[len(traj) // 2:]) if c is not None)

    def side_lum(t):  # mean luminance of the first side frame at or after time t (header time when hdr)
        return side_frame(t - c_img)[0]

    chosen = None
    for strict in (True, False):  # first require odometry displacement to agree with speed x time
        for c in cands:
            if strict and not 0.75 <= c["ratio"] <= 1.25:
                continue
            lum = side_lum((c["t0"] + c["t1"]) / 2)
            c["side_lum_mid"] = lum
            if lum is not None and lum >= BLACK_LUM:
                chosen = c
                break
        if chosen:
            if not strict:
                notes.append("no pass had odometry displacement consistent with speed x time (ratio 0.75-1.25); longest moving run used")
            break
    if not chosen:
        raise RuntimeError(f"no corridor pass with a non-black side camera found; candidates: {cands[:6]}")
    # 3. pull the window (+3 s margin), keep every side frame, then trim to speed > 0.4 at full odometry rate
    w0, w1 = max(chosen["t0"] - 3, 0), chosen["t1"] + 3
    lag = 0.0  # how long after the images odometry with the same stamp is recorded (header clock only)
    if hdr:
        lag = max(0.0, max(p[5] - (p[0] - c_img) for p in traj if w0 - 30 <= p[0] <= w1 + 30))
    odo, fix, info, imgs = bag.conn_ids(ODO), bag.conn_ids(FIX), bag.conn_ids(INFO), []
    odos, fixes, K = [], [], None
    for cid, t, msg in bag.window(max(w0 - c_img, 0), w1 - c_img + (lag + 2 if hdr else 0)):
        if cid in side:
            imgs.append((t, msg))
        elif cid in odo:
            m = ts.deserialize_ros1(msg, "nav_msgs/msg/Odometry")
            odos.append((m.header.stamp.sec + m.header.stamp.nanosec * 1e-9 if hdr else t, m.twist.twist.linear.x))
        elif cid in fix:
            m = ts.deserialize_ros1(msg, "sensor_msgs/msg/NavSatFix")
            fixes.append((t, m.header.stamp.sec + m.header.stamp.nanosec * 1e-9, int(m.status.status)))
        elif cid in info and K is None:
            m = ts.deserialize_ros1(msg, "sensor_msgs/msg/CameraInfo")
            K = dict(width=int(m.width), fx=float(m.K[0]), fy=float(m.K[4]), cx=float(m.K[2]), cy=float(m.K[5]))
    ot, ov = np.array([o[0] for o in odos]), np.array([o[1] for o in odos])
    moving = ot[ov > 0.4]
    ta, tb = (float(moving.min()), float(moving.max())) if len(moving) else (w0, w1)
    ta, tb = max(ta, chosen["t0"] - 3), min(tb, chosen["t1"] + 3)
    dec = []
    for t, msg in imgs:
        if hdr or ta <= t <= tb:
            m = ts.deserialize_ros1(msg, "sensor_msgs/msg/CompressedImage")
            hs = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
            if hdr and not ta <= hs <= tb:
                continue
            data = bytes(m.data)
            lum = mean_lum(cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_REDUCED_GRAYSCALE_8))
            dec.append((t, hs, lum, data))
    n_all = len(dec)
    kept = [d for d in dec if d[2] >= BLACK_LUM]
    n_black = n_all - len(kept)
    tc = 1 if hdr else 0
    fps = (len(kept) - 1) / (kept[-1][tc] - kept[0][tc]) if len(kept) > 1 else None
    idx, k = sample_indices(len(kept), fps)
    frames, fmeta = [], []
    ft = np.array([f[0] for f in fixes]) if fixes else np.array([])
    fh = np.array([f[1] for f in fixes]) if fixes else np.array([])
    for i in idx:
        t, hs, lum, data = kept[i]
        im = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        native = im.shape[1]
        frames.append((f"bag_time={t:.3f}s header_stamp={hs:.3f}", resize_to_width(im), None))
        fmeta.append(dict(bag_time_s=round(t, 3), header_stamp=round(hs, 3),
                          fix_gap_bagtime_s=float(np.abs(ft - t).min()) if len(ft) else None,
                          fix_gap_header_s=float(np.abs(fh - hs).min()) if len(fh) else None))
    gb = [f["fix_gap_bagtime_s"] for f in fmeta if f["fix_gap_bagtime_s"] is not None]
    gh = [f["fix_gap_header_s"] for f in fmeta if f["fix_gap_header_s"] is not None]
    anchor = dict(dataset="blt", sequence=seq["id"], anchor_type="RTK-GNSS fix (/gps/fix)", n_fix_in_window=len(fixes),
                  fix_status_counts=dict(Counter(f[2] for f in fixes)),
                  share_gap_lt_0p2_bagtime=float(np.mean(np.array(gb) < 0.2)) if gb else None,
                  share_gap_lt_0p2_header=float(np.mean(np.array(gh) < 0.2)) if gh else None,
                  median_gap_bagtime_s=float(np.median(gb)) if gb else None, median_gap_header_s=float(np.median(gh)) if gh else None)
    write_json(results / "anchors_parts" / f"{seq['id']}.json", anchor)
    scale = frames[0][1].shape[1] / native
    if hdr:
        notes.append(f"corridor pass selected on message header stamps: image bag time {ta - c_img:.1f}-{tb - c_img:.1f} s "
                     f"({tb - ta:.1f} s), the {'run overlapping ' + same + ' most' if same else 'longest run'} "
                     f"with odometry speed > 0.4 m/s and heading change < 15 deg "
                     f"(3 s odometry sampling, then trimmed at full rate); odometry displacement {chosen['disp']:.1f} m, "
                     f"displacement/(speed x time) = {chosen['ratio']:.2f}; odometry is recorded up to {lag:.1f} s after "
                     f"the images with the same stamp, so bag time was not used")
    else:
        notes.append(f"corridor pass: bag time {ta:.1f}-{tb:.1f} s ({tb - ta:.1f} s), chosen as the longest run with odometry speed > 0.4 m/s "
                     f"and heading change < 15 deg (3 s odometry sampling, then trimmed at full rate); "
                     f"odometry displacement {chosen['disp']:.1f} m, displacement/(speed x time) = {chosen['ratio']:.2f}")
    notes.append(f"{n_black} of {n_all} side-camera frames in the pass were black (mean luminance < {BLACK_LUM:g}) and excluded before sampling")
    meta = dict(source=f"LCAS share {seq['bag']} topic {SIDE}", n_frames_total=len(kept), fps_source=round(fps, 3) if fps else None, k=k,
                frame_spacing_s=round(k / fps, 4) if fps else None, native_size=f"{native}x{round(frames[0][1].shape[0] / scale)}",
                black_frames_share=n_black / n_all if n_all else None, black_share_basis="all side frames in the pass, before exclusion",
                focal_px_doc=round(K["fx"] * scale, 2) if K else None,
                focal_doc_source="side camera_info K[0] scaled to the width used" if K else "undetermined: no camera_info in window",
                pass_window_s=[round(ta - c_img, 2), round(tb - c_img, 2)], candidates=cands[:8], bytes_fetched_gb=round(bag.bytes_fetched / 1e9, 2),
                notes=notes, frames=fmeta)
    if hdr:
        meta.update(clock="header", pass_window_header_s=[round(ta, 3), round(tb, 3)], odometry_record_lag_max_s=round(lag, 2))
    finish(seq, work, results, frames, meta)


def prep_blt_front(seq, work, results):
    """Front camera over the SAME corridor window as the committed side pass of `base` (the
    BLT masking ablations run on front passes; the window must not be re-derived)."""
    from rosbags.typesys import Stores, get_typestore

    ts = get_typestore(Stores.ROS1_NOETIC)
    committed = REPO / "results/characterisation"
    side_meta = json.loads((committed / "per_sequence" / f"{seq['base']}.meta.json").read_text())
    ta, tb = side_meta["pass_window_s"]
    hw = side_meta.get("pass_window_header_s")  # side pass cut on header stamps: select front frames the same way
    FRONT = "/front/zed_node/rgb/image_rect_color/compressed"
    INFO = "/front/zed_node/rgb/camera_info"
    bag = RemoteBag(seq["bag"])
    front, info = bag.conn_ids(FRONT), bag.conn_ids(INFO)
    dec, K = [], None
    for cid, t, msg in bag.window(ta - (2 if hw else 0), tb + (2 if hw else 0)):
        if cid in front:
            m = ts.deserialize_ros1(msg, "sensor_msgs/msg/CompressedImage")
            hs = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
            if hw and not hw[0] <= hs <= hw[1]:
                continue
            data = bytes(m.data)
            lum = mean_lum(cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_REDUCED_GRAYSCALE_8))
            dec.append((t, hs, lum, data))
        elif cid in info and K is None:
            m = ts.deserialize_ros1(msg, "sensor_msgs/msg/CameraInfo")
            K = dict(fx=float(m.K[0]))
    n_all = len(dec)
    kept = [d for d in dec if d[2] >= BLACK_LUM]
    n_black = n_all - len(kept)
    if not kept:
        raise RuntimeError(f"front camera black or absent in window {ta}-{tb} of {seq['bag']}")
    fps = (len(kept) - 1) / (kept[-1][0] - kept[0][0]) if len(kept) > 1 else None
    idx, k = sample_indices(len(kept), fps)
    frames, fmeta, native = [], [], None
    for i in idx:
        t, hs, lum, data = kept[i]
        im = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        native = im.shape[1]
        frames.append((f"bag_time={t:.3f}s header_stamp={hs:.3f}", resize_to_width(im), None))
        fmeta.append(dict(bag_time_s=round(t, 3), header_stamp=round(hs, 3)))
    scale = frames[0][1].shape[1] / native
    notes = [f"front camera over the committed side pass window {ta}-{tb} s of {seq['base']}"
             + (f" (selected on header stamps {hw[0]}-{hw[1]})" if hw else ""),
             f"{n_black} of {n_all} front frames in the window were black and excluded before sampling"]
    meta = dict(source=f"LCAS share {seq['bag']} topic {FRONT}", n_frames_total=len(kept),
                fps_source=round(fps, 3) if fps else None, k=k,
                frame_spacing_s=round(k / fps, 4) if fps else None,
                native_size=f"{native}x{round(frames[0][1].shape[0] / scale)}",
                black_frames_share=n_black / n_all if n_all else None,
                black_share_basis="all front frames in the window, before exclusion",
                focal_px_doc=round(K["fx"] * scale, 2) if K else None,
                focal_doc_source="front camera_info K[0] scaled to the width used" if K else
                "undetermined: no camera_info in window",
                pass_window_s=[ta, tb], bytes_fetched_gb=round(bag.bytes_fetched / 1e9, 2),
                notes=notes, frames=fmeta)
    finish(seq, work, results, frames, meta)


# ------------------------------------------------------------------ stills helper
def stills_from_bytes(items, by_model=True):
    """items: iterable of (name, bytes). Loaded WITHOUT applying EXIF rotation. Returns frames + per-image info."""
    frames, info = [], []
    for name, data in items:
        im = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        if im is None:
            info.append(dict(name=name, error="could not decode"))
            continue
        h, w = im.shape[:2]
        tags = exif_tags(data)
        model = (str(tags.get("Image Make", "")).strip() + " " + str(tags.get("Image Model", "")).strip()).strip()
        fpx, fsrc = exif_focal_px(tags, w, h)
        out = resize_to_width(im)
        frames.append((name, out, (f"{out.shape[1]}x{out.shape[0]}", re.sub(r"[^A-Za-z0-9]+", "-", model) if model else None)))
        info.append(dict(name=name, native=f"{w}x{h}", used=f"{out.shape[1]}x{out.shape[0]}", model=model or None,
                         focal_px_doc_used=round(fpx * out.shape[1] / w, 1) if fpx else None, focal_doc_source=fsrc,
                         gps=exif_gps(tags), datetime=str(tags.get("EXIF DateTimeOriginal", "")) or None, lum=mean_lum(out)))
    # share intrinsics per image size, split further by EXIF camera model only when every image names one
    use_model = by_model and all(g[1] for _, _, g in frames)
    frames = [(n, im, f"{g[0]}_{g[1]}" if use_model else g[0]) for n, im, g in frames]
    return frames, info


def stills_meta(source, info, extra_notes=()):
    ok = [i for i in info if "error" not in i]
    docs = [i["focal_px_doc_used"] for i in ok if i["focal_px_doc_used"]]
    per_group = {}
    for i in ok:
        if i["focal_px_doc_used"]:
            per_group.setdefault(i["used"] + ("_" + re.sub(r"[^A-Za-z0-9]+", "-", i["model"]) if i["model"] else ""), []).append(i["focal_px_doc_used"])
    return dict(source=source, n_frames_total=len(info), fps_source=None, k=1, frame_spacing_s=None,
                native_size=dict(Counter(i["native"] for i in ok)), black_frames_share=float(np.mean([i["lum"] < BLACK_LUM for i in ok])),
                black_share_basis="all images", focal_px_doc=round(float(np.median(docs)), 1) if docs else None,
                focal_px_doc_by_group={g: round(float(np.median(v)), 1) for g, v in per_group.items()},
                focal_doc_source=(Counter(i["focal_doc_source"] for i in ok if i["focal_doc_source"]).most_common(1)[0][0] if docs
                                  else "undetermined: no focal information in EXIF"),
                notes=["stills: 'consecutive' means file-name order; EXIF rotation not applied, so sensor dimensions are kept"] + list(extra_notes),
                frames=[{k: v for k, v in i.items() if k != "lum"} for i in info])


def prep_emb_stills(seq, work, results):
    z = zipfile.ZipFile(download(zenodo_url(3361736, "thsant/wgisd-1.0.0.zip"), work / "src/wgisd.zip"))
    names = sorted(n for n in z.namelist() if re.search(r"/data/[^/]+\.jpg$", n))
    frames, info = stills_from_bytes((Path(n).name, z.read(n)) for n in names)
    finish(seq, work, results, frames, stills_meta("zenodo 3361736 wgisd-1.0.0.zip data/*.jpg (the 300 annotated images, 2048 px wide in the deposit)", info))


def prep_gcs(seq, work, results):
    z = zipfile.ZipFile(GCS_ZIP)
    names = sorted(n for n in z.namelist() if "/Dataset5/" in n and "/Segmented/" not in n and n.lower().endswith(".jpg"))
    frames, info = stills_from_bytes((Path(n).name, z.read(n)) for n in names)
    finish(seq, work, results, frames, stills_meta("CSU GrapeDatabase.zip Dataset5, originals only (Segmented/ copies excluded)", info))


def prep_gst(seq, work, results):
    folder = "GrapeSet/Images/Blue/WhiteBackground/Setup1/HighResolution/"
    z = zipfile.ZipFile(download(zenodo_url(14019981, "GrapeSet.zip"), work / "src/GrapeSet.zip"))
    names = sorted(n for n in z.namelist() if n.startswith(folder) and n.lower().endswith((".jpeg", ".jpg")))
    frames, info = stills_from_bytes((Path(n).name, z.read(n)) for n in names)
    finish(seq, work, results, frames, stills_meta(f"zenodo 14019981 GrapeSet.zip {folder}", info,
                                                    ["40 images = 10 three-bunch combinations x 4 side views 90 deg apart"]))


def prep_esc(seq, work, results):
    import py7zr
    import shapefile
    from pyproj import Transformer

    arc = download(zenodo_url(10362568, "PLANTs_SmartPhonePhotos_JPG.7z"), work / "src/photos.7z")
    with py7zr.SevenZipFile(arc) as a:
        a.extractall(work / "src/photos")
    files = sorted(p for p in (work / "src/photos").rglob("*") if p.suffix.lower() in (".jpg", ".jpeg"))
    frames, info = stills_from_bytes((str(p.relative_to(work / "src/photos")), p.read_bytes()) for p in files)
    # anchor: share of photos whose GPS tag lies within 50 m of a trunk
    tz = download(zenodo_url(10362568, "trunk_locations_VineyardB7.zip"), work / "src/trunks.zip")
    zipfile.ZipFile(tz).extractall(work / "src/trunks")
    shp = next((work / "src/trunks").rglob("*.shp"))
    prj = shp.with_suffix(".prj").read_text() if shp.with_suffix(".prj").exists() else None
    pts = np.array([s.points[0] for s in shapefile.Reader(str(shp)).shapes() if s.points])
    to_utm = Transformer.from_crs(prj or "EPSG:4326", "EPSG:32629", always_xy=True)
    tx, ty = to_utm.transform(pts[:, 0], pts[:, 1])
    trunks = np.column_stack([tx, ty])
    g = [(i["name"], i["gps"]) for i in info if i.get("gps")]
    dmin = []
    if g:
        px, py = Transformer.from_crs("EPSG:4326", "EPSG:32629", always_xy=True).transform([x[1][1] for x in g], [x[1][0] for x in g])
        P = np.column_stack([px, py])
        dmin = [float(np.sqrt(((trunks - p) ** 2).sum(1)).min()) for p in P]
    write_json(results / "anchors_parts" / f"{seq['id']}.json",
               dict(dataset="esc", sequence=seq["id"], anchor_type="EXIF GPS per photo + trunk shapefile", n_photos=len(info),
                    n_with_gps=len(g), n_trunks=int(len(trunks)), trunk_crs=(prj or "no .prj; assumed EPSG:4326")[:120],
                    share_within_50m=float(np.mean(np.array(dmin) < 50)) if dmin else None,
                    dist_m_p50=float(np.median(dmin)) if dmin else None, dist_m_p95=float(np.percentile(dmin, 95)) if dmin else None,
                    dist_m_max=float(np.max(dmin)) if dmin else None))
    finish(seq, work, results, frames, stills_meta("zenodo 10362568 PLANTs_SmartPhonePhotos_JPG.7z, all photos", info))


def prep_emb_video(seq, work, results):
    import requests

    tree = requests.get("https://api.github.com/repos/thsant/wgisd/git/trees/HEAD?recursive=1", timeout=60).json()
    paths = sorted(e["path"] for e in tree["tree"] if e["path"].startswith("extra/video_demo_frames/") and e["type"] == "blob")
    idx, k = sample_indices(len(paths), None)
    frames, lums = [], []
    for i in idx:
        p = download(f"https://raw.githubusercontent.com/thsant/wgisd/{tree['sha']}/{paths[i]}", work / "src/frames" / Path(paths[i]).name)
        im = cv2.imread(str(p), cv2.IMREAD_COLOR)
        native = f"{im.shape[1]}x{im.shape[0]}"
        im = resize_to_width(im)
        lums.append(mean_lum(im))
        frames.append((Path(paths[i]).name, im, None))
    meta = dict(source=f"github thsant/wgisd tree {tree['sha'][:7]} extra/video_demo_frames/ (not in the Zenodo deposit)", n_frames_total=len(paths),
                fps_source=None, k=k, frame_spacing_s=None, native_size=native, black_frames_share=float(np.mean(np.array(lums) < BLACK_LUM)),
                black_share_basis="sampled frames", focal_px_doc=None, focal_doc_source="undetermined: frames carry no EXIF",
                notes=["source frame rate unknown (files are every 2nd frame of a demo video), so the 2 fps cap could not be applied; k from the 200-frame rule only",
                       f"sampled frames cover files 0..{idx[-1]} of {len(paths)} in name order"],
                frames=[dict(file=Path(paths[i]).name) for i in idx])
    finish(seq, work, results, frames, meta)


# ------------------------------------------------------------------ Botrytis (green band)
PANELS_DOC = {"0_V1": 65, "30_V1": 45, "45_V1": 40, "0_V2": 20}


def prep_bot(seq, work, results):
    import shapefile
    from pyproj import Transformer
    from scipy.spatial import ConvexHull, Delaunay

    flight = seq["sequence_id"]
    z = zipfile.ZipFile(download(zenodo_url(7383601, seq["key"]), work / "src" / seq["key"]))
    green = sorted(n for n in z.namelist() if re.search(r"IMG_\d+_2\.tif$", n, re.I))
    panel = [n for n in green if "/PANEL/" in n.upper()]  # the deposit keeps reflectance-panel captures in <flight>/PANEL/
    caps = []
    for n in green:
        if n in panel:
            continue
        data = z.read(n)
        tags = exif_tags(data)
        im = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
        caps.append(dict(name=n, gps=exif_gps(tags), dt=str(tags.get("EXIF DateTimeOriginal", "")), sub=str(tags.get("EXIF SubSecTime", "")),
                         p999=float(np.percentile(im, 99.9)), mean=float(im.mean()), shape=im.shape, dtype=str(im.dtype),
                         focal=exif_focal_px(tags, im.shape[1], im.shape[0])))
    caps.sort(key=lambda c: (c["dt"], c["sub"].zfill(6), c["name"]))  # capture-time order
    keep = caps
    alt = np.array([c["gps"][2] if c["gps"] and c["gps"][2] is not None else np.nan for c in caps])
    notes = [f"{len(panel)} reflectance-panel captures excluded (all files under {flight}/PANEL/: "
             f"{Path(panel[0]).stem[:8] if panel else '-'}..{Path(panel[-1]).stem[:8] if panel else '-'}); the data paper gives {PANELS_DOC.get(flight)} for this flight",
             f"{len(caps)} captures under {flight}/SET/ kept, in capture-time order; GPS altitude median {np.nanmedian(alt):.1f} m, "
             f"min {np.nanmin(alt):.1f} m, max {np.nanmax(alt):.1f} m (take-off and landing legs are part of SET and were not removed)"]
    # capture interval -> fps
    from datetime import datetime
    tt = []
    for c in keep:
        try:
            tt.append(datetime.strptime(c["dt"], "%Y:%m:%d %H:%M:%S").timestamp() + (float("0." + c["sub"]) if c["sub"].isdigit() else 0.0))
        except ValueError:
            pass
    dt = float(np.median(np.diff(tt))) if len(tt) > 2 else None
    fps = 1.0 / dt if dt and dt > 0 else None
    idx, k = sample_indices(len(keep), fps)
    raws = [cv2.imdecode(np.frombuffer(z.read(keep[i]["name"]), np.uint8), cv2.IMREAD_UNCHANGED) for i in idx]
    scale = float(np.percentile(np.concatenate([r.ravel()[::16] for r in raws]), 99.9))
    frames = [(keep[i]["name"], np.clip(r.astype(np.float32) / scale * 255.0, 0, 255).astype(np.uint8), None) for i, r in zip(idx, raws)]
    notes.append(f"green band only (IMG_*_2.tif), {raws[0].dtype} scaled to 8 bit by the 99.9th percentile of the sampled frames ({scale:.0f} DN)")
    # inspection sheet of the excluded panel captures
    if panel:
        pick = [panel[i] for i in sorted(set(np.linspace(0, len(panel) - 1, min(10, len(panel))).astype(int)))]
        tiles = []
        for n in pick:
            r = cv2.imdecode(np.frombuffer(z.read(n), np.uint8), cv2.IMREAD_UNCHANGED)
            t = cv2.resize(np.clip(r.astype(np.float32) / max(float(np.percentile(r, 99.9)), 1) * 255, 0, 255).astype(np.uint8), (320, 240))
            cv2.putText(t, Path(n).stem, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, 255, 2)
            tiles.append(t)
        while len(tiles) % 5:
            tiles.append(np.zeros_like(tiles[0]))
        sheet = np.vstack([np.hstack(tiles[i:i + 5]) for i in range(0, len(tiles), 5)])
        (results / "contact_sheets").mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(results / "contact_sheets" / f"{seq['id']}_panel_captures.jpg"), sheet)
    # anchors: EXIF GPS on sampled frames, GCPs inside the convex hull of image positions
    gz = download(zenodo_url(7383601, "GCPs.zip"), work / "src/GCPs.zip")
    zipfile.ZipFile(gz).extractall(work / "src/gcps")
    shp = next((work / "src/gcps").rglob("*.shp"))
    prj = shp.with_suffix(".prj").read_text() if shp.with_suffix(".prj").exists() else None
    gp = np.array([s.points[0] for s in shapefile.Reader(str(shp)).shapes() if s.points])
    glon, glat = Transformer.from_crs(prj or "EPSG:4326", "EPSG:4326", always_xy=True).transform(gp[:, 0], gp[:, 1])
    pos = np.array([[c["gps"][1], c["gps"][0]] for c in keep if c["gps"]])
    inside = Delaunay(pos[ConvexHull(pos).vertices]).find_simplex(np.column_stack([glon, glat])) >= 0 if len(pos) > 3 else np.array([])
    write_json(results / "anchors_parts" / f"{seq['id']}.json",
               dict(dataset="bot", sequence=seq["id"], anchor_type="EXIF GPS per capture + GCP shapefile",
                    sampled_with_gps_share=float(np.mean([keep[i]["gps"] is not None for i in idx])), n_gcp=int(len(gp)),
                    gcp_inside_hull=int(inside.sum()) if len(inside) else None, gcp_crs=(prj or "no .prj; assumed EPSG:4326")[:120],
                    gcp_lonlat=[[round(float(a), 7), round(float(b), 7)] for a, b in zip(glon, glat)]))
    fd = [c["focal"][0] for c in keep if c["focal"][0]]
    meta = dict(source=f"zenodo 7383601 {seq['key']} green band", n_frames_total=len(keep), fps_source=round(fps, 4) if fps else None, k=k,
                frame_spacing_s=round(k * dt, 3) if dt else None, native_size=f"{raws[0].shape[1]}x{raws[0].shape[0]}",
                black_frames_share=float(np.mean([c["mean"] / max(scale, 1) * 255 < BLACK_LUM for c in keep])), black_share_basis="all kept captures",
                focal_px_doc=round(float(np.median(fd)), 1) if fd else None,
                focal_doc_source=keep[0]["focal"][1] if fd else "undetermined: no focal information in EXIF",
                n_captures_in_zip=len(green), n_panel_excluded=len(panel), n_panel_documented=PANELS_DOC.get(flight),
                excluded_capture_ids=[Path(n).stem for n in panel],
                notes=notes, frames=[dict(name=keep[i]["name"]) for i in idx])
    finish(seq, work, results, frames, meta)


def prep_bot_rgb(seq, work, results):
    """Composed-RGB quality row (full method): the same captures as the committed green-band
    frame list of `base`, composed by botrytis_compose mode "full"."""
    import botrytis_compose as BC

    committed = REPO / "results/characterisation"  # base artefacts live in the repo, not the job dir
    base_list = committed / "frame_lists" / f"{seq['base']}.txt"
    stems, labels = [], []
    for ln in base_list.read_text().splitlines():
        if ln.strip():
            lab = ln.split("\t")[1]
            stems.append(lab.rsplit("_", 1)[0])
            labels.append(lab.rsplit("_", 1)[0])
    zf = zipfile.ZipFile(download(zenodo_url(7383601, seq["key"]), work / "src" / seq["key"]))
    align, stretch, gray_world, crop, bands = BC.MODES["full"]
    median_H, used = BC.median_alignment(zf, stems, bands, bands[1])
    members = set(zf.namelist())
    frames, skipped = [], []
    for stem, lab in zip(stems, labels):
        if not {f"{stem}_{b}.tif" for b in set(bands)} <= members:
            skipped.append(stem.split("/")[-1])
            frames.append((lab, None, None))
            continue
        frames.append((lab, BC.compose(zf, stem, "full", median_H)[0], None))
    base_meta = json.loads((committed / "per_sequence" / f"{seq['base']}.meta.json").read_text())
    # keep index-bound names: a skipped capture leaves a gap instead of renaming the rest
    kept = [(f"frame_{i:06d}.png", lab, img) for i, (lab, img, _) in enumerate(frames)]
    img_dir = work / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    for name, _lab, img in kept:
        if img is not None:
            cv2.imwrite(str(img_dir / name), img)
    text = "".join(f"{n}\t{lab}\n" for n, lab, _ in kept)
    fl = results / "frame_lists" / f"{seq['id']}.txt"
    fl.parent.mkdir(parents=True, exist_ok=True)
    if not fl.exists():
        fl.write_text(text)
    meta = dict(source=f"zenodo 7383601 {seq['key']}, composed RGB (full method)",
                n_frames_total=base_meta["n_frames_total"], fps_source=base_meta["fps_source"],
                k=base_meta["k"], frame_spacing_s=base_meta["frame_spacing_s"],
                native_size=f"{frames[0][1].shape[1]}x{frames[0][1].shape[0]}",
                black_frames_share=None, black_share_basis="not recomputed for the composite",
                focal_px_doc=base_meta.get("focal_px_doc"), focal_doc_source=base_meta.get("focal_doc_source"),
                n_images=sum(1 for _, _, img in kept if img is not None),
                skipped_missing_bands=skipped,
                notes=[f"captures = committed {seq['base']} frame list ({len(stems)} captures)",
                       f"{len(skipped)} capture(s) skipped for missing band files: {skipped}" if skipped else
                       "no captures skipped",
                       "composition: median-homography band alignment (Blue/Red -> Green), joint 2/98 "
                       "percentile stretch, gray-world white balance, 30 px border crop",
                       f"alignment captures used {used}; shifts px "
                       + str({b: [round(float(H[0, 2]), 1), round(float(H[1, 2]), 1)] for b, H in median_H.items()})],
                frames=[dict(name=s_) for s_ in stems])
    meta.update(dataset=seq["dataset"], sequence_id=seq["sequence_id"], id=seq["id"],
                population=seq["population"], width_used=kept[0][2].shape[1] if kept[0][2] is not None
                else 1280, widths={}, camera_groups={"None": meta["n_images"]}, camera_mode="SINGLE")
    write_json(work / "meta.json", meta)
    write_json(results / "per_sequence" / f"{seq['id']}.meta.json", meta)
    print(json.dumps({k: v for k, v in meta.items() if k != "frames"}, default=str)[:1200])


PREP = dict(video=prep_video, blt=prep_blt, blt_front=prep_blt_front, bot=prep_bot, bot_rgb=prep_bot_rgb, esc=prep_esc, emb_video=prep_emb_video, emb_stills=prep_emb_stills, gcs=prep_gcs, gst=prep_gst)

if __name__ == "__main__":
    key, work, results = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    seq = next(s for s in SEQUENCES if str(s["index"]) == key or s["id"] == key)
    work.mkdir(parents=True, exist_ok=True)
    PREP[seq["kind"]](seq, work, results)
