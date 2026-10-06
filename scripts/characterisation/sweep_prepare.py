"""Frames for one requirement-sweep run: one property varied, the rest at the standard protocol.

Sweeps (the shared level 2 fps / 1600 px / whole span is the sequence's quality row, not rerun):
    fps:<target>     frame spacing - k = round(fps_source / target), whole span, width 1600
    width:<px>       image width   - 2 fps sampling, saved at <px> (never upscaled); SIFT stays
                     capped at 1600 by the fixed COLMAP configuration, so levels above 1600
                     change training and scoring only
    count:<n>        frame count   - 2 fps sampling, the first <n> frames (same start, shorter span)

usage: sweep_prepare.py <sequence index> <kind:value> <work_dir>
Writes <work>/images, <work>/meta.json, <work>/frame_list.txt; prints the meta.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from common import TARGET_FPS, TARGET_WIDTH, download, resize_to_width, zenodo_url  # noqa: E402
from prepare import LOCAL_VIDEOS  # noqa: E402
from sequences import SEQUENCES  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("index", type=int)
    ap.add_argument("spec")
    ap.add_argument("work", type=Path)
    a = ap.parse_args()
    kind, value = a.spec.split(":")
    seq = SEQUENCES[a.index]
    assert seq["kind"] == "video", "sweeps run on video sequences"

    paths = [LOCAL_VIDEOS.get(k) or download(zenodo_url(seq["record"], k), a.work / "src" / k)
             for k in seq["keys"]]
    caps = [cv2.VideoCapture(str(p)) for p in paths]
    fps = caps[0].get(cv2.CAP_PROP_FPS)
    for c in caps:
        c.release()

    width = TARGET_WIDTH
    target_fps = TARGET_FPS
    n_cap = None
    if kind == "fps":
        target_fps = float(value)
    elif kind == "width":
        width = int(value)
    elif kind == "count":
        n_cap = int(value)
    else:
        raise SystemExit(f"unknown sweep kind {kind}")
    k = max(1, round(fps / target_fps))

    img_dir = a.work / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    rows, g, written = [], 0, 0
    for p in paths:
        cap = cv2.VideoCapture(str(p))
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            if g % k == 0 and (n_cap is None or written < n_cap):
                name = f"frame_{written:06d}.png"
                cv2.imwrite(str(img_dir / name), resize_to_width(fr, width))
                rows.append((name, f"{p.name}:{g} (global {g})", g))
                written += 1
            g += 1
        cap.release()

    (a.work / "frame_list.txt").write_text("".join(f"{n}\t{lab}\n" for n, lab, _ in rows))
    first = cv2.imread(str(img_dir / rows[0][0]))
    span_s = (rows[-1][2] - rows[0][2]) / fps
    meta = dict(id=f"{seq['id']}_sw_{kind}{value}", dataset=seq["dataset"],
                sequence_id=f"{seq['sequence_id']}_sw_{kind}{value}", population="sweep",
                n_images=written, camera_mode="SINGLE", sweep_kind=kind, sweep_value=value,
                k=k, fps_source=round(fps, 3), fps_actual=round(fps / k, 3),
                train_width=first.shape[1], sfm_width=min(first.shape[1], 1600),
                span_s=round(span_s, 2), span_frames_source=rows[-1][2] - rows[0][2],
                notes=[f"sweep {kind}={value}; other properties at the standard protocol",
                       "span in metres is not determinable: the sequence carries no metric anchor"])
    (a.work / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
