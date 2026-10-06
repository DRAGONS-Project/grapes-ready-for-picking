"""Frame-level quantities for one prepared sequence (no reconstruction).

usage: frame_quality.py <work_dir> <results_dir>

Reads <work_dir>/images and meta.json; writes <results_dir>/per_sequence/<id>.quality.json,
<results_dir>/static_masks/<id>.png (sequences only) and <results_dir>/contact_sheets/<id>.jpg.
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

from common import BLACK_LUM, write_json

BLUR_THRESHOLD = 100.0
STATIC_TOL = 10.0
STATIC_SHARE = 0.9


def main(work: Path, results: Path) -> None:
    meta = json.loads((work / "meta.json").read_text())
    sid = meta["id"]
    labels = dict(l.split("\t", 1) for l in (results / "frame_lists" / f"{sid}.txt").read_text().splitlines())
    files = sorted((work / "images").rglob("*.png"), key=lambda p: str(p.relative_to(work / "images")) if meta["camera_mode"] == "SINGLE" else p.name)
    grays = [cv2.imread(str(f), cv2.IMREAD_GRAYSCALE) for f in files]
    blur = np.array([cv2.Laplacian(g, cv2.CV_64F).var() for g in grays])
    clip = np.array([float(((g >= 250) | (g <= 5)).mean()) for g in grays])
    lum = np.array([float(g.mean()) for g in grays])
    out = dict(id=sid, n_frames=len(files), blur_median=float(np.median(blur)), blur_share_below=float((blur < BLUR_THRESHOLD).mean()),
               blur_p10=float(np.percentile(blur, 10)), blur_p50=float(np.percentile(blur, 50)), blur_p90=float(np.percentile(blur, 90)),
               clip_share_median=float(np.median(clip)), clip_frames_gt5=float((clip > 0.05).mean()), lum_mean_sigma=float(lum.std()),
               lum_mean=float(lum.mean()), black_frames_share_sample=float((lum < BLACK_LUM).mean()),
               black_frames_share=meta.get("black_frames_share"), black_share_basis=meta.get("black_share_basis"))
    # static content: sequences only (stills populations get n/a)
    shapes = {g.shape for g in grays}
    if meta["population"] == "stills":
        out.update(static_share="n/a", static_note="stills population: static content is a property of sequences")
    elif len(shapes) > 1:
        out.update(static_share="undetermined", static_note="frames differ in size")
    else:
        stack = np.stack(grays).astype(np.float32)
        med = np.median(stack, axis=0)
        within = (np.abs(stack - med) < STATIC_TOL).mean(axis=0)
        mask = (within > STATIC_SHARE).astype(np.uint8) * 255
        (results / "static_masks").mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(results / "static_masks" / f"{sid}.png"), mask)
        h = mask.shape[0]
        out.update(static_share=float((mask > 0).mean()),
                   static_share_top_third=float((mask[: h // 3] > 0).mean()), static_share_bottom_third=float((mask[2 * h // 3:] > 0).mean()))
    # contact sheet: 10 evenly spaced frames, 2 x 5, tiles <= 480 px wide, source label printed
    pick = sorted(set(np.linspace(0, len(files) - 1, min(10, len(files))).astype(int).tolist()))
    tiles = []
    for i in pick:
        im = cv2.imread(str(files[i]), cv2.IMREAD_COLOR)
        tw = 480
        th = round(im.shape[0] * tw / im.shape[1])
        t = cv2.resize(im, (tw, th), interpolation=cv2.INTER_AREA)
        key = str(files[i].relative_to(work / "images"))
        text = f"#{i} " + labels.get(key, key)[:44]
        cv2.rectangle(t, (0, 0), (tw, 22), (0, 0, 0), -1)
        cv2.putText(t, text, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(t)
    hmax = max(t.shape[0] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, hmax - t.shape[0], 0, 0, cv2.BORDER_CONSTANT) for t in tiles]
    while len(tiles) % 5:
        tiles.append(np.zeros_like(tiles[0]))
    sheet = np.vstack([np.hstack(tiles[i:i + 5]) for i in range(0, len(tiles), 5)])
    (results / "contact_sheets").mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(results / "contact_sheets" / f"{sid}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
    write_json(results / "per_sequence" / f"{sid}.quality.json", out)
    print(json.dumps(out)[:900])


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
