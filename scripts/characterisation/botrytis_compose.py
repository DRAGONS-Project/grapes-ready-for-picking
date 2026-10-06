"""Compose 3-band images from the Botrytis Micasense capture zips, in every ablation mode.

The RedEdge-MX has five separate lenses (band 1=Blue, 2=Green, 3=Red, 4=NIR, 5=Red Edge), so
the bands need registration before they can be stacked; see
src/reconstruction/extract_botrytis_rgb.py for the investigation. This module exposes each
step of that method as a switch so the ablation conditions are single, named deviations:

    mode            alignment   stretch    gray-world   border crop   bands
    naive           none        per band   no           0             B/G/R
    align_median    median      per band   no           0             B/G/R
    align_perframe  per frame   per band   no           0             B/G/R
    radiometry_only none        joint      yes          0             B/G/R
    perband_wb      median      per band   yes          30            B/G/R
    full            median      joint      yes          30            B/G/R
    full_nocrop     median      joint      yes          0             B/G/R
    falsecolor      median      joint      yes          30            NIR/RedEdge/Red

Per-frame alignment falls back to the identity when RANSAC finds under 80 inliers (the
failure mode the manuscript describes); the failure share goes into the report.

usage: botrytis_compose.py <zip> <out_dir> --frame-list <committed list> --mode full
       [--report report.json]
Writes frame_000000.png ... in the frame list's order, plus the report.
"""

import argparse
import json
import zipfile
from pathlib import Path

import cv2
import numpy as np

BLUE, GREEN, RED, NIR, REDEDGE = 1, 2, 3, 4, 5
MODES = {  # align, stretch, gray_world, crop, (bands composed as the image's B, G, R channels)
    "naive": ("none", "perband", False, 0, (BLUE, GREEN, RED)),
    "align_median": ("median", "perband", False, 0, (BLUE, GREEN, RED)),
    "align_perframe": ("perframe", "perband", False, 0, (BLUE, GREEN, RED)),
    "radiometry_only": ("none", "joint", True, 0, (BLUE, GREEN, RED)),
    "perband_wb": ("median", "perband", True, 30, (BLUE, GREEN, RED)),
    "full": ("median", "joint", True, 30, (BLUE, GREEN, RED)),
    "full_nocrop": ("median", "joint", True, 0, (BLUE, GREEN, RED)),
    # shown as red=NIR, green=Red Edge, blue=Red (classic false-colour order)
    "falsecolor": ("median", "joint", True, 30, (RED, REDEDGE, NIR)),
}
MIN_INLIERS = 80


def _stretch_u8(band: np.ndarray, low: float = 2.0, high: float = 98.0) -> np.ndarray:
    lo, hi = np.percentile(band, [low, high])
    hi = hi if hi > lo else lo + 1
    return (np.clip((band.astype(np.float32) - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)


def _read(zf: zipfile.ZipFile, stem: str, band: int) -> np.ndarray:
    raw = np.frombuffer(zf.read(f"{stem}_{band}.tif"), np.uint8)
    return cv2.imdecode(raw, cv2.IMREAD_ANYDEPTH)


def _homography(zf, stem: str, band: int, ref_band: int, orb, bf):
    """H aligning `band` to `ref_band` for one capture, or None (too few inliers)."""
    ref8 = _stretch_u8(_read(zf, stem, ref_band))
    kp_ref, des_ref = orb.detectAndCompute(ref8, None)
    src8 = _stretch_u8(_read(zf, stem, band))
    kp, des = orb.detectAndCompute(src8, None)
    if des_ref is None or des is None or len(des_ref) < 10 or len(des) < 10:
        return None
    matches = sorted(bf.match(des, des_ref), key=lambda m: m.distance)[:500]
    if len(matches) < 10:
        return None
    src = np.float32([kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst = np.float32([kp_ref[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    if H is None or mask is None or int(mask.sum()) < MIN_INLIERS:
        return None
    return H


def median_alignment(zf, stems, bands, ref_band, calib_n: int = 25):
    """One homography per band: the median over up to calib_n captures (the method's step)."""
    rng = np.random.default_rng(0)
    sample = (stems if len(stems) <= calib_n else
              [stems[i] for i in sorted(rng.choice(len(stems), calib_n, replace=False))])
    orb, bf = cv2.ORB_create(4000), cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    members = set(zf.namelist())
    out, used = {}, {}
    for band in bands:
        if band == ref_band:
            continue
        ok = [s for s in sample
              if {f"{s}_{band}.tif", f"{s}_{ref_band}.tif"} <= members]
        hs = [h for s in ok
              if (h := _homography(zf, s, band, ref_band, orb, bf)) is not None]
        if not hs:
            raise RuntimeError(f"no capture passed the {MIN_INLIERS}-inlier threshold for band {band}")
        out[band] = np.median(np.array(hs), axis=0)
        used[band] = f"{len(hs)}/{len(sample)}"
    return out, used


def compose(zf, stem: str, mode: str, median_H=None, orb=None, bf=None):
    """One capture -> (8-bit 3-channel image, number of per-frame alignment failures)."""
    align, stretch, gray_world, crop, bands = MODES[mode]
    ref_band = bands[1]  # the middle channel is the registration reference
    planes = {b: _read(zf, stem, b).astype(np.float32) for b in set(bands)}
    failures = 0
    for b in bands:
        if b == ref_band:
            continue
        H = None
        if align == "median":
            H = median_H[b]
        elif align == "perframe":
            H = _homography(zf, stem, b, ref_band, orb, bf)
            if H is None:
                failures += 1  # the manuscript's failure mode: fall back to no alignment
        if H is not None:
            h, w = planes[b].shape
            planes[b] = cv2.warpPerspective(planes[b], H, (w, h), flags=cv2.INTER_LINEAR)
    if stretch == "joint":
        lo, hi = np.percentile(np.concatenate([planes[b].ravel() for b in bands]), [2.0, 98.0])
        hi = hi if hi > lo else lo + 1
        chans = [np.clip((planes[b] - lo) / (hi - lo), 0, 1) for b in bands]
    else:
        chans = [_stretch_u8(planes[b]).astype(np.float32) / 255.0 for b in bands]
    if gray_world:
        means = [c.mean() for c in chans]
        target = sum(means) / len(means)
        chans = [np.clip(c * (target / m), 0, 1) if m > 0 else c
                 for c, m in zip(chans, means, strict=True)]
    img = cv2.merge([(c * 255).astype(np.uint8) for c in chans])
    h, w = img.shape[:2]
    return img[crop:h - crop, crop:w - crop], failures


def compose_list(zip_path: Path, out_dir: Path, frame_list: Path, mode: str,
                 report_path: Path | None):
    """Compose every capture named in a committed frame list, keeping its order and names."""
    stems = []
    for ln in frame_list.read_text().splitlines():
        if ln.strip():
            label = ln.split("\t")[1]
            stems.append(label.rsplit("_", 1)[0])  # "45_V1/SET/IMG_0060_2.tif" -> ".../IMG_0060"
    out_dir.mkdir(parents=True, exist_ok=True)
    zf = zipfile.ZipFile(zip_path)
    median_H, used = (None, {})
    if MODES[mode][0] == "median":
        median_H, used = median_alignment(zf, stems, MODES[mode][4], MODES[mode][4][1])
    orb, bf = cv2.ORB_create(4000), cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    members = set(zf.namelist())
    failures, skipped = 0, []
    for i, stem in enumerate(stems):
        needed = {f"{stem}_{b}.tif" for b in set(MODES[mode][4])}
        if not needed <= members:
            # a handful of captures lack some band files in the deposit (e.g. 30_V1 IMG_0204
            # has only bands 1-2); skip them, keeping every other frame's index-bound name
            skipped.append(stem.split("/")[-1])
            continue
        img, f = compose(zf, stem, mode, median_H, orb, bf)
        failures += f
        cv2.imwrite(str(out_dir / f"frame_{i:06d}.png"), img)
    report = dict(
        mode=mode, zip=zip_path.name, n_frames=len(stems) - len(skipped),
        skipped_missing_bands=skipped, frame_list=frame_list.name,
        align=MODES[mode][0], stretch=MODES[mode][1], gray_world=MODES[mode][2],
        crop_px=MODES[mode][3], bands_bgr=list(MODES[mode][4]),
        calib_captures_used=used,
        median_shifts_px={str(b): [round(float(H[0, 2]), 2), round(float(H[1, 2]), 2)]
                          for b, H in (median_H or {}).items()},
        perframe_failures=failures,
        perframe_failure_share=round(
            failures / max((len(stems) - len(skipped)) * (len(set(MODES[mode][4])) - 1), 1), 4)
        if MODES[mode][0] == "perframe" else None,
    )
    if report_path:
        report_path.write_text(json.dumps(report, indent=1))
    print(json.dumps(report))
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("zip_path", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--frame-list", type=Path, required=True)
    ap.add_argument("--mode", choices=sorted(MODES), required=True)
    ap.add_argument("--report", type=Path, default=None)
    a = ap.parse_args()
    compose_list(a.zip_path, a.out_dir, a.frame_list, a.mode, a.report)
