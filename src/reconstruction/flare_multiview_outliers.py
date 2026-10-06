"""Detect flare/glare-affected frames per COLMAP 3D point via multi-view outlier
statistics, with the outlier threshold estimated from the observed pixel data
rather than a fixed constant.

For each sparse point with a track of >= min_track_len observations, sample a
small patch's color at each observing image, then compute a per-track robust
center (median) and scale (MAD) and express each observation's deviation from
its own track as a modified z-score (Iglewicz & Hoaglin):

    z = (x - median) / (1.4826 * MAD + eps)

Two scoring modes (--score-mode):

  veil (default): only counts "brighter and more desaturated than this point
      usually looks" (max(zV, 0) + max(zS, 0) in HSV), i.e. veiling-glare
      washout. This is the original arm described in docs/flare_fix_research.md.

  color: a direction-agnostic anomaly magnitude in Lab space
      (sqrt(zL^2 + za^2 + zb^2)). Veiling glare pushes points in one
      consistent direction (brighter/desaturated) so `veil` mode catches it
      well, but a lens *ghost* -- a secondary internal-lens reflection that
      tints a region a saturated color rather than washing it out -- doesn't
      fit that one-directional model. `color` mode flags any large deviation
      from a point's own multi-view history, regardless of direction, so it
      also catches ghost-tinted observations that `veil` mode misses.

Pooling flare_score across every observation in every track gives an empirical
distribution with a large bulk near 0 (ordinary view-to-view brightness
variation) and a tail (genuinely affected observations). Otsu's method on that
pooled histogram picks the split point automatically -- the threshold is
estimated from this scene's own pixel data, not hand-tuned.

With --masks-dir, flagged observations are rasterized into per-frame nerfstudio
-convention masks (white = keep, black = excluded), one flagged point becoming
a filled disk of --mask-radius, for direct comparison against a single-image
detector (e.g. generate_ghost_masks.py). Note this is inherently sparse: COLMAP
only triangulates points where SIFT finds texture, and a smooth ghost blob is
exactly the kind of low-texture region SIFT under-detects, so the mask will
tend to cover a ghost's edges/textured interior more densely than its smoothest
center.

Usage:
    uv run --extra recon python src/reconstruction/flare_multiview_outliers.py \
        <sparse_dir> <images_dir> [--min-track-len 3] [--patch 5] \
        [--score-mode veil|color] [--masks-dir <dir>] [--mask-radius 20]
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pycolmap


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("sparse_dir", type=Path)
    p.add_argument("images_dir", type=Path)
    p.add_argument("--min-track-len", type=int, default=3)
    p.add_argument("--patch", type=int, default=5, help="Odd patch size for color sampling.")
    p.add_argument("--score-mode", choices=["veil", "color"], default="veil")
    p.add_argument("--out", type=Path, default=None, help="Optional path to dump per-observation results as JSON.")
    p.add_argument("--masks-dir", type=Path, default=None, help="If set, rasterize flagged points into per-frame nerfstudio-convention masks.")
    p.add_argument("--mask-radius", type=int, default=20, help="Disk radius (px) drawn around each flagged 2D point.")
    p.add_argument("--mask-dilate-px", type=int, default=9)
    args = p.parse_args()

    recon = pycolmap.Reconstruction(str(args.sparse_dir))
    half = args.patch // 2

    # Cache per-image HSV V/S and Lab L/a/b planes (cheap: 60 images at ~1600x900).
    plane_cache: dict[str, dict[str, np.ndarray]] = {}
    shape_cache: dict[str, tuple[int, int]] = {}

    def get_planes(image_name: str) -> dict[str, np.ndarray]:
        if image_name not in plane_cache:
            img = cv2.imread(str(args.images_dir / image_name))
            if img is None:
                raise RuntimeError(f"Failed to read {image_name}")
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            plane_cache[image_name] = {
                "V": hsv[..., 2].astype(np.float32) / 255.0,
                "S": hsv[..., 1].astype(np.float32) / 255.0,
                "L": lab[..., 0].astype(np.float32),
                "a": lab[..., 1].astype(np.float32),
                "b": lab[..., 2].astype(np.float32),
            }
            shape_cache[image_name] = img.shape[:2]
        return plane_cache[image_name]

    # Gather per-observation (point3D_id, image_name, x, y, V, S, L, a, b).
    observations = []
    for point3D_id, point in recon.points3D.items():
        elements = list(point.track.elements)
        if len(elements) < args.min_track_len:
            continue
        obs = []
        for el in elements:
            image = recon.images[el.image_id]
            xy = image.points2D[el.point2D_idx].xy
            x, y = xy[0], xy[1]
            planes = get_planes(image.name)
            h, w = planes["V"].shape
            xi, yi = int(round(x)), int(round(y))
            if xi - half < 0 or yi - half < 0 or xi + half >= w or yi + half >= h:
                continue
            sl = (slice(yi - half, yi + half + 1), slice(xi - half, xi + half + 1))
            obs.append({
                "point3D_id": point3D_id, "image": image.name, "x": x, "y": y,
                "V": float(planes["V"][sl].mean()), "S": float(planes["S"][sl].mean()),
                "L": float(planes["L"][sl].mean()), "a": float(planes["a"][sl].mean()), "b": float(planes["b"][sl].mean()),
            })
        if len(obs) < args.min_track_len:
            continue

        if args.score_mode == "veil":
            # Floors (not a tiny epsilon) on the MAD denominator: half of tracks
            # have only 3 observations, where MAD is often exactly 0 by chance,
            # which would otherwise blow z up to +-1000s and let a handful of
            # such tracks dominate the pooled Otsu threshold. Floor values are
            # the ~5th percentile of MAD observed on longer (>=5-obs) tracks in
            # this sequence, i.e. the low end of *real* multi-view color noise.
            v_vals = np.array([o["V"] for o in obs])
            s_vals = np.array([o["S"] for o in obs])
            med_v, med_s = np.median(v_vals), np.median(s_vals)
            mad_v = np.median(np.abs(v_vals - med_v))
            mad_s = np.median(np.abs(s_vals - med_s))
            for o in obs:
                zv = (o["V"] - med_v) / max(1.4826 * mad_v, 0.02)
                zs = (med_s - o["S"]) / max(1.4826 * mad_s, 0.02)
                o["flare_score"] = max(zv, 0.0) + max(zs, 0.0)
        else:  # color
            floors = {"L": 1.0, "a": 0.3, "b": 0.3}
            meds, mads = {}, {}
            for ch in ("L", "a", "b"):
                vals = np.array([o[ch] for o in obs])
                meds[ch] = np.median(vals)
                mads[ch] = np.median(np.abs(vals - meds[ch]))
            for o in obs:
                z2 = 0.0
                for ch in ("L", "a", "b"):
                    z = (o[ch] - meds[ch]) / max(1.4826 * mads[ch], floors[ch])
                    z2 += z * z
                o["flare_score"] = float(np.sqrt(z2))

        observations.extend(obs)

    scores = np.array([o["flare_score"] for o in observations], dtype=np.float32)
    print(f"{len(observations)} observations from tracks with length >= {args.min_track_len} (score-mode={args.score_mode})")
    print(f"flare_score: mean {scores.mean():.3f}, median {np.median(scores):.3f}, "
          f"p90 {np.percentile(scores, 90):.3f}, p99 {np.percentile(scores, 99):.3f}, max {scores.max():.3f}")

    # Otsu threshold on the pooled distribution (scaled to uint8 range for cv2's
    # implementation, since it only operates on 8-bit histograms).
    scaled = np.clip(scores / scores.max() * 255, 0, 255).astype(np.uint8)
    otsu_thresh_scaled, _ = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    otsu_thresh = otsu_thresh_scaled / 255 * scores.max()
    print(f"Otsu-estimated flare_score threshold: {otsu_thresh:.3f}")

    flagged = scores > otsu_thresh
    print(f"Flagged as multi-view flare outliers: {flagged.sum()} / {len(observations)} "
          f"({flagged.mean() * 100:.1f}%)")

    per_frame_flagged = {}
    for o, is_flagged in zip(observations, flagged):
        if is_flagged:
            per_frame_flagged.setdefault(o["image"], []).append((o["x"], o["y"], o["flare_score"]))
    print(f"Frames with >=1 flagged point: {len(per_frame_flagged)} / {len(recon.images)}")

    if args.out:
        args.out.write_text(json.dumps({
            "otsu_threshold": float(otsu_thresh),
            "observations": [
                {**{k: v for k, v in o.items() if k != "point3D_id"}, "flagged": bool(f)}
                for o, f in zip(observations, flagged)
            ],
        }))
        print(f"Wrote per-observation results to {args.out}")

    if args.masks_dir:
        args.masks_dir.mkdir(parents=True, exist_ok=True)
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (args.mask_dilate_px, args.mask_dilate_px))
        for image_name in sorted(shape_cache):
            h, w = shape_cache[image_name]
            canvas = np.zeros((h, w), dtype=np.uint8)
            for x, y, _ in per_frame_flagged.get(image_name, []):
                cv2.circle(canvas, (int(round(x)), int(round(y))), args.mask_radius, 255, -1)
            if args.mask_dilate_px > 0 and canvas.any():
                canvas = cv2.dilate(canvas, dilate_kernel)
            keep_mask = 255 - canvas
            out_path = args.masks_dir / (Path(image_name).stem + ".png")
            cv2.imwrite(str(out_path), keep_mask)
        print(f"Wrote {len(shape_cache)} rasterized masks to {args.masks_dir}")


if __name__ == "__main__":
    main()
