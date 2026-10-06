"""Generate per-frame binary masks to exclude lens-flare *ghost* regions from GS
training -- distinct from generate_flare_masks.py, which targets veiling glare
(bright, desaturated washout). A ghost is a secondary internal-lens reflection:
a smooth, low-texture, saturated-colored blob (green in this sequence) that sits
on top of the scene. It fails generate_flare_masks.py's `v > v_thresh & s <
s_thresh` test outright (a ghost is saturated, not washed-out), and plain
excess-green thresholding is too weak a signal on its own in a vineyard scene
where the whole frame is already green -- see docs/flare_fix_research.md.

Signature used here: local excess-green (G - max(R,B)) elevated above its own
large-radius local baseline (adapts to backlit/shadow context instead of a
fixed threshold), AND low small-radius local texture in V (a ghost is a smooth
gradient; real sunlit leaf detail/specular speckle is high-frequency and gets
rejected by this). Brightness floor drops shadowed/dark false positives.

Usage:
    uv run --extra recon python src/reconstruction/generate_ghost_masks.py \
        <images_dir> <masks_dir>
"""

import argparse
from pathlib import Path

import cv2
import numpy as np


def _box(x: np.ndarray, radius: int) -> np.ndarray:
    k = 2 * radius + 1
    return cv2.boxFilter(x, cv2.CV_32F, (k, k))


def _local_std(x: np.ndarray, radius: int) -> np.ndarray:
    mean = _box(x, radius)
    sq_mean = _box(x * x, radius)
    return np.sqrt(np.clip(sq_mean - mean * mean, 0, None))


def ghost_mask(
    image_bgr: np.ndarray,
    baseline_radius: int = 45,
    texture_radius: int = 5,
    elevation_thresh: float = 0.025,
    texture_thresh: float = 0.09,
    v_floor: float = 0.45,
    dilate_px: int = 9,
    min_blob_area_frac: float = 0.002,
) -> np.ndarray:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    v = hsv[..., 2].astype(np.float32) / 255.0
    b, g, r = (image_bgr[..., i].astype(np.float32) for i in range(3))
    excess_green = (g - np.maximum(r, b)) / 255.0
    excess_green = cv2.GaussianBlur(excess_green, (5, 5), 0)

    baseline = _box(excess_green, baseline_radius)
    elevation = excess_green - baseline
    tex_v = _local_std(v, texture_radius)

    candidate = (elevation > elevation_thresh) & (tex_v < texture_thresh) & (v > v_floor)
    candidate_u8 = (candidate * 255).astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    closed = cv2.morphologyEx(candidate_u8, cv2.MORPH_CLOSE, kernel)

    min_area = int(min_blob_area_frac * image_bgr.shape[0] * image_bgr.shape[1])
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    cleaned = np.zeros_like(closed)
    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == label] = 255

    if dilate_px > 0:
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px, dilate_px))
        cleaned = cv2.dilate(cleaned, dilate_kernel)

    # Invert: nerfstudio convention is white = keep, black = excluded.
    keep_mask = 255 - cleaned
    return keep_mask


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("images_dir", type=Path)
    p.add_argument("masks_dir", type=Path)
    p.add_argument("--baseline-radius", type=int, default=45)
    p.add_argument("--texture-radius", type=int, default=5)
    p.add_argument("--elevation-thresh", type=float, default=0.025)
    p.add_argument("--texture-thresh", type=float, default=0.09)
    p.add_argument("--v-floor", type=float, default=0.45)
    p.add_argument("--dilate-px", type=int, default=9)
    p.add_argument("--min-blob-area-frac", type=float, default=0.002)
    args = p.parse_args()

    args.masks_dir.mkdir(parents=True, exist_ok=True)
    image_paths = sorted(args.images_dir.glob("*.jpg")) + sorted(args.images_dir.glob("*.png"))

    total_frac = []
    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            raise RuntimeError(f"Failed to read {img_path}")
        mask = ghost_mask(
            img, args.baseline_radius, args.texture_radius, args.elevation_thresh,
            args.texture_thresh, args.v_floor, args.dilate_px, args.min_blob_area_frac,
        )
        out_path = args.masks_dir / (img_path.stem + ".png")
        cv2.imwrite(str(out_path), mask)
        excluded_frac = 1.0 - (mask > 0).mean()
        total_frac.append(excluded_frac)
        print(f"{img_path.name}: {excluded_frac * 100:.2f}% excluded -> {out_path.name}")

    total_frac = np.array(total_frac)
    print(f"\n{len(image_paths)} masks written to {args.masks_dir}")
    print(f"Excluded fraction: mean {total_frac.mean() * 100:.2f}%, min {total_frac.min() * 100:.2f}%, max {total_frac.max() * 100:.2f}%")


if __name__ == "__main__":
    main()
