"""Generate per-frame binary masks to exclude lens-flare/glare regions from GS training.

The MOTS NoPathPlanning_1 sequence has a low, backlit sun in frame for most/all
of the sequence, producing streaking lens flare and veiling glare (bright,
desaturated regions) concentrated in one part of each frame. Sunlit foliage
stays saturated (green) even when bright; flare washes pixels toward white
(high value, low saturation), which is what we threshold on.

nerfstudio mask convention (see get_image_mask_tensor_from_path /
splatfacto.py get_loss_dict): mask is loaded as a boolean tensor from a
single-channel image and multiplied elementwise into both gt and rendered
images before the loss. White (nonzero) = keep, black (0) = excluded from
the loss.

Usage:
    uv run --extra recon python src/reconstruction/generate_flare_masks.py \
        <images_dir> <masks_dir> [--v-thresh 0.80] [--s-thresh 0.28]
"""

import argparse
from pathlib import Path

import cv2
import numpy as np


def flare_mask(
    image_bgr: np.ndarray, v_thresh: float, s_thresh: float, dilate_px: int, min_blob_area_frac: float
) -> np.ndarray:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    s = hsv[..., 1].astype(np.float32) / 255.0
    v = hsv[..., 2].astype(np.float32) / 255.0

    flare = (v > v_thresh) & (s < s_thresh)
    flare_u8 = (flare * 255).astype(np.uint8)

    # Close small holes first.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    flare_u8 = cv2.morphologyEx(flare_u8, cv2.MORPH_CLOSE, kernel)

    # Drop small connected components: real flare/glare is large contiguous
    # washed-out regions (sky glow, streaks); backlit-leaf specular
    # highlights are small scattered speckles that pass the HSV threshold
    # but shouldn't be excluded from training.
    min_area = int(min_blob_area_frac * image_bgr.shape[0] * image_bgr.shape[1])
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(flare_u8, connectivity=8)
    cleaned = np.zeros_like(flare_u8)
    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == label] = 255
    flare_u8 = cleaned

    if dilate_px > 0:
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px, dilate_px))
        flare_u8 = cv2.dilate(flare_u8, dilate_kernel)

    # Invert: nerfstudio convention is white = keep, black = excluded.
    keep_mask = 255 - flare_u8
    return keep_mask


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("images_dir", type=Path)
    p.add_argument("masks_dir", type=Path)
    p.add_argument("--v-thresh", type=float, default=0.90, help="Value threshold (0-1) above which a pixel is flare-eligible.")
    p.add_argument("--s-thresh", type=float, default=0.15, help="Saturation threshold (0-1) below which a pixel is flare-eligible.")
    p.add_argument("--dilate-px", type=int, default=9, help="Structuring element size for dilating the flare region.")
    p.add_argument("--min-blob-area-frac", type=float, default=0.0008, help="Minimum connected-component area as a fraction of image area; smaller blobs are dropped (treated as leaf specular highlights, not flare).")
    args = p.parse_args()

    args.masks_dir.mkdir(parents=True, exist_ok=True)
    image_paths = sorted(args.images_dir.glob("*.jpg")) + sorted(args.images_dir.glob("*.png"))

    total_frac = []
    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            raise RuntimeError(f"Failed to read {img_path}")
        mask = flare_mask(img, args.v_thresh, args.s_thresh, args.dilate_px, args.min_blob_area_frac)
        out_path = args.masks_dir / (img_path.stem + ".png")
        cv2.imwrite(str(out_path), mask)
        excluded_frac = 1.0 - (mask > 0).mean()
        total_frac.append(excluded_frac)
        print(f"{img_path.name}: {excluded_frac * 100:.1f}% excluded -> {out_path.name}")

    total_frac = np.array(total_frac)
    print(f"\n{len(image_paths)} masks written to {args.masks_dir}")
    print(f"Excluded fraction: mean {total_frac.mean() * 100:.1f}%, min {total_frac.min() * 100:.1f}%, max {total_frac.max() * 100:.1f}%")


if __name__ == "__main__":
    main()
