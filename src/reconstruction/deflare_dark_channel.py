"""Remove veiling-glare haze from MOTS NoPathPlanning_1 frames via a dark-channel-prior
correction, applied per-pixel across the whole frame rather than excluding a hard-thresholded
hotspot from the training loss (see docs/flare_fix_research.md for why the exclusion-mask
attempts failed: multi-view outlier stats show the damage is a low-level haze spread through
the whole frame, not confined to the obvious bright hotspot near the top edge).

Physical model (He, Sun, Tang, "Single Image Haze Removal Using Dark Channel Prior," CVPR 2009):
    I(x) = J(x) t(x) + A (1 - t(x))
where I is the observed (hazy/flared) image, J is the true scene radiance, A is the
atmospheric/veil light color, and t is the transmission map. We read A here as the glare-veil
color (bright, desaturated -- the same signature the HSV mask thresholded on) rather than
atmospheric airlight; the imaging equation is identical for both phenomena. Recovering
J = (I - A) / max(t, t0) + A gives the deflared image.

Transmission is refined with a from-scratch guided filter (He, Sun, Tang, ECCV 2010) since the
project's opencv-python-headless build has no ximgproc/guided-filter module.

Usage:
    uv run --extra recon python src/reconstruction/deflare_dark_channel.py \
        <images_dir> <output_dir> [--masks-dir <hsv_masks_dir>] \
        [--sanity-grid <path.png>] [--sanity-n 6]
"""

import argparse
from pathlib import Path

import cv2
import numpy as np


def dark_channel(img: np.ndarray, patch_radius: int) -> np.ndarray:
    min_channel = img.min(axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * patch_radius + 1, 2 * patch_radius + 1))
    return cv2.erode(min_channel, kernel)


def estimate_veil_light(
    img: np.ndarray, dc: np.ndarray, top_frac: float = 0.001, atmo: str = "mean"
) -> np.ndarray:
    h, w = dc.shape
    n_pixels = max(1, int(h * w * top_frac))
    flat_dc = dc.reshape(-1)
    idx = np.argpartition(flat_dc, -n_pixels)[-n_pixels:]
    flat_img = img.reshape(-1, 3)
    candidates = flat_img[idx]
    if atmo == "brightest":
        # ablation: the naive single-pixel rule (brightest pixel among the dark-channel top
        # candidates), the variant the text says produces green/magenta fringing
        return candidates[np.argmax(candidates.sum(axis=1))]
    # Mean rather than a single brightest pixel: the single-pixel argmax was
    # frequently a sensor-clipped outlier (one or two channels pinned at 255),
    # whose per-channel ratio is not a valid estimate of the veil color and
    # produced green/magenta fringing in the corrected frame wherever t is
    # small -- see docs/flare_fix_research.md.
    return candidates.mean(axis=0)


def guided_filter(guide: np.ndarray, src: np.ndarray, radius: int, eps: float) -> np.ndarray:
    mean_g = cv2.boxFilter(guide, cv2.CV_32F, (radius, radius))
    mean_s = cv2.boxFilter(src, cv2.CV_32F, (radius, radius))
    mean_gs = cv2.boxFilter(guide * src, cv2.CV_32F, (radius, radius))
    cov_gs = mean_gs - mean_g * mean_s
    mean_gg = cv2.boxFilter(guide * guide, cv2.CV_32F, (radius, radius))
    var_g = mean_gg - mean_g * mean_g

    a = cov_gs / (var_g + eps)
    b = mean_s - a * mean_g

    mean_a = cv2.boxFilter(a, cv2.CV_32F, (radius, radius))
    mean_b = cv2.boxFilter(b, cv2.CV_32F, (radius, radius))
    return mean_a * guide + mean_b


def deflare(
    img_bgr_u8: np.ndarray,
    patch_radius: int = 7,
    omega: float = 0.90,
    t0: float = 0.15,
    guided_radius: int = 40,
    guided_eps: float = 1e-3,
    atmo: str = "mean",
    guided: bool = True,
) -> np.ndarray:
    img = img_bgr_u8.astype(np.float32) / 255.0

    dc = dark_channel(img, patch_radius)
    veil = estimate_veil_light(img, dc, atmo=atmo)
    veil = np.clip(veil, 0.5, 1.0)  # veil light should be bright by construction; guard degenerate scenes

    normalized = img / veil[None, None, :]
    dc_normalized = dark_channel(normalized, patch_radius)
    t_raw = 1.0 - omega * dc_normalized

    if guided:
        guide_gray = cv2.cvtColor(img_bgr_u8, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        t_refined = guided_filter(guide_gray, t_raw, guided_radius, guided_eps)
    else:
        t_refined = t_raw  # ablation: raw block-wise transmission, no edge-aware refinement
    t_refined = np.clip(t_refined, t0, 1.0)

    recovered = (img - veil[None, None, :]) / t_refined[..., None] + veil[None, None, :]
    recovered = np.clip(recovered, 0.0, 1.0)
    return (recovered * 255).astype(np.uint8)


def make_sanity_grid(
    images_dir: Path, output_paths: list[tuple[str, np.ndarray, np.ndarray]], grid_path: Path
) -> None:
    """output_paths: list of (name, original_bgr, deflared_bgr), most-affected first."""
    rows = []
    for name, orig, defl in output_paths:
        h, w = orig.shape[:2]
        pair = np.hstack([orig, defl])
        cv2.putText(pair, f"{name}: original | deflared", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        cv2.putText(pair, f"{name}: original | deflared", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        rows.append(pair)
    grid = np.vstack(rows)
    grid_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(grid_path), grid)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("images_dir", type=Path)
    p.add_argument("output_dir", type=Path)
    p.add_argument("--patch-radius", type=int, default=7)
    p.add_argument("--omega", type=float, default=0.90)
    p.add_argument("--t0", type=float, default=0.15)
    p.add_argument("--guided-radius", type=int, default=40)
    p.add_argument("--guided-eps", type=float, default=1e-3)
    p.add_argument("--atmo", choices=["mean", "brightest"], default="mean",
                   help="Veil colour: mean of the top dark-channel candidates, or the single "
                        "brightest pixel (the ablation variant)")
    p.add_argument("--no-guided", action="store_true",
                   help="Skip the guided-filter refinement of the transmission map (ablation)")
    p.add_argument("--masks-dir", type=Path, default=None, help="Existing HSV flare masks, used only to rank frames for the sanity grid (most-excluded first).")
    p.add_argument("--sanity-grid", type=Path, default=None, help="If set, also write a before/after comparison grid for the most-affected frames.")
    p.add_argument("--sanity-n", type=int, default=6)
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    image_paths = sorted(args.images_dir.glob("*.jpg")) + sorted(args.images_dir.glob("*.png"))

    severities = []
    for img_path in image_paths:
        if args.masks_dir is not None:
            mask_path = args.masks_dir / (img_path.stem + ".png")
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            severity = 1.0 - (mask > 0).mean() if mask is not None else 0.0
        else:
            severity = None
        severities.append(severity)

    sanity_candidates = []
    for img_path, severity in zip(image_paths, severities):
        img = cv2.imread(str(img_path))
        if img is None:
            raise RuntimeError(f"Failed to read {img_path}")
        defl = deflare(img, args.patch_radius, args.omega, args.t0, args.guided_radius,
                       args.guided_eps, atmo=args.atmo, guided=not args.no_guided)
        out_path = args.output_dir / img_path.name
        cv2.imwrite(str(out_path), defl)
        mean_abs_delta = np.abs(img.astype(np.float32) - defl.astype(np.float32)).mean()
        print(f"{img_path.name}: mean|delta|={mean_abs_delta:.2f} -> {out_path.name}")
        rank_key = severity if severity is not None else mean_abs_delta
        sanity_candidates.append((rank_key, img_path.stem, img, defl))

    print(f"\n{len(image_paths)} deflared frames written to {args.output_dir}")

    if args.sanity_grid is not None:
        sanity_candidates.sort(key=lambda t: t[0], reverse=True)
        top = sanity_candidates[: args.sanity_n]
        make_sanity_grid(args.images_dir, [(name, orig, defl) for _, name, orig, defl in top], args.sanity_grid)
        print(f"Sanity grid ({len(top)} most-affected frames) written to {args.sanity_grid}")


if __name__ == "__main__":
    main()
