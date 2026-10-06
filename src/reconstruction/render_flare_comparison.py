"""Render a before/after qualitative comparison on the flare-affected region
of MOTS NoPathPlanning_1, for a held-out view.

The quantitative eval (eval_nopathplanning_masked.sh,
eval_nopathplanning_region_compare.sh) shows flare-masked training is
statistically indistinguishable from baseline on PSNR/SSIM, even when scoring
only the non-flare pixels. This script renders the actual held-out frame
with the largest flare-excluded area from both the baseline and the
flare-masked checkpoint, crops to the flare region, and saves a side-by-side
panel (ground truth / baseline / masked) so the qualitative difference (or
lack thereof) in the flare region itself can be inspected directly.

Usage:
    uv run --extra recon-ns python src/reconstruction/render_flare_comparison.py \
        --baseline-config <config.yml> --masked-config <config.yml> \
        --masks-dir <ns_data_masked/masks> --output-dir <dir>
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import numpy._core.multiarray
from PIL import Image, ImageDraw

torch.serialization.add_safe_globals([numpy._core.multiarray.scalar])
_orig_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_load(*args, **kwargs)
torch.load = _patched_load

from nerfstudio.utils.eval_utils import eval_setup  # noqa: E402


def find_most_flared_frame(config_path: Path, masks_dir: Path):
    """Iterate the eval set once, tracking the frame with the largest excluded
    (flare) area, and render GT + prediction only for that frame."""
    config, pipeline, _, _ = eval_setup(config_path)
    model = pipeline.model
    eval_dataset = pipeline.datamanager.eval_dataset

    best = None  # (excluded_frac, image_name, gt_np, pred_np, keep_mask)
    for camera, batch in pipeline.datamanager.fixed_indices_eval_dataloader:
        image_idx = int(batch["image_idx"])
        image_name = Path(eval_dataset.image_filenames[image_idx]).name
        mask_path = masks_dir / (Path(image_name).stem + ".png")
        keep_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if keep_mask is None:
            raise RuntimeError(f"Missing flare mask for {image_name} at {mask_path}")
        excluded_frac = 1.0 - (keep_mask > 0).mean()

        if best is None or excluded_frac > best[0]:
            outputs = model.get_outputs_for_camera(camera=camera)
            gt_rgb = model.composite_with_background(model.get_gt_img(batch["image"]), outputs["background"])
            pred_rgb = outputs["rgb"]
            gt_np = gt_rgb.detach().cpu().numpy()
            pred_np = pred_rgb.detach().cpu().numpy()
            if keep_mask.shape != gt_np.shape[:2]:
                keep_mask = cv2.resize(keep_mask, (gt_np.shape[1], gt_np.shape[0]),
                                        interpolation=cv2.INTER_NEAREST)
            best = (excluded_frac, image_name, gt_np, pred_np, keep_mask)

    return best  # excluded_frac, image_name, gt_np, pred_np, keep_mask


def render_for_frame(config_path: Path, image_name: str):
    """Render the prediction for one specific (already-known) held-out frame."""
    config, pipeline, _, _ = eval_setup(config_path)
    model = pipeline.model
    eval_dataset = pipeline.datamanager.eval_dataset

    for camera, batch in pipeline.datamanager.fixed_indices_eval_dataloader:
        image_idx = int(batch["image_idx"])
        name = Path(eval_dataset.image_filenames[image_idx]).name
        if name != image_name:
            continue
        outputs = model.get_outputs_for_camera(camera=camera)
        pred_rgb = outputs["rgb"]
        return pred_rgb.detach().cpu().numpy()
    raise RuntimeError(f"Held-out frame {image_name} not found in this checkpoint's eval set.")


def flare_bbox(keep_mask: np.ndarray, pad_frac: float = 0.15) -> tuple[int, int, int, int]:
    """Bounding box (x0, y0, x1, y1) of the excluded (flare) region, padded."""
    ys, xs = np.where(keep_mask == 0)
    if len(xs) == 0:
        h, w = keep_mask.shape
        return 0, 0, w, h
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    h, w = keep_mask.shape
    pad_x = int((x1 - x0) * pad_frac)
    pad_y = int((y1 - y0) * pad_frac)
    return (max(0, x0 - pad_x), max(0, y0 - pad_y),
            min(w, x1 + pad_x), min(h, y1 + pad_y))


def to_uint8(img: np.ndarray) -> np.ndarray:
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def label(img: Image.Image, text: str) -> Image.Image:
    out = img.copy()
    draw = ImageDraw.Draw(out)
    draw.rectangle([0, 0, out.width, 18], fill=(0, 0, 0))
    draw.text((4, 2), text, fill=(255, 255, 255))
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--baseline-config", type=Path, required=True)
    p.add_argument("--masked-config", type=Path, required=True)
    p.add_argument("--masks-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=== Pass 1: baseline checkpoint, selecting most-flared held-out frame ===")
    excluded_frac, image_name, gt_np, baseline_np, keep_mask = find_most_flared_frame(
        args.baseline_config, args.masks_dir
    )
    print(f"Selected {image_name}: {excluded_frac * 100:.1f}% of frame excluded by flare mask")

    print("=== Pass 2: masked checkpoint, rendering the same held-out frame ===")
    masked_np = render_for_frame(args.masked_config, image_name)

    # Full-frame panel.
    gt_img = Image.fromarray(to_uint8(gt_np))
    baseline_img = Image.fromarray(to_uint8(baseline_np))
    masked_img = Image.fromarray(to_uint8(masked_np))

    full_panel = Image.new("RGB", (gt_img.width * 3 + 20, gt_img.height))
    for i, (name, im) in enumerate([("ground truth", gt_img), ("baseline", baseline_img), ("masked", masked_img)]):
        full_panel.paste(label(im, name), (i * (gt_img.width + 10), 0))
    full_panel_path = args.output_dir / "flare_full_frame_comparison.png"
    full_panel.save(full_panel_path)

    # Cropped panel, zoomed to the flare-affected region.
    x0, y0, x1, y1 = flare_bbox(keep_mask)
    crop_h = y1 - y0
    zoom = max(1, 400 // max(crop_h, 1))
    crops = []
    for name, arr in [("ground truth", gt_np), ("baseline", baseline_np), ("masked", masked_np)]:
        crop = to_uint8(arr[y0:y1, x0:x1])
        im = Image.fromarray(crop).resize((crop.shape[1] * zoom, crop.shape[0] * zoom), Image.LANCZOS)
        crops.append((name, im))

    crop_w, crop_h = crops[0][1].size
    crop_panel = Image.new("RGB", (crop_w * 3 + 20, crop_h))
    for i, (name, im) in enumerate(crops):
        crop_panel.paste(label(im, name), (i * (crop_w + 10), 0))
    crop_panel_path = args.output_dir / "flare_region_comparison.png"
    crop_panel.save(crop_panel_path)

    print(f"Saved full-frame comparison to {full_panel_path}")
    print(f"Saved flare-region crop comparison to {crop_panel_path}")
    print(f"Frame: {image_name}, bbox: ({x0},{y0})-({x1},{y1})")


if __name__ == "__main__":
    main()
