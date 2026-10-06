"""Evaluate a trained splatfacto checkpoint restricted to the non-flare pixels.

nerfstudio's own eval (ns-eval / ns_eval_patched.py) computes PSNR/SSIM/LPIPS
over the *entire* held-out frame: SplatfactoModel.get_image_metrics_and_images
never looks at "mask" (only get_loss_dict does, during training). That makes
baseline vs. flare-masked training runs an unfair comparison: the masked model
never received gradient signal on the excluded flare pixels, so it gets
penalized there at eval time for a region it was deliberately told to ignore,
while the baseline model can directly (if spuriously) fit the flare artefact
and look better on a metric that includes those same pixels.

This script re-renders each held-out view and computes PSNR/SSIM twice: once
over the full frame (matches ns-eval, for reference) and once restricted to
the kept (non-flare) pixels of the flare mask -- using the SAME external mask
files for every checkpoint compared, regardless of whether that checkpoint's
own training data had mask_path wired in. This isolates reconstruction
quality on the pixels every run was actually trying to get right.

LPIPS is not included in the masked comparison: it's a whole-image perceptual
metric without a natural per-pixel decomposition compatible with an arbitrary
mask shape.

Usage:
    uv run --extra recon-ns python src/reconstruction/ns_eval_masked_region.py \
        --load-config <config.yml> --masks-dir <dir of frame_XXXXXX.png> \
        --output-path <metrics.json>
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import numpy._core.multiarray
from skimage.metrics import structural_similarity

torch.serialization.add_safe_globals([numpy._core.multiarray.scalar])
_orig_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_load(*args, **kwargs)
torch.load = _patched_load

from nerfstudio.utils.eval_utils import eval_setup  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--load-config", type=Path, required=True)
    p.add_argument("--masks-dir", type=Path, required=True, help="Dir of frame_XXXXXX.png flare masks (white=keep).")
    p.add_argument("--output-path", type=Path, required=True)
    args = p.parse_args()

    config, pipeline, checkpoint_path, _ = eval_setup(args.load_config)
    model = pipeline.model
    eval_dataset = pipeline.datamanager.eval_dataset

    full_psnrs, full_ssims = [], []
    masked_psnrs, masked_ssims = [], []
    kept_fracs = []

    for camera, batch in pipeline.datamanager.fixed_indices_eval_dataloader:
        image_idx = int(batch["image_idx"])
        image_name = Path(eval_dataset.image_filenames[image_idx]).name

        mask_path = args.masks_dir / (Path(image_name).stem + ".png")
        keep_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if keep_mask is None:
            raise RuntimeError(f"Missing flare mask for {image_name} at {mask_path}")
        keep_mask = keep_mask > 0  # True = keep (non-flare)

        outputs = model.get_outputs_for_camera(camera=camera)
        gt_rgb = model.composite_with_background(model.get_gt_img(batch["image"]), outputs["background"])
        pred_rgb = outputs["rgb"]

        gt_np = gt_rgb.detach().cpu().numpy().astype(np.float64)
        pred_np = pred_rgb.detach().cpu().numpy().astype(np.float64)
        if keep_mask.shape != gt_np.shape[:2]:
            # Undistortion can crop a 1px border, shifting (H, W) slightly.
            keep_mask = cv2.resize(
                keep_mask.astype(np.uint8), (gt_np.shape[1], gt_np.shape[0]), interpolation=cv2.INTER_NEAREST
            ) > 0

        # Full-frame metrics (matches ns-eval, kept for reference/sanity-check).
        mse_full = np.mean((gt_np - pred_np) ** 2)
        full_psnrs.append(-10 * np.log10(mse_full))
        full_ssims.append(structural_similarity(gt_np, pred_np, data_range=1.0, channel_axis=2))

        # Masked: restrict PSNR to kept pixels; restrict SSIM by computing the
        # full per-pixel SSIM map and averaging only over kept pixels.
        diff2 = (gt_np - pred_np) ** 2
        mse_masked = diff2[keep_mask].mean()
        masked_psnrs.append(-10 * np.log10(mse_masked))

        _, ssim_map = structural_similarity(gt_np, pred_np, data_range=1.0, channel_axis=2, full=True)
        masked_ssims.append(ssim_map.mean(axis=-1)[keep_mask].mean())

        kept_fracs.append(keep_mask.mean())

    def mean_std(xs):
        arr = np.array(xs)
        return float(arr.mean()), float(arr.std())

    full_psnr_mean, full_psnr_std = mean_std(full_psnrs)
    full_ssim_mean, full_ssim_std = mean_std(full_ssims)
    masked_psnr_mean, masked_psnr_std = mean_std(masked_psnrs)
    masked_ssim_mean, masked_ssim_std = mean_std(masked_ssims)

    results = {
        "experiment_name": config.experiment_name,
        "method_name": config.method_name,
        "checkpoint": str(checkpoint_path),
        "n_eval_images": len(full_psnrs),
        "mean_kept_fraction": float(np.mean(kept_fracs)),
        "results": {
            "psnr_full_frame": full_psnr_mean, "psnr_full_frame_std": full_psnr_std,
            "ssim_full_frame": full_ssim_mean, "ssim_full_frame_std": full_ssim_std,
            "psnr_nonflare_only": masked_psnr_mean, "psnr_nonflare_only_std": masked_psnr_std,
            "ssim_nonflare_only": masked_ssim_mean, "ssim_nonflare_only_std": masked_ssim_std,
        },
    }
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    print(f"Saved results to: {args.output_path}")


if __name__ == "__main__":
    main()
