"""Compute PSNR, SSIM, and LPIPS from gt_*.png / pred_*.png pairs in an eval/ directory.

Prints mean ± std for each metric and writes results to metrics_full.json in the same dir.

With --mask-dir the three metrics are computed over the white pixels of each view's mask only
(PSNR from the masked squared error, SSIM and LPIPS as the mean of their per-pixel maps over the
mask), so conditions can be compared on the same pixels or a region scored on its own. Views
whose mask is missing or empty are skipped and listed.

Usage:
    uv run --extra recon python src/reconstruction/compute_metrics.py <eval_dir> [--device cpu|cuda]
        [--mask-dir DIR] [--out metrics_region.json]
"""

import argparse
import json
from pathlib import Path

import lpips
import numpy as np
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def load_image(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def to_tensor(img: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device) * 2 - 1  # [-1, 1] for lpips


def find_mask(mask_dir: Path, view_name: str) -> Path | None:
    """Mask for a view: same name as the image, with or without the eval files' extra .png."""
    stem = view_name
    candidates = [stem]
    while stem.lower().endswith((".png", ".jpg", ".jpeg")):
        stem = stem.rsplit(".", 1)[0]
        candidates += [stem + ".png", stem + ".jpg.png", stem + ".png.png"]
    for name in candidates:
        if (mask_dir / name).exists():
            return mask_dir / name
    return None


def masked_metrics(gt: np.ndarray, pred: np.ndarray, keep: np.ndarray, loss_fn, device) -> dict:
    """PSNR / SSIM / LPIPS over the pixels where `keep` is True."""
    mse = float(((gt - pred) ** 2)[keep].mean())
    psnr = float(10 * np.log10(1.0 / mse)) if mse > 0 else float("inf")
    _, ssim_map = structural_similarity(gt, pred, data_range=1.0, channel_axis=2, full=True)
    # skimage leaves a 3-pixel border out of its mean (7-pixel window); do the same here, so a
    # mask that keeps everything reproduces the unmasked value
    inner = np.zeros_like(keep)
    inner[3:-3, 3:-3] = keep[3:-3, 3:-3]
    ssim = float(ssim_map.mean(axis=2)[inner if inner.any() else keep].mean())
    with torch.no_grad():
        lpips_map = loss_fn(to_tensor(gt, device), to_tensor(pred, device))[0, 0].cpu().numpy()
    return {"psnr": psnr, "ssim": ssim, "lpips": float(lpips_map[keep].mean())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("eval_dir", type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--net", default="alex", choices=["alex", "vgg"], help="LPIPS backbone")
    parser.add_argument(
        "--mask-dir", type=Path, default=None, help="Score only each mask's white pixels"
    )
    parser.add_argument("--out", default="metrics_full.json", help="Output file name in eval_dir")
    args = parser.parse_args()

    device = torch.device(args.device)
    loss_fn = lpips.LPIPS(net=args.net, spatial=args.mask_dir is not None).to(device)

    pairs = sorted(
        (gt, args.eval_dir / f"pred_{gt.name[3:]}")
        for gt in args.eval_dir.glob("gt_*.png")
    )
    pairs = [(gt, pred) for gt, pred in pairs if pred.exists()]
    if not pairs:
        raise FileNotFoundError(f"No gt_*/pred_* pairs found in {args.eval_dir}")

    results = []
    skipped = []
    for gt_path, pred_path in pairs:
        gt = load_image(gt_path)
        pred = load_image(pred_path)

        if args.mask_dir is not None:
            name = gt_path.name[3:]
            mask_path = find_mask(args.mask_dir, name)
            keep = None
            if mask_path is not None:
                mask = Image.open(mask_path).convert("L").resize(gt.shape[1::-1], Image.NEAREST)
                keep = np.asarray(mask) > 127
            if keep is None or not keep.any():
                skipped.append(name)
                continue
            m = masked_metrics(gt, pred, keep, loss_fn, device)
            results.append({"name": name, **m, "mask_share": float(keep.mean())})
            print(
                f"  {name:30s}  PSNR {m['psnr']:6.2f}  SSIM {m['ssim']:.4f}"
                f"  LPIPS {m['lpips']:.4f}  mask {100 * keep.mean():5.1f} %"
            )
            continue

        psnr = float(peak_signal_noise_ratio(gt, pred, data_range=1.0))
        ssim = float(structural_similarity(gt, pred, data_range=1.0, channel_axis=2))

        gt_t = to_tensor(gt, device)
        pred_t = to_tensor(pred, device)
        with torch.no_grad():
            lpips_val = float(loss_fn(gt_t, pred_t).item())

        results.append({"name": gt_path.name[3:], "psnr": psnr, "ssim": ssim, "lpips": lpips_val})
        print(f"  {gt_path.name[3:]:30s}  PSNR {psnr:6.2f}  SSIM {ssim:.4f}  LPIPS {lpips_val:.4f}")

    if not results:
        raise SystemExit(f"No view could be scored: {len(skipped)} masks missing or empty")

    psnrs = [r["psnr"] for r in results]
    ssims = [r["ssim"] for r in results]
    lpipss = [r["lpips"] for r in results]

    summary = {
        "n_views": len(results),
        "lpips_net": args.net,
        "psnr_mean": float(np.mean(psnrs)),
        "psnr_std":  float(np.std(psnrs)),
        "ssim_mean": float(np.mean(ssims)),
        "ssim_std":  float(np.std(ssims)),
        "lpips_mean": float(np.mean(lpipss)),
        "lpips_std":  float(np.std(lpipss)),
        "per_view": results,
    }
    if args.mask_dir is not None:
        summary["mask_dir"] = str(args.mask_dir)
        summary["mask_share_mean"] = float(np.mean([r["mask_share"] for r in results]))
        summary["skipped_views"] = skipped

    out_path = args.eval_dir / args.out
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'─'*55}")
    print(f"  n={summary['n_views']}  net={args.net}")
    print(f"  PSNR  {summary['psnr_mean']:6.2f} ± {summary['psnr_std']:.2f}")
    print(f"  SSIM  {summary['ssim_mean']:.4f} ± {summary['ssim_std']:.4f}")
    print(f"  LPIPS {summary['lpips_mean']:.4f} ± {summary['lpips_std']:.4f}")
    print(f"  Saved → {out_path}")


if __name__ == "__main__":
    main()
