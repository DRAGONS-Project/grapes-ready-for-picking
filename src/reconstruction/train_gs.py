"""Train a 3D Gaussian Splatting model from a COLMAP sparse reconstruction using gsplat.

Uses gsplat's DefaultStrategy for adaptive density control (clone/split/prune/opacity
reset), per-parameter learning rates following standard 3DGS conventions, and an
L1 + D-SSIM photometric loss.
"""

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import pycolmap
import torch
from gsplat import rasterization
from gsplat.strategy import DefaultStrategy
from PIL import Image
from pytorch_msssim import ssim
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from config import load_config


PINHOLE_MODELS = ("SIMPLE_PINHOLE", "PINHOLE")


def load_mask(mask_dir: Path, image_name: str, size: tuple[int, int]) -> np.ndarray:
    """Loss mask of one image (white = pixels used). A missing file means the whole image is used."""
    for name in (image_name, image_name + ".png"):
        if (mask_dir / name).exists():
            mask = Image.open(mask_dir / name).convert("L").resize(size, Image.NEAREST)
            return np.asarray(mask) > 127
    return np.ones(size[::-1], dtype=bool)


def load_colmap_scene(
    sparse_dir: Path,
    image_dir: Path,
    device: torch.device,
    allow_distorted: bool = False,
    mask_dir: Path | None = None,
) -> dict:
    """Load camera intrinsics/extrinsics, images, and the sparse point cloud.

    Images are returned sorted by filename for a deterministic train/test split.

    The rasterizer is a pinhole camera: only the focal length and principal point are used.
    A COLMAP model with distortion parameters (OPENCV, SIMPLE_RADIAL, ...) has to be
    undistorted first (pycolmap.undistort_images), otherwise the model is fitted to images
    its camera cannot reproduce. `allow_distorted` skips the check to reproduce earlier runs.
    """
    recon = pycolmap.Reconstruction(str(sparse_dir))
    distorted = sorted({cam.model.name for cam in recon.cameras.values()} - set(PINHOLE_MODELS))
    if distorted and not allow_distorted:
        raise SystemExit(
            f"Camera model(s) {distorted} carry lens distortion, which this trainer ignores. "
            "Undistort the model and images first, or pass --allow-distorted."
        )

    viewmats, Ks, images, names, masks = [], [], [], [], []
    width = height = None

    for image in sorted(recon.images.values(), key=lambda im: im.name):
        cam = recon.cameras[image.camera_id]
        width, height = cam.width, cam.height

        K = np.eye(3, dtype=np.float32)
        params = cam.params
        if cam.model.name in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL"):
            f, cx, cy = params[0], params[1], params[2]
            K[0, 0] = K[1, 1] = f
        else:  # PINHOLE, OPENCV, ...
            fx, fy, cx, cy = params[0], params[1], params[2], params[3]
            K[0, 0], K[1, 1] = fx, fy
        K[0, 2], K[1, 2] = cx, cy
        Ks.append(K)

        cam_from_world = image.cam_from_world()
        world_to_cam = np.eye(4, dtype=np.float32)
        world_to_cam[:3, :3] = cam_from_world.rotation.matrix()
        world_to_cam[:3, 3] = cam_from_world.translation
        viewmats.append(world_to_cam)

        img = Image.open(image_dir / image.name).convert("RGB").resize((width, height))
        images.append(np.asarray(img, dtype=np.float32) / 255.0)
        names.append(image.name)
        if mask_dir is not None:
            masks.append(load_mask(mask_dir, image.name, (width, height)))

    xyz = np.stack([p.xyz for p in recon.points3D.values()]).astype(np.float32)
    rgb = np.stack([p.color for p in recon.points3D.values()]).astype(np.float32) / 255.0

    return {
        "viewmats": torch.tensor(np.stack(viewmats), device=device),
        "Ks": torch.tensor(np.stack(Ks), device=device),
        "images": torch.tensor(np.stack(images), device=device),
        "names": names,
        "masks": torch.tensor(np.stack(masks), device=device) if masks else None,
        "width": width,
        "height": height,
        "points_xyz": torch.tensor(xyz, device=device),
        "points_rgb": torch.tensor(rgb, device=device),
    }


def filter_by_min_camera_dist(scene: dict, min_dist: float) -> list[int]:
    """Return indices of frames whose camera centres are at least min_dist apart.

    Greedy sequential pass: keep a frame only if it is farther than min_dist
    from every previously kept frame.  Preserves sorted order, so the
    train/test holdout applied afterwards remains deterministic.
    """
    if min_dist <= 0:
        return list(range(scene["viewmats"].shape[0]))

    viewmats = scene["viewmats"].cpu().numpy()  # (N, 4, 4)
    # Camera centre in world coords: -R^T t
    centers = np.einsum("nij,nj->ni", -viewmats[:, :3, :3].transpose(0, 2, 1), viewmats[:, :3, 3])

    kept = []
    for i, c in enumerate(centers):
        if not kept or np.min(np.linalg.norm(centers[kept] - c, axis=1)) >= min_dist:
            kept.append(i)
    return kept


def split_train_test(n_views: int, holdout_every: int) -> tuple[list[int], list[int]]:
    """Hold out every Nth view (by sorted order) for evaluation."""
    if holdout_every <= 0:
        return list(range(n_views)), []
    test = list(range(0, n_views, holdout_every))
    train = [i for i in range(n_views) if i not in test]
    return train, test


def split_by_names(names: list[str], holdout_names: set[str]) -> tuple[list[int], list[int]]:
    """Hold out the views whose image name is listed, so every run on a sequence is scored on
    the same frames whatever it registered. Listed names that did not register are skipped."""
    test = [i for i, name in enumerate(names) if name in holdout_names]
    train = [i for i, name in enumerate(names) if name not in holdout_names]
    return train, test


def view_overlap(scene: dict, train_idx: list[int], test_idx: list[int]) -> list[dict]:
    """How close each held-out view is to the training set: distance from its camera centre to
    the nearest training camera, as a fraction of the camera path length (views in name order),
    and the angle between the two optical axes."""
    viewmats = scene["viewmats"].cpu().numpy().astype(np.float64)
    rot = viewmats[:, :3, :3]
    centers = np.einsum("nij,nj->ni", -rot.transpose(0, 2, 1), viewmats[:, :3, 3])
    axes = rot[:, 2, :]
    path_len = float(np.linalg.norm(np.diff(centers, axis=0), axis=1).sum())
    overlap = []
    for i in test_idx:
        dists = np.linalg.norm(centers[train_idx] - centers[i], axis=1)
        j = train_idx[int(np.argmin(dists))]
        cos = float(np.clip(axes[i] @ axes[j], -1.0, 1.0))
        overlap.append({
            "name": scene["names"][i],
            "nn_name": scene["names"][j],
            "nn_dist_frac": float(dists.min() / path_len) if path_len > 0 else float("nan"),
            "nn_angle_deg": float(np.degrees(np.arccos(cos))),
        })
    return overlap


def compute_scene_scale(points_xyz: torch.Tensor) -> float:
    """Robust scene extent (95th percentile distance from centroid), used to scale
    the means' learning rate to the scene's physical size."""
    centroid = points_xyz.mean(dim=0)
    dists = torch.linalg.norm(points_xyz - centroid, dim=-1)
    return float(torch.quantile(dists, 0.95).clamp(min=1e-6))


def knn_scales(points_xyz: torch.Tensor, k: int = 3) -> torch.Tensor:
    """Per-point initial size: RMS distance to the k nearest neighbours, as in the reference
    3DGS implementation, so the initialization follows the (arbitrary) scale of the SfM model."""
    from scipy.spatial import cKDTree

    pts = points_xyz.detach().cpu().numpy()
    dists, _ = cKDTree(pts).query(pts, k=k + 1)
    rms = np.sqrt((dists[:, 1:] ** 2).mean(axis=1))
    return torch.tensor(rms, dtype=torch.float32, device=points_xyz.device).clamp(min=1e-7)


def create_splats(
    points_xyz: torch.Tensor,
    points_rgb: torch.Tensor,
    init_scale: str = "knn",
    fixed_scale: float = 0.01,
) -> torch.nn.ParameterDict:
    """Create the optimizable Gaussian primitives, initialized from a sparse point cloud.

    init_scale="knn" sizes each Gaussian from its neighbours; "fixed" gives every Gaussian
    `fixed_scale` in SfM units (the behaviour of earlier runs).
    """
    n = points_xyz.shape[0]
    device = points_xyz.device
    if init_scale == "knn":
        scales = knn_scales(points_xyz)[:, None].repeat(1, 3)
    else:
        scales = torch.full((n, 3), fixed_scale, device=device)

    quats = torch.zeros((n, 4), device=device)
    quats[:, 0] = 1.0

    return torch.nn.ParameterDict({
        "means": torch.nn.Parameter(points_xyz.clone()),
        "scales": torch.nn.Parameter(scales.log()),
        "quats": torch.nn.Parameter(quats),
        "opacities": torch.nn.Parameter(torch.logit(torch.full((n,), 0.5, device=device))),
        "colors": torch.nn.Parameter(points_rgb.clone()),
    })


def create_optimizers(params: torch.nn.ParameterDict, scene_scale: float) -> dict[str, torch.optim.Optimizer]:
    """One Adam optimizer per parameter group, with standard 3DGS learning rates."""
    lrs = {
        "means": 1.6e-4 * scene_scale,
        "scales": 5e-3,
        "quats": 1e-3,
        "opacities": 5e-2,
        "colors": 2.5e-3,
    }
    return {
        name: torch.optim.Adam([{"params": params[name], "lr": lr, "name": name}], eps=1e-15)
        for name, lr in lrs.items()
    }


def render(params: torch.nn.ParameterDict, viewmat: torch.Tensor, K: torch.Tensor, width: int, height: int, **kwargs):
    renders, alphas, info = rasterization(
        means=params["means"],
        quats=params["quats"],
        scales=params["scales"].exp(),
        opacities=torch.sigmoid(params["opacities"]),
        colors=params["colors"],
        viewmats=viewmat[None],
        Ks=K[None],
        width=width,
        height=height,
        **kwargs,
    )
    return renders[0], alphas[0], info


def train(
    scene: dict,
    train_idx: list[int],
    output_dir: Path,
    num_iters: int,
    save_every: int,
    init_scale: str = "knn",
) -> torch.nn.ParameterDict:
    device = scene["points_xyz"].device
    params = create_splats(scene["points_xyz"], scene["points_rgb"], init_scale=init_scale)
    params = params.to(device)
    init = params["scales"].detach().exp()[:, 0]
    print(f"Initial Gaussian size ({init_scale}): median {init.median().item():.4g}, "
          f"range {init.min().item():.3g} to {init.max().item():.3g} (SfM units)")

    scene_scale = compute_scene_scale(scene["points_xyz"])
    optimizers = create_optimizers(params, scene_scale)
    means_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizers["means"], gamma=0.01 ** (1.0 / num_iters)
    )

    strategy = DefaultStrategy(refine_stop_iter=min(num_iters // 2, 15_000), verbose=True)
    strategy.check_sanity(params, optimizers)
    strategy_state = strategy.initialize_state(scene_scale=scene_scale)

    renders_dir = output_dir / "renders"
    renders_dir.mkdir(parents=True, exist_ok=True)

    order = train_idx.copy()
    random.shuffle(order)

    for it in range(num_iters):
        if not order:
            order = train_idx.copy()
            random.shuffle(order)
        view_idx = order.pop()

        rendered, _, info = render(
            params, scene["viewmats"][view_idx], scene["Ks"][view_idx], scene["width"], scene["height"], packed=True
        )
        strategy.step_pre_backward(params, optimizers, strategy_state, it, info)

        target = scene["images"][view_idx]
        if scene.get("masks") is not None:
            rendered = torch.where(scene["masks"][view_idx][..., None], rendered, target)
        l1_loss = torch.nn.functional.l1_loss(rendered, target)
        ssim_loss = 1.0 - ssim(
            rendered.permute(2, 0, 1)[None], target.permute(2, 0, 1)[None], data_range=1.0, size_average=True
        )
        loss = 0.8 * l1_loss + 0.2 * ssim_loss

        for optimizer in optimizers.values():
            optimizer.zero_grad(set_to_none=True)
        loss.backward()

        for optimizer in optimizers.values():
            optimizer.step()
        means_lr_scheduler.step()

        strategy.step_post_backward(params, optimizers, strategy_state, it, info, packed=True)

        if (it + 1) % save_every == 0 or it == num_iters - 1:
            print(
                f"iter {it + 1}/{num_iters}  loss={loss.item():.4f}  "
                f"n_gaussians={params['means'].shape[0]}"
            )
            img = (rendered.clamp(0, 1).detach().cpu().numpy() * 255).astype(np.uint8)
            Image.fromarray(img).save(renders_dir / f"iter_{it + 1:06d}_view{view_idx:03d}.png")

    return params


@torch.no_grad()
def evaluate(params: torch.nn.ParameterDict, scene: dict, test_idx: list[int], output_dir: Path) -> dict:
    """Render held-out views and compute PSNR/SSIM against ground truth."""
    eval_dir = output_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    per_view = []
    for view_idx in test_idx:
        rendered, _, _ = render(
            params, scene["viewmats"][view_idx], scene["Ks"][view_idx], scene["width"], scene["height"]
        )
        pred = rendered.clamp(0, 1).cpu().numpy()
        target = scene["images"][view_idx].cpu().numpy()

        psnr = float(peak_signal_noise_ratio(target, pred, data_range=1.0))
        ssim_val = float(structural_similarity(target, pred, data_range=1.0, channel_axis=2))
        per_view.append({"name": scene["names"][view_idx], "psnr": psnr, "ssim": ssim_val})

        Image.fromarray((pred * 255).astype(np.uint8)).save(
            eval_dir / f"pred_{scene['names'][view_idx]}.png"
        )
        Image.fromarray((target * 255).astype(np.uint8)).save(
            eval_dir / f"gt_{scene['names'][view_idx]}.png"
        )

    metrics = {
        "n_test_views": len(test_idx),
        "psnr_mean": float(np.mean([v["psnr"] for v in per_view])) if per_view else None,
        "ssim_mean": float(np.mean([v["ssim"] for v in per_view])) if per_view else None,
        "per_view": per_view,
    }
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics


def save_ply(params: torch.nn.ParameterDict, output_path: Path) -> None:
    from plyfile import PlyData, PlyElement

    means = params["means"].detach().cpu().numpy()
    colors = (params["colors"].detach().cpu().numpy().clip(0, 1) * 255).astype(np.uint8)

    vertex = np.empty(
        means.shape[0],
        dtype=[
            ("x", "f4"), ("y", "f4"), ("z", "f4"),
            ("red", "u1"), ("green", "u1"), ("blue", "u1"),
        ],
    )
    vertex["x"], vertex["y"], vertex["z"] = means[:, 0], means[:, 1], means[:, 2]
    vertex["red"], vertex["green"], vertex["blue"] = colors[:, 0], colors[:, 1], colors[:, 2]

    PlyData([PlyElement.describe(vertex, "vertex")]).write(str(output_path))


def main():
    config = load_config()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sparse_dir", type=Path, help="COLMAP sparse reconstruction directory")
    parser.add_argument("image_dir", type=Path, help="Directory of input images")
    parser.add_argument(
        "output_dir", type=Path, help="Directory to write renders, checkpoint, and point cloud"
    )
    parser.add_argument("--iters", type=int, default=config.train.iters, help="Number of training iterations")
    parser.add_argument(
        "--save-every", type=int, default=config.train.save_every, help="Save a render every N iterations"
    )
    parser.add_argument(
        "--holdout-every",
        type=int,
        default=config.train.holdout_every,
        help="Hold out every Nth view (sorted order) for evaluation; 0 disables holdout",
    )
    parser.add_argument(
        "--min-camera-dist",
        type=float,
        default=0.0,
        help="Drop frames whose camera centre is closer than this (metres) to any already-kept frame. "
             "Use to remove near-duplicate views from stationary robot segments.",
    )
    parser.add_argument(
        "--holdout-names",
        type=Path,
        default=None,
        help="Text file of image names to hold out (first column); replaces --holdout-every",
    )
    parser.add_argument(
        "--mask-dir",
        type=Path,
        default=None,
        help="Loss masks named like the images (white = used); black pixels are left out of the loss",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed for view order and densification")
    parser.add_argument(
        "--init-scale",
        choices=["knn", "fixed"],
        default="knn",
        help="Initial Gaussian size: from nearest-neighbour distances, or fixed 0.01 (earlier runs)",
    )
    parser.add_argument(
        "--allow-distorted",
        action="store_true",
        help="Train on a COLMAP model with lens distortion as it is (earlier runs)",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scene = load_colmap_scene(
        args.sparse_dir, args.image_dir, device, args.allow_distorted, args.mask_dir
    )
    n_total = scene["viewmats"].shape[0]

    kept_idx = filter_by_min_camera_dist(scene, args.min_camera_dist)
    if len(kept_idx) < n_total:
        dropped = n_total - len(kept_idx)
        print(f"Dropped {dropped} near-duplicate frames (min_camera_dist={args.min_camera_dist}m); "
              f"{len(kept_idx)} remaining")
        scene = {k: (v[kept_idx] if isinstance(v, torch.Tensor) and v.shape[0] == n_total else v)
                 for k, v in scene.items()}
        scene["names"] = [scene["names"][i] for i in kept_idx]

    n_views = scene["viewmats"].shape[0]
    if args.holdout_names is not None:
        listed = {ln.split("\t")[0].strip() for ln in args.holdout_names.read_text().splitlines()}
        listed.discard("")
        train_idx, test_idx = split_by_names(scene["names"], listed)
        print(f"Held-out names: {len(listed)} listed, {len(test_idx)} registered")
    else:
        train_idx, test_idx = split_train_test(n_views, args.holdout_every)
    print(
        f"Loaded {n_total} views → {n_views} after dedup "
        f"({len(train_idx)} train / {len(test_idx)} test), "
        f"{scene['points_xyz'].shape[0]} points"
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    params = train(scene, train_idx, args.output_dir, args.iters, args.save_every, args.init_scale)
    save_ply(params, args.output_dir / "point_cloud.ply")
    torch.save({k: v.detach() for k, v in params.items()}, args.output_dir / "model.pt")

    if test_idx:
        metrics = evaluate(params, scene, test_idx, args.output_dir)
        overlap = view_overlap(scene, train_idx, test_idx)
        metrics["overlap"] = overlap
        metrics["nn_dist_frac"] = float(np.mean([o["nn_dist_frac"] for o in overlap]))
        metrics["nn_angle_deg"] = float(np.mean([o["nn_angle_deg"] for o in overlap]))
        with open(args.output_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Held-out PSNR: {metrics['psnr_mean']:.2f}  SSIM: {metrics['ssim_mean']:.4f}")


if __name__ == "__main__":
    main()
