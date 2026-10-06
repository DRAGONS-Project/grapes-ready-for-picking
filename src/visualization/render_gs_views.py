"""Render a 4×3 grid of novel views from a trained GS model for angle selection.

Supports two input formats:
  - model.pt  : custom training-loop checkpoint (means/quats/scales/opacities/colors)
  - *.ckpt    : nerfstudio splatfacto checkpoint (DC-only color from gauss_params)

All camera matrices follow the gsplat / COLMAP / OpenCV convention:
  +X right,  +Y down in image,  +Z into scene.

Usage (via slurm/render_gs_views.sh):
    uv run --extra recon python src/visualization/render_gs_views.py \\
        --model <path/to/model.pt or step-XXXXXX.ckpt> \\
        --output outputs/teaser/views_grid_4x3.png \\
        [--colmap-sparse outputs/reconstruction/.../colmap/sparse/1]  # for world-up
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image


# ── coordinate helpers ────────────────────────────────────────────────────────

def look_at(cam_pos: np.ndarray, lookat: np.ndarray, up: np.ndarray) -> np.ndarray:
    """World-to-camera 4×4 (OpenCV/gsplat: +Z into scene, +Y down in image).

    Compared with the standard OpenGL look_at:
      - z points INTO the scene (lookat - eye), not away from it
      - y = down in image  →  row 1 of R maps to world 'down' = -up
    """
    z = lookat - cam_pos
    z_n = np.linalg.norm(z)
    z = z / z_n if z_n > 1e-6 else np.array([0., 0., 1.])

    # Degenerate: camera looking along the 'up' axis
    if abs(np.dot(up, z)) > 0.999:
        up = np.array([0., 0., 1.]) if abs(z[2]) < 0.9 else np.array([1., 0., 0.])

    x = np.cross(z, up)          # +X_cam = right  (z × world_up for Y-up scenes)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)           # +Y_cam = down in image  (z × x)

    R = np.stack([x, y, z], axis=0).astype(np.float32)  # rows = cam-axis dirs in world
    t = (-R @ cam_pos).astype(np.float32)
    mat = np.eye(4, dtype=np.float32)
    mat[:3, :3] = R
    mat[:3, 3] = t
    return mat


def spherical_cam(center: np.ndarray, dist: float, az_deg: float, el_deg: float,
                  world_up: np.ndarray) -> np.ndarray:
    """Camera position on a sphere, with the 'pole' pointing along world_up."""
    az = np.radians(az_deg)
    el = np.radians(el_deg)

    # Unit sphere in local Y-up frame: el=0 = equator, el=90 = top (+Y)
    local = np.array([np.cos(el) * np.cos(az),
                      np.sin(el),
                      np.cos(el) * np.sin(az)])

    # Rotate so +Y_local → world_up
    y_hat = np.array([0., 1., 0.])
    up = world_up / (np.linalg.norm(world_up) + 1e-8)

    dot = float(np.dot(y_hat, up))
    if abs(dot) > 0.999:
        # Already (anti-)aligned
        offset = local * dist if dot > 0 else np.array([-local[0], -local[1], local[2]]) * dist
    else:
        v = np.cross(y_hat, up)
        s = np.linalg.norm(v)
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.eye(3) + vx + vx @ vx * ((1 - dot) / (s ** 2 + 1e-10))
        offset = R @ local * dist

    return center + offset


# ── world-up detection ────────────────────────────────────────────────────────

def world_up_from_colmap(sparse_dir: Path) -> np.ndarray:
    """Average -R[1,:] across COLMAP cameras (row 1 = image-down direction)."""
    try:
        import pycolmap
        recon = pycolmap.Reconstruction(str(sparse_dir))
        votes = []
        for img in recon.images.values():
            R = img.cam_from_world().rotation.matrix()
            votes.append(-R[1])           # negate image-down → image-up = world-up
        avg = np.mean(votes, axis=0)
        n = np.linalg.norm(avg)
        if n > 0.1:
            print(f"world_up from COLMAP: {(avg/n).round(3)}")
            return (avg / n).astype(np.float32)
    except Exception as e:
        print(f"COLMAP world-up detection failed ({e}), defaulting to Y-up")
    return np.array([0., 1., 0.], dtype=np.float32)


def world_up_from_transforms(transforms_json: Path) -> np.ndarray:
    """Average camera Y-col (nerfstudio 'up') across transforms.json frames,
    then apply the dataparser rotation so it matches the splatfacto model space."""
    import json
    with open(transforms_json) as f:
        t = json.load(f)

    # dataparser rotation (3×3 upper-left of the 3×4 transform matrix)
    dp_T = np.array(t.get("transform",
                           [[1,0,0,0],[0,1,0,0],[0,0,1,0]]), dtype=np.float32)
    # strip any translation column if present; keep only 3×4 or 3×3
    if dp_T.shape == (4, 4):
        R_dp = dp_T[:3, :3]
    elif dp_T.ndim == 2 and dp_T.shape[1] >= 3:
        R_dp = dp_T[:3, :3]
    else:
        R_dp = np.eye(3, dtype=np.float32)

    votes = []
    for frame in t.get("frames", []):
        c2w = np.array(frame["transform_matrix"], dtype=np.float32)
        # Column 1 = Y_cam = 'up' direction in nerfstudio OpenGL convention
        up_raw = c2w[:3, 1]
        votes.append(up_raw)

    if not votes:
        return np.array([0., 1., 0.], dtype=np.float32)

    avg = np.mean(votes, axis=0)
    # Transform to dataparser (model) space
    up_dp = R_dp @ avg
    n = np.linalg.norm(up_dp)
    if n < 0.1:
        return np.array([0., 1., 0.], dtype=np.float32)
    result = (up_dp / n).astype(np.float32)
    print(f"world_up from transforms.json → dataparser space: {result.round(3)}")
    return result


# ── model loading ─────────────────────────────────────────────────────────────

_C0 = 0.28209479177387814   # zeroth-order SH coefficient


def load_model_pt(path: Path, device: torch.device) -> dict:
    """Load custom training-loop checkpoint → {means, quats, scales, opacities, colors}."""
    raw = torch.load(path, map_location=device)
    params = {k: v.to(device) for k, v in raw.items()}
    print(f"  Loaded model.pt: {params['means'].shape[0]:,} Gaussians")
    return params


def load_splatfacto_ckpt(path: Path, device: torch.device) -> dict:
    """Extract Gaussian params from a nerfstudio splatfacto checkpoint.

    Uses DC-only SH (view-independent colour), which is sufficient for orbit renders.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    # The nerfstudio checkpoint stores state_dict with dotted flat keys inside
    # ckpt["pipeline"], e.g. "_model.gauss_params.means".
    # Try nested dict first, then flat dotted keys.
    gp = None
    nested_paths = [
        lambda c: c["pipeline"]["_model"]["gauss_params"],
        lambda c: c["pipeline"]["model"]["gauss_params"],
        lambda c: c["model"]["gauss_params"],
        lambda c: c["gauss_params"],
    ]
    for fn in nested_paths:
        try:
            gp = fn(ckpt)
            print(f"  gauss_params keys (nested): {list(gp.keys())[:8]}")
            break
        except (KeyError, TypeError):
            continue

    if gp is None:
        # Try flat dotted keys: ckpt["pipeline"]["_model.gauss_params.<name>"]
        pipeline = ckpt.get("pipeline", {})
        prefix = "_model.gauss_params."
        flat = {k[len(prefix):]: v for k, v in pipeline.items()
                if k.startswith(prefix)}
        if flat:
            gp = flat
            print(f"  gauss_params keys (flat): {list(gp.keys())}")
        else:
            def _keys(d, depth=0):
                if not isinstance(d, dict) or depth > 3:
                    return
                for k, v in list(d.items())[:30]:
                    print("  " * depth + str(k), "→", type(v).__name__,
                          getattr(v, "shape", ""))
                    _keys(v, depth + 1)
            print("Checkpoint structure:")
            _keys(ckpt)
            raise KeyError("Could not locate gauss_params in checkpoint.")

    means = gp["means"].to(device)          # (N, 3)
    scales = gp["scales"].to(device)        # (N, 3) log-scales

    quats = gp["quats"].to(device)
    if quats.ndim == 3:
        quats = quats.squeeze(1)            # (N,1,4) → (N,4)

    opacities = gp["opacities"].to(device)
    if opacities.ndim == 2:
        opacities = opacities.squeeze(-1)   # (N,1) → (N,)

    # DC SH → RGB: c = C0 * f_dc + 0.5
    f_dc = gp["features_dc"].to(device)    # (N,3) or (N,1,3)
    if f_dc.ndim == 3:
        f_dc = f_dc.squeeze(1)
    colors = (f_dc * _C0 + 0.5).clamp(0, 1)

    print(f"  Loaded splatfacto ckpt: {means.shape[0]:,} Gaussians")
    return {
        "means": means,
        "quats": quats,
        "scales": scales,
        "opacities": opacities,
        "colors": colors,
    }


# ── rendering ─────────────────────────────────────────────────────────────────

def render_view(params: dict, viewmat: np.ndarray, K: np.ndarray,
                width: int, height: int, device: torch.device) -> np.ndarray:
    from gsplat import rasterization

    vm = torch.tensor(viewmat, device=device).unsqueeze(0)
    Kt = torch.tensor(K,       device=device).unsqueeze(0)

    with torch.no_grad():
        renders, _, _ = rasterization(
            means=params["means"],
            quats=params["quats"],
            scales=params["scales"].exp(),
            opacities=torch.sigmoid(params["opacities"]),
            colors=params["colors"],
            viewmats=vm,
            Ks=Kt,
            width=width,
            height=height,
            packed=False,
        )
    rgb = renders[0].clamp(0, 1).cpu().numpy()
    return (rgb * 255).astype(np.uint8)


# ── argument parsing ──────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model",        required=True, type=Path,
                   help="Path to model.pt or nerfstudio .ckpt")
    p.add_argument("--output",       default="outputs/teaser/views_grid_4x3.png", type=Path)
    p.add_argument("--width",        type=int, default=640)
    p.add_argument("--height",       type=int, default=480)
    p.add_argument("--fov",          type=float, default=60.,
                   help="Horizontal FOV in degrees")
    p.add_argument("--colmap-sparse", type=Path, default=None,
                   help="COLMAP sparse dir (for world-up; model.pt only)")
    p.add_argument("--transforms-json", type=Path, default=None,
                   help="nerfstudio transforms.json (for world-up; .ckpt only)")
    # Single-view mode: if both --az and --el are given, renders one image (no grid/labels).
    p.add_argument("--az",  type=float, default=None, help="Azimuth  (deg) for single-view render")
    p.add_argument("--el",  type=float, default=None, help="Elevation (deg) for single-view render")
    p.add_argument("--dist-scale", type=float, default=1.1,
                   help="Orbit distance as multiple of max horizontal scene extent (default 1.1)")
    p.add_argument("--min-opacity", type=float, default=0.0,
                   help="Pre-filter Gaussians with sigmoid(opacity) below this value (0–1). "
                        "Useful to remove floaters for teaser renders (try 0.05).")
    p.add_argument("--lookat-shift", type=float, nargs=3, default=[0., 0., 0.],
                   metavar=("DX", "DY", "DZ"),
                   help="Shift the scene lookat (centroid) by this world-space vector "
                        "before rendering.  Useful to re-centre on a specific vine row.")
    p.add_argument("--clip-bbox", action="store_true",
                   help="Remove Gaussians outside the scene P5-P95 bounding box "
                        "(eliminates below-ground and edge floaters).")
    p.add_argument("--bbox-pad", type=float, default=0.0,
                   help="Expand the clip bbox by this fraction of each axis extent "
                        "(e.g. 0.1 adds 10%% padding on each side). Default 0.")
    return p.parse_args()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    repo = Path(__file__).resolve().parents[2]

    model_path  = args.model  if args.model.is_absolute()  else repo / args.model
    output_path = args.output if args.output.is_absolute() else repo / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Model:  {model_path}")

    # ── load params ──
    if model_path.suffix == ".ckpt":
        params = load_splatfacto_ckpt(model_path, device)
    else:
        params = load_model_pt(model_path, device)

    # ── opacity pruning (floater removal) ──
    if args.min_opacity > 0.0:
        with torch.no_grad():
            opac = torch.sigmoid(params["opacities"].float()).ravel()
        keep = opac >= args.min_opacity
        n_before = params["means"].shape[0]
        params = {k: v[keep] if v.shape[0] == n_before else v
                  for k, v in params.items()}
        print(f"Opacity pruning (≥{args.min_opacity}): {n_before:,} → {params['means'].shape[0]:,} Gaussians")

    # ── spatial bbox clipping (removes below-ground / edge floaters) ──
    if args.clip_bbox:
        means_cpu = params["means"].detach().cpu().numpy()
        lo_clip = np.percentile(means_cpu, 5,  axis=0)
        hi_clip = np.percentile(means_cpu, 95, axis=0)
        pad = (hi_clip - lo_clip) * args.bbox_pad
        lo_clip -= pad;  hi_clip += pad
        mask = np.all((means_cpu >= lo_clip) & (means_cpu <= hi_clip), axis=1)
        mask_t = torch.from_numpy(mask).to(params["means"].device)
        n_before = params["means"].shape[0]
        params = {k: v[mask_t] if v.shape[0] == n_before else v
                  for k, v in params.items()}
        print(f"BBox clip ({lo_clip.round(2)} … {hi_clip.round(2)}): "
              f"{n_before:,} → {params['means'].shape[0]:,} Gaussians")

    # ── scene geometry ──
    means_np = params["means"].detach().cpu().numpy()

    # Opacity-weighted centroid: focus on solid content, ignore sparse floaters.
    with torch.no_grad():
        opac_np = torch.sigmoid(params["opacities"].float()).cpu().numpy().ravel()
    thresh = np.percentile(opac_np, 40)   # top-60% opacity Gaussians define the center
    dense = means_np[opac_np > thresh]
    center = dense.mean(axis=0).astype(np.float32)
    shift = np.array(args.lookat_shift, dtype=np.float32)
    if np.any(shift != 0):
        center = center + shift
        print(f"Lookat shift applied: {shift} → new centre {center.round(3)}")

    # Extent from P5-P95 for camera distance (still use full distribution for scale).
    lo = np.percentile(means_np, 5,  axis=0)
    hi = np.percentile(means_np, 95, axis=0)
    extent = hi - lo
    print(f"Scene centre (opacity-wtd): {center.round(3)},  P5-P95 extent: {extent.round(2)}")

    # ── world-up ──
    if model_path.suffix == ".ckpt":
        tfile = args.transforms_json
        if tfile is None:
            # Try to find transforms.json next to the splatfacto outputs
            tfile = model_path.parents[3] / "work" / "ns_data" / "transforms.json"
        world_up = world_up_from_transforms(tfile) if tfile.exists() else np.array([0., 1., 0.])
    else:
        sdir = args.colmap_sparse
        if sdir is None:
            best_txt = model_path.parents[2] / "colmap" / "best_sparse_dir.txt"
            if best_txt.exists():
                sdir = Path(best_txt.read_text().strip())
        world_up = world_up_from_colmap(sdir) if (sdir and sdir.exists()) else np.array([0., 1., 0.])

    # ── intrinsics ──
    fx = args.width / (2 * np.tan(np.radians(args.fov / 2)))
    K = np.array([[fx, 0, args.width / 2],
                  [0, fx, args.height / 2],
                  [0,  0,              1]], dtype=np.float32)

    # Camera orbit distances — single scale factor for all elevation tiers.
    xz_span = max(extent[0], extent[2])   # horizontal span perpendicular to world_up
    dist = xz_span * args.dist_scale

    # ── single-view mode ──────────────────────────────────────────────────────
    if args.az is not None and args.el is not None:
        cam_pos = spherical_cam(center, dist, args.az, args.el, world_up)
        viewmat = look_at(cam_pos, center, world_up)
        print(f"  Single view  az={args.az}°  el={args.el}°  cam={cam_pos.round(3)}")
        img = render_view(params, viewmat, K, args.width, args.height, device)
        Image.fromarray(img).save(str(output_path))
        print(f"Saved: {output_path}")
        return

    # ── grid mode: 12 views, 4 columns × 3 rows ──────────────────────────────
    dist_high = dist * 0.8
    dist_mid  = dist
    dist_low  = dist * 1.3

    views = [
        # Row 0: bird's-eye (60°), four cardinal azimuths
        ("60° N",   0,  60, dist_high),
        ("60° E",  90,  60, dist_high),
        ("60° S", 180,  60, dist_high),
        ("60° W", 270,  60, dist_high),
        # Row 1: mid elevation (35°), four azimuths
        ("35° N",   0,  35, dist_mid),
        ("35° NE", 45,  35, dist_mid),
        ("35° E",  90,  35, dist_mid),
        ("35° SE",135,  35, dist_mid),
        # Row 2: low drone-like (15°), diagonal azimuths
        ("15° N",   0,  15, dist_low),
        ("15° NE", 45,  15, dist_low),
        ("15° E",  90,  15, dist_low),
        ("15° SE",135,  15, dist_low),
    ]

    imgs = []
    for label, az, el, d in views:
        cam_pos = spherical_cam(center, d, az, el, world_up)
        viewmat = look_at(cam_pos, center, world_up)
        print(f"  {label:8s}  az={az:4d}°  el={el:2d}°  cam={cam_pos.round(1)}")
        img = render_view(params, viewmat, K, args.width, args.height, device)
        imgs.append((label, img))

    cols, rows_grid = 4, 3
    pad, label_h = 4, 20
    cell_w = args.width  + pad
    cell_h = args.height + pad + label_h
    grid = np.zeros((rows_grid * cell_h, cols * cell_w, 3), dtype=np.uint8)

    from PIL import ImageDraw
    for i, (label, img) in enumerate(imgs):
        r, c = divmod(i, cols)
        y0, x0 = r * cell_h, c * cell_w
        grid[y0 + label_h: y0 + label_h + args.height, x0: x0 + args.width] = img
        cell = Image.fromarray(grid[y0: y0 + cell_h, x0: x0 + cell_w])
        ImageDraw.Draw(cell).text((4, 2), label, fill=(255, 255, 100))
        grid[y0: y0 + cell_h, x0: x0 + cell_w] = np.array(cell)

    Image.fromarray(grid).save(str(output_path))
    print(f"Saved grid: {output_path}  ({cols*cell_w}×{rows_grid*cell_h} px)")

    for label, img in imgs:
        slug = label.replace(" ", "_").replace("°", "deg")
        Image.fromarray(img).save(str(output_path.parent / f"view_{slug}.png"))
    print(f"Individual views: {output_path.parent}/view_*.png")


if __name__ == "__main__":
    main()
