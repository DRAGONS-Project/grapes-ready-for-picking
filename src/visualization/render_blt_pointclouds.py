"""Render point-cloud images for the BLT raw vs. preprocessed comparison.

Uses a pure-numpy pinhole rasteriser with a z-buffer — fully headless,
no EGL/display required.  Two image sets are produced:

  outputs/figures/blt_raw/          — raw (unmasked) point cloud
  outputs/figures/blt_preprocessed/ — preprocessed (masked) point cloud

Viewpoints:
  * overview_{top,side,angle}   — external orbital shots
  * row_{entrance,mid,exit}     — "inside the row" walking along the row
  * robot_pov_{0..4}            — actual COLMAP camera positions (preprocessed
                                   reconstruction; same poses reused for raw)
"""

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d
from PIL import Image

try:
    import pycolmap
    HAS_PYCOLMAP = True
except ImportError:
    HAS_PYCOLMAP = False


# ── point cloud loading ───────────────────────────────────────────────────────

def load_cloud(path: Path):
    pc = o3d.io.read_point_cloud(str(path))
    pts = np.asarray(pc.points, dtype=np.float64)
    if pc.has_colors():
        rgb = (np.asarray(pc.colors, dtype=np.float32) * 255).astype(np.uint8)
    else:
        rgb = np.full((len(pts), 3), 153, dtype=np.uint8)
    return pts, rgb


# ── camera math ───────────────────────────────────────────────────────────────

def look_at_matrix(eye: np.ndarray, target: np.ndarray,
                   up: np.ndarray) -> np.ndarray:
    """World-to-camera 3×3 rotation (OpenCV: +X right, +Y down, +Z into scene)."""
    z = target - eye
    z /= np.linalg.norm(z) + 1e-12

    if abs(np.dot(up, z)) > 0.999:
        up = np.array([1., 0., 0.]) if abs(z[0]) < 0.9 else np.array([0., 0., 1.])

    x = np.cross(z, up)
    x /= np.linalg.norm(x) + 1e-12
    y = np.cross(z, x)    # +Y_cam = down in image

    R = np.stack([x, y, z], axis=0)   # rows = camera axis dirs in world
    return R


# ── rasteriser ────────────────────────────────────────────────────────────────

def render(pts: np.ndarray, rgb: np.ndarray,
           eye: np.ndarray, target: np.ndarray, up: np.ndarray,
           fov_deg: float = 70., width: int = 1280, height: int = 960,
           point_radius: int = 2, bg: tuple = (20, 20, 20)) -> np.ndarray:
    """Project 3-D points onto an image using a pinhole camera + z-buffer.

    Returns an H×W×3 uint8 array.
    """
    R = look_at_matrix(eye, target, up)
    t = -R @ eye                        # translation in camera frame

    # Transform to camera space
    p_cam = (R @ pts.T).T + t          # (N, 3)

    # Keep only points in front of the camera
    mask = p_cam[:, 2] > 0.1
    p_cam = p_cam[mask]
    c_vis = rgb[mask]

    if len(p_cam) == 0:
        img = np.full((height, width, 3), bg, dtype=np.uint8)
        return img

    # Intrinsics (square pixels)
    fx = width / (2.0 * np.tan(np.radians(fov_deg / 2)))
    cx, cy = width / 2.0, height / 2.0

    x_px = (fx * p_cam[:, 0] / p_cam[:, 2] + cx).astype(np.float32)
    y_px = (fx * p_cam[:, 1] / p_cam[:, 2] + cy).astype(np.float32)
    depth = p_cam[:, 2].astype(np.float32)

    # Filter to image bounds (with point_radius margin)
    r = point_radius
    in_bounds = ((x_px >= r) & (x_px < width - r) &
                 (y_px >= r) & (y_px < height - r))
    x_px, y_px, depth, c_vis = x_px[in_bounds], y_px[in_bounds], depth[in_bounds], c_vis[in_bounds]

    # Sort far → near so close points paint over distant ones
    order = np.argsort(-depth)
    xi = x_px[order].astype(np.int32)
    yi = y_px[order].astype(np.int32)
    col = c_vis[order]

    img = np.full((height, width, 3), bg, dtype=np.uint8)

    # Paint filled circles via a square brush of size (2r+1)²
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dx * dx + dy * dy > r * r:
                continue
            xw = np.clip(xi + dx, 0, width - 1)
            yw = np.clip(yi + dy, 0, height - 1)
            img[yw, xw] = col

    return img


# ── viewpoint definitions ─────────────────────────────────────────────────────

def autofit_dist(half_extent: float, fov_deg: float, margin: float = 1.25) -> float:
    """Camera distance so that `half_extent` fills half the FOV, with margin."""
    return half_extent / np.tan(np.radians(fov_deg / 2)) * margin


def colmap_cameras(colmap_dir) -> list:
    """Return list of (center, forward, up) numpy arrays from a COLMAP sparse dir."""
    if colmap_dir is None or not HAS_PYCOLMAP:
        return []
    try:
        recon = pycolmap.Reconstruction(str(colmap_dir))
        cams = []
        for img in sorted(recon.images.values(), key=lambda i: i.name):
            T = img.cam_from_world()
            R = T.rotation.matrix()
            t = T.translation
            center  = -R.T @ t
            forward =  R.T @ np.array([0., 0., 1.])
            up      = -R.T @ np.array([0., 1., 0.])
            cams.append((center, forward, up))
        return cams
    except Exception as e:
        print(f"  (COLMAP pose extraction failed: {e})")
        return []


def build_viewpoints(pts: np.ndarray, colmap_dir=None, fov_deg: float = 70.):
    """Return list of (name, eye, target, up) tuples."""

    # Tight scene bounds (P10-P90) to ignore outlier background points
    lo  = np.percentile(pts, 10, axis=0)
    hi  = np.percentile(pts, 90, axis=0)
    center = (lo + hi) / 2
    extent = hi - lo                          # per-axis P10-P90 range

    world_up = np.array([0., -1., 0.])       # -Y = up in COLMAP convention

    # Row direction: longest PCA axis
    sample = pts if len(pts) <= 5000 else pts[np.random.choice(len(pts), 5000, replace=False)]
    _, _, Vt = np.linalg.svd(sample - sample.mean(0), full_matrices=False)
    row_dir = Vt[0]
    if row_dir[2] < 0:
        row_dir = -row_dir

    side_dir = np.cross(row_dir, world_up)
    side_dir /= np.linalg.norm(side_dir) + 1e-12

    # Auto-fit orbit distances per view axis
    half_xz   = max(extent[0], extent[2]) / 2
    half_full  = max(extent)              / 2
    dist_top  = autofit_dist(half_xz,  fov_deg)     # top-down: fit XZ footprint
    dist_side = autofit_dist(half_full, fov_deg)     # side/angle: fit largest dim

    views = []

    # ── overview shots ────────────────────────────────────────────────────────
    # Top-down: camera straight above, look down, row_dir as "up" in image
    views.append(("overview_top",
                  center + world_up * (-dist_top),   # -Y = above (Y is down)
                  center,
                  row_dir))

    # Side elevation: look along the row from the side
    views.append(("overview_side",
                  center - side_dir * dist_side + world_up * (-dist_side * 0.25),
                  center,
                  world_up))

    # Angled 3/4 bird's-eye
    views.append(("overview_angle",
                  center
                  - side_dir * dist_side * 0.6
                  + world_up * (-dist_side * 0.55)
                  - row_dir  * dist_side * 0.25,
                  center,
                  world_up))

    # ── COLMAP camera poses (inside-row shots) ────────────────────────────────
    cams = colmap_cameras(colmap_dir)

    if cams:
        n = len(cams)
        # Sort by position along the row direction so entrance/exit make sense
        proj = [float(np.dot(c[0], row_dir)) for c in cams]
        order = np.argsort(proj)
        cams_sorted = [cams[i] for i in order]

        # Entrance: first camera, look toward end of row
        c0, f0, u0 = cams_sorted[0]
        row_end_pos = cams_sorted[-1][0]
        views.append(("row_entrance", c0, row_end_pos, u0))

        # Exit: last camera, look back
        cn, fn, un = cams_sorted[-1]
        views.append(("row_exit", cn, cams_sorted[0][0], un))

        # Mid-row looking across (perpendicular) — pick 3 cameras
        for frac, label in [(0.25, "row_quarter"), (0.5, "row_mid"), (0.75, "row_three_quarter")]:
            idx = int(frac * (n - 1))
            ci, fi, ui = cams_sorted[idx]
            # Look perpendicular to row direction (toward the vine wall)
            perp_target = ci + side_dir * float(extent[0]) * 0.5
            views.append((label, ci, perp_target, ui))

        # 5 evenly-spaced robot-POV shots with original look direction
        for k, idx in enumerate([int(n * f) for f in (0.0, 0.25, 0.5, 0.75, 1.0)]):
            idx = min(idx, n - 1)
            ci, fi, ui = cams_sorted[idx]
            views.append((f"robot_pov_{k}", ci, ci + fi * 3.0, ui))
    else:
        # Fallback: geometry-based traversal views along the row
        # Robot height: Y is down, so camera sits slightly below the cloud median in Y
        robot_y = center[1] + extent[1] * 0.25

        proj_row = np.dot(pts - center, row_dir)
        row_half = float(proj_row.max() - proj_row.min()) / 2 * 0.85
        side_half = float((np.dot(pts - center, side_dir)).max()
                          - (np.dot(pts - center, side_dir)).min()) / 2

        # Place 5 cameras evenly along the row, looking toward the other end
        for k, t in enumerate([0.05, 0.25, 0.5, 0.75, 0.95]):
            cam_pos = center + row_dir * ((-1 + 2 * t) * row_half)
            cam_pos[1] = robot_y
            look_fwd = row_dir if t < 0.5 else -row_dir
            views.append((f"robot_pov_{k}", cam_pos, cam_pos + look_fwd * 5.0, world_up))

        # Entrance and exit
        rs = center - row_dir * row_half; rs = rs.copy(); rs[1] = robot_y
        re = center + row_dir * row_half; re = re.copy(); re[1] = robot_y
        views.append(("row_entrance", rs, re, world_up))
        views.append(("row_exit",     re, rs, world_up))

        # Mid-row across
        rm = center.copy(); rm[1] = robot_y
        views.append(("row_mid", rm, rm + side_dir * side_half, world_up))

    return views


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args():
    repo = Path(__file__).resolve().parents[2]
    base_ply = repo / "outputs" / "reconstruction" / "blt_sfm_comparison"
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw",        default=base_ply / "blt_raw_sparse.ply",         type=Path)
    p.add_argument("--prep",       default=base_ply / "blt_preprocessed_sparse.ply", type=Path)
    p.add_argument("--colmap-dir", default=Path("/home/grid/users/plgmwlodarzc/synthetic-grapes/data/preprocessed/colmap/sparse/0"), type=Path)
    p.add_argument("--out-dir",    default=repo / "outputs" / "figures",             type=Path)
    p.add_argument("--width",      default=1280, type=int)
    p.add_argument("--height",     default=960,  type=int)
    p.add_argument("--fov",        default=70.,  type=float)
    p.add_argument("--radius",     default=2,    type=int)
    return p.parse_args()


def render_cloud(tag: str, ply_path: Path, colmap_dir, args):
    out_dir = args.out_dir / f"blt_{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== {tag} ({ply_path.name}) ===")
    pts, rgb = load_cloud(ply_path)
    print(f"  {len(pts):,} points loaded")

    views = build_viewpoints(pts, colmap_dir, fov_deg=args.fov)

    for name, eye, target, up in views:
        img = render(pts, rgb, eye, target, up,
                     fov_deg=args.fov, width=args.width, height=args.height,
                     point_radius=args.radius)
        out_path = out_dir / f"{name}.png"
        Image.fromarray(img).save(str(out_path))
        print(f"  saved: {out_path.name}")

    print(f"  → {len(views)} images in {out_dir}")


def main():
    args = parse_args()
    colmap_dir = args.colmap_dir if args.colmap_dir.exists() else None
    if colmap_dir is None:
        print("Warning: COLMAP sparse dir not found; robot_pov views skipped.")

    render_cloud("raw",          args.raw,  None,       args)  # no COLMAP for raw (temp dir gone)
    render_cloud("preprocessed", args.prep, colmap_dir, args)


if __name__ == "__main__":
    main()
