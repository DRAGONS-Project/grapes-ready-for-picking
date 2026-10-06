"""
Teaser figure: 3DGS vineyard + Franka Panda arm in Genesis.

Pipeline:
  1. Load model.pt → extract 2.5M Gaussian centers + colors → dense PLY
  2. Poisson surface reconstruction → static OBJ mesh (cached)
  3. Genesis scene: mesh as vineyard backdrop + Franka Panda in harvesting pose
  4. Render a single high-resolution PNG for the paper teaser

The dense Gaussian point cloud (step 1) gives far better Poisson quality than
the sparse 64k-point COLMAP export (point_cloud.ply).

Requirements:
  - A Genesis venv (genesis-world 0.4.7) with open3d and Pillow installed.
    Default: /lustre/pd03/plgrid/plgdragons/synthetic-grapes/venv/.venv-genesis
    To reinstall: bash slurm/install_genesis.sh  (from synthetic-grapes repo)

Usage (via SLURM — see slurm/render_teaser.sh):
    <genesis-venv>/bin/python src/visualization/teaser_gs_genesis.py \\
        --model outputs/reconstruction/mots_pathplanning_1/runs/5355824/model.pt \\
        --output outputs/teaser/teaser.png

Options:
    --model PATH         Path to model.pt from train_gs.py          [required]
    --mesh-cache PATH    Cached mesh .obj (built once, reused after)
    --output PATH        Output PNG                                  (default: outputs/teaser/teaser.png)
    --azimuth F          Camera azimuth in degrees, 0=+x CCW        (default: -35)
    --elevation F        Camera elevation in degrees                 (default: 28)
    --fov F              Camera field of view                        (default: 42)
    --robot-side STR     Which side of the scene to place the arm:
                         "left" (+y), "right" (-y), "front" (+x), "back" (-x)
                         (default: right)
    --width N            Render width px                             (default: 1920)
    --height N           Render height px                            (default: 1080)
    --voxel-size F       Voxel downsample size for Poisson          (default: 0.015)
    --poisson-depth N    Poisson octree depth                       (default: 9)
    --max-points N       Cap on Gaussian centers to use (0 = all)   (default: 0)
"""

import argparse
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model",          required=True, type=Path)
    p.add_argument("--mesh-cache",     type=Path, default=None)
    p.add_argument("--output",         default="outputs/teaser/teaser.png")
    p.add_argument("--azimuth",        type=float, default=-35.0)
    p.add_argument("--elevation",      type=float, default=28.0)
    p.add_argument("--fov",            type=float, default=42.0)
    p.add_argument("--robot-side",     default="right",
                   choices=["left", "right", "front", "back"])
    p.add_argument("--width",          type=int,   default=1920)
    p.add_argument("--height",         type=int,   default=1080)
    p.add_argument("--voxel-size",     type=float, default=0.015)
    p.add_argument("--poisson-depth",  type=int,   default=9)
    p.add_argument("--max-points",     type=int,   default=0)
    return p.parse_args()


# ── Step 1: extract dense point cloud from GS model ──────────────────────────

def extract_dense_ply(model_path: Path, output_path: Path, max_points: int = 0) -> None:
    """Extract Gaussian centers + colors from model.pt → colored PLY."""
    import torch
    import open3d as o3d

    print(f"Loading model from {model_path} …")
    params = torch.load(model_path, map_location="cpu")

    means  = params["means"].numpy().astype(np.float64)   # (N, 3)
    colors = params["colors"].numpy().astype(np.float64)  # (N, 3) in [0,1]
    colors = np.clip(colors, 0.0, 1.0)

    if max_points > 0 and len(means) > max_points:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(means), max_points, replace=False)
        means  = means[idx]
        colors = colors[idx]

    print(f"Extracted {len(means):,} Gaussian centers")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(means)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(str(output_path), pcd)
    print(f"Dense point cloud saved: {output_path}")


# ── Step 2: Poisson mesh reconstruction ──────────────────────────────────────

def pointcloud_to_mesh(ply_path: Path, obj_path: Path,
                       voxel_size: float, depth: int) -> None:
    import open3d as o3d

    print(f"Poisson reconstruction (voxel={voxel_size}, depth={depth}) …")
    pcd = o3d.io.read_point_cloud(str(ply_path))
    pcd = pcd.voxel_down_sample(voxel_size)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=voxel_size * 3, max_nn=30
        )
    )
    pcd.orient_normals_consistent_tangent_plane(k=15)
    mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)

    # Crop to the convex hull of the original points to remove Poisson artifacts.
    hull, _ = o3d.geometry.PointCloud.compute_convex_hull(pcd)
    bbox = hull.get_oriented_bounding_box()
    mesh = mesh.crop(bbox)

    obj_path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_triangle_mesh(str(obj_path), mesh)
    print(f"Mesh saved: {obj_path}  ({len(mesh.triangles):,} triangles)")


# ── Step 3: Genesis render ────────────────────────────────────────────────────

def render_teaser(
    mesh_path: Path,
    output_path: Path,
    azimuth: float,
    elevation: float,
    fov: float,
    robot_side: str,
    width: int,
    height: int,
) -> None:
    import open3d as o3d
    import genesis as gs

    # Scene bounding box from the mesh.
    mesh_o3d = o3d.io.read_triangle_mesh(str(mesh_path))
    bbox_min = np.array(mesh_o3d.get_min_bound())
    bbox_max = np.array(mesh_o3d.get_max_bound())
    center   = (bbox_min + bbox_max) / 2.0
    extent   = bbox_max - bbox_min
    print(f"Scene bbox: min={bbox_min}, max={bbox_max}, extent={extent}")

    # ── Camera placement ──────────────────────────────────────────────────────
    distance = 1.6 * max(extent[0], extent[1], 1.0)
    lookat   = center.copy()

    az = np.radians(azimuth)
    el = np.radians(elevation)
    cam_pos = lookat + distance * np.array([
        np.cos(el) * np.cos(az),
        np.cos(el) * np.sin(az),
        np.sin(el),
    ])

    # ── Robot placement ───────────────────────────────────────────────────────
    # Place the Panda base at the edge of the scene on the chosen side,
    # offset outward by 15% of the scene extent so it's clearly "next to" the vines.
    side_offset = 0.15
    if robot_side == "right":
        robot_base = np.array([center[0],
                                bbox_min[1] - side_offset * extent[1],
                                bbox_min[2]])
        # Rotate to face the vines (toward +y)
        robot_yaw_deg = 90.0
    elif robot_side == "left":
        robot_base = np.array([center[0],
                                bbox_max[1] + side_offset * extent[1],
                                bbox_min[2]])
        robot_yaw_deg = -90.0
    elif robot_side == "front":
        robot_base = np.array([bbox_min[0] - side_offset * extent[0],
                                center[1],
                                bbox_min[2]])
        robot_yaw_deg = 0.0
    else:  # back
        robot_base = np.array([bbox_max[0] + side_offset * extent[0],
                                center[1],
                                bbox_min[2]])
        robot_yaw_deg = 180.0

    yaw = np.radians(robot_yaw_deg)
    robot_quat = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])  # wxyz

    print(f"Robot base: {robot_base}  yaw={robot_yaw_deg} deg")
    print(f"Camera: pos={cam_pos}, lookat={lookat}, az={azimuth} deg, el={elevation} deg")

    # ── Genesis ───────────────────────────────────────────────────────────────
    gs.init(backend=gs.cuda, logging_level="warning")

    scene = gs.Scene(
        show_viewer=False,
        renderer=gs.renderers.Rasterizer(),
        sim_options=gs.options.SimOptions(gravity=(0, 0, -9.81)),
    )

    # Vineyard scene — static visual backdrop, no physics.
    scene.add_entity(
        gs.morphs.Mesh(file=str(mesh_path), fixed=True, collision=False),
        material=gs.materials.Kinematic(),
    )

    # Franka Panda — bundled with genesis-world.
    import importlib.util
    genesis_spec = importlib.util.find_spec("genesis")
    genesis_root = Path(genesis_spec.origin).parent
    panda_urdf = genesis_root / "assets" / "urdf" / "panda_bullet" / "panda.urdf"
    if not panda_urdf.exists():
        # Fallback: search common asset paths
        candidates = list(genesis_root.rglob("panda.urdf"))
        if not candidates:
            raise FileNotFoundError(
                f"panda.urdf not found under {genesis_root}. "
                "Ensure genesis-world 0.4.7 is installed."
            )
        panda_urdf = candidates[0]
    print(f"Panda URDF: {panda_urdf}")

    robot = scene.add_entity(
        gs.morphs.URDF(
            file=str(panda_urdf),
            fixed=False,
            pos=tuple(float(v) for v in robot_base),
            quat=tuple(float(v) for v in robot_quat),
        ),
    )

    cam = scene.add_camera(
        res=(width, height),
        pos=tuple(float(v) for v in cam_pos),
        lookat=tuple(float(v) for v in lookat),
        fov=fov,
    )

    scene.build()

    # ── Robot pose: arm extended reaching toward the vine row ─────────────────
    # 7 arm joints + 2 finger joints = 9 DOFs for the Panda.
    # This pose has the elbow up and end-effector pointing forward-down,
    # suggesting the gripper is approaching a grape cluster.
    arm_pose   = np.array([0.0, -0.3, 0.0, -2.0, 0.0, 1.8, 0.785])
    finger_val = 0.025  # slightly open (max ~0.04 m)
    dofs_target = np.concatenate([arm_pose, [finger_val, finger_val]])

    n_dofs = robot.n_dofs
    print(f"Panda DOFs: {n_dofs}")

    # Free-floating base adds 6 DOFs at the front; arm+fingers are the last 9.
    arm_dofs_idx = np.arange(n_dofs - 9, n_dofs)

    # Step once to settle the arm pose before rendering.
    robot.set_pos(tuple(float(v) for v in robot_base), zero_velocity=True)
    robot.set_quat(tuple(float(v) for v in robot_quat), zero_velocity=True)
    robot.set_dofs_position(dofs_target, dofs_idx_local=arm_dofs_idx)
    scene.step()

    # Render a single frame.
    rgb, _, _, _ = cam.render()

    from PIL import Image
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(str(output_path))
    print(f"Saved: {output_path}  ({width}×{height} px)")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    repo_root   = Path(__file__).resolve().parents[2]
    model_path  = args.model if args.model.is_absolute() else repo_root / args.model
    output_path = Path(args.output) if Path(args.output).is_absolute() else repo_root / args.output

    # Intermediate files live next to model.pt so they're easy to find/reuse.
    run_dir = model_path.parent
    dense_ply = run_dir / "dense_gaussians.ply"
    mesh_path = args.mesh_cache
    if mesh_path is None:
        mesh_path = run_dir / "teaser_mesh.obj"
    elif not mesh_path.is_absolute():
        mesh_path = repo_root / mesh_path

    # Step 1 — dense point cloud from GS means (cached).
    if dense_ply.exists():
        print(f"Reusing dense point cloud: {dense_ply}")
    else:
        extract_dense_ply(model_path, dense_ply, max_points=args.max_points)

    # Step 2 — Poisson mesh (cached).
    if mesh_path.exists():
        print(f"Reusing cached mesh: {mesh_path}")
    else:
        pointcloud_to_mesh(dense_ply, mesh_path, args.voxel_size, args.poisson_depth)

    # Step 3 — Genesis render.
    render_teaser(
        mesh_path=mesh_path,
        output_path=output_path,
        azimuth=args.azimuth,
        elevation=args.elevation,
        fov=args.fov,
        robot_side=args.robot_side,
        width=args.width,
        height=args.height,
    )


if __name__ == "__main__":
    main()
