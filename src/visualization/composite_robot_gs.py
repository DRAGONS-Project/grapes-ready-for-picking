"""
Composite a Franka Panda robot arm onto 3DGS eval renders for the paper teaser.

Pipeline:
  1. Render Panda-only in Genesis with green-screen background (no vineyard mesh)
  2. Chroma-key the robot → RGBA PNG
  3. Composite onto each provided background image at a specified position/scale

The robot camera uses az=80°, el=35° to match the GS eval render perspective.

Usage:
    <genesis-venv>/bin/python src/visualization/composite_robot_gs.py \\
        --backgrounds outputs/teaser/teaser_eval_04.png \\
                      outputs/teaser/teaser_eval_05.png \\
                      outputs/teaser/teaser_eval_06.png \\
        --output-dir  outputs/teaser/composite

Options:
    --backgrounds PATH [PATH ...]   Background images (GS eval renders)
    --output-dir  PATH              Directory to write composite PNGs
    --azimuth     F                 Robot camera azimuth in degrees (default: 80)
    --elevation   F                 Robot camera elevation in degrees (default: 35)
    --fov         F                 Robot camera FOV (default: 50)
    --robot-scale F                 Scale factor applied when pasting robot onto BG
                                    (default: 0.40 → robot ≈ 40% of image height)
    --paste-x     F                 Horizontal center of pasted robot, 0–1 (default: 0.75)
    --paste-y     F                 Vertical bottom of robot in image, 0–1 (default: 0.92)
    --robot-side  STR               left/right/front/back (default: right)
"""

import argparse
from pathlib import Path

import numpy as np


CHROMA_KEY = (0, 255, 0)   # pure green — background color in Genesis render


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backgrounds", nargs="+", required=True, type=Path)
    p.add_argument("--output-dir", type=Path, default=Path("outputs/teaser/composite"))
    p.add_argument("--azimuth",    type=float, default=80.0)
    p.add_argument("--elevation",  type=float, default=35.0)
    p.add_argument("--fov",        type=float, default=50.0)
    p.add_argument("--robot-scale", type=float, default=0.40)
    p.add_argument("--paste-x",    type=float, default=0.75)
    p.add_argument("--paste-y",    type=float, default=0.92)
    p.add_argument("--robot-side", default="right",
                   choices=["left", "right", "front", "back"])
    p.add_argument("--dist",       type=float, default=3.0,
                   help="Camera distance from robot in Genesis (m, default 3.0)")
    p.add_argument("--crop",       type=float, nargs=4, default=None,
                   metavar=("L", "T", "R", "B"),
                   help="Crop background image (fractions 0-1) before compositing")
    p.add_argument("--render-res", type=int, nargs=2, default=[1280, 960],
                   metavar=("W", "H"))
    return p.parse_args()


def render_robot_greenscreen(azimuth, elevation, fov, robot_side, width, height, dist=3.0):
    """Render Franka Panda alone against bright-green background → (H,W,3) uint8."""
    import genesis as gs

    from genesis.options.vis import DirectionalLight, VisOptions as _VisOptions

    gs.init(backend=gs.cuda, logging_level="warning")

    scene = gs.Scene(
        show_viewer=False,
        renderer=gs.renderers.Rasterizer(),
        sim_options=gs.options.SimOptions(gravity=(0, 0, -9.81)),
        vis_options=_VisOptions(
            background_color=(0.0, 1.0, 0.0),   # chroma-key green
            ambient_light=(0.01, 0.01, 0.015),  # near-zero → one side stays dark
            shadow=True,
            lights=[
                # Sun: strong, warm, from upper-right-front — matches outdoor vineyard
                DirectionalLight(
                    dir=(-0.4, 0.5, -1.0), color=(1.0, 0.93, 0.78), intensity=30.0
                ),
                # Sky scatter: very weak cool fill from opposite side only
                DirectionalLight(
                    dir=(0.3, -0.4, -0.3), color=(0.35, 0.45, 0.65), intensity=2.0
                ),
            ],
        ),
    )

    # ── Franka Panda ──────────────────────────────────────────────────────────
    import importlib.util
    genesis_spec = importlib.util.find_spec("genesis")
    genesis_root = Path(genesis_spec.origin).parent
    panda_urdf = genesis_root / "assets" / "urdf" / "panda_bullet" / "panda.urdf"
    if not panda_urdf.exists():
        candidates = list(genesis_root.rglob("panda.urdf"))
        if not candidates:
            raise FileNotFoundError("panda.urdf not found")
        panda_urdf = candidates[0]
    print(f"Panda URDF: {panda_urdf}")

    # Robot faces toward the camera (+x) by default; rotate to face the vines.
    yaw_map = {"right": 90.0, "left": -90.0, "front": 0.0, "back": 180.0}
    yaw = np.radians(yaw_map[robot_side])
    robot_quat = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])  # wxyz

    robot = scene.add_entity(
        gs.morphs.URDF(
            file=str(panda_urdf),
            fixed=False,
            pos=(0.0, 0.0, 0.0),
            quat=tuple(float(v) for v in robot_quat),
        ),
    )

    # ── Camera: az/el around origin ───────────────────────────────────────────
    az_r = np.radians(azimuth)
    el_r = np.radians(elevation)
    cam_pos = np.array([
        dist * np.cos(el_r) * np.cos(az_r),
        dist * np.cos(el_r) * np.sin(az_r),
        dist * np.sin(el_r),
    ])
    # For top-down views the arm extends toward +y; aim between base and arm tip
    lookat_y = 0.3 if elevation > 60 else 0.0
    cam = scene.add_camera(
        res=(width, height),
        pos=tuple(float(v) for v in cam_pos),
        lookat=(0.0, lookat_y, 0.4),
        fov=fov,
    )

    scene.build()

    # Arm reaching horizontally toward the vine row (visible from top-down)
    # j0=0 keeps arm in robot's forward dir (+y world); j1=-0.2 tilts slightly
    # forward; j3=-1.5 moderately extends elbow; j5=0.8 angles wrist toward grapes
    arm_pose   = np.array([0.0, -0.2, 0.0, -1.5, 0.0, 0.8, 0.785])
    finger_val = 0.025
    dofs_target = np.concatenate([arm_pose, [finger_val, finger_val]])
    n_dofs = robot.n_dofs
    arm_dofs_idx = np.arange(n_dofs - 9, n_dofs)
    robot.set_pos((0.0, 0.0, 0.0), zero_velocity=True)
    robot.set_quat(tuple(float(v) for v in robot_quat), zero_velocity=True)
    robot.set_dofs_position(dofs_target, dofs_idx_local=arm_dofs_idx)
    scene.step()

    rgb, _, _, _ = cam.render()
    print(f"Robot render: {rgb.shape}  dtype={rgb.dtype}")
    return rgb


def chroma_key(rgb):
    """Return RGBA uint8 where green pixels become transparent."""
    r = rgb[:, :, 0].astype(np.int32)
    g = rgb[:, :, 1].astype(np.int32)
    b = rgb[:, :, 2].astype(np.int32)

    # A pixel is "green screen" when green is dominant and bright.
    is_bg = (g > 80) & (g > r * 1.5) & (g > b * 1.5)

    alpha = np.where(is_bg, 0, 255).astype(np.uint8)
    rgba = np.dstack([rgb, alpha])
    return rgba


def composite(bg_path: Path, robot_rgba: np.ndarray,
              paste_x: float, paste_y: float, robot_scale: float,
              output_path: Path, crop: tuple | None = None) -> None:
    from PIL import Image as PILImage2

    bg_img = PILImage2.open(bg_path).convert("RGB")
    orig_W, orig_H = bg_img.size
    if crop is not None:
        cl, ct, cr, cb = crop
        box = (int(cl * orig_W), int(ct * orig_H),
               int(cr * orig_W), int(cb * orig_H))
        bg_img = bg_img.crop(box).resize((orig_W, orig_H), PILImage2.LANCZOS)
    bg = np.array(bg_img)
    H, W = bg.shape[:2]

    # Crop out empty border rows/cols from robot render (tight crop around robot).
    alpha = robot_rgba[:, :, 3]
    rows = np.any(alpha > 0, axis=1)
    cols = np.any(alpha > 0, axis=0)
    if not rows.any():
        print(f"WARNING: robot mask is empty for {bg_path.name}, skipping")
        return
    r0, r1 = np.where(rows)[0][[0, -1]]
    c0, c1 = np.where(cols)[0][[0, -1]]
    robot_crop = robot_rgba[r0:r1+1, c0:c1+1]
    rh, rw = robot_crop.shape[:2]

    # Scale robot so its height = robot_scale * image height
    target_h = int(H * robot_scale)
    scale = target_h / rh
    target_w = int(rw * scale)
    from PIL import Image as PILImage
    robot_img = PILImage.fromarray(robot_crop).resize(
        (target_w, target_h), PILImage.LANCZOS
    )
    robot_arr = np.array(robot_img)

    # Placement: paste_x is horizontal center, paste_y is bottom edge (0=top, 1=bottom)
    x_center = int(W * paste_x)
    y_bottom = int(H * paste_y)
    x0 = x_center - target_w // 2
    y0 = y_bottom - target_h

    # Clamp to image bounds
    rx0 = max(0, -x0);        x0 = max(0, x0)
    ry0 = max(0, -y0);        y0 = max(0, y0)
    x1 = min(W, x0 + target_w - rx0)
    y1 = min(H, y0 + target_h - ry0)
    rx1 = rx0 + (x1 - x0)
    ry1 = ry0 + (y1 - y0)

    if x1 <= x0 or y1 <= y0:
        print(f"WARNING: robot paste out of bounds for {bg_path.name}")
        return

    robot_patch = robot_arr[ry0:ry1, rx0:rx1]
    a = robot_patch[:, :, 3:4].astype(np.float32) / 255.0
    bg_patch    = bg[y0:y1, x0:x1].astype(np.float32)
    robot_rgb   = robot_patch[:, :, :3].astype(np.float32)
    blended     = (a * robot_rgb + (1 - a) * bg_patch).clip(0, 255).astype(np.uint8)
    bg[y0:y1, x0:x1] = blended

    output_path.parent.mkdir(parents=True, exist_ok=True)
    PILImage2.fromarray(bg).save(str(output_path))
    print(f"Saved composite: {output_path}")


def main():
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[2]

    W, H = args.render_res
    rgb = render_robot_greenscreen(
        azimuth=args.azimuth,
        elevation=args.elevation,
        fov=args.fov,
        robot_side=args.robot_side,
        width=W,
        height=H,
        dist=args.dist,
    )

    # Save the raw robot render and RGBA mask for debugging
    from PIL import Image
    robot_render_path = repo_root / "outputs" / "teaser" / "robot_render.png"
    Image.fromarray(rgb).save(str(robot_render_path))
    print(f"Raw robot render saved: {robot_render_path}")

    robot_rgba = chroma_key(rgb)
    robot_rgba_path = repo_root / "outputs" / "teaser" / "robot_rgba.png"
    Image.fromarray(robot_rgba, mode="RGBA").save(str(robot_rgba_path))
    print(f"Robot RGBA saved: {robot_rgba_path}")

    out_dir = args.output_dir if args.output_dir.is_absolute() else repo_root / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    for bg_path in args.backgrounds:
        bg_path = bg_path if bg_path.is_absolute() else repo_root / bg_path
        out_name = f"composite_{bg_path.stem}.png"
        composite(
            bg_path=bg_path,
            robot_rgba=robot_rgba,
            paste_x=args.paste_x,
            paste_y=args.paste_y,
            robot_scale=args.robot_scale,
            output_path=out_dir / out_name,
            crop=tuple(args.crop) if args.crop else None,
        )

    print("Done.")


if __name__ == "__main__":
    main()
