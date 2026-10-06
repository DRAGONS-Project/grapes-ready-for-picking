"""Convert a pycolmap sparse reconstruction into a Nerfstudio dataset
(transforms.json + sparse_pc.ply + images/), so it can be used with
`ns-train splatfacto`.

This bypasses `ns-process-data`, which hard-requires the COLMAP CLI binary
even with --skip-colmap. The conversion replicates the coordinate-convention
logic of nerfstudio.process_data.colmap_utils.colmap_to_json().
"""

import argparse
import shutil
from pathlib import Path

import numpy as np
import pycolmap


def parse_camera_params(camera: pycolmap.Camera) -> dict:
    if camera.model_name != "OPENCV":
        raise ValueError(f"Unsupported camera model: {camera.model_name}")
    fx, fy, cx, cy, k1, k2, p1, p2 = camera.params
    return {
        "w": camera.width,
        "h": camera.height,
        "fl_x": float(fx),
        "fl_y": float(fy),
        "cx": float(cx),
        "cy": float(cy),
        "k1": float(k1),
        "k2": float(k2),
        "p1": float(p1),
        "p2": float(p2),
        "camera_model": "OPENCV",
    }


def write_sparse_pc(recon: pycolmap.Reconstruction, applied_transform: np.ndarray, out_path: Path) -> None:
    points = recon.points3D.values()
    xyz = np.array([p.xyz for p in points], dtype=np.float32)
    rgb = np.array([p.color for p in points], dtype=np.uint8)
    xyz = xyz @ applied_transform[:3, :3].T + applied_transform[:3, 3]

    with open(out_path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(xyz)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uint8 red\n")
        f.write("property uint8 green\n")
        f.write("property uint8 blue\n")
        f.write("end_header\n")
        for (x, y, z), (r, g, b) in zip(xyz, rgb, strict=True):
            f.write(f"{x} {y} {z} {r} {g} {b}\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sparse_dir", type=Path)
    parser.add_argument("images_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    recon = pycolmap.Reconstruction(args.sparse_dir)
    cameras = recon.cameras
    single_camera = len(cameras) == 1
    out = parse_camera_params(next(iter(cameras.values()))) if single_camera else {}

    frames = []
    for image in recon.images.values():
        cam_from_world = image.cam_from_world()
        w2c = np.eye(4)
        w2c[:3, :3] = cam_from_world.rotation.matrix()
        w2c[:3, 3] = cam_from_world.translation

        c2w = np.linalg.inv(w2c)
        # COLMAP (OpenCV) -> Nerfstudio (OpenGL) camera convention
        c2w[0:3, 1:3] *= -1
        # COLMAP world -> Nerfstudio world (z-up) convention
        c2w = c2w[np.array([0, 2, 1, 3]), :]
        c2w[2, :] *= -1

        frame = {
            "file_path": f"./images/{image.name}",
            "transform_matrix": c2w.tolist(),
        }
        if not single_camera:
            frame.update(parse_camera_params(cameras[image.camera_id]))
        frames.append(frame)
    out["frames"] = frames

    applied_transform = np.eye(4)[:3, :]
    applied_transform = applied_transform[np.array([0, 2, 1]), :]
    applied_transform[2, :] *= -1
    out["applied_transform"] = applied_transform.tolist()
    out["ply_file_path"] = "sparse_pc.ply"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    images_out = args.output_dir / "images"
    images_out.mkdir(exist_ok=True)
    for image in recon.images.values():
        shutil.copy(args.images_dir / image.name, images_out / image.name)

    write_sparse_pc(recon, applied_transform, args.output_dir / "sparse_pc.ply")

    import json

    with open(args.output_dir / "transforms.json", "w") as f:
        json.dump(out, f, indent=2)

    print(f"Wrote {len(frames)} frames to {args.output_dir}")


if __name__ == "__main__":
    main()
