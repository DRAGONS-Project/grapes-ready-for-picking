"""Export a model.pt checkpoint to standard 3DGS PLY format.

The output is compatible with SuperSplat (https://supersplat.dev),
gsplat-viewer, and any other viewer that reads the original 3DGS .ply format.

Usage:
    uv run --extra recon python src/reconstruction/export_gs_ply.py \\
        outputs/reconstruction/mots_pathplanning_1/model.pt \\
        outputs/reconstruction/mots_pathplanning_1/gaussians.ply
"""

import argparse
import struct
from pathlib import Path

import numpy as np
import torch


# Zeroth-order SH coefficient: RGB color in [0,1] ↔ f_dc via
#   color = C0 * f_dc + 0.5  →  f_dc = (color - 0.5) / C0
C0 = 0.28209479177387814


def export(model_path: Path, output_path: Path) -> None:
    params = torch.load(model_path, map_location="cpu")

    means    = params["means"].numpy().astype(np.float32)       # (N, 3)
    scales   = params["scales"].numpy().astype(np.float32)      # (N, 3) log-scales
    quats    = params["quats"].numpy().astype(np.float32)       # (N, 4) wxyz
    opacities = params["opacities"].numpy().astype(np.float32)  # (N,)  logit
    colors   = params["colors"].numpy().astype(np.float32)      # (N, 3) [0,1]

    N = means.shape[0]
    print(f"Exporting {N:,} Gaussians …")

    # Colors → SH DC coefficients (f_dc_0/1/2).
    f_dc = (colors - 0.5) / C0                                  # (N, 3)

    # Higher-order SH coefficients: all zero (we didn't learn them).
    # Standard degree-3 SH has 15 coefficients per channel = 45 total.
    f_rest = np.zeros((N, 45), dtype=np.float32)

    # Property names exactly as expected by 3DGS viewers.
    properties = (
        ["x", "y", "z", "nx", "ny", "nz"]
        + ["f_dc_0", "f_dc_1", "f_dc_2"]
        + [f"f_rest_{i}" for i in range(45)]
        + ["opacity"]
        + ["scale_0", "scale_1", "scale_2"]
        + ["rot_0", "rot_1", "rot_2", "rot_3"]
    )

    normals = np.zeros((N, 3), dtype=np.float32)

    # Stack all columns in property order.
    data = np.concatenate([
        means,                          # x y z
        normals,                        # nx ny nz
        f_dc,                           # f_dc_0/1/2
        f_rest,                         # f_rest_0…44
        opacities[:, None],             # opacity
        scales,                         # scale_0/1/2
        quats,                          # rot_0/1/2/3  (w x y z)
    ], axis=1).astype(np.float32)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "wb") as f:
        # PLY header.
        header = (
            "ply\n"
            "format binary_little_endian 1.0\n"
            f"element vertex {N}\n"
        )
        for prop in properties:
            header += f"property float {prop}\n"
        header += "end_header\n"
        f.write(header.encode("ascii"))
        f.write(data.tobytes())

    size_mb = output_path.stat().st_size / 1e6
    print(f"Saved {output_path}  ({size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_pt", type=Path, help="Path to model.pt")
    parser.add_argument("output_ply", type=Path, help="Output .ply path")
    args = parser.parse_args()
    export(args.model_pt, args.output_ply)


if __name__ == "__main__":
    main()
