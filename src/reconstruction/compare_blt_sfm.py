"""Compare COLMAP SfM on the raw (unmasked) vs. preprocessed (masked) BLT frames.

Runs SfM on the raw BLT images (robot body visible) and exports its sparse
point cloud, alongside the point cloud from the already-existing preprocessed
SfM reconstruction — for a side-by-side "SfM geometry" comparison figure.
"""

import argparse
import json
from pathlib import Path

import pycolmap

from sfm import run_sfm


def export_summary(recon: pycolmap.Reconstruction, ply_path: Path) -> dict:
    recon.export_PLY(str(ply_path))
    return {
        "num_reg_images": recon.num_reg_images(),
        "num_points3D": recon.num_points3D(),
        "mean_reprojection_error": recon.compute_mean_reprojection_error(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_image_dir", type=Path, help="Directory of raw (unmasked) BLT images")
    parser.add_argument("preprocessed_sparse_dir", type=Path, help="Existing preprocessed COLMAP sparse/0 dir")
    parser.add_argument("work_dir", type=Path, help="Scratch dir for raw COLMAP database/sparse output")
    parser.add_argument("output_dir", type=Path, help="Directory to write comparison point clouds + summary")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # --- Raw: run SfM from scratch ---
    raw_sparse_dir = run_sfm(args.raw_image_dir, args.work_dir, camera_model="OPENCV")
    raw_recon = pycolmap.Reconstruction(str(raw_sparse_dir))
    raw_summary = export_summary(raw_recon, args.output_dir / "blt_raw_sparse.ply")

    # --- Preprocessed: existing SfM reconstruction ---
    prep_recon = pycolmap.Reconstruction(str(args.preprocessed_sparse_dir))
    prep_summary = export_summary(prep_recon, args.output_dir / "blt_preprocessed_sparse.ply")

    summary = {"raw": raw_summary, "preprocessed": prep_summary}
    with open(args.output_dir / "sfm_comparison.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
