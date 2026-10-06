"""Run COLMAP structure-from-motion on a directory of images via pycolmap."""

import argparse
from pathlib import Path

import pycolmap

from config import load_config


def run_sfm(image_dir: Path, output_dir: Path, camera_model: str = "OPENCV") -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    database_path = output_dir / "database.db"
    sparse_dir = output_dir / "sparse"
    sparse_dir.mkdir(parents=True, exist_ok=True)

    pycolmap.extract_features(
        database_path, image_dir, reader_options=pycolmap.ImageReaderOptions(camera_model=camera_model)
    )
    pycolmap.match_exhaustive(database_path)

    reconstructions = pycolmap.incremental_mapping(database_path, image_dir, sparse_dir)
    if not reconstructions:
        raise RuntimeError("COLMAP reconstruction failed: no models produced")

    # Keep the reconstruction with the most registered images.
    best_idx = max(reconstructions, key=lambda i: reconstructions[i].num_reg_images())
    best = reconstructions[best_idx]
    best_dir = sparse_dir / str(best_idx)
    print(
        f"Reconstruction {best_idx}: {best.num_reg_images()} images registered, "
        f"{best.num_points3D()} points"
    )
    (output_dir / "best_sparse_dir.txt").write_text(str(best_dir))
    return best_dir


def main():
    config = load_config()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image_dir", type=Path, help="Directory of input images")
    parser.add_argument(
        "output_dir", type=Path, help="Directory for the COLMAP database and sparse model"
    )
    parser.add_argument(
        "--camera-model", default=config.colmap.camera_model, help="COLMAP camera model"
    )
    args = parser.parse_args()

    sparse_dir = run_sfm(args.image_dir, args.output_dir, args.camera_model)
    print(f"Sparse model written to {sparse_dir}")


if __name__ == "__main__":
    main()
