"""Undistort a COLMAP model and its images to a pinhole camera (COLMAP's image undistorter).

usage: undistort.py <sparse_dir> <image_dir> <output_dir>
Writes <output_dir>/images and <output_dir>/sparse; prints the sparse path. Image names are kept.
"""

import sys
from pathlib import Path

import pycolmap

sparse, images, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
pycolmap.undistort_images(output_path=out, input_path=sparse, image_path=images)
rec = pycolmap.Reconstruction(str(out / "sparse"))
cam = next(iter(rec.cameras.values()))
params = [round(float(p), 2) for p in cam.params]
print(
    f"undistorted {rec.num_reg_images()} images -> {cam.model.name} {cam.width}x{cam.height}, "
    f"params {params}",
    file=sys.stderr,
)
print(out / "sparse")
