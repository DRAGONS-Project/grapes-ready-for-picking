"""Build a nerfstudio data directory with per-frame flare masks wired in.

Copies an existing ns_data directory (images + sparse_pc.ply, via symlink) into
a new directory, generates flare/glare masks for every frame with
generate_flare_masks.flare_mask, and writes a transforms.json with a
"mask_path" entry added per frame (nerfstudio's NerfstudioDataParser convention:
see get_image_mask_tensor_from_path / nerfstudio_dataparser.py). The source
ns_data directory is left untouched so existing baseline/bilateral experiments
that read from it are unaffected.

Usage:
    uv run --extra recon python src/reconstruction/prepare_masked_dataset.py \
        <src_ns_data_dir> <dst_ns_data_dir>
"""

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

from generate_flare_masks import flare_mask


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("src_ns_data", type=Path)
    p.add_argument("dst_ns_data", type=Path)
    p.add_argument("--v-thresh", type=float, default=0.90)
    p.add_argument("--s-thresh", type=float, default=0.15)
    p.add_argument("--dilate-px", type=int, default=9)
    p.add_argument("--min-blob-area-frac", type=float, default=0.0008)
    args = p.parse_args()

    src, dst = args.src_ns_data, args.dst_ns_data
    dst.mkdir(parents=True, exist_ok=True)

    images_link = dst / "images"
    if not images_link.exists():
        images_link.symlink_to((src / "images").resolve())

    transforms = json.loads((src / "transforms.json").read_text())
    ply_src = (src / transforms["ply_file_path"]).resolve()
    ply_dst = dst / transforms["ply_file_path"]
    if not ply_dst.exists():
        ply_dst.symlink_to(ply_src)

    masks_dir = dst / "masks"
    masks_dir.mkdir(exist_ok=True)

    excluded_fracs = []
    for frame in transforms["frames"]:
        img_path = src / frame["file_path"]
        img = cv2.imread(str(img_path))
        if img is None:
            raise RuntimeError(f"Failed to read {img_path}")
        mask = flare_mask(img, args.v_thresh, args.s_thresh, args.dilate_px, args.min_blob_area_frac)
        mask_name = Path(frame["file_path"]).stem + ".png"
        cv2.imwrite(str(masks_dir / mask_name), mask)
        frame["mask_path"] = f"./masks/{mask_name}"
        excluded_fracs.append(1.0 - (mask > 0).mean())

    (dst / "transforms.json").write_text(json.dumps(transforms, indent=2))

    excluded_fracs = np.array(excluded_fracs)
    print(f"{len(transforms['frames'])} frames, masks written to {masks_dir}")
    print(f"Excluded fraction: mean {excluded_fracs.mean() * 100:.1f}%, "
          f"min {excluded_fracs.min() * 100:.1f}%, max {excluded_fracs.max() * 100:.1f}%")
    print(f"Masked dataset ready at {dst}")


if __name__ == "__main__":
    main()
