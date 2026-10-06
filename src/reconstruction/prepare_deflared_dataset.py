"""Build a nerfstudio data directory pointing at dark-channel-deflared frames
(see deflare_dark_channel.py) instead of the raw MOTS NoPathPlanning_1 images.

Reuses the *same* transforms.json (camera poses, intrinsics, eval split) as
ns_data/ns_data_masked -- only the images/ directory changes, to isolate the
pixel-correction effect from any pose/eval-split confound (same approach as
prepare_masked_dataset.py, minus the mask_path field: this arm corrects pixels
rather than excluding them, so no mask is needed).

Usage:
    uv run --extra recon python src/reconstruction/prepare_deflared_dataset.py \
        <src_ns_data_dir> <deflared_images_dir> <dst_ns_data_dir>
"""

import argparse
import json
import shutil
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("src_ns_data", type=Path)
    p.add_argument("deflared_images_dir", type=Path)
    p.add_argument("dst_ns_data", type=Path)
    args = p.parse_args()

    src, deflared, dst = args.src_ns_data, args.deflared_images_dir, args.dst_ns_data
    dst.mkdir(parents=True, exist_ok=True)

    images_dst = dst / "images"
    if images_dst.exists():
        shutil.rmtree(images_dst) if images_dst.is_dir() and not images_dst.is_symlink() else images_dst.unlink()
    images_dst.symlink_to(deflared.resolve())

    transforms = json.loads((src / "transforms.json").read_text())
    ply_src = (src / transforms["ply_file_path"]).resolve()
    ply_dst = dst / transforms["ply_file_path"]
    ply_dst.parent.mkdir(parents=True, exist_ok=True)
    if not ply_dst.exists():
        ply_dst.symlink_to(ply_src)

    missing = [f["file_path"] for f in transforms["frames"] if not (deflared / Path(f["file_path"]).name).exists()]
    if missing:
        raise RuntimeError(f"{len(missing)} frames missing from {deflared}, e.g. {missing[:3]}")

    (dst / "transforms.json").write_text(json.dumps(transforms, indent=2))
    print(f"{len(transforms['frames'])} frames. Deflared dataset ready at {dst} (images -> {deflared})")


if __name__ == "__main__":
    main()
