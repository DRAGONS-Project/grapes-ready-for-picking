"""Build a nerfstudio data directory combining dark-channel-deflared frames
(see deflare_dark_channel.py) with pre-generated lens-ghost masks (see
generate_ghost_masks.py): images/ points at the deflared frames, and each
frame gets a mask_path excluding its ghost region from the training loss.

Deflaring fixes the whole-frame veiling-glare haze but not the lens ghost (a
discrete secondary reflection outside the dark-channel-prior's model, and
actually made *more* visible by dehaze's contrast/saturation boost -- see
docs/flare_fix_research.md). This arm layers ghost-region exclusion on top of
the already-deflared images to also address that, the same way
prepare_masked_dataset.py layered veiling-glare exclusion onto the raw
baseline.

Reuses the *same* transforms.json (camera poses, intrinsics, eval split) as
ns_data/ns_data_masked/ns_data_deflare -- only images/ and the added
mask_path change, to isolate this arm's effect from any pose/eval-split
confound.

Usage:
    uv run --extra recon python src/reconstruction/prepare_deflared_ghostmasked_dataset.py \
        <src_ns_data_dir> <deflared_images_dir> <ghost_masks_dir> <dst_ns_data_dir>
"""

import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("src_ns_data", type=Path)
    p.add_argument("deflared_images_dir", type=Path)
    p.add_argument("ghost_masks_dir", type=Path)
    p.add_argument("dst_ns_data", type=Path)
    args = p.parse_args()

    src, deflared, ghost_masks, dst = args.src_ns_data, args.deflared_images_dir, args.ghost_masks_dir, args.dst_ns_data
    dst.mkdir(parents=True, exist_ok=True)

    images_dst = dst / "images"
    if images_dst.is_symlink() or images_dst.exists():
        images_dst.unlink()
    images_dst.symlink_to(deflared.resolve())

    masks_dst = dst / "masks"
    if masks_dst.is_symlink() or masks_dst.exists():
        masks_dst.unlink()
    masks_dst.symlink_to(ghost_masks.resolve())

    transforms = json.loads((src / "transforms.json").read_text())
    ply_src = (src / transforms["ply_file_path"]).resolve()
    ply_dst = dst / transforms["ply_file_path"]
    ply_dst.parent.mkdir(parents=True, exist_ok=True)
    if not ply_dst.exists():
        ply_dst.symlink_to(ply_src)

    missing_img = [f["file_path"] for f in transforms["frames"] if not (deflared / Path(f["file_path"]).name).exists()]
    if missing_img:
        raise RuntimeError(f"{len(missing_img)} frames missing from {deflared}, e.g. {missing_img[:3]}")

    for frame in transforms["frames"]:
        stem = Path(frame["file_path"]).stem
        mask_name = stem + ".png"
        if not (ghost_masks / mask_name).exists():
            raise RuntimeError(f"Missing ghost mask for {stem} at {ghost_masks / mask_name}")
        frame["mask_path"] = f"./masks/{mask_name}"

    (dst / "transforms.json").write_text(json.dumps(transforms, indent=2))
    print(f"{len(transforms['frames'])} frames. Deflared+ghost-masked dataset ready at {dst} "
          f"(images -> {deflared}, masks -> {ghost_masks})")


if __name__ == "__main__":
    main()
