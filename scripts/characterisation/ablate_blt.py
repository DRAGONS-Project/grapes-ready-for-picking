"""CPU stage of one BLT masking condition: frames, the mask set, the COLMAP gate, staging.

Mask kinds (all white = keep):
    fixed_region  the author's UK spec (results/masks/hand/uk_front.png, one mask for all
                  frames - the chassis is rigid in the frame)
    greek         the author-approved generous Greek regions (results/masks/hand/greek_front.png)
    hsv           the author's historical HSV thresholds (synthetic-grapes commit 458ff88),
                  applied per frame by the ORIGINAL ColorMaskModel implementation

usage: ablate_blt.py <seq_index> <condition> <mask_kind> <sfm_mask 0|1> <loss_mask 0|1>
                     <work_dir> <out_root> [--threads N]
Stages <out_root>/<id>/<condition>/_stage/{images,sparse[,loss_masks]} when the gate passes.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from common import write_json  # noqa: E402
from sequences import SEQUENCES  # noqa: E402

REPO = HERE.parent.parent
GATE = 0.5

# the committed HSV spec, synthetic-grapes 458ff88 configs/model/color_mask_preprocessing.yaml
HSV_COLORS = {
    "white panel": {"lower": [0, 0, 205], "upper": [179, 40, 255]},
    "red object": {"lower": [0, 80, 40], "upper": [20, 255, 255],
                   "lower2": [150, 80, 40], "upper2": [179, 255, 255]},
    "black tube": {"lower": [0, 0, 0], "upper": [179, 80, 30]},
}
HSV_REGIONS = [{"x1": 0.88, "y1": 0.82, "x2": 1.0, "y2": 1.0}]


def run(cmd, **kw):
    print("+", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, **kw)


def build_masks(kind: str, images: Path, masks: Path, masks_colmap: Path):
    masks.mkdir(parents=True, exist_ok=True)
    masks_colmap.mkdir(parents=True, exist_ok=True)
    names = sorted(p.name for p in images.iterdir())
    if kind in ("fixed_region", "greek"):
        src = REPO / "results/masks/hand" / \
            ("uk_front.png" if kind == "fixed_region" else "greek_front.png")
        keep = cv2.imread(str(src), cv2.IMREAD_GRAYSCALE)
        first = cv2.imread(str(images / names[0]))
        if keep.shape != first.shape[:2]:
            keep = cv2.resize(keep, first.shape[1::-1], interpolation=cv2.INTER_NEAREST)
        for n in names:
            cv2.imwrite(str(masks / n), keep)
    elif kind == "hsv":
        sys.path.insert(0, "/home/grid/users/plgmwlodarzc/synthetic-grapes/src")
        from model.color_mask import ColorMaskModel
        model = ColorMaskModel(colors=HSV_COLORS, morph_kernel_size=25,
                               always_mask_regions=HSV_REGIONS, bridge_kernel_size=120)
        for n in names:
            bgr = cv2.imread(str(images / n))
            remove = model.forward(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))["combined"]
            cv2.imwrite(str(masks / n), ((~remove.astype(bool)) * 255).astype(np.uint8))
    else:
        raise SystemExit(f"unknown mask kind {kind}")
    shares = []
    for n in names:
        m = cv2.imread(str(masks / n), cv2.IMREAD_GRAYSCALE)
        shares.append(1.0 - (m > 127).mean())
        dst = masks_colmap / f"{n}.png"
        if not dst.exists():
            os.link(masks / n, dst)
    return dict(mask_kind=kind, n_masks=len(names),
                masked_share_mean=round(float(np.mean(shares)), 4),
                masked_share_max=round(float(np.max(shares)), 4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("index", type=int)
    ap.add_argument("condition")
    ap.add_argument("mask_kind")
    ap.add_argument("sfm_mask", type=int)
    ap.add_argument("loss_mask", type=int)
    ap.add_argument("work", type=Path)
    ap.add_argument("out_root", type=Path)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()

    seq = SEQUENCES[a.index]
    cid = f"{seq['id']}_c_{a.condition}"
    out = a.out_root / seq["id"] / a.condition
    out.mkdir(parents=True, exist_ok=True)
    py = sys.executable

    base = a.work / "base"
    run([py, HERE / "prepare.py", a.index, base, a.work / "res"])
    images = base / "images"
    masks, masks_colmap = a.work / "masks", a.work / "masks_colmap"
    report = build_masks(a.mask_kind, images, masks, masks_colmap)
    report.update(sfm_mask=bool(a.sfm_mask), loss_mask=bool(a.loss_mask))
    write_json(out / "mask_report.json", report)
    print(json.dumps(report), flush=True)

    wdir = a.work / "c"
    (wdir / "images").mkdir(parents=True, exist_ok=True)
    for n in sorted(p.name for p in images.iterdir()):
        dst = wdir / "images" / n
        if not dst.exists():
            os.link(images / n, dst)
    write_json(wdir / "meta.json", dict(id=cid, dataset=seq["dataset"],
                                        sequence_id=f"{seq['sequence_id']}_c_{a.condition}",
                                        population="ablation", n_images=report["n_masks"],
                                        camera_mode="SINGLE", notes=[]))
    cmd = [py, HERE / "register.py", wdir, wdir / "res", "--threads", a.threads]
    if a.sfm_mask:
        cmd += ["--sfm-masks", masks_colmap]
    run(cmd)
    reg_path = wdir / "res" / "per_sequence" / f"{cid}.registration.json"
    reg = json.loads(reg_path.read_text())
    shutil.copy(reg_path, out / "registration.json")
    pose = wdir / "res" / "poses" / f"{cid}.csv"
    if pose.exists():
        shutil.copy(pose, out / "poses.csv")
    largest = max(reg.get("model_sizes") or [0])
    gate = largest / reg["n_images"] >= GATE
    print(json.dumps(dict(condition=a.condition, n_registered=reg.get("n_registered"),
                          model_sizes=reg.get("model_sizes"), gate_pass=gate)), flush=True)
    if not (gate and reg.get("best_sparse_dir") and Path(reg["best_sparse_dir"]).is_dir()):
        print("gate failed: registration row only")
        return
    stage = out / "_stage"
    shutil.rmtree(stage, ignore_errors=True)
    run([py, HERE / "undistort.py", reg["best_sparse_dir"], wdir / "images", stage])
    if a.loss_mask:
        mu = a.work / "mu"
        run([py, HERE / "undistort.py", reg["best_sparse_dir"], masks, mu])
        shutil.move(str(mu / "images"), str(stage / "loss_masks"))
    for extra in ("stereo", "run-colmap-geometric.sh", "run-colmap-photometric.sh"):
        p = stage / extra
        shutil.rmtree(p, ignore_errors=True) if p.is_dir() else p.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
