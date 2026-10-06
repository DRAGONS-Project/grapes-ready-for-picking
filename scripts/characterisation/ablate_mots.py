"""CPU part of the MOTS ablations for one sequence: frames, dehaze variants, ghost masks,
the COLMAP gate per condition, undistortion of images and masks, staging for the training loop.

Conditions: raw | dehaze | ghost | dehaze_ghost | dehaze_bright | dehaze_noguided.
Masks are white = keep. Ghost conditions use the mask in SfM and in the loss; every condition
gets the same mask undistorted with its own model as the common-pixel scoring mask (the union
of exclusions over all conditions is the ghost mask, since nothing else masks).

usage: ablate_mots.py <sequence index> <work_dir> <out_root> [--threads N]
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from common import write_json  # noqa: E402
from sequences import SEQUENCES  # noqa: E402

GATE = 0.5
CONDITIONS = {  # name -> (deflare args or None, use masks)
    "raw": (None, False),
    "dehaze": ([], False),
    "ghost": (None, True),
    "dehaze_ghost": ([], True),
    "dehaze_bright": (["--atmo", "brightest"], False),
    "dehaze_noguided": (["--no-guided"], False),
}


def run(cmd, **kw):
    print("+", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("index", type=int)
    ap.add_argument("work", type=Path)
    ap.add_argument("out_root", type=Path)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()

    seq = SEQUENCES[a.index]
    repo = HERE.parent.parent
    py = sys.executable
    base = a.work / "base"
    run([py, HERE / "prepare.py", a.index, base, a.work / "res"])
    images = base / "images"
    names = sorted(p.name for p in images.iterdir())

    # the three dehaze variants, computed once and shared by their conditions
    variants = {"": images}
    for tag, args in (("dehaze", []), ("dehaze_bright", ["--atmo", "brightest"]),
                      ("dehaze_noguided", ["--no-guided"])):
        out = a.work / f"img_{tag}"
        run([py, repo / "src/reconstruction/deflare_dark_channel.py", images, out, *args])
        variants[tag] = out

    # ghost masks (white = keep), named like the images; doubled names for COLMAP's convention
    masks = a.work / "masks"
    run([py, repo / "src/reconstruction/generate_ghost_masks.py", images, masks])
    masks_colmap = a.work / "masks_colmap"
    masks_colmap.mkdir(exist_ok=True)
    for n in names:
        dst = masks_colmap / f"{n}.png"
        if not dst.exists():
            os.link(masks / n, dst)

    summary = {}
    for cond, (deflare_args, use_masks) in CONDITIONS.items():
        cid = f"{seq['id']}_c_{cond}"
        out = a.out_root / cond
        out.mkdir(parents=True, exist_ok=True)
        wdir = a.work / f"c_{cond}"
        (wdir / "images").mkdir(parents=True, exist_ok=True)
        variant_of = {"raw": "", "ghost": "", "dehaze": "dehaze", "dehaze_ghost": "dehaze",
                      "dehaze_bright": "dehaze_bright", "dehaze_noguided": "dehaze_noguided"}
        src = variants[variant_of[cond]]
        for n in names:
            dst = wdir / "images" / n
            if not dst.exists():
                os.link(src / n, dst)
        write_json(wdir / "meta.json", dict(id=cid, dataset=seq["dataset"],
                                            sequence_id=f"{seq['sequence_id']}_c_{cond}",
                                            population="ablation", n_images=len(names),
                                            camera_mode="SINGLE", notes=[]))
        cmd = [py, HERE / "register.py", wdir, wdir / "res", "--threads", a.threads]
        if use_masks:
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
        summary[cond] = dict(n_registered=reg.get("n_registered"),
                             model_sizes=reg.get("model_sizes"), gate_pass=gate,
                             sfm_masked=use_masks)
        write_json(a.out_root / "cpu_stage_summary.json", summary)
        if not (gate and reg.get("best_sparse_dir") and Path(reg["best_sparse_dir"]).is_dir()):
            continue
        stage = out / "_stage"
        shutil.rmtree(stage, ignore_errors=True)
        run([py, HERE / "undistort.py", reg["best_sparse_dir"], wdir / "images", stage])
        # masks follow the frames through the same undistortion (this condition's model)
        mu = a.work / f"mu_{cond}"
        run([py, HERE / "undistort.py", reg["best_sparse_dir"], masks, mu])
        shutil.move(str(mu / "images"), str(stage / "common_masks"))
        if use_masks:
            shutil.copytree(stage / "common_masks", stage / "loss_masks")
        for extra in ("stereo", "run-colmap-geometric.sh", "run-colmap-photometric.sh"):
            p = stage / extra
            shutil.rmtree(p, ignore_errors=True) if p.is_dir() else p.unlink(missing_ok=True)
        shutil.rmtree(wdir, ignore_errors=True)
        shutil.rmtree(mu, ignore_errors=True)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
