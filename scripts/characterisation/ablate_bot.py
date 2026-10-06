"""CPU stage of the Botrytis ablations: one flight, every condition.

Downloads the capture zip once, builds each condition's frames (botrytis_compose modes, plus
the green single-band control), runs the fixed COLMAP configuration per condition, undistorts
the gate-passing models, and stages what the GPU stage needs under
<out_root>/<condition>/_stage/. Registration rows are written for every condition.

usage: ablate_bot.py <flight> <work_dir> <out_root> [--threads N] [--conditions a,b,...]
"""

import argparse
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from botrytis_compose import _read, compose_list  # noqa: E402
from common import download, write_json, zenodo_url  # noqa: E402

GATE = 0.5  # train only when the largest model holds at least this share of the frames


def green_frames(zf: zipfile.ZipFile, stems: list[str], out: Path) -> None:
    """The single-band control: green band scaled to 8 bit by the 99.9th percentile of the
    sampled frames, exactly as the characterisation (and quality) green rows were built."""
    out.mkdir(parents=True, exist_ok=True)
    raws = [_read(zf, s, 2) for s in stems]
    scale = float(np.percentile(np.concatenate([r.ravel()[::16] for r in raws]), 99.9))
    for i, r in enumerate(raws):
        img = np.clip(r.astype(np.float32) / scale * 255.0, 0, 255).astype(np.uint8)
        cv2.imwrite(str(out / f"frame_{i:06d}.png"), img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("flight", choices=["0_V1", "30_V1", "45_V1", "0_V2"])
    ap.add_argument("work", type=Path)
    ap.add_argument("out_root", type=Path)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--conditions", default=None, help="comma list; default = the agreed set")
    a = ap.parse_args()

    conditions = (a.conditions.split(",") if a.conditions else
                  ["naive", "align_median", "align_perframe", "radiometry_only",
                   "perband_wb", "full", "full_nocrop", "green"]
                  + (["falsecolor"] if a.flight == "45_V1" else []))
    repo = HERE.parent.parent
    base_list = repo / "results/characterisation/frame_lists" / f"bot_{a.flight}.txt"
    stems = [ln.split("\t")[1].rsplit("_", 1)[0]
             for ln in base_list.read_text().splitlines() if ln.strip()]

    zip_path = download(zenodo_url(7383601, f"{a.flight}.zip"), a.work / "src" / f"{a.flight}.zip")
    zf = zipfile.ZipFile(zip_path)
    summary = {}
    for cond in conditions:
        cid = f"bot_{a.flight}_c_{cond}"
        wdir = a.work / cond
        out = a.out_root / cond
        out.mkdir(parents=True, exist_ok=True)
        print(f"== condition {cond}", flush=True)
        if cond == "green":
            green_frames(zf, stems, wdir / "images")
            report = dict(mode="green", n_frames=len(stems),
                          note="band 2 only, joint 99.9th percentile scale")
        else:
            report = compose_list(zip_path, wdir / "images", base_list, cond, None)
        write_json(out / "compose_report.json", report)
        write_json(wdir / "meta.json",
                   dict(id=cid, dataset="bot", sequence_id=f"{a.flight}_c_{cond}",
                        population="ablation", n_images=len(stems),
                        camera_mode="SINGLE", notes=[]))
        res = wdir / "res"
        r = subprocess.run([sys.executable, str(HERE / "register.py"), str(wdir), str(res),
                            "--threads", str(a.threads)])
        reg_path = res / "per_sequence" / f"{cid}.registration.json"
        if r.returncode != 0 or not reg_path.exists():
            summary[cond] = dict(outcome="register_failed")
            write_json(a.out_root / "cpu_stage_summary.json", summary)
            continue
        reg = json.loads(reg_path.read_text())
        shutil.copy(reg_path, out / "registration.json")
        pose = res / "poses" / f"{cid}.csv"
        if pose.exists():
            shutil.copy(pose, out / "poses.csv")
        largest = max(reg.get("model_sizes") or [0])
        gate = largest / reg["n_images"] >= GATE
        summary[cond] = dict(n_registered=reg.get("n_registered"),
                             model_sizes=reg.get("model_sizes"), gate_pass=gate)
        if gate and reg.get("best_sparse_dir") and Path(reg["best_sparse_dir"]).is_dir():
            stage = out / "_stage"
            shutil.rmtree(stage, ignore_errors=True)
            subprocess.run([sys.executable, str(HERE / "undistort.py"), reg["best_sparse_dir"],
                            str(wdir / "images"), str(stage)], check=True)
            # the GPU stage needs only images/ and sparse/
            for extra in ("stereo", "run-colmap-geometric.sh", "run-colmap-photometric.sh"):
                p = stage / extra
                shutil.rmtree(p, ignore_errors=True) if p.is_dir() else p.unlink(missing_ok=True)
        shutil.rmtree(wdir, ignore_errors=True)
        write_json(a.out_root / "cpu_stage_summary.json", summary)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
