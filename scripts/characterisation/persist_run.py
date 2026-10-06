"""Keep the small outputs of one training run: metrics, held-out renders (JPEG q90), a summary.

usage: persist_run.py <train_dir> <out_dir> <registration.json> [--renders all|pred|none]
       [--gpu-mem-log file] [--set k=v ...]
Checkpoints and point clouds stay in the job's scratch directory and are discarded with it.
"""

import argparse
import json
import shutil
from pathlib import Path

from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("train_dir", type=Path)
ap.add_argument("out_dir", type=Path)
ap.add_argument("registration", type=Path)
ap.add_argument("--renders", choices=["all", "pred", "none"], default="all")
ap.add_argument("--gpu-mem-log", type=Path)
ap.add_argument("--set", nargs="*", default=[], help="extra summary fields, key=value (JSON)")
a = ap.parse_args()

a.out_dir.mkdir(parents=True, exist_ok=True)
for f in (a.train_dir / "metrics.json", a.train_dir / "eval" / "metrics_full.json",
          a.train_dir / "eval" / "metrics_common.json"):
    if f.exists():
        shutil.copy(f, a.out_dir / f.name)

if a.renders != "none":
    (a.out_dir / "eval_renders").mkdir(exist_ok=True)
    for p in sorted((a.train_dir / "eval").glob("*.png")):
        if a.renders == "pred" and not p.name.startswith("pred_"):
            continue
        name = p.stem.replace(".jpg", "").replace(".png", "") + ".jpg"
        Image.open(p).convert("RGB").save(a.out_dir / "eval_renders" / name, quality=90)

n_gauss = None
ply = a.train_dir / "point_cloud.ply"
if ply.exists():
    with open(ply, "rb") as fh:
        for line in fh:
            if line.startswith(b"element vertex"):
                n_gauss = int(line.split()[-1])
                break
            if line.startswith(b"end_header"):
                break

mem = []
if a.gpu_mem_log and a.gpu_mem_log.exists():
    mem = [int(x) for x in a.gpu_mem_log.read_text().split() if x.strip().isdigit()]

reg = json.loads(a.registration.read_text())
full_path = a.train_dir / "eval" / "metrics_full.json"
full = json.loads(full_path.read_text()) if full_path.exists() else {}
# overlap of the held-out views with the training views, written by the trainer
own_path = a.train_dir / "metrics.json"
own = json.loads(own_path.read_text()) if own_path.exists() else {}
extra = {k: own[k] for k in ("n_test_views", "nn_dist_frac", "nn_angle_deg") if k in own}
for kv in a.set:
    k, v = kv.split("=", 1)
    try:
        extra[k] = json.loads(v)
    except json.JSONDecodeError:
        extra[k] = v

summary = dict(
    id=reg["id"],
    trainer="src/reconstruction/train_gs.py (gsplat)",
    n_images=reg["n_images"],
    n_registered=reg.get("n_registered"),
    model_sizes=reg.get("model_sizes"),
    t_extract_s=reg.get("t_extract"),
    t_match_s=reg.get("t_match"),
    t_map_s=reg.get("t_map"),
    n_gaussians=n_gauss,
    gpu_mem_peak_mib=max(mem) if mem else None,
    **extra,
    **{k: v for k, v in full.items() if not isinstance(v, list)},
)
summary["disk_kept_mb"] = round(
    sum(f.stat().st_size for f in a.out_dir.rglob("*") if f.is_file()) / 1e6, 1
)
(a.out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
print(json.dumps(summary))
