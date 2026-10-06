"""Build results/sweeps/sweeps.csv: one row per sweep run, plus each sequence's shared level.

usage: collect_sweeps.py
"""

import csv
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# both sweep sequences are UAV row passes on the same estate (NoPathPlanning_1: frontal,
# backlit, at harvest; Terras Gauda Row7.2_p3: 3 m AGL, 60 deg tilt, pre-harvest dense canopy)
SEQUENCE_KIND = {"mots_NoPathPlanning_1": "uav_row_pass", "btg_Row7.2_p3": "uav_row_pass"}

COLS = ["sequence", "sequence_kind", "sweep", "value", "sfm_width", "train_width",
        "fps_actual", "n_frames",
        "span_s", "n_registered", "n_models", "model_sizes", "tri_angle_median",
        "baseline_frac", "psnr", "ssim", "lpips", "psnr_at640", "ssim_at640", "lpips_at640",
        "nn_dist_frac", "nn_angle_deg", "n_gaussians", "t_train_s", "note"]


def jread(p):
    return json.loads(p.read_text()) if p.exists() else {}


def main():
    root = REPO / "results/sweeps"
    rows = []
    for seq_dir in sorted(p for p in root.iterdir() if p.is_dir() and p.name != "_mask_cache"):
        for run in sorted(p for p in seq_dir.iterdir() if p.is_dir()):
            meta, reg = jread(run / "meta.json"), jread(run / "registration.json")
            s, a = jread(run / "summary.json"), jread(run / "metrics_at640.json")
            note = jread(run / "README.json").get("note", "")
            if note:  # the shared level: metrics live in its quality row
                s = jread(REPO / jread(run / "README.json")["summary_json"])
                reg = {k: s.get(k) for k in ("n_registered", "model_sizes")}
                meta = dict(sweep_kind=run.name.split("_")[0],
                            sweep_value=run.name.split("_", 1)[1], sfm_width=1600,
                            train_width=(s.get("train_image_size") or [None])[0],
                            n_images=s.get("n_images"))
            sizes = reg.get("model_sizes") or []
            rows.append(dict(
                sequence=seq_dir.name, sequence_kind=SEQUENCE_KIND.get(seq_dir.name, ""),
                sweep=meta.get("sweep_kind"), value=meta.get("sweep_value"),
                sfm_width=meta.get("sfm_width"), train_width=meta.get("train_width"),
                fps_actual=meta.get("fps_actual"), n_frames=meta.get("n_images"),
                span_s=meta.get("span_s"), n_registered=reg.get("n_registered"),
                n_models=len(sizes) or None, model_sizes=sizes,
                tri_angle_median=reg.get("tri_angle_median"),
                baseline_frac=reg.get("baseline_frac"),
                psnr=s.get("psnr_mean"), ssim=s.get("ssim_mean"), lpips=s.get("lpips_mean"),
                psnr_at640=a.get("psnr_mean"), ssim_at640=a.get("ssim_mean"),
                lpips_at640=a.get("lpips_mean"),
                nn_dist_frac=s.get("nn_dist_frac"), nn_angle_deg=s.get("nn_angle_deg"),
                n_gaussians=s.get("n_gaussians"), t_train_s=s.get("t_train_s"), note=note))
    with open(root / "sweeps.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} rows -> {root / 'sweeps.csv'}")


if __name__ == "__main__":
    main()
