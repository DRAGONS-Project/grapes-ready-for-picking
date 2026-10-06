"""Build results/quality/quality.csv: one row per quality sequence, both held-out splits.

Scene groups say which rows look at the same scene content and are therefore not independent
samples (MOTS sequences share one plot; Greek BLT sessions share one vineyard; Terras Gauda
parts share their clip's row; Botrytis flights share their field, and the composed-RGB rows
share their green rows' captures).

usage: collect_quality.py [results_root]
"""

import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def scene_group(sid: str) -> str:
    if sid.startswith("mots_"):
        return "mots_plot"
    if sid.startswith("blt_gr_"):
        return "blt_gerovassiliou"
    if sid.startswith("blt_uk_"):
        return "blt_riseholme"
    if sid.startswith("btg_"):
        return "btg_" + sid.removeprefix("btg_").split("_p")[0]  # parts group with their clip
    if sid.startswith("bot_"):
        return "bot_V2" if "V2" in sid else "bot_V1"
    if sid.startswith("emb_"):
        return "emb"
    if sid.startswith("slam_"):
        return "slam"
    return sid


SPLIT_COLS = ["psnr_mean", "psnr_std", "ssim_mean", "ssim_std", "lpips_mean", "lpips_std",
              "n_views", "nn_dist_frac", "nn_angle_deg", "split_rule", "n_gaussians",
              "t_train_s", "gpu_mem_peak_mib", "train_image_size"]
REG_COLS = ["n_images", "n_registered", "reg_rate", "n_models", "model_sizes", "reproj_err",
            "track_len", "n_points", "tri_angle_median", "baseline_frac", "view_coverage_sr",
            "outcome"]


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "results/full_program/B"
    out = REPO / "results/quality"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        row = {"id": d.name, "dataset": d.name.split("_")[0], "scene_group": scene_group(d.name)}
        reg_f = next(iter(d.glob("*.registration.json")), None)
        if reg_f:
            reg = json.loads(reg_f.read_text())
            for c in REG_COLS:
                row[c] = reg.get(c)
        for split in ("interp", "block"):
            f = d / split / "summary.json"
            if not f.exists():
                continue
            s = json.loads(f.read_text())
            for c in SPLIT_COLS:
                v = s.get(c, s.get(c.replace("n_views", "n_test_views")))
                row[f"{split}_{c}"] = v
            row[f"{split}_per_view_json"] = str((d / split / "metrics_full.json").relative_to(REPO))
            row[f"{split}_job"] = s.get("job")
        rows.append(row)
    cols = ["id", "dataset", "scene_group", *REG_COLS]
    for split in ("interp", "block"):
        cols += [f"{split}_{c}" for c in SPLIT_COLS] + [f"{split}_per_view_json", f"{split}_job"]
    with open(out / "quality.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    done = sum(1 for r in rows if r.get("interp_psnr_mean") is not None)
    both = sum(1 for r in rows if r.get("block_psnr_mean") is not None)
    print(f"{len(rows)} rows ({done} with interp metrics, {both} with block)")
    print(out / "quality.csv")


if __name__ == "__main__":
    main()
