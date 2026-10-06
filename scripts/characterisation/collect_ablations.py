"""Build the ablation CSVs: one row per (sequence, condition, seed), registration first.

Reads results/full_program/C/<set>/<sequence>/<condition>/ written by the ablation jobs:
registration.json always; compose_report.json (Botrytis); seed<k>/summary.json and
metrics_common.json where the gate passed and training ran. A condition with a registration
row and no training rows is reported with empty metric columns, not dropped.

usage: collect_ablations.py [botrytis|mots|blt ...]
"""

import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "results/full_program/C"

REG_COLS = ["n_images", "n_registered", "reg_rate", "n_models", "model_sizes",
            "reproj_err", "track_len", "n_points", "tri_angle_median", "baseline_frac",
            "view_coverage_sr", "outcome"]
MET = ["psnr_mean", "psnr_std", "ssim_mean", "ssim_std", "lpips_mean", "lpips_std", "n_views"]


def rows_for(setname: str):
    rows = []
    for seq_dir in sorted((ROOT / setname).iterdir()):
        if not seq_dir.is_dir():
            continue
        for cond_dir in sorted(p for p in seq_dir.iterdir() if p.is_dir()):
            base = dict(sequence=seq_dir.name, condition=cond_dir.name)
            reg_f = cond_dir / "registration.json"
            if not reg_f.exists():
                # raw conditions persisted by reconstruct_b.sh keep the registration under the id
                named = sorted(cond_dir.glob("*.registration.json"))
                reg_f = named[0] if named else reg_f
            if reg_f.exists():
                reg = json.loads(reg_f.read_text())
                base.update({c: reg.get(c) for c in REG_COLS})
                sizes = reg.get("model_sizes") or [0]
                base["gate_pass"] = max(sizes) / reg["n_images"] >= 0.5
            rep_f = cond_dir / "compose_report.json"
            if rep_f.exists():
                rep = json.loads(rep_f.read_text())
                base["compose_mode"] = rep.get("mode")
                base["perframe_failure_share"] = rep.get("perframe_failure_share")
                base["median_shifts_px"] = json.dumps(rep.get("median_shifts_px")) \
                    if rep.get("median_shifts_px") else None
            seeds = sorted(cond_dir.glob("seed*/summary.json"))
            if not seeds:
                rows.append(dict(base, seed=None))
                continue
            for sf in seeds:
                s = json.loads(sf.read_text())
                r = dict(base, seed=s.get("seed"),
                         **{m: s.get(m) for m in MET},
                         n_gaussians=s.get("n_gaussians"), t_train_s=s.get("t_train_s"),
                         loss_masked=s.get("loss_masked"), job=s.get("job"))
                cm = sf.parent / "metrics_common.json"
                if cm.exists():
                    c = json.loads(cm.read_text())
                    r.update(psnr_common=c.get("psnr_mean"), ssim_common=c.get("ssim_mean"),
                             lpips_common=c.get("lpips_mean"),
                             common_mask_share=c.get("mask_share_mean"))
                rows.append(r)
    return rows


def repeat_rates():
    """(sequence, sfm_masked) -> registration success rate from the repeat study, if run."""
    path = REPO / "results/repeats/summary.csv"
    if not path.exists():
        return {}
    with open(path) as f:
        return {(r["sequence"], r["mask_kind"] != "none"): float(r["success_rate"])
                for r in csv.DictReader(f)}


def main():
    sets = sys.argv[1:] or [p.name for p in ROOT.iterdir() if p.is_dir()]
    out_dir = REPO / "results/ablations"
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = ["sequence", "condition", "seed", "gate_pass", *REG_COLS,
            "compose_mode", "perframe_failure_share", "median_shifts_px",
            *MET, "psnr_common", "ssim_common", "lpips_common", "common_mask_share",
            "n_gaussians", "t_train_s", "loss_masked", "job",
            "registration_success_rate", "pending_repeats"]
    for setname in sets:
        if not (ROOT / setname).is_dir():
            continue
        rows = rows_for(setname)
        if setname == "blt":
            # single-run registration on the front passes is bistable: every row carries the
            # repeat-study success rate of its SfM configuration, or pending_repeats
            rates = repeat_rates()
            for r in rows:
                sfm_masked = r["condition"] in ("hand_both", "hand_sfm_only")
                rate = rates.get((r["sequence"], sfm_masked))
                if r["condition"] == "hsv_both":
                    rate = None  # the HSV-mask configuration is not part of the repeat study
                r["registration_success_rate"] = rate
                r["pending_repeats"] = rate is None
        with open(out_dir / f"{setname}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        trained = sum(1 for r in rows if r.get("psnr_mean") is not None)
        print(f"{setname}: {len(rows)} rows ({trained} trained) -> results/ablations/{setname}.csv")


if __name__ == "__main__":
    main()
