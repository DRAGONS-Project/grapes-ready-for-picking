"""Collect the registration repeat study.

results/repeats/repeats.csv   one row per run (outcome, seeds, initial-pair record)
results/repeats/summary.csv   one row per condition (success rate, registered range,
                              distinct initial pairs); success = largest model >= 50 % of frames
results/joined.csv            gains registration_stability (raw-condition success rate) and
                              registration_stability_masked for the repeated sequences

usage: collect_repeats.py
"""

import csv
import json
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "results/repeats"


def pair_str(p):
    if not p:
        return ""
    return f"{p['names'][0]}+{p['names'][1]} (gap {p['frame_gap']}, inliers {p['inliers']})"


def main():
    runs = []
    for f in sorted((ROOT / "runs").glob("*/run_*.json")):
        runs.append(json.loads(f.read_text()))
    cols = ["condition", "sequence", "mask_kind", "masked_share", "run", "colmap_seed",
            "pythonhashseed", "threads", "n_images", "n_registered", "largest_model", "n_models",
            "model_sizes", "success", "n_init_attempts", "n_no_pair_found", "discards",
            "first_init_pair", "first_init_gap", "first_init_inliers",
            "largest_model_init_pair", "largest_init_gap", "largest_init_inliers",
            "largest_init_registered_after", "t_extract", "t_match", "t_map", "outcome"]
    with open(ROOT / "repeats.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in runs:
            fp, lp = r.get("first_init_pair"), r.get("largest_model_init_pair")
            w.writerow(dict(
                r, discards=json.dumps(r.get("discards", {})),
                first_init_pair=pair_str(fp), first_init_gap=fp and fp["frame_gap"],
                first_init_inliers=fp and fp["inliers"],
                largest_model_init_pair=pair_str(lp), largest_init_gap=lp and lp["frame_gap"],
                largest_init_inliers=lp and lp["inliers"],
                largest_init_registered_after=lp and lp["n_registered_after"]))

    by = {}
    for r in runs:
        by.setdefault(r["condition"], []).append(r)
    summ = []
    for cond, rs in by.items():
        reg = [r.get("n_registered") or 0 for r in rs]
        succ = [bool(r.get("success")) for r in rs]
        # a pair is its two frame names, whatever their order or the run's inlier count
        key = lambda p: tuple(sorted(p["names"])) if p else None
        firsts = {key(r.get("first_init_pair")) for r in rs}
        larg = {key(r.get("largest_model_init_pair")) for r in rs}
        summ.append(dict(
            condition=cond, sequence=rs[0]["sequence"], mask_kind=rs[0]["mask_kind"],
            n=len(rs), n_images=rs[0]["n_images"], success_count=sum(succ),
            success_rate=round(sum(succ) / len(rs), 3),
            registered_median=statistics.median(reg), registered_min=min(reg),
            registered_max=max(reg), registered_all=reg,
            distinct_first_init_pairs=len(firsts), distinct_largest_init_pairs=len(larg),
            init_attempts_median=statistics.median([r.get("n_init_attempts") or 0 for r in rs])))
    with open(ROOT / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summ[0].keys()))
        w.writeheader()
        w.writerows(summ)

    # registration_stability into joined.csv (raw conditions) + masked variant
    stab, stab_m = {}, {}
    for s in summ:
        target = stab if s["mask_kind"] == "none" else stab_m
        target[s["sequence"]] = s["success_rate"]
    jp = REPO / "results/joined.csv"
    with open(jp) as f:
        rows = list(csv.DictReader(f))
    fields = list(rows[0].keys())
    for c in ("registration_stability", "registration_stability_masked"):
        if c not in fields:
            fields.append(c)
    have = {r["id"] for r in rows}
    for r in rows:
        r["registration_stability"] = stab.get(r["id"], r.get("registration_stability", ""))
        r["registration_stability_masked"] = stab_m.get(
            r["id"], r.get("registration_stability_masked", ""))
    for sid in sorted((set(stab) | set(stab_m)) - have):
        rows.append({"id": sid, "registration_stability": stab.get(sid, ""),
                     "registration_stability_masked": stab_m.get(sid, "")})
    with open(jp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"{len(runs)} runs, {len(summ)} conditions")
    for s in summ:
        print(f"  {s['condition']:24s} {s['success_count']}/{s['n']} success, registered "
              f"{s['registered_all']} of {s['n_images']}")


if __name__ == "__main__":
    main()
