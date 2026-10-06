"""Collect the BLT standing-tail ablation (Table 10) and draw its figure.

Four arms per front pass (15 Sep and 13 Jul 2022):
  raw, masked                    the repeat-study arms (results/repeats/runs/), registration only;
                                 their image metrics come from the masking block (three trainer
                                 seeds on one registration), rescored on the held-out views the
                                 trimmed arms keep
  trimmed, masked + trimmed      the frames before the standing tail (results/ablations/blt_tail/),
                                 five COLMAP runs each, one training (trainer seed 0) per run that
                                 passes the 50 % gate
  corrected cut, raw / masked    the passes re-cut on header stamps, same row (results/repeats/runs/
                                 *_hdr_front_*), registration only: no masking-block condition on the
                                 corrected cut passed the gate, so nothing was trained
outputs (results/ablations/blt_tail/):
  table10_blt_ablation_runs.csv       one row per pass, arm and COLMAP run
  table10_blt_ablation_extended.csv   one row per pass and arm: runs 0-4 side by side, mean, SD
  results/figures/blt_tail/blt_tail.{pdf,png}   left: motion per frame with the trim boundary;
                                 right: registration per run for the four arms
usage: collect_tail.py [--no-figure]
"""

import csv
import json
import re
import statistics as st
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RES = REPO / "results"
TAIL = RES / "ablations/blt_tail"
PASSES = [  # label, sequence, keep_before, repeat conditions, trimmed conditions,
    #           corrected-cut repeat conditions, corrected-cut sequence (header-stamp re-cut, same row)
    ("15 Sep", "blt_gr_20220915_front", 95, ("sep15_front_raw", "sep15_front_hand"),
     ("sep15_front_raw_trim", "sep15_front_hand_trim"),
     ("sep15_hdr_front_raw", "sep15_hdr_front_hand"), "blt_gr_20220915_hdr_front"),
    ("13 Jul", "blt_gr_20220713_front", 131, ("jul13_front_raw", "jul13_front_hand"),
     ("jul13_front_raw_trim", "jul13_front_hand_trim"),
     ("jul13_hdr_front_raw", "jul13_hdr_front_hand"), "blt_gr_20220713_hdr_front"),
]
ARMS = ["raw", "masked", "trimmed", "masked + trimmed", "corrected cut, raw", "corrected cut, masked"]
MET = ("psnr", "ssim", "lpips")


def idx(name):
    return int(re.search(r"(\d+)\.\w+", name).group(1))


def jread(p):
    return json.loads(p.read_text()) if p.exists() else None


def view_means(metrics_full, keep):
    """Mean of the per-view metrics over the held-out views before the trim."""
    views = [v for v in metrics_full["per_view"] if idx(v["name"]) < keep]
    if not views:
        return None
    return dict(n_views=len(views), **{m: sum(v[m] for v in views) / len(views) for m in MET})


def sd(v):
    return st.stdev(v) if len(v) > 1 else ""


def r4(x):
    return round(x, 4) if isinstance(x, float) else x


def collect():
    runs_rows, wide = [], []
    for label, seq0, keep, rep, trim, hdr, hseq in PASSES:
        n0 = sum(1 for _ in open(RES / "characterisation/frame_lists" / f"{seq0}.txt"))
        for arm, cond in zip(ARMS, rep + trim + hdr):
            trimmed = cond.endswith("_trim")
            corrected = cond in hdr
            seq = hseq if corrected else seq0
            n_frames = sum(1 for _ in open(RES / "characterisation/frame_lists" / f"{seq}.txt"))
            rdir = (TAIL / "runs" / cond) if trimmed else (RES / "repeats/runs" / cond)
            runs = [jread(rdir / f"run_{r}.json") for r in range(5)]
            per = []
            for r, d in enumerate(runs):
                row = dict(pass_=label, sequence=seq, arm=arm, condition=cond, run=r,
                           frames=keep if trimmed else n_frames, sfm_mask="masked" in arm,
                           loss_mask=arm == "masked + trimmed", status="pending" if d is None else "done")
                if d:
                    row.update(n_registered=d.get("n_registered"), largest_model=d.get("largest_model"),
                               reg_frac=round(d["largest_model"] / d["n_images"], 4),
                               success=d.get("success"), n_models=d.get("n_models"),
                               model_sizes=json.dumps(d.get("model_sizes")), n_points=d.get("n_points", ""),
                               n_init_attempts=d.get("n_init_attempts"))
                    tr = TAIL / "train" / cond / f"run_{r}"
                    mf = jread(tr / "metrics_full.json")
                    if trimmed and mf:
                        vm = view_means(mf, keep)
                        row.update(n_views=vm["n_views"], **{m: vm[m] for m in MET},
                                   metrics_from="this run, trainer seed 0")
                    elif trimmed and d.get("success"):
                        row.update(metrics_from="training pending")
                per.append(row)
            runs_rows += per
            done = [p for p in per if p["status"] == "done"]
            w = dict(pass_=label, sequence=seq, arm=arm, condition=cond,
                     frames=keep if trimmed else n_frames, sfm_mask="masked" in arm,
                     loss_mask_in_training=arm in ("masked", "masked + trimmed"),
                     runs_done=len(done), successes=sum(bool(p.get("success")) for p in done))
            for p in per:
                r = p["run"]
                w[f"reg_frac_run{r}"] = p.get("reg_frac", "")
                w[f"n_registered_run{r}"] = p.get("largest_model", "")
                w[f"model_sizes_run{r}"] = p.get("model_sizes", "")
                w[f"n_points_run{r}"] = p.get("n_points", "")
                for m in MET:
                    w[f"{m}_run{r}"] = r4(p.get(m, ""))
            for key in ("reg_frac", "largest_model", "n_points") + MET:
                v = [p[key] for p in done if p.get(key) not in (None, "")]
                name = "n_registered" if key == "largest_model" else key
                w[f"{name}_mean"] = r4(sum(v) / len(v)) if v else ""
                w[f"{name}_sd"] = r4(sd(v)) if v else ""
            w["n_trained"] = sum(1 for p in done if p.get("psnr") is not None)
            # masking-block comparator: three trainer seeds on one registration, shared views
            mb = {"raw": "raw", "masked": "hand_both"}.get(arm)
            vals = []
            if mb:
                for s in range(3):
                    mf = jread(RES / "full_program/C/blt" / seq / mb / f"seed{s}/metrics_full.json")
                    if mf:
                        vals.append(view_means(mf, keep))
            w["masking_block_trainer_seeds"] = len(vals)
            for m in MET:
                v = [x[m] for x in vals]
                w[f"masking_block_{m}_mean"] = r4(sum(v) / len(v)) if v else ""
                w[f"masking_block_{m}_min"] = r4(min(v)) if v else ""
                w[f"masking_block_{m}_max"] = r4(max(v)) if v else ""
            w["shared_heldout_views"] = "" if corrected else len([i for i in range(0, n0, 8) if i < keep])
            wide.append(w)
    TAIL.mkdir(parents=True, exist_ok=True)
    for name, rows in (("table10_blt_ablation_runs.csv", runs_rows), ("table10_blt_ablation_extended.csv", wide)):
        cols = []
        for r in rows:
            cols += [c for c in r if c not in cols]
        with open(TAIL / name, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=[c.rstrip("_") for c in cols])
            wr.writeheader()
            wr.writerows({k.rstrip("_"): v for k, v in r.items()} for r in rows)
    print(f"{len(runs_rows)} run rows, {len(wide)} arm rows -> {TAIL}")
    return runs_rows, wide


def figure(runs_rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    motion = list(csv.DictReader(open(RES / "ablations/blt_stationary/blt_stationary_start.csv")))
    fig = plt.figure(figsize=(7.2, 3.2))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.15], hspace=0.35, wspace=0.62)
    bx = fig.add_subplot(gs[:, 1])
    xmax = max(p[2] for p in PASSES) + 20
    for row, (label, seq, keep, *_) in enumerate(PASSES):
        ax = fig.add_subplot(gs[row, 0])
        rows = [m for m in motion if m["sequence"] == seq and m["rtk_step_ratio"]]
        x = [int(m["frame"]) for m in rows]
        y = [min(float(m["rtk_step_ratio"]), 1.4) for m in rows]
        ax.axvspan(keep - 0.5, len(x) + 0.5, color="0.93", lw=0, zorder=1)
        ax.plot(x, y, color="#9467bd", lw=1.0, zorder=3)
        ax.axvline(keep - 0.5, color="0.35", lw=0.8, ls="--", zorder=2)
        ax.axhline(0.1, color="0.6", lw=0.6, ls=":", zorder=2)
        ax.text(2, 1.3, f"BLT {label}, front camera", fontsize=7, color="0.15", va="top")
        ax.text(keep + 1.5, 1.3, f"dropped:\nframes {keep}-{len(x)}", fontsize=6, color="0.3", va="top")
        ax.set_xlim(0, xmax)
        ax.set_ylim(0, 1.4)
        ax.set_yticks([0, 0.5, 1.0])
        ax.tick_params(labelsize=7)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        if row == 0:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("frame index (0.47 s apart)", fontsize=7.5)
            ax.text(2, 0.14, "one tenth of the median step", fontsize=5.5, color="0.45")
    fig.text(0.015, 0.55, "step between frames / median step (RTK)", rotation=90, va="center", fontsize=7.5)
    # right: one bar per pass and arm, "k of n", and the five runs as squares
    order, i = [], 0
    for label, seq, keep, rep, trim, hdr, _ in PASSES:
        for arm, cond in zip(ARMS, rep + trim + hdr):
            rr = [r for r in runs_rows if r["condition"] == cond and r["status"] == "done"]
            order.append((i, label, arm, rr))
            i += 1
        i += 0.6
    for y, label, arm, rr in order:
        n, k = len(rr), sum(bool(r.get("success")) for r in rr)
        col = "#c5b0d5" if "masked" in arm else "#9467bd"
        hatch = "////" if "trimmed" in arm else ("xxx" if "corrected" in arm else None)
        if n:
            bx.barh(y, k / n * 100, color=col, height=0.62, zorder=3, hatch=hatch, edgecolor="white", lw=0)
            bx.text(k / n * 100 + 1.5, y, f"{k} of {n}", va="center", fontsize=7, color="0.25")
        for r in rr:
            ok = bool(r.get("success"))
            bx.scatter(120 + 4 * r["run"], y, marker="s", s=14, zorder=4,
                       facecolor="0.25" if ok else "white", edgecolor="0.25", lw=0.7)
    bx.set_yticks([o[0] for o in order])
    bx.set_yticklabels([f"{o[1]}, {o[2]}" for o in order], fontsize=7)
    bx.invert_yaxis()
    bx.set_xlim(0, 140)
    bx.set_xticks([0, 20, 40, 60, 80, 100])
    bx.tick_params(labelsize=7)
    bx.text(128, -0.95, "runs 0-4", ha="center", fontsize=6.5, color="0.25")
    bx.set_xlabel("runs in which the frames registered, %", fontsize=7.5)
    bx.grid(axis="x", color="0.92", zorder=0)
    for s in ("top", "right"):
        bx.spines[s].set_visible(False)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.93, bottom=0.14)
    out = RES / "figures/blt_tail"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "blt_tail.pdf")
    fig.savefig(out / "blt_tail.png", dpi=200)
    print("figure ->", out)


if __name__ == "__main__":
    rr, _ = collect()
    if "--no-figure" not in sys.argv:
        figure(rr)
