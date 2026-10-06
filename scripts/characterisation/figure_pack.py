"""Figure pack for the paper: median- and best-PSNR held-out views of named rows, from the
kept seed-0 renders (no training).

Each figure row gets `results/figures/<name>/` with, per panel, the render and ground truth
at the run resolution (`<panel>_<frame>_{pred,gt}.png` copies of the kept q90 JPEGs), a
`meta.json` (frame names, per-view metrics, split, selection rule) and a `sheet.jpg` contact
sheet (render | GT | abs-diff) for quick inspection.

usage: figure_pack.py [row ...]   (no args = every row whose inputs exist)
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
REPO = HERE.parent.parent
FIG = REPO / "results/figures"


def per_view(run: Path):
    mf = json.loads((run / "metrics_full.json").read_text())
    return mf["per_view"], mf.get("lpips_net", "alex")


def pick(views, rule):
    """rule: 'median' | 'best' | an exact frame name."""
    have = [v for v in views]
    if rule == "best":
        return max(have, key=lambda v: v["psnr"])
    if rule == "median":
        s = sorted(have, key=lambda v: v["psnr"])
        return s[len(s) // 2]
    return next(v for v in have if v["name"].startswith(rule))


def frame_stem(view_name: str) -> str:
    # eval names look like frame_000032.png.png; renders are pred_frame_000032.jpg
    return Path(view_name.removesuffix(".png")).stem


def load_pair(run: Path, stem: str):
    gt = cv2.imread(str(run / "eval_renders" / f"gt_{stem}.jpg"))
    pred = cv2.imread(str(run / "eval_renders" / f"pred_{stem}.jpg"))
    return gt, pred


def export_row(name: str, panels: list[dict], note: str = ""):
    """panels: dicts with run (Path), label, rule ('median'/'best'/frame), optional overlay
    ('cluster' draws the dataset grape mask on a third panel)."""
    out = FIG / name
    out.mkdir(parents=True, exist_ok=True)
    meta = {"note": note, "panels": []}
    sheets = []
    for p in panels:
        run = Path(p["run"])
        views, net = per_view(run)
        v = pick(views, p["rule"])
        stem = frame_stem(v["name"])
        gt, pred = load_pair(run, stem)
        if gt is None or pred is None:
            print(f"  {name}: no renders for {run} {stem}, skipped")
            return False
        for kind, img in (("gt", gt), ("pred", pred)):
            cv2.imwrite(str(out / f"{p['label']}_{stem}_{kind}.png"), img)
        diff = cv2.absdiff(gt, pred)
        row = [pred, gt, cv2.applyColorMap(cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY),
                                           cv2.COLORMAP_INFERNO)]
        extra = {}
        if p.get("overlay") == "cluster":
            from region_metric import cluster_mask, run_camera, undistort_mask
            rid, cam = run_camera(run)
            labels = {}
            flist = REPO / "results/characterisation/frame_lists" / \
                (rid.split("_c_")[0] + ".txt")
            import re
            for ln in flist.read_text().splitlines():
                if ln.strip():
                    n_, lab = ln.split("\t")
                    g = re.search(r"global (\d+)", lab)
                    labels[n_] = int(g.group(1)) if g else None
            gidx = labels.get(stem.split("_", 0)[0] if False else f"{stem}.png")
            m = cluster_mask(rid.split("_c_")[0], gidx,
                             REPO / "results/regions/_mask_cache")
            mu8 = cv2.resize((m * 255).astype(np.uint8), (cam.width, cam.height),
                             interpolation=cv2.INTER_NEAREST)
            keep = undistort_mask(mu8, cam)
            if keep.shape != gt.shape[:2]:
                keep = cv2.resize(keep.astype(np.uint8), gt.shape[1::-1],
                                  interpolation=cv2.INTER_NEAREST) > 0
            ov = gt.copy()
            ov[keep] = (0.35 * ov[keep] + 0.65 * np.array([0, 0, 255])).astype(np.uint8)
            cv2.imwrite(str(out / f"{p['label']}_{stem}_clustermask.png"), ov)
            row.append(ov)
            extra["cluster_share"] = float(keep.mean())
        h = 360
        row = [cv2.resize(x, (round(x.shape[1] * h / x.shape[0]), h)) for x in row]
        strip = cv2.hconcat(row)
        cv2.putText(strip, f"{p['label']}  {stem}  PSNR {v['psnr']:.2f}", (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        sheets.append(strip)
        meta["panels"].append(dict(label=p["label"], run=str(run.relative_to(REPO)),
                                   rule=p["rule"], frame=stem, lpips_net=net,
                                   split=json.loads((run / "summary.json").read_text()
                                                    ).get("split", "interp")
                                   if (run / "summary.json").exists() else None,
                                   psnr=v["psnr"], ssim=v["ssim"], lpips=v.get("lpips"),
                                   **extra))
    w = max(s.shape[1] for s in sheets)
    sheets = [cv2.copyMakeBorder(s, 0, 0, 0, w - s.shape[1], cv2.BORDER_CONSTANT)
              for s in sheets]
    cv2.imwrite(str(out / "sheet.jpg"), cv2.vconcat(sheets),
                [cv2.IMWRITE_JPEG_QUALITY, 88])
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"  {name}: {len(panels)} panel(s) -> {out.relative_to(REPO)}")
    return True


def median_frame_of(run: Path) -> str:
    views, _ = per_view(Path(run))
    return frame_stem(pick(views, "median")["name"])


def rows():
    B = REPO / "results/full_program/B"
    C = REPO / "results/full_program/C"
    S = REPO / "results/sweeps"
    r = {}
    for seq in ("mots_NoPathPlanning_1", "mots_PathPlanning_8"):
        for split in ("interp", "block"):
            r[f"{seq}_{split}"] = dict(panels=[
                dict(run=B / seq / split, label=f"median_{split}", rule="median"),
                dict(run=B / seq / split, label=f"best_{split}", rule="best")],
                note="row pass" if "NoPath" in seq else "orbit")
    r["btg_Row7.2_p3_interp"] = dict(panels=[
        dict(run=B / "btg_Row7.2_p3/interp", label="median", rule="median"),
        dict(run=B / "btg_Row7.2_p3/interp", label="best", rule="best")],
        note="UAV row pass, 3 m AGL, 60 deg tilt")
    # same held-out frame across conditions: the median frame of the raw run
    raw = C / "mots/mots_NoPathPlanning_1/raw/seed0"
    if (raw / "metrics_full.json").exists():
        f = median_frame_of(raw) + ""
        r["mots_raw_vs_dehazeghost"] = dict(panels=[
            dict(run=raw, label="raw", rule=f, overlay="cluster"),
            dict(run=C / "mots/mots_NoPathPlanning_1/dehaze_ghost/seed0",
                 label="dehaze_ghost", rule=f, overlay="cluster")],
            note="same held-out frame; cluster mask overlaid; per-region PSNR in "
                 "results/regions/mots_regions.csv")
    bot = C / "botrytis/45_V1"
    if (bot / "full/seed0/metrics_full.json").exists():
        f = median_frame_of(bot / "full/seed0")
        r["botrytis_naive_vs_full"] = dict(panels=[
            dict(run=bot / "naive/seed0", label="naive", rule=f),
            dict(run=bot / "full/seed0", label="full", rule=f)],
            note="same held-out frame (median of the full condition); naive registers "
                 "109/200 so the frame may be absent there - then its median is used")
    for d, lab in (("blt_gr_20220323_front", "23mar"), ("blt_gr_20220608_front", "08jun"),
                   ("blt_gr_20220915_front", "15sep")):
        run = C / "blt" / d / "raw/seed0"
        if (run / "metrics_full.json").exists():
            r.setdefault("blt_greek_fronts_raw", dict(panels=[], note=(
                "median held-out view per date; corridor positions are fractional "
                "(the pass windows differ per date), not geographically matched")))
            r["blt_greek_fronts_raw"]["panels"].append(
                dict(run=run, label=lab, rule="median"))
    for seq, levels in (("mots_NoPathPlanning_1",
                         [("fps_1", "lowest"), ("width_1600", None), ("fps_4", "highest"),
                          ("width_640", "lowest-width"), ("count_15", "lowest-count")]),
                        ("btg_Row7.2_p3",
                         [("fps_0.5", "lowest"), ("width_1600", None), ("fps_4", "highest"),
                          ("width_640", "lowest-width"), ("count_15", "lowest-count")])):
        panels = []
        for lv, _ in levels:
            run = S / seq / lv
            if lv.startswith("width_1600"):
                run = REPO / "results/full_program/B" / seq / "interp"
            if (run / "metrics_full.json").exists():
                panels.append(dict(run=run, label=lv, rule="median"))
        if panels:
            r[f"sweep_{seq}"] = dict(panels=panels,
                                     note="median held-out render per level; width_1600 "
                                          "is the shared level (= the quality row)")
    return r


def main():
    want = sys.argv[1:]
    FIG.mkdir(parents=True, exist_ok=True)
    done, skipped = [], []
    for name, spec in rows().items():
        if want and name not in want:
            continue
        try:
            ok = export_row(name, spec["panels"], spec.get("note", ""))
            (done if ok else skipped).append(name)
        except FileNotFoundError as e:
            print(f"  {name}: missing input ({e}), skipped")
            skipped.append(name)
    print(f"{len(done)} rows written, {len(skipped)} skipped")


if __name__ == "__main__":
    main()
