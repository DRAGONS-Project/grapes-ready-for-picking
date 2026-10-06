"""Registration repeat study: N independent runs of the fixed COLMAP configuration on one
condition, each with its own PYTHONHASHSEED and COLMAP random seed (run r uses seed r; seed 0
is the instrument's own). Nothing else changes: same frames, same masks, same options.

Per run it records the outcome and what the incremental mapper did: every initial image pair
it tried (from the mapper log), how many images each attempt registered, the pair behind the
largest model, and why other attempts were discarded.

usage: repeat_registration.py <condition> <source> <mask_kind> <n_runs> <work_dir> <out_dir>
       [--threads N] [--keep-before N] [--stage DIR [--loss-masks]]
  source:    seq:<index>              a sequence from sequences.py (committed frame list)
             sweep:<index>:<kind:val> a sweep level (sweep_prepare.py)
  mask_kind: none | fixed_region | greek   (SfM feature masks, white = keep)
  --keep-before N  use only the frames whose index is below N (the frame names are unchanged)
  --stage DIR      for every run that passes the gate, write the undistorted model and frames to
                   DIR/run_<r>/ for training (and, with --loss-masks, the undistorted masks)
"""

import argparse
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from common import write_json  # noqa: E402

MAX_ID = 2147483647  # COLMAP pair_id = id1 * MAX_ID + id2 (id1 < id2)


def frame_index(name: str):
    m = re.search(r"(\d+)\.\w+$", name)
    return int(m.group(1)) if m else None


def parse_mapper_log(log: str):
    """Initial-pair attempts in order, with how many images each went on to register."""
    attempts, cur = [], None
    discards = {}
    for ln in log.splitlines():
        m = re.search(r"Registering initial image pair #(\d+) and #(\d+)", ln)
        if m:
            cur = dict(ids=[int(m.group(1)), int(m.group(2))], n_registered_after=0)
            attempts.append(cur)
            continue
        if cur is not None and re.search(r"Registering image #\d+", ln):
            cur["n_registered_after"] += 1
        m = re.search(r"Discarding reconstruction due to (.+)$", ln)
        if m:
            reason = m.group(1).strip()
            discards[reason] = discards.get(reason, 0) + 1
            if cur is not None and "discarded" not in cur:
                cur["discarded"] = reason
    no_pair = len(re.findall(r"No good initial image pair found", log))
    return attempts, discards, no_pair


def pair_info(db: Path, ids):
    """Names, frame gap and verified inlier count of an image pair."""
    con = sqlite3.connect(db)
    names = {}
    for i in ids:
        row = con.execute("select name from images where image_id=?", (i,)).fetchone()
        names[i] = row[0] if row else None
    a, b = sorted(ids)
    row = con.execute("select rows from two_view_geometries where pair_id=?",
                      (a * MAX_ID + b,)).fetchone()
    con.close()
    fa, fb = frame_index(names[ids[0]] or ""), frame_index(names[ids[1]] or "")
    return dict(ids=ids, names=[names[ids[0]], names[ids[1]]],
                frame_gap=abs(fa - fb) if fa is not None and fb is not None else None,
                inliers=row[0] if row else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("condition")
    ap.add_argument("source")
    ap.add_argument("mask_kind", choices=["none", "fixed_region", "greek"])
    ap.add_argument("n_runs", type=int)
    ap.add_argument("work", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--keep-before", type=int)
    ap.add_argument("--stage", type=Path)
    ap.add_argument("--loss-masks", action="store_true")
    a = ap.parse_args()
    py = sys.executable
    a.out.mkdir(parents=True, exist_ok=True)

    base = a.work / "base"
    kind, _, rest = a.source.partition(":")
    if kind == "seq":
        subprocess.run([py, str(HERE / "prepare.py"), rest, str(base), str(a.work / "res")],
                       check=True)
        from sequences import SEQUENCES
        seq_id = SEQUENCES[int(rest)]["id"]
    else:
        idx, _, spec = rest.partition(":")
        subprocess.run([py, str(HERE / "sweep_prepare.py"), idx, spec, str(base)], check=True)
        seq_id = json.loads((base / "meta.json").read_text())["id"]
    images = base / "images"
    names = sorted(p.name for p in images.iterdir())
    n_prepared = len(names)
    if a.keep_before is not None:
        kept = base / "images_kept"
        kept.mkdir(exist_ok=True)
        names = [n for n in names if frame_index(n) < a.keep_before]
        for n in names:
            if not (kept / n).exists():
                os.link(images / n, kept / n)
        images = kept

    masks_colmap = None
    mask_report = None
    if a.mask_kind != "none":
        from ablate_blt import build_masks
        masks_colmap = a.work / "masks_colmap"
        mask_report = build_masks(a.mask_kind, images, a.work / "masks", masks_colmap)

    for r in range(a.n_runs):
        rid = f"{a.condition}_r{r}"
        wdir = a.work / f"run_{r}"
        (wdir / "images").mkdir(parents=True, exist_ok=True)
        for n in names:
            dst = wdir / "images" / n
            if not dst.exists():
                os.link(images / n, dst)
        write_json(wdir / "meta.json", dict(id=rid, dataset="repeat", sequence_id=rid,
                                            population="repeat", n_images=len(names),
                                            camera_mode="SINGLE", notes=[]))
        cmd = [py, str(HERE / "register.py"), str(wdir), str(wdir / "res"),
               "--threads", str(a.threads), "--seed", str(r)]
        if masks_colmap is not None:
            cmd += ["--sfm-masks", str(masks_colmap)]
        env = dict(os.environ, PYTHONHASHSEED=str(r))
        log_path = wdir / "register.err"
        with open(log_path, "w") as lf:
            proc = subprocess.run(cmd, env=env, stderr=lf, stdout=subprocess.PIPE, text=True)
        reg_path = wdir / "res" / "per_sequence" / f"{rid}.registration.json"
        row = dict(condition=a.condition, sequence=seq_id, mask_kind=a.mask_kind, run=r,
                   colmap_seed=r, pythonhashseed=r, threads=a.threads,
                   n_images=len(names), returncode=proc.returncode)
        if a.keep_before is not None:
            row.update(keep_before=a.keep_before, n_prepared=n_prepared)
        if mask_report:
            row["masked_share"] = mask_report["masked_share_mean"]
        if reg_path.exists():
            reg = json.loads(reg_path.read_text())
            sizes = reg.get("model_sizes") or []
            largest = max(sizes) if sizes else 0
            row.update(n_registered=reg.get("n_registered"), n_models=len(sizes),
                       model_sizes=sizes, largest_model=largest,
                       success=largest / len(names) >= 0.5,
                       n_points=reg.get("n_points"), reproj_err=reg.get("reproj_err"),
                       track_len=reg.get("track_len"), tri_angle_median=reg.get("tri_angle_median"),
                       baseline_frac=reg.get("baseline_frac"),
                       t_extract=reg.get("t_extract"), t_match=reg.get("t_match"),
                       t_map=reg.get("t_map"), outcome=reg.get("outcome"))
            if a.stage and row["success"] and reg.get("best_sparse_dir"):
                st = a.stage / f"run_{r}"
                shutil.rmtree(st, ignore_errors=True)
                py_run = lambda *args: subprocess.run([py, str(HERE / "undistort.py"), *map(str, args)],
                                                      check=True, stdout=subprocess.DEVNULL)
                py_run(reg["best_sparse_dir"], wdir / "images", st)
                if a.loss_masks and masks_colmap is not None:
                    mu = a.work / f"mu_{r}"
                    py_run(reg["best_sparse_dir"], a.work / "masks", mu)
                    shutil.move(str(mu / "images"), str(st / "loss_masks"))
                    shutil.rmtree(mu, ignore_errors=True)
                for extra in ("stereo", "run-colmap-geometric.sh", "run-colmap-photometric.sh"):
                    p = st / extra
                    shutil.rmtree(p, ignore_errors=True) if p.is_dir() else p.unlink(missing_ok=True)
                shutil.copy(reg_path, st / "registration.json")
                row["staged"] = str(st)
            attempts, discards, no_pair = parse_mapper_log(log_path.read_text(errors="replace"))
            db = Path(reg["colmap_dir"]) / "database.db"
            for at in attempts:
                at.update(pair_info(db, at["ids"]))
            best = max(attempts, key=lambda t: t["n_registered_after"]) if attempts else None
            row.update(n_init_attempts=len(attempts), n_no_pair_found=no_pair,
                       discards=discards, init_attempts=attempts,
                       first_init_pair=attempts[0] if attempts else None,
                       largest_model_init_pair=best)
            shutil.rmtree(Path(reg["colmap_dir"]), ignore_errors=True)
        write_json(a.out / f"run_{r}.json", row)
        brief = {k: row.get(k) for k in ("condition", "run", "n_registered", "model_sizes",
                                         "success", "n_init_attempts")}
        if row.get("first_init_pair"):
            fp = row["first_init_pair"]
            brief["first_pair"] = (fp["names"], fp["frame_gap"], fp["inliers"])
        print(json.dumps(brief), flush=True)
        shutil.rmtree(wdir, ignore_errors=True)


if __name__ == "__main__":
    main()
