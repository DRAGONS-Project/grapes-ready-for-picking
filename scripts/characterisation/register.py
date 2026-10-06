"""Registration gate: one fixed COLMAP configuration, then geometry quantities from the largest model.

usage: register.py <work_dir> <results_dir> [--threads N] [--mapper-threads M]

Configuration (identical for every sequence): OPENCV camera model, SIFT with max image size 1600 and
8192 features, exhaustive matching, incremental mapper defaults, random seed 0. One shared camera per
sequence; for stills with mixed image dimensions, one shared camera per dimension group (PER_FOLDER).
Writes <results_dir>/per_sequence/<id>.registration.json and <results_dir>/poses/<id>.csv.
"""

import argparse
import csv
import itertools
import json
import os
import time
from pathlib import Path

import numpy as np
import pycolmap

from common import write_json

SEED = 0
CAP_DEG = 15.0
MC_DIRECTIONS = 20000


def features_and_matches(img: Path, work: Path, mode: str, threads: int, masks: Path | None = None) -> dict:
    db = work / "database.db"
    pycolmap.set_random_seed(SEED)
    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = "OPENCV"
    if masks is not None:
        # COLMAP convention: <masks>/<image name>.png, no features where the mask is black
        reader.mask_path = str(masks)
    ext = pycolmap.FeatureExtractionOptions()
    ext.max_image_size = 1600
    ext.sift.max_num_features = 8192
    ext.num_threads = threads
    ext.use_gpu = False
    t0 = time.time()
    pycolmap.extract_features(db, img, camera_mode=getattr(pycolmap.CameraMode, mode), reader_options=reader, extraction_options=ext)
    t1 = time.time()
    mo = pycolmap.FeatureMatchingOptions()
    mo.num_threads = threads
    mo.use_gpu = False
    pycolmap.match_exhaustive(db, matching_options=mo)
    return dict(t_extract=t1 - t0, t_match=time.time() - t1)


def map_once(img: Path, work: Path, sparse_name: str, mapper_threads: int) -> tuple[dict, float]:
    sparse = work / sparse_name
    sparse.mkdir(parents=True, exist_ok=True)
    pycolmap.set_random_seed(SEED)
    po = pycolmap.IncrementalPipelineOptions()
    po.num_threads = mapper_threads
    po.mapper.num_threads = mapper_threads
    po.mapper.random_seed = SEED
    t0 = time.time()
    maps = pycolmap.incremental_mapping(work / "database.db", img, sparse, options=po)
    return maps, time.time() - t0


def geometry(rec: pycolmap.Reconstruction) -> tuple[dict, list]:
    imgs = {i: im for i, im in rec.images.items() if im.has_pose}
    C = {i: np.asarray(im.projection_center()) for i, im in imgs.items()}
    A = {i: np.asarray(im.viewing_direction()) for i, im in imgs.items()}
    angs = []
    for p in rec.points3D.values():
        cams = [e.image_id for e in p.track.elements if e.image_id in C][:5]
        best = 0.0
        for a, b in itertools.combinations(cams, 2):
            ra, rb = p.xyz - C[a], p.xyz - C[b]
            cosang = np.dot(ra, rb) / (np.linalg.norm(ra) * np.linalg.norm(rb) + 1e-12)
            best = max(best, float(np.degrees(np.arccos(np.clip(cosang, -1, 1)))))
        angs.append(best)
    order = sorted(imgs, key=lambda i: imgs[i].name)
    steps = [np.linalg.norm(C[order[j + 1]] - C[order[j]]) for j in range(len(order) - 1)]
    P = np.array([p.xyz for p in rec.points3D.values()])
    lo, hi = np.percentile(P, 5, axis=0), np.percentile(P, 95, axis=0)
    axes = np.array([A[i] for i in order])
    axes /= np.linalg.norm(axes, axis=1, keepdims=True)
    # solid angle of the union of 15-degree caps around every optical axis (Monte Carlo, fixed seed)
    rng = np.random.default_rng(SEED)
    d = rng.normal(size=(MC_DIRECTIONS, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    covered = (d @ axes.T >= np.cos(np.radians(CAP_DEG))).any(axis=1)
    mean_axis = axes.mean(axis=0)
    mean_axis /= np.linalg.norm(mean_axis) + 1e-12
    dev = np.degrees(np.arccos(np.clip(axes @ mean_axis, -1, 1)))
    cams = {}
    for i in order:
        cam = rec.cameras[imgs[i].camera_id]
        cams.setdefault(imgs[i].camera_id, dict(width=cam.width, height=cam.height, fx=float(cam.params[0]), fy=float(cam.params[1]),
                                                params=[round(float(x), 5) for x in cam.params], n_registered=0))["n_registered"] += 1
    fx = [c["fx"] for c in cams.values()]
    g = dict(tri_angle_median=float(np.median(angs)) if angs else None,
             baseline_frac=float(np.median(steps) / (np.linalg.norm(hi - lo) + 1e-12)) if steps else None,
             view_coverage_sr=float(covered.mean() * 4 * np.pi), view_spread_p90=float(np.percentile(dev, 90)),
             focal_px_est=float(np.median(fx)), focal_px_est_min=float(min(fx)), focal_px_est_max=float(max(fx)), cameras=cams)
    poses = [(imgs[i].name, *C[i].tolist(), *A[i].tolist()) for i in order]
    return g, poses


def summarise(meta, maps, base: dict, results: Path, tag: str) -> dict:
    sid, out = meta["id"], dict(base)
    sizes = sorted((m.num_reg_images() for m in maps.values()), reverse=True)
    out.update(n_models=len(maps), model_sizes=sizes)
    if maps:
        best_idx = max(maps, key=lambda k: maps[k].num_reg_images())
        best = maps[best_idx]
        n = best.num_reg_images()
        out["best_sparse_dir"] = f"{base['colmap_dir']}/sparse{tag}/{best_idx}"
        out.update(n_registered=n, reg_rate=n / meta["n_images"], n_points=best.num_points3D(),
                   reproj_err=round(best.compute_mean_reprojection_error(), 4), track_len=round(best.compute_mean_track_length(), 3))
        if n >= 3:
            g, poses = geometry(best)
            out.update(g)
            (results / ("poses" + tag)).mkdir(parents=True, exist_ok=True)
            with open(results / ("poses" + tag) / f"{sid}.csv", "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["image", "cx", "cy", "cz", "ax", "ay", "az"])
                w.writerows([[p[0]] + [f"{v:.6f}" for v in p[1:]] for p in poses])
            doc = meta.get("focal_px_doc")
            out.update(focal_px_doc=doc, focal_doc_source=meta.get("focal_doc_source"), focal_ratio=(g["focal_px_est"] / doc) if doc else None)
            if meta.get("focal_px_doc_by_group"):
                out["focal_px_doc_by_group"] = meta["focal_px_doc_by_group"]
    else:
        out.update(n_registered=0, reg_rate=0.0, n_points=0)
    n = out["n_registered"]
    out["outcome"] = "failed" if n < 3 else ("partial" if (n / meta["n_images"] < 0.5 or len(maps) > 1) else "ok")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("work", type=Path)
    ap.add_argument("results", type=Path)
    ap.add_argument("--threads", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    ap.add_argument("--mapper-threads", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=0, help="COLMAP random seed (0 = the fixed instrument; other values only for the repeat study)")
    ap.add_argument("--sfm-masks", type=Path, default=None, help="masks for feature extraction: <dir>/<image name>.png, black = no features")
    ap.add_argument("--det-test", action="store_true", help="run the mapper single-threaded twice and multi-threaded once on one database")
    a = ap.parse_args()
    global SEED
    SEED = a.seed
    meta = json.loads((a.work / "meta.json").read_text())
    sid = meta["id"]
    cwork = a.work / "colmap"
    cwork.mkdir(parents=True, exist_ok=True)
    ps = a.results / "per_sequence"
    base = dict(id=sid, n_images=meta["n_images"], camera_mode=meta["camera_mode"], threads=a.threads, mapper_threads=a.mapper_threads, seed=a.seed,
                pycolmap=pycolmap.__version__, colmap=str(pycolmap.COLMAP_version), colmap_dir=str(cwork))
    write_json(ps / f"{sid}.registration.json", dict(base, outcome="started"))  # visible even if the run is killed at the timeout
    img = a.work / "images"
    base.update({k: round(v, 1) for k, v in features_and_matches(img, cwork, meta["camera_mode"], a.threads, a.sfm_masks).items()})
    variants = [(1, "_det1a"), (1, "_det1b"), (-1, "_multi")] if a.det_test else [(a.mapper_threads, "")]
    for mt, tag in variants:
        maps, t_map = map_once(img, cwork, "sparse" + tag, mt)
        out = summarise(meta, maps, dict(base, mapper_threads=mt, t_map=round(t_map, 1),
                                         wall_s=round(base["t_extract"] + base["t_match"] + t_map, 1)), a.results, tag)
        write_json(ps / f"{sid}.registration{tag}.json", out)
        if tag == "_multi":  # the multi-threaded variant is the one reported in registration.csv
            write_json(ps / f"{sid}.registration.json", out)
            summarise(meta, maps, out, a.results, "")
        print(tag or "run", json.dumps({k: v for k, v in out.items() if k != "cameras"})[:1100], flush=True)
    os._exit(0)  # skip interpreter teardown: pycolmap/OpenBLAS can segfault there after results are written


if __name__ == "__main__":
    main()
